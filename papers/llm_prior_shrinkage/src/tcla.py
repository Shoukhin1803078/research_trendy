"""Faithful NumPy reimplementation of TCLA (arXiv:2607.09562), transcribed from the
authors' code at github.com/SkyCol/TCLA, not from the paper prose.

Provenance of every constant, so deviations are auditable:
  layers (5,7,9,11)            methods/rlc.py default
  beta_base = 10.0             methods/rlc.py default (sharp kernel)
  reliability_gamma = 5.0      methods/rlc.py compute_prototype_reliabilities
  lambda_reg = 1.0             methods/rlc.py fit_tcla_proto_kernel_residual
  eta clamp [0, 3]             same
  standardize_kernel = False   same (default)
  top_k layers = 4, force 11   methods/prototypes.py select_best_layers
  lf_mode aggressive           sharp kernel, correction_strength = 1.0
  lf_mode conservative         reliable kernel, strength = shot/(shot+1)
  metric                       balanced accuracy (utils/metrics.py)
  protocol                     k-shot per class sampled from ALL data; test = the
                               remainder (utils/data.py FewShotImageFolder). There is
                               no fixed held-out test split.
"""
import numpy as np


def _n(X, eps=1e-12):
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), eps)


# ---------- prototypes ----------

def mahalanobis_weighted_prototype(feats_c):
    """RobustPrototype.compute_mahalanobis_weighted. 1-shot returns the feature itself."""
    F = _n(feats_c)
    if F.shape[0] == 1:
        return F[0]
    mean = F.mean(0, keepdims=True)
    cov = np.cov(F.T) + 1e-6 * np.eye(F.shape[1])
    inv = np.linalg.pinv(cov)
    diff = F - mean
    mahal = np.sqrt(np.maximum(((diff @ inv) * diff).sum(1), 0.0))
    w = np.exp(-mahal - np.max(-mahal))          # softmax(-mahal), shifted for stability
    w /= w.sum()
    return _n((w[:, None] * F).sum(0))


def build_prototypes(feats, ys, C):
    P = np.zeros((C, feats.shape[1]), dtype=np.float64)
    for c in range(C):
        m = ys == c
        if m.any():
            P[c] = mahalanobis_weighted_prototype(feats[m].astype(np.float64))
    return _n(P)


# ---------- layer selection ----------

def layer_separability(protos):
    """score(l) = 1 - mean_{i<j} cos(p_i, p_j).  Higher = better separated."""
    P = _n(protos); C = P.shape[0]
    S = P @ P.T
    off = [S[i, j] for i in range(C) for j in range(i + 1, C)]
    return 1.0 - float(np.mean(off)) if off else 1e-6


def select_layers_and_weights(layer_protos, top_k=4, final_layer=11, tau=1.0):
    """select_best_layers + compute_adaptive_layer_weights."""
    scores = {l: layer_separability(P) for l, P in layer_protos.items()}
    others = sorted([(l, s) for l, s in scores.items() if l != final_layer],
                    key=lambda x: -x[1])
    sel = [l for l, _ in others[:max(top_k - 1, 0)]]
    if final_layer in scores:
        sel.append(final_layer)
    w = np.maximum(np.array([scores.get(l, 1e-6) for l in sel], dtype=np.float64), 1e-6)
    w = tau * w; w /= w.sum()
    return sel, w, scores


# ---------- reliability ----------

def prototype_reliabilities(layer_feats_sup, ys, protos_by_layer, sel, gamma=5.0, eps=1e-6):
    """r[l,c] = ((K_c-1)/K_c) * exp(-gamma * mean ||z_i - p_c||^2), clipped to [0,1]."""
    C = protos_by_layer[sel[0]].shape[0]
    R = np.zeros((len(sel), C))
    for li, l in enumerate(sel):
        P = _n(protos_by_layer[l])
        for c in range(C):
            m = ys == c
            k = int(m.sum())
            if k <= 1:
                R[li, c] = 0.0
                continue
            Fc = _n(layer_feats_sup[l][m].astype(np.float64))
            compact = ((Fc - P[c][None]) ** 2).sum(1).mean()
            R[li, c] = np.clip(((k - 1.0) / (k + eps)) * np.exp(-gamma * compact), 0.0, 1.0)
    return R


# ---------- kernel ----------

def proto_kernel(layer_feats, protos_by_layer, sel, weights, rel=None,
                 beta=10.0, kernel_mode="sharp"):
    """K_PROTO [N, C] = sum_l w_l * K_l, where
         sharp    K = exp(beta*(cos-1))
         reliable K = (1-r)*0.5*(cos+1) + r*sharp"""
    K = None
    for li, l in enumerate(sel):
        Fl = _n(layer_feats[l].astype(np.float64))
        Pl = _n(protos_by_layer[l])
        A = Fl @ Pl.T
        sharp = np.exp(beta * (A - 1.0))
        if kernel_mode == "sharp":
            Kl = sharp
        elif kernel_mode == "reliable":
            r = np.zeros((1, A.shape[1])) if rel is None else rel[li][None, :]
            Kl = (1.0 - r) * (0.5 * (A + 1.0)) + r * sharp
        else:
            raise ValueError(kernel_mode)
        K = weights[li] * Kl if K is None else K + weights[li] * Kl
    return K


# ---------- text prototypes (PDA) ----------

def pda_text_prototypes(text_feats_per_prompt, layer_protos_proj, sel, weights, beta_t=10.0):
    """t_c^(l) = sum_j softmax_j(beta_t * s_{c,j}^(l)) t_{c,j};  t_c = sum_l w_l t_c^(l).
    s_{c,j}^(l) = cos(prompt j of class c, projected class prototype at layer l)."""
    C = len(text_feats_per_prompt)
    D = text_feats_per_prompt[0].shape[1]
    out = np.zeros((C, D))
    for li, l in enumerate(sel):
        Pp = _n(layer_protos_proj[l])
        for c in range(C):
            T = _n(text_feats_per_prompt[c].astype(np.float64))   # [P, D]
            s = T @ Pp[c]
            w = np.exp(beta_t * (s - s.max())); w /= w.sum()
            out[c] += weights[li] * (w[:, None] * T).sum(0)
    return _n(out)


# ---------- RLC: closed-form residual logit correction ----------

def fit_rlc(Z_zs_sup, K_sup, ys_sup, C, lambda_reg=1.0, eps=1e-6,
            standardize_kernel=False, calibrate_eta=True, eta_min=0.0, eta_max=3.0):
    """R = Y - Z_zs;  W = (K'K + lambda I)^-1 K'R;  eta = <R,KW>/||KW||^2 clamped."""
    Y = np.eye(C)[ys_sup]
    R = Y - Z_zs_sup
    if standardize_kernel:
        km = K_sup.mean(0, keepdims=True)
        ks = np.maximum(K_sup.std(0, keepdims=True), eps)
        Kf = (K_sup - km) / ks
    else:
        km = np.zeros((1, K_sup.shape[1])); ks = np.ones((1, K_sup.shape[1])); Kf = K_sup
    W = np.linalg.solve(Kf.T @ Kf + lambda_reg * np.eye(Kf.shape[1]), Kf.T @ R)
    Rh = Kf @ W
    if calibrate_eta:
        eta = float(np.clip((R * Rh).sum() / max((Rh ** 2).sum(), eps), eta_min, eta_max))
    else:
        eta = 1.0
    q = float(np.clip(1.0 - ((R - eta * Rh) ** 2).sum() / max((R ** 2).sum(), eps), 0.0, 1.0))
    return {"W": W, "k_mean": km, "k_std": ks, "eta": eta, "q": q,
            "standardize_kernel": standardize_kernel}


def apply_rlc(Z_zs, K, state, correction_strength=1.0):
    Ku = (K - state["k_mean"]) / state["k_std"] if state["standardize_kernel"] else K
    return Z_zs + float(correction_strength) * state["eta"] * (Ku @ state["W"])


def lf_mode_config(lf_mode, shot):
    if lf_mode == "aggressive":
        return "sharp", 1.0
    if lf_mode == "conservative":
        return "reliable", shot / (shot + 1.0)
    raise ValueError(lf_mode)


# ================= Phase B: ours =================
#
# Diagnosis from the Phase A reproduction: TCLA computes
#     Z = Z_ZS + mu * eta * K W,   W fit to the residual R = Y - Z_ZS
# which anchors every prediction to the zero-shot logits with a hard-coded coefficient of
# 1.0. On BUSI the zero-shot signal is weak (BACC 41.3) while the prototype kernel is strong
# (73.6), so from 2-shot upward the anchor is pure drag and deleting RLC outright beats TCLA.
#
# Fix: put Z_ZS in the design matrix as a FEATURE instead of fixing it as an offset, and
# choose the ridge penalty by analytic leave-one-out rather than hard-coding lambda = 1.0.
# This strictly generalises both baselines - the fit can recover TCLA (coefficient ~1 on
# Z_ZS) or proto_only (coefficient ~0) - and decides per task and per shot.

def _ridge_loo_error(X, Y, lam):
    """Leave-one-out squared error for ridge, in closed form via the hat matrix.

    For ridge, LOO residuals are e_i / (1 - h_ii) where h_ii is the leverage, so the whole
    LOO curve costs one solve per lambda and no refitting. That is what makes per-task
    lambda selection free here.
    """
    n, p = X.shape
    G = X.T @ X + lam * np.eye(p)
    try:
        Ginv = np.linalg.inv(G)
    except np.linalg.LinAlgError:
        Ginv = np.linalg.pinv(G)
    B = Ginv @ (X.T @ Y)
    H_diag = np.einsum("ij,jk,ik->i", X, Ginv, X)       # leverages
    denom = np.clip(1.0 - H_diag, 1e-6, None)[:, None]
    resid = (Y - X @ B) / denom
    return float((resid ** 2).sum()), B


def fit_anchor_ridge(Z_zs_sup, K_sup, ys_sup, C, lambdas=None, use_bias=True,
                     include_zs=True):
    """OURS. Joint ridge over [K | Z_ZS | 1] -> one-hot Y, lambda by analytic LOO.

    include_zs=False reduces to a prototype-kernel-only ridge, which is the ablation that
    isolates what the zero-shot features contribute.
    """
    Y = np.eye(C)[ys_sup]
    parts = [K_sup]
    if include_zs:
        parts.append(Z_zs_sup)
    if use_bias:
        parts.append(np.ones((len(K_sup), 1)))
    X = np.concatenate(parts, axis=1)
    # scale-free grid; K and Z_ZS live on different scales so standardise columns
    mu = X.mean(0, keepdims=True); sd = np.maximum(X.std(0, keepdims=True), 1e-8)
    if use_bias:
        mu[0, -1] = 0.0; sd[0, -1] = 1.0            # leave the bias column alone
    Xs = (X - mu) / sd
    if lambdas is None:
        lambdas = np.logspace(-4, 3, 22)
    best = None
    for lam in lambdas:
        err, B = _ridge_loo_error(Xs, Y, float(lam))
        if best is None or err < best[0]:
            best = (err, float(lam), B)
    _, lam, B = best
    return {"B": B, "mu": mu, "sd": sd, "lam": lam, "use_bias": use_bias,
            "include_zs": include_zs, "n_basis": Xs.shape[1]}


def apply_anchor_ridge(Z_zs, K, state):
    parts = [K]
    if state["include_zs"]:
        parts.append(Z_zs)
    if state["use_bias"]:
        parts.append(np.ones((len(K), 1)))
    X = np.concatenate(parts, axis=1)
    return ((X - state["mu"]) / state["sd"]) @ state["B"]
