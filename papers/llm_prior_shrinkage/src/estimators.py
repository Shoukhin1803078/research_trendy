"""Training-free few-shot classifiers on frozen features. All closed-form, CPU, no gradients.

Baselines reproduce the published family; the last is ours.

IMPORTANT structural constraint, handled explicitly rather than hidden:
Psi(G) = sum_{i<j} G_ij (t_i - t_j)(t_i - t_j)^T has rank <= C-1 (it lives in the span of
class-embedding differences). With d=512 and C=5 that is rank <= 4. So Psi CANNOT replace
the isotropic shrinkage target - (1-a)S + a*Psi would still be singular. It must AUGMENT it:

    Sigma = (1-alpha) * S_hat  +  alpha * [ (1-gamma) * I  +  gamma * Psi_norm ]

The claim is therefore not "replace the identity prior" but "the identity prior is not the
only option, and adding the label geometry to it helps where the labels are ordered."
"""
import numpy as np
from scipy.linalg import pinvh


def _norm_rows(X, eps=1e-12):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), eps)



def _alpha(alpha, S, X, ys, C):
    """Shrinkage weight, guarded against the k=1 case.

    The earlier _scale() fix was INCOMPLETE. At k=1 the within-class scatter S is exactly
    zero AND Ledoit-Wolf on all-zero centred data returns alpha=0, so
    Sigma = (1-0)*0 + 0*mu*I = 0 is STILL the zero matrix: pinvh gives 0, every logit is 0,
    and the classifier silently degenerates to a constant predictor (QWK exactly 0.0000,
    one distinct predicted class). When S carries no information alpha must be 1 - the
    prior is all there is.
    """
    if np.trace(S) <= 1e-12:
        return 1.0
    if alpha is None:
        from sklearn.covariance import ledoit_wolf
        _, alpha = ledoit_wolf(X - class_means(X, ys, C)[ys], assume_centered=True)
    if not np.isfinite(alpha) or alpha <= 0:
        return 1e-3
    return float(min(alpha, 1.0))


def _scale(S, Xsn, d):
    """Isotropic scale for the shrinkage target.

    trace(S)/d is the natural choice, but at k=1 every class has one sample so S is
    EXACTLY zero, mu becomes 0, Sigma becomes the zero matrix, pinvh gives 0, and every
    logit is 0 -> the classifier silently degenerates to predicting class 0 for all
    inputs. (That bug produced a spurious "accuracy 0.5000, AUROC 0.5000, QWK 0.0000" on
    dr_regular, whose test split happens to be 50% class 0.)
    Fall back to the TOTAL scatter scale, which is well defined for any k >= 1.
    """
    mu = float(np.trace(S) / d)
    if mu > 1e-12:
        return mu
    Z = Xsn - Xsn.mean(0, keepdims=True)
    mu_tot = float((Z * Z).sum() / (max(len(Xsn) - 1, 1) * d))
    return mu_tot if mu_tot > 1e-12 else 1.0 / d


def class_means(Xs, ys, C):
    d = Xs.shape[1]
    M = np.zeros((C, d))
    for c in range(C):
        m = ys == c
        M[c] = Xs[m].mean(0) if m.any() else 0.0
    return M


def pooled_within_scatter(Xs, ys, C):
    d = Xs.shape[1]
    S = np.zeros((d, d)); n = 0
    for c in range(C):
        Xc = Xs[ys == c]
        if len(Xc) < 2:
            continue
        Z = Xc - Xc.mean(0, keepdims=True)
        S += Z.T @ Z; n += len(Xc) - 1
    return S / max(n, 1)


# ---------------- baselines ----------------

def zero_shot(Xq, text_feats, **_):
    """BiomedCLIP zero-shot: cosine similarity to class text embeddings."""
    return _norm_rows(Xq) @ _norm_rows(text_feats).T


def proto_adapter(Xs, ys, Xq, C, **_):
    """Proto-Adapter / nearest class mean on normalized features."""
    M = _norm_rows(class_means(_norm_rows(Xs), ys, C))
    return _norm_rows(Xq) @ M.T


def tip_adapter(Xs, ys, Xq, C, beta=5.5, alpha=1.0, text_feats=None, **_):
    """Tip-Adapter (ECCV'22), training-free: key-value cache + zero-shot prior.
    A = exp(-beta(1 - Xq @ Xs^T)); logits = A @ onehot(ys) * alpha + zero_shot."""
    Xsn, Xqn = _norm_rows(Xs), _norm_rows(Xq)
    A = np.exp(-beta * (1.0 - Xqn @ Xsn.T))
    L = np.eye(C)[ys]
    cache_logits = A @ L
    if text_feats is not None:
        return 100.0 * (Xqn @ _norm_rows(text_feats).T) + alpha * cache_logits
    return cache_logits


def linear_probe(Xs, ys, Xq, C, Creg=1.0, seed=0, **_):
    """Logistic regression on frozen features. Convex, but iterative - included because it
    is the benchmark's reference adapter."""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, C=Creg, random_state=seed)
    clf.fit(_norm_rows(Xs), ys)
    P = clf.predict_proba(_norm_rows(Xq))
    full = np.zeros((len(Xq), C)); full[:, clf.classes_] = P
    return full


def gda_ledoit_wolf(Xs, ys, Xq, C, alpha=None, **_):
    """Gaussian discriminant analysis with ISOTROPIC shrinkage toward I.
    This is the estimator the 2026 closed-form adapters rest on, and the target of Claim 1."""
    Xsn, Xqn = _norm_rows(Xs), _norm_rows(Xq)
    S = pooled_within_scatter(Xsn, ys, C)
    d = S.shape[0]
    alpha = _alpha(alpha, S, Xsn, ys, C)
    mu = _scale(S, Xsn, d)
    Sig = (1.0 - alpha) * S + alpha * mu * np.eye(d)
    P = pinvh(Sig)
    M = class_means(Xsn, ys, C)
    # -0.5 * Mahalanobis, equal priors
    return np.stack([-0.5 * np.einsum("ij,jk,ik->i", Xqn - M[c], P, Xqn - M[c])
                     for c in range(C)], axis=1)


# ---------------- ours ----------------

def ordinal_graph(C, decay="inverse", mode="adjacent"):
    """G_ij large for ADJACENT classes, small for distant ones.

    Rationale: adjacent-grade distinctions are the intrinsically confusable, fine-grained
    ones. Putting large prior variance along their difference directions makes the
    Mahalanobis metric DOWN-weight them, so the classifier leans on the global severity
    axis instead. That trades off-by-one errors (clinically tolerable) against
    maximal-distance errors (catastrophic) - exactly what QWK and MAE reward.
    """
    G = np.zeros((C, C))
    for i in range(C):
        for j in range(C):
            if i == j:
                continue
            dist = abs(i - j)
            if mode == "adjacent":      # weight ~ 1/dist: adjacent pairs dominate
                w = 1.0 / dist if decay == "inverse" else np.exp(-(dist - 1))
            elif mode == "distant":     # inverted: distant pairs dominate
                w = float(dist)
            elif mode == "uniform":     # control: no ordinal information at all
                w = 1.0
            else:
                raise ValueError(mode)
            G[i, j] = w
    # normalize so gamma means the same thing across modes
    return G / G.sum()


def psi_from_graph(text_feats, G):
    """Psi = sum_{i<j} G_ij (t_i - t_j)(t_i - t_j)^T.  PSD, rank <= C-1, trace-normalized."""
    T = _norm_rows(text_feats)
    C, d = T.shape
    Psi = np.zeros((d, d))
    for i in range(C):
        for j in range(i + 1, C):
            v = (T[i] - T[j])[:, None]
            Psi += G[i, j] * (v @ v.T)
    tr = np.trace(Psi)
    return Psi * (d / tr) if tr > 1e-12 else Psi


def gda_structured(Xs, ys, Xq, C, text_feats, alpha=None, gamma=0.5,
                   G=None, **_):
    """OURS: GDA whose shrinkage target carries the label geometry.

        Sigma = (1-alpha) S  +  alpha [ (1-gamma) I  +  gamma Psi_norm ]

    gamma=0 reduces exactly to gda_ledoit_wolf, which makes the ablation clean:
    any gain is attributable to the geometry term alone.
    """
    Xsn, Xqn = _norm_rows(Xs), _norm_rows(Xq)
    S = pooled_within_scatter(Xsn, ys, C)
    d = S.shape[0]
    alpha = _alpha(alpha, S, Xsn, ys, C)
    if G is None:
        G = ordinal_graph(C)
    Psi = psi_from_graph(text_feats, G)
    mu = _scale(S, Xsn, d)
    target = (1.0 - gamma) * np.eye(d) + gamma * (Psi / d)
    Sig = (1.0 - alpha) * S + alpha * mu * target
    P = pinvh(Sig)
    M = class_means(Xsn, ys, C)
    return np.stack([-0.5 * np.einsum("ij,jk,ik->i", Xqn - M[c], P, Xqn - M[c])
                     for c in range(C)], axis=1)


if __name__ == "__main__":
    # Structural sanity checks, no data needed.
    rng = np.random.default_rng(0)
    C, k, d = 5, 8, 512
    Xs = rng.standard_normal((C * k, d)); ys = np.repeat(np.arange(C), k)
    Xq = rng.standard_normal((40, d)); T = rng.standard_normal((C, d))

    G = ordinal_graph(C)
    print("ordinal graph G (row 0 = class 0 vs others):", np.round(G[0], 3))
    assert G[0, 1] > G[0, 4], "adjacent must outweigh distant"

    Psi = psi_from_graph(T, G)
    ev = np.linalg.eigvalsh(Psi)
    rank = int((ev > 1e-8 * ev.max()).sum())
    print(f"Psi: PSD={ev.min() > -1e-8}, rank={rank}, C-1={C-1}, trace/d={np.trace(Psi)/d:.3f}")
    assert rank <= C - 1, "Psi rank must be <= C-1"
    print("  -> confirms Psi alone is rank-deficient, so it must augment I, not replace it")

    for name, fn in [("zero_shot", lambda: zero_shot(Xq, T)),
                     ("proto_adapter", lambda: proto_adapter(Xs, ys, Xq, C)),
                     ("tip_adapter", lambda: tip_adapter(Xs, ys, Xq, C, text_feats=T)),
                     ("linear_probe", lambda: linear_probe(Xs, ys, Xq, C)),
                     ("gda_ledoit_wolf", lambda: gda_ledoit_wolf(Xs, ys, Xq, C)),
                     ("gda_structured", lambda: gda_structured(Xs, ys, Xq, C, T))]:
        L = fn()
        assert L.shape == (40, C), f"{name} bad shape {L.shape}"
        print(f"  {name:<18} ok  logits{L.shape}  finite={np.isfinite(L).all()}")

    # gamma=0 must reduce EXACTLY to the isotropic baseline
    a = gda_ledoit_wolf(Xs, ys, Xq, C, alpha=0.3)
    b = gda_structured(Xs, ys, Xq, C, T, alpha=0.3, gamma=0.0)
    print(f"\ngamma=0 reduces to Ledoit-Wolf baseline: max|diff|={np.abs(a-b).max():.2e}")
    assert np.allclose(a, b, atol=1e-6), "gamma=0 must reduce to the baseline exactly"
    print("ALL STRUCTURAL CHECKS PASSED")


# ---------------- decision-level ordinal methods (second attempt at Claim 2) ----------------
#
# The Psi(G) route failed because it passes the ordering through a text tower whose class
# embeddings are near-collinear (cos 0.909-0.994), so the difference vectors carry no usable
# signal. These methods use the SAME LLM-supplied ordering but never touch the embeddings.
#
# Mechanism, and why it should help rather than just reshuffle: a "grade > j" split POOLS
# classes, so each binary problem sees far more samples per side than the C-way problem.
# With C=5 and k=8 the C-way problem gives 8 samples per class (within-class scatter rank
# <= 35 of 512); the binary split gives 16 vs 24 (rank <= 38) but over only TWO means, so
# the per-parameter conditioning is much better. This attacks the degeneracy directly.

def _binary_gda_scores(Xs, ys_bin, Xq, alpha=None):
    """Signed score for a single 'grade > j' discriminant. Shared-covariance GDA, 2 classes."""
    S = pooled_within_scatter(Xs, ys_bin, 2)
    d = S.shape[0]
    alpha = _alpha(alpha, S, Xs, ys_bin, 2)
    mu = _scale(S, Xs, d)
    P = pinvh((1.0 - alpha) * S + alpha * mu * np.eye(d))
    M = class_means(Xs, ys_bin, 2)
    dlo = np.einsum("ij,jk,ik->i", Xq - M[0], P, Xq - M[0])
    dhi = np.einsum("ij,jk,ik->i", Xq - M[1], P, Xq - M[1])
    return 0.5 * (dlo - dhi)          # >0 means 'grade > j'


def ordinal_frank_hall(Xs, ys, Xq, C, order=None, **_):
    """Frank & Hall monotone reduction over the LLM-supplied class order.

    Builds C-1 binary discriminants P(rank > j), forces monotonicity by cumulative
    minimum, then differences them into class probabilities. `order` is the permutation
    the LLM provides; passing a wrong order should destroy performance, which is the
    ablation that proves the ordering is what carries the information.
    """
    Xsn, Xqn = _norm_rows(Xs), _norm_rows(Xq)
    order = np.arange(C) if order is None else np.asarray(order)
    rank_of = np.empty(C, dtype=int)
    rank_of[order] = np.arange(C)          # class label -> position on the scale
    r = rank_of[ys]

    surv = np.ones((len(Xqn), C + 1))      # P(rank > j), j = -1..C-1
    for j in range(C - 1):
        yb = (r > j).astype(int)
        if yb.min() == yb.max():           # degenerate split, fall back to the prior
            surv[:, j + 1] = float(yb.max())
            continue
        s = _binary_gda_scores(Xsn, yb, Xqn)
        surv[:, j + 1] = 1.0 / (1.0 + np.exp(-s))
    surv[:, 0] = 1.0
    surv[:, C] = 0.0
    # enforce monotone non-increasing survival (isotonic in the simplest form)
    surv = np.minimum.accumulate(surv, axis=1)
    probs_by_rank = surv[:, :-1] - surv[:, 1:]          # P(rank == j)
    probs_by_rank = np.clip(probs_by_rank, 1e-12, None)
    probs_by_rank /= probs_by_rank.sum(1, keepdims=True)
    out = np.zeros_like(probs_by_rank)
    out[:, order] = probs_by_rank                        # map ranks back to class labels
    return np.log(out)


def ordinal_expected_grade(Xs, ys, Xq, C, base=None, order=None, **_):
    """Rank-aware decoding of an existing classifier: predict round(E[rank]).

    Parameter-free, so there is nothing to tune and no leakage. Converts a confident-but-
    wrong argmax into a hedged middle grade, which is what MAE and QWK reward.
    """
    base = base or gda_ledoit_wolf
    L = base(Xs=Xs, ys=ys, Xq=Xq, C=C)
    Z = L - L.max(1, keepdims=True)
    Pp = np.exp(Z); Pp /= Pp.sum(1, keepdims=True)
    order = np.arange(C) if order is None else np.asarray(order)
    rank_of = np.empty(C, dtype=int); rank_of[order] = np.arange(C)
    exp_rank = Pp @ rank_of.astype(float)
    pred_rank = np.clip(np.rint(exp_rank), 0, C - 1).astype(int)
    pred_class = order[pred_rank]
    out = np.full((len(Xq), C), -1e3)
    out[np.arange(len(Xq)), pred_class] = 0.0
    return out


def ordinal_frank_hall_shuffled(Xs, ys, Xq, C, seed=0, **_):
    """ABLATION: Frank-Hall with a WRONG class order.

    If the LLM-supplied ordering is what carries the information, this must perform much
    worse than ordinal_frank_hall. If it performs the same, the gain is from the binary
    pooling alone and the LLM contributes nothing - which we would have to report.
    """
    rng = np.random.default_rng(1000 + seed)
    bad = rng.permutation(C)
    while C > 2 and np.array_equal(bad, np.arange(C)):
        bad = rng.permutation(C)
    return ordinal_frank_hall(Xs, ys, Xq, C, order=bad)


def majority_class(Xs, ys, Xq, C, **_):
    """TRIVIAL BASELINE: always predict the most frequent class in the support set.

    Omitted from the first results table, which was a mistake: on dr_regular the test
    split is 50% grade 0, so a constant predictor scores 0.500 accuracy and several
    "real" methods do not beat it. Any accuracy number must be read against this line.
    """
    counts = np.bincount(ys, minlength=C)
    out = np.full((len(Xq), C), -1e3)
    out[:, int(counts.argmax())] = 0.0
    return out
