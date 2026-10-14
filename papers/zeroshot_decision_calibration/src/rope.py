"""
ROPE - Rank-based Operating-Point Estimation for zero-shot clinical screening.

The claim this code exists to test: when a zero-shot foundation model is reported
to "fail" at a clinical screening task, the failure is often in the DECISION RULE
(where the threshold lands) and not in the SCORE (how well it ranks patients).
A decision rule that uses only the unlabelled cohort's score ranks recovers most
of the loss, with zero labels.

Everything here is pure numpy, closed-form, and CPU-cheap.

Terminology
-----------
s        raw score, higher = more likely positive (e.g. logit difference, or the
         LLM's log P("1") - log P("0"))
native   the model's own decision, s > 0 (argmax over the two label options)
pi       prevalence (positive rate)
H        headroom: balanced accuracy available at a well-placed threshold, minus
         the balanced accuracy the native rule actually achieves
"""
import numpy as np
from math import sqrt

SQRT2 = sqrt(2.0)


# ---------------------------------------------------------------- basic metrics

def auroc(y, s):
    """Rank-based AUROC with tie correction. y in {0,1}."""
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ss = s[order]
    i = 0
    while i < len(ss):                      # average ranks within ties
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n0 * n1)


def sens_spec(y, yhat):
    y = np.asarray(y).astype(int); yhat = np.asarray(yhat).astype(int)
    p, n = (y == 1), (y == 0)
    sens = yhat[p].mean() if p.any() else np.nan
    spec = (1 - yhat[n]).mean() if n.any() else np.nan
    return sens, spec


def bacc(y, yhat):
    se, sp = sens_spec(y, yhat)
    return 0.5 * (se + sp)


def brier(y, p):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def ece(y, p, bins=10):
    """Expected calibration error, equal-width bins."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for b in range(bins):
        m = (p > edges[b]) & (p <= edges[b + 1]) if b else (p >= 0) & (p <= edges[1])
        if m.sum():
            out += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(out)


def majority_bacc(y):
    """Balanced accuracy of always predicting the majority class: always 0.5."""
    return 0.5


def majority_acc(y):
    y = np.asarray(y).astype(int)
    return float(max(y.mean(), 1 - y.mean()))


# ----------------------------------------------------------- the diagnostic (H)

def bacc_binormal(a, allow_flip=False):
    """Balanced accuracy reachable at the optimal threshold under a binormal
    (equal-variance Gaussian) model with the given AUROC.

    allow_flip=False (the default, and the only honest one for a label-free
    rule): the score is used in the direction the model already points. An
    AUROC below 0.5 therefore yields a reachable BACC below 0.5 -- no
    threshold on a monotone increasing rule can rescue an inverted score.
    allow_flip=True reports what would be reachable if the sign were known,
    which costs labels to establish and is kept only for reference."""
    if allow_flip:
        a = max(a, 1.0 - a)
    a = min(max(a, 1e-6), 1 - 1e-6)
    from statistics import NormalDist
    N = NormalDist()
    d = SQRT2 * N.inv_cdf(a)
    return float(N.cdf(d / 2.0))


def bacc_oracle(y, s):
    """Best balanced accuracy over every threshold. Exact but optimistically
    biased (the threshold is chosen on the same data), so it is an upper bound,
    reported alongside the binormal estimate rather than instead of it."""
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    n1, n0 = int(y.sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    tp = np.cumsum(ys == 1); fp = np.cumsum(ys == 0)
    b = 0.5 * (tp / n1 + 1.0 - fp / n0)
    return float(max(b.max(), 0.5))         # 0.5 from predicting all-one-class


def headroom(y, s):
    """H = reachable BACC (binormal, same sign as the model) - BACC of the
    model's native s>0 rule. `inverted` flags AUROC < 0.5, where the score
    ranks patients backwards and no label-free monotone rule can help."""
    a = auroc(y, s)
    nat = bacc(y, (np.asarray(s, float) > 0).astype(int))
    reach = bacc_binormal(a)
    return {
        "auroc": a,
        "inverted": bool(a < 0.5),
        "bacc_native": nat,
        "bacc_binormal": reach,
        "bacc_binormal_if_sign_known": bacc_binormal(a, allow_flip=True),
        "bacc_oracle": bacc_oracle(y, s),
        "H": reach - nat,
        "H_oracle": bacc_oracle(y, s) - nat,
    }


# ------------------------------------------------------- the zero-label rules

def ranks01(s):
    """Transductive rank normalisation over the unlabelled cohort -> (0,1).
    Destroys the model's arbitrary scale and offset, keeps its ordering."""
    s = np.asarray(s, float)
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), float)
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return (r + 0.5) / len(s)


def predict_quantile(s, pi, r=None):
    """ROPE-Q: call the top `pi` fraction of the cohort positive. Zero labels.
    `r` is the precomputed rank vector; pass it to avoid re-ranking in a loop."""
    pi = float(min(max(pi, 1e-6), 1 - 1e-6))
    r = ranks01(s) if r is None else r
    return (r > 1.0 - pi).astype(int)


def estimate_pi_mixture(s, iters=300, tol=1e-9, seed=0):
    """ROPE-M: prevalence from the score distribution alone - a two-component
    equal-variance Gaussian mixture fitted by EM. No labels, no outside
    knowledge. Returns the weight of the higher-mean component."""
    s = np.asarray(s, float)
    x = (s - s.mean()) / (s.std() + 1e-12)
    rng = np.random.default_rng(seed)
    lo, hi = np.quantile(x, [0.25, 0.75])
    mu = np.array([lo, hi]); var = float(x.var()) + 1e-9; w = np.array([0.5, 0.5])
    prev = None
    for _ in range(iters):
        d = np.exp(-0.5 * (x[:, None] - mu[None, :]) ** 2 / var) * w[None, :]
        d /= d.sum(1, keepdims=True) + 1e-300
        nk = d.sum(0) + 1e-12
        w = nk / len(x)
        mu = (d * x[:, None]).sum(0) / nk
        var = float((d * (x[:, None] - mu[None, :]) ** 2).sum() / len(x)) + 1e-9
        ll = float(np.log((np.exp(-0.5 * (x[:, None] - mu[None, :]) ** 2 / var)
                           * w[None, :]).sum(1) + 1e-300).sum())
        if prev is not None and abs(ll - prev) < tol:
            break
        prev = ll
    return float(w[int(np.argmax(mu))])


# ------------------------------------------- the k-label rule and the shrinkage

def fit_pi_labels(y_k):
    """Prevalence from k labels alone."""
    y_k = np.asarray(y_k).astype(int)
    return float(y_k.mean()) if len(y_k) else 0.5


def _bacc_from_pi(s_eval, y_eval, s_cohort, pi):
    """Apply the quantile rule calibrated on the whole cohort, score on y_eval."""
    thr = np.quantile(ranks01(s_cohort), 1.0 - min(max(pi, 1e-6), 1 - 1e-6))
    r = ranks01(s_cohort)
    idx = np.searchsorted(np.sort(s_cohort), s_eval, side="left")
    idx = np.clip(idx, 0, len(s_cohort) - 1)
    yhat = (np.sort(r)[idx] > thr).astype(int)
    return bacc(y_eval, yhat)


def shrink_pi_loo(s_cohort, idx_k, y_k, pi_prior, grid=None):
    """ROPE-S: blend the zero-label prevalence with the k-label one,

        pi(w) = w * pi_prior + (1 - w) * pi_labels,

    choosing w by leave-one-out balanced accuracy on the k labels only. w -> 1
    at k = 0 and is free to go to 0 as k grows, so the rule degrades gracefully
    instead of lurching between the prior and the data."""
    s_cohort = np.asarray(s_cohort, float)
    idx_k = np.asarray(idx_k, int); y_k = np.asarray(y_k).astype(int)
    k = len(y_k)
    if grid is None:
        grid = np.linspace(0.0, 1.0, 21)
    if k < 2 or y_k.sum() in (0, k):
        return 1.0, pi_prior            # cannot learn a prevalence from this
    r = ranks01(s_cohort)
    r_k = r[idx_k]
    npos = int(y_k.sum())
    best_w, best_sc = 1.0, -np.inf
    for w in grid:
        sc = 0.0
        for i in range(k):
            # prevalence from the other k-1 labels, computed without array copies
            pi_lab = (npos - y_k[i]) / (k - 1)
            pi = w * pi_prior + (1 - w) * pi_lab
            yh = 1 if r_k[i] > 1.0 - min(max(pi, 1e-6), 1 - 1e-6) else 0
            sc += 1.0 if yh == y_k[i] else 0.0
        if sc > best_sc:
            best_sc, best_w = sc, w
    pi = best_w * pi_prior + (1 - best_w) * fit_pi_labels(y_k)
    return float(best_w), float(pi)


# --------------------------------------------------------------------- reporting

def evaluate(y, s, pi_hat, name):
    """Score one decision rule. `pi_hat` None means use the model's native s>0."""
    y = np.asarray(y).astype(int)
    yhat = (np.asarray(s, float) > 0).astype(int) if pi_hat is None \
        else predict_quantile(s, pi_hat)
    se, sp = sens_spec(y, yhat)
    return {"rule": name, "pi_used": pi_hat, "bacc": bacc(y, yhat),
            "sens": se, "spec": sp, "acc": float((yhat == y).mean()),
            "pred_pos_rate": float(yhat.mean())}


def bootstrap_ci(fn, y, s, n=2000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for a statistic fn(y, s)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); s = np.asarray(s); m = len(y)
    vals = []
    for _ in range(n):
        i = rng.integers(0, m, m)
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(fn(y[i], s[i]))
    vals = np.array([v for v in vals if np.isfinite(v)])
    if not len(vals):
        return (np.nan, np.nan)
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))
