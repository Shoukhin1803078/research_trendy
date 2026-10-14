"""
AUDIT A1 - is the headroom validation circular?

H        = reachable(AUROC_test) - BACC_native_test
gain     = BACC_fitted_test      - BACC_native_test

Both are differences from the SAME baseline term, measured on the SAME test set.
Any spread in BACC_native therefore induces correlation between them before the
diagnostic does any work. This script measures how much of the reported
correlation survives once that shared term is removed.
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
import rope

rows = json.load(open(os.path.join(HERE, "results", "budget.json")))
ok = [r for r in rows if not r["inverted"]]
print(f"endpoints with AUROC > 0.5: n = {len(ok)}  (all 9: {len(rows)})\n")


def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


def spearman(a, b):
    return pearson(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))


def perm_p(a, b, stat, n=200000, seed=0):
    """Exact-ish two-sided permutation p-value. With n<=8 this enumerates."""
    from itertools import permutations
    a = np.asarray(a, float); b = np.asarray(b, float)
    obs = stat(a, b)
    if len(a) <= 8:
        alls = [stat(a, np.array(p)) for p in permutations(b)]
        return float(np.mean(np.abs(alls) >= abs(obs) - 1e-12)), len(alls)
    rng = np.random.default_rng(seed)
    cnt = sum(abs(stat(a, rng.permutation(b))) >= abs(obs) - 1e-12 for _ in range(n))
    return float(cnt / n), n


def boot_ci(a, b, stat, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float); b = np.asarray(b, float)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(a), len(a))
        if len(np.unique(a[i])) < 3 or len(np.unique(b[i])) < 3:
            continue
        vals.append(stat(a[i], b[i]))
    vals = np.array(vals)
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)), len(vals)


for k in ("8", "32", "64"):
    native = np.array([r["bacc_native"] for r in ok])
    reach = np.array([r["bacc_native"] + r["H"] for r in ok])      # reachable
    achieved = np.array([r["shots"][k]["thresh_k"]["mean"] for r in ok])
    H = reach - native
    gain = achieved - native

    print(f"=== k = {k} ===")
    print(f"  spread of the shared baseline BACC_native: "
          f"min {native.min():.3f} max {native.max():.3f} sd {native.std():.3f}")

    r1, s1 = pearson(H, gain), spearman(H, gain)
    p1, n1 = perm_p(H, gain, spearman)
    lo1, hi1, _ = boot_ci(H, gain, pearson)
    print(f"  AS REPORTED   H vs gain          pearson {r1:+.3f} [{lo1:+.3f},{hi1:+.3f}]"
          f"   spearman {s1:+.3f}  perm p = {p1:.4f} (over {n1} perms)")

    # (a) remove the shared term: correlate the two LEVELS, not the differences
    r2, s2 = pearson(reach, achieved), spearman(reach, achieved)
    p2, _ = perm_p(reach, achieved, spearman)
    lo2, hi2, _ = boot_ci(reach, achieved, pearson)
    print(f"  SHARED TERM OUT  reachable vs achieved  pearson {r2:+.3f} "
          f"[{lo2:+.3f},{hi2:+.3f}]   spearman {s2:+.3f}  perm p = {p2:.4f}")

    # (b) partial correlation of H and gain, controlling for native
    def resid(v, x):
        x = np.asarray(x, float); v = np.asarray(v, float)
        A = np.vstack([np.ones_like(x), x]).T
        beta, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ beta
    rp = pearson(resid(H, native), resid(gain, native))
    print(f"  PARTIAL (control native)  pearson {rp:+.3f}")

    # (c) how much correlation would pure noise produce, given this baseline spread?
    rng = np.random.default_rng(0)
    null = []
    for _ in range(20000):
        fake_reach = rng.permutation(reach)        # break the link to the endpoint
        fake_gain = rng.permutation(achieved) - native
        null.append(pearson(fake_reach - native, fake_gain))
    null = np.array(null)
    print(f"  NULL from shared baseline alone: mean {null.mean():+.3f}, "
          f"95th pct {np.quantile(null,0.95):+.3f}, "
          f"P(null >= observed) = {np.mean(null >= r1):.4f}")
    print()
