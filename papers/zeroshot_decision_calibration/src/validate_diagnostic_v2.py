"""
P3/S3 - does the headroom estimate predict the realised gain, stated honestly?

Fixes three things the audit found in the original validation:

 1. H and the realised gain share the term -BACC_native and are computed on the
    same test split, so part of their correlation is mechanical. We therefore
    report BOTH the difference form (as before) and the level form
    (`reachable` vs `achieved`), which has no shared term.
 2. The original used 6 usable endpoints. We use every endpoint the cached
    features support, with a minimum of 10 positives.
 3. Endpoints are NOT independent: several are nested cut-points on one label
    column. We bootstrap by CLUSTER (dataset::label column) and also report a
    `primary` subset of one endpoint per column.
"""
import json, os, sys
from itertools import permutations
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(HERE, "results")

d = json.load(open(f"{R}/endpoints_v2.json"))
eps = [e for e in d["endpoints"] if e["auroc"] > 0.5]


def pearson(a, b):
    a = np.asarray(a, float) - np.mean(a); b = np.asarray(b, float) - np.mean(b)
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-12 else np.nan


def spearman(a, b):
    return pearson(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))


def perm_p(a, b, stat):
    a = np.asarray(a, float); b = np.asarray(b, float)
    obs = stat(a, b)
    if len(a) <= 8:
        vals = [stat(a, np.array(p)) for p in permutations(b)]
        return float(np.mean(np.abs(vals) >= abs(obs) - 1e-12)), len(vals)
    rng = np.random.default_rng(0)
    n = 100000
    c = sum(abs(stat(a, rng.permutation(b))) >= abs(obs) - 1e-12 for _ in range(n))
    return float(c / n), n


def cluster_boot(eps_, xf, yf, stat, n=20000, seed=0):
    """Resample whole label columns, not individual endpoints."""
    rng = np.random.default_rng(seed)
    cl = sorted(set(e["cluster"] for e in eps_))
    by = {c: [e for e in eps_ if e["cluster"] == c] for c in cl}
    vals = []
    for _ in range(n):
        pick = [by[cl[i]] for i in rng.integers(0, len(cl), len(cl))]
        pool = [e for grp in pick for e in grp]
        if len(pool) < 3:
            continue
        x = [xf(e) for e in pool]; y = [yf(e) for e in pool]
        if len(set(x)) < 3 or len(set(y)) < 3:
            continue
        v = stat(x, y)
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < 100:
        return (np.nan, np.nan, len(vals))
    return float(np.quantile(vals, .025)), float(np.quantile(vals, .975)), len(vals)


def report(eps_, label, k):
    H = [e["H"] for e in eps_]
    gain = [e["shots"][k]["mean"] - e["bacc_native"] for e in eps_]
    reach = [e["bacc_binormal"] for e in eps_]
    ach = [e["shots"][k]["mean"] for e in eps_]
    ncl = len(set(e["cluster"] for e in eps_))
    print(f"\n  {label}  (n = {len(eps_)} endpoints, {ncl} label columns, k = {k})")

    for nm, x, y in (("H vs gain          (shared -BACC_native)", H, gain),
                     ("reachable vs achieved (no shared term) ", reach, ach)):
        rp, sp = pearson(x, y), spearman(x, y)
        p, nperm = perm_p(x, y, spearman)
        if nm.startswith("H vs"):
            lo, hi, nb = cluster_boot(eps_, lambda e: e["H"],
                                      lambda e: e["shots"][k]["mean"] - e["bacc_native"],
                                      pearson)
        else:
            lo, hi, nb = cluster_boot(eps_, lambda e: e["bacc_binormal"],
                                      lambda e: e["shots"][k]["mean"], pearson)
        sig = "" if p < 0.05 else "   NOT SIGNIFICANT"
        print(f"    {nm}  pearson {rp:+.3f} cluster-CI [{lo:+.3f},{hi:+.3f}]"
              f"   spearman {sp:+.3f}  perm p = {p:.4f}{sig}")


if __name__ == "__main__":
    print("=== Headroom validation, expanded endpoint set ===")
    print(f"  kept {len(d['endpoints'])} endpoints, {len(eps)} with AUROC > 0.5, "
          f"{len(d['skipped'])} skipped for too few positives")
    for nm, why in d["skipped"]:
        print(f"    skipped: {nm} ({why})")

    prim = [e for e in eps if e["primary"]]
    for k in ("8", "64"):
        report(eps, "ALL usable endpoints", k)
        report(prim, "PRIMARY only (one per label column)", k)

    print("\n=== Sign agreement: does H > +0.02 predict a gain? ===")
    for k in ("8", "64"):
        tp = sum(1 for e in eps if e["H"] > 0.02 and
                 e["shots"][k]["mean"] - e["bacc_native"] > 0)
        npos = sum(1 for e in eps if e["H"] > 0.02)
        tn = sum(1 for e in eps if e["H"] <= 0.02 and
                 e["shots"][k]["mean"] - e["bacc_native"] <= 0.02)
        nneg = sum(1 for e in eps if e["H"] <= 0.02)
        print(f"  k={k:>2}: of {npos} endpoints with H>+0.02, {tp} gained; "
              f"of {nneg} with H<=+0.02, {tn} did not")

    print("\n=== Per-endpoint detail (k=64) ===")
    print(f"  {'endpoint':32} {'npos':>5} {'AUROC':>6} {'nat':>6} {'H':>7} "
          f"{'gain':>7}  cluster")
    for e in sorted(eps, key=lambda x: -x["H"]):
        g = e["shots"]["64"]["mean"] - e["bacc_native"]
        print(f"  {e['endpoint']:32} {e['n_pos_test']:5} {e['auroc']:6.3f} "
              f"{e['bacc_native']:6.3f} {e['H']:+7.3f} {g:+7.3f}  {e['cluster']}")

    out = {"n_endpoints_kept": len(d["endpoints"]), "n_usable": len(eps),
           "n_clusters": len(set(e["cluster"] for e in eps)),
           "n_primary": len(prim)}
    json.dump(out, open(f"{R}/validate_v2_summary.json", "w"), indent=1)
