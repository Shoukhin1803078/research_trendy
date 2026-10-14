"""
P3 - expand the validation set from 9 endpoints to as many as the cached
features support, to answer feedback.md section 3: is nine enough?

Writes NEW files (`*_v2.json`); it does not overwrite the original run.

IMPORTANT ON INDEPENDENCE. The extra endpoints are nested cut-points on the same
label columns (DR level >=1, >=2, >=3, =4 all come from one set of scores and one
label vector). They are therefore strongly correlated and must NOT be treated as
independent observations. Every analysis downstream reports two versions:

  all        every endpoint, with the caveat
  primary    one endpoint per (dataset, label column) - the conservative set

and bootstraps by CLUSTER (dataset, label column) rather than by endpoint.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rope
from run_arm_a import load, zero_shot_scores

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
SHOTS = (4, 8, 16, 32, 64)
SEEDS = 50

# (dataset, label key, positive class INDICES, name, is_primary)
# `is_primary` marks the one endpoint per label column used in the conservative
# analysis; it is chosen as the clinically standard cut-point, fixed in advance.
ENDPOINTS = [
    ("bus", "malignancy",       [1],          "BUS: malignant finding",        True),
    ("bus", "case category",    [2],          "BUS: malignant vs rest",        True),
    ("bus", "case category",    [1, 2],       "BUS: any lesion vs normal",     False),

    ("dr_regular", "Overall quality",  [0],          "DR: inadequate quality",  True),

    ("dr_regular", "DR level", [1, 2, 3, 4], "DR: any retinopathy (>=1)",       False),
    ("dr_regular", "DR level", [2, 3, 4],    "DR: referable DR (>=2)",          True),
    ("dr_regular", "DR level", [3, 4],       "DR: sight-threatening (>=3)",     False),
    ("dr_regular", "DR level", [4],          "DR: proliferative (=4)",          False),

    ("dr_regular", "Artifact", [1, 2, 3, 4, 5], "DR: any artifact",             True),
    ("dr_regular", "Artifact", [2, 3, 4, 5],    "DR: moderate+ artifact",       False),
    ("dr_regular", "Artifact", [4, 5],          "DR: severe artifact",          False),

    ("dr_regular", "Clarity", [0, 1],       "DR: poor clarity",                 True),
    ("dr_regular", "Clarity", [0],          "DR: worst clarity",                False),
    ("dr_regular", "Clarity", [0, 1, 2, 3], "DR: clarity below best",           False),

    ("dr_regular", "Field definition", [0, 1],       "DR: poor field def",      True),
    ("dr_regular", "Field definition", [0],          "DR: worst field def",     False),
    ("dr_regular", "Field definition", [0, 1, 2, 3], "DR: field def below best", False),
]

MIN_POS = 10          # fewer positives than this and the estimates are noise


def thresh_on_support(s_sup, y_sup, s_eval):
    """One parameter fitted on the support labels only. No test labels used."""
    if len(np.unique(y_sup)) < 2:
        return (s_eval > 0).astype(int)
    c = np.unique(s_sup)
    c = np.concatenate([[c[0] - 1], (c[:-1] + c[1:]) / 2, [c[-1] + 1]])
    best, bt = -1, 0.0
    for t in c:
        b = rope.bacc(y_sup, (s_sup > t).astype(int))
        if b > best:
            best, bt = b, t
    return (s_eval > bt).astype(int)


def run(ds, key, pos, name, primary):
    img, txt = load(ds)
    y_all = img[f"labels__{key}"]
    if y_all.ndim > 1:
        return None, "multilabel"
    s_all = zero_shot_scores(img, txt, key, pos)
    yb = np.isin(y_all, pos).astype(int)
    tr, te = img["split__train"], img["split__test"]
    y_te, s_te = yb[te], s_all[te]
    y_tr, s_tr = yb[tr], s_all[tr]
    npos = int(y_te.sum())
    if npos < MIN_POS or npos > len(y_te) - MIN_POS:
        return None, f"only {npos} positives of {len(y_te)}"

    d = rope.headroom(y_te, s_te)
    res = {"endpoint": name, "dataset": ds, "label_column": key,
           "cluster": f"{ds}::{key}", "primary": primary,
           "pos_classes": pos, "n_test": int(len(y_te)), "n_pos_test": npos,
           "prevalence": float(y_te.mean()), **d,
           "auroc_ci95": rope.bootstrap_ci(rope.auroc, y_te, s_te, n=2000),
           "shots": {}}
    for k in SHOTS:
        accs = []
        for sd in range(SEEDS):
            r = np.random.default_rng(10_000 * sd + k)
            idx = r.choice(len(y_tr), size=min(k, len(y_tr)), replace=False)
            accs.append(rope.bacc(y_te, thresh_on_support(s_tr[idx], y_tr[idx], s_te)))
        res["shots"][k] = {"mean": float(np.mean(accs)), "sd": float(np.std(accs)),
                           "per_seed": [float(a) for a in accs]}
    return res, None


if __name__ == "__main__":
    out, skipped = [], []
    for ds, key, pos, name, primary in ENDPOINTS:
        r, why = run(ds, key, pos, name, primary)
        if r is None:
            skipped.append((name, why)); print(f"SKIP {name}: {why}", flush=True); continue
        out.append(r)
        print(f"{name:32} n={r['n_test']:4} pos={r['n_pos_test']:4} "
              f"pi={r['prevalence']:.3f} AUROC={r['auroc']:.3f} "
              f"native={r['bacc_native']:.3f} H={r['H']:+.3f} "
              f"fit64={r['shots'][64]['mean']:.3f}{'  [primary]' if primary else ''}",
              flush=True)
    json.dump({"endpoints": out, "skipped": skipped},
              open(f"{OUT}/endpoints_v2.json", "w"), indent=1, default=float)
    us = [r for r in out if r["auroc"] > 0.5]
    print(f"\n{len(out)} endpoints kept ({len(skipped)} skipped), "
          f"{len(us)} with AUROC > 0.5, "
          f"{len(set(r['cluster'] for r in us))} independent label columns")
    print(f"wrote {OUT}/endpoints_v2.json")
