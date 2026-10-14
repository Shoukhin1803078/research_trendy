"""Claim 1's experiment: does the published ranking survive order-aware evaluation?

Protocol mirrors medimeta-fsl-benchmark: balanced k-shot support drawn from the TRAIN
split, evaluated on the full TEST split, averaged over seeds. We change exactly one
thing - we also score with order-aware metrics - and then compare the two rankings.
"""
import os, json, sys, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import estimators as E
from metrics import evaluate

FEAT = "/mnt/extra-storage/features"
METHODS = ["majority_class", "zero_shot", "proto_adapter", "tip_adapter", "linear_probe",
           "gda_ledoit_wolf", "gda_structured",
           # second attempt at Claim 2: ordering used at the DECISION level
           "ordinal_frank_hall", "ordinal_expected_grade",
           "ordinal_frank_hall_shuffled"]


def sample_balanced(ys, idx, C, k, rng):
    """Balanced k-shot, as task_presampling.py does (n_support_samples_per_class)."""
    picked = []
    for c in range(C):
        pool = idx[ys[idx] == c]
        if len(pool) == 0:
            return None
        rep = len(pool) < k
        picked.append(rng.choice(pool, size=k, replace=rep))
    return np.concatenate(picked)


def run(dataset_id, task, ks=(1, 2, 4, 8, 16), seeds=range(5)):
    z = np.load(os.path.join(FEAT, f"{dataset_id}_biomedclip.npz"), allow_pickle=True)
    t = np.load(os.path.join(FEAT, f"{dataset_id}_text.npz"), allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    X = z["features"]
    y = z[f"labels__{task}"].astype(int).ravel()
    tr, te = z["split__train"], z["split__test"]
    T = t[f"feats__{task}"]
    raw_keys = [int(s) for s in t[f"keys__{task}"]]
    # Remap possibly non-contiguous label values (e.g. {0,1,4,6,8,10}) to 0..C-1,
    # preserving order. Essential: ordinal metrics need contiguous ranks.
    remap = {v: i for i, v in enumerate(raw_keys)}
    keep = np.isin(y, raw_keys)
    y = np.array([remap[v] for v in y[keep]])
    reindex = -np.ones(len(keep), dtype=int); reindex[np.where(keep)[0]] = np.arange(keep.sum())
    X = X[keep]
    tr = reindex[tr[keep[tr]]]; te = reindex[te[keep[te]]]
    C = len(raw_keys)
    target = meta["tasks"][task]["target"]
    is_ord = target == "ORDINAL_REGRESSION"

    print(f"\n{'='*86}")
    print(f"{dataset_id} / '{task}'   target={target}   C={C}   "
          f"train={len(tr)} test={len(te)}")
    print(f"{'='*86}")

    rows = []
    for k, seed in itertools.product(ks, seeds):
        rng = np.random.default_rng(seed)
        sup = sample_balanced(y, tr, C, k, rng)
        if sup is None:
            continue
        Xs, ys_, Xq, yq = X[sup], y[sup], X[te], y[te]
        for m in METHODS:
            fn = getattr(E, m)
            # Every estimator takes **_ and ignores what it does not need, so pass the
            # full kwarg set verbatim. (An earlier filter here was a no-op; removed.)
            L = fn(Xs=Xs, ys=ys_, Xq=Xq, C=C, text_feats=T, seed=seed)
            yhat = L.argmax(1)
            # softmax for AUROC
            Z = L - L.max(1, keepdims=True)
            P = np.exp(Z); P /= P.sum(1, keepdims=True)
            r = evaluate(yq, yhat, proba=P, n_classes=C)
            r.update(method=m, k=k, seed=seed)
            rows.append(r)

    # aggregate
    import collections
    agg = collections.defaultdict(dict)
    for m in METHODS:
        for k in ks:
            sel = [r for r in rows if r["method"] == m and r["k"] == k]
            if sel:
                agg[m][k] = {mt: float(np.mean([s[mt] for s in sel]))
                             for mt in sel[0] if mt not in ("method", "k", "seed")}

    flat_key, ord_key = "accuracy", ("qwk" if is_ord else "accuracy")
    for k in ks:
        print(f"\n--- k={k} ---")
        print(f"{'method':<18} {'accuracy':>9} {'auroc':>8} | {'QWK':>8} {'MAE':>7} {'adj_acc':>8}")
        print("-" * 70)
        for m in METHODS:
            if k in agg[m]:
                a = agg[m][k]
                print(f"{m:<18} {a['accuracy']:>9.4f} {a['auroc']:>8.4f} | "
                      f"{a['qwk']:>8.4f} {a['mae']:>7.3f} {a['adjacent_acc']:>8.4f}")
        if is_ord:
            by_flat = sorted([m for m in METHODS if k in agg[m]],
                             key=lambda m: -agg[m][k]["accuracy"])
            by_auroc = sorted([m for m in METHODS if k in agg[m]],
                              key=lambda m: -agg[m][k]["auroc"])
            by_ord = sorted([m for m in METHODS if k in agg[m]],
                            key=lambda m: -agg[m][k]["qwk"])
            print(f"  rank by accuracy : {' > '.join(by_flat)}")
            print(f"  rank by AUROC    : {' > '.join(by_auroc)}")
            print(f"  rank by QWK      : {' > '.join(by_ord)}")
            print(f"  >>> ranking CHANGES (acc vs QWK): {by_flat != by_ord}"
                  f"   winner acc='{by_flat[0]}' vs QWK='{by_ord[0]}'")

    os.makedirs("results", exist_ok=True)
    out = f"results/{dataset_id}__{task.replace(' ','_')}.json"
    json.dump({"dataset": dataset_id, "task": task, "target": target, "C": C,
               "n_train": int(len(tr)), "n_test": int(len(te)),
               "rows": rows, "agg": {m: {str(k): v for k, v in d.items()} for m, d in agg.items()}},
              open(out, "w"), indent=2)
    print(f"\nwrote {out}")
    return agg


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "dr_regular"
    task = sys.argv[2] if len(sys.argv) > 2 else "DR level"
    nseeds = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    run(ds, task, seeds=range(nseeds))
