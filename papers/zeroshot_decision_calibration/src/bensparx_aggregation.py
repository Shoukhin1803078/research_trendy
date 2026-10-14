"""
Two of paper 3's limitations, tested on its own public data. No LLM needed.

arXiv:2605.24806 makes two protocol choices that it does not justify:

 (1) AGGREGATION. It forces one hard 0/1 token per 10-second segment and takes a
     MAJORITY VOTE over a subject's 7-8 segments. That discards the magnitude of
     each segment's evidence twice: once when the score becomes a sign, again
     when the signs are counted. We test the aggregation rule directly, holding
     the underlying scores fixed, so the comparison isolates the rule.

 (2) FEATURE PIPELINE. The paper lists as a limitation that "handcrafted
     features were derived from a predefined pipeline rather than optimized
     individually for each corpus". We measure how much that costs by scoring
     feature subsets, with the selection done INSIDE each training fold so the
     number is not inflated.

Protocol: subject-grouped stratified 5-fold, 5 seeds. Splitting by row would
leak a subject across train and test, since the 900 rows are 7-8 segments from
each of 120 subjects; we quantify that leak too.
"""
import csv, json, os, sys, warnings, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rope

warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

CSV = "/mnt/extra-storage/data/bensparx/BenSParX.csv"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
SEEDS = 5

# the paper describes its features as time-domain, frequency-domain and cepstral
GROUPS = {
    "jitter_shimmer": lambda n: ("Jitter" in n or "Shimmer" in n),
    "voice_quality":  lambda n: n in ("AutoCorrHarmonicity", "meanAutoCorrHarmonicity",
                                      "NHR", "HNR"),
    "pitch_intensity": lambda n: ("Pitch" in n or "Intensity" in n or n in
                                  ("f1", "f2", "f3", "f4")),
    "mfcc":           lambda n: "MFCC" in n,
    "energy":         lambda n: "energy" in n.lower(),
    "pulses":         lambda n: ("Pulses" in n or "PeriodPulses" in n),
}


def load():
    rows = list(csv.reader(open(CSV)))
    hdr, data = rows[0], rows[1:]
    X = np.array([[float(v) for v in r[1:-1]] for r in data])
    y = np.array([int(r[-1]) for r in data])
    g = np.array([r[0].rsplit("_", 1)[0] for r in data])
    return X, y, g, np.array(hdr[1:-1])


def aggregate(p_seg, y_seg, g_seg, thr=0.5):
    """Four subject-level rules over the same segment probabilities."""
    subs = sorted(set(g_seg))
    ys = np.array([y_seg[g_seg == u][0] for u in subs])
    out = {}
    vote = np.array([(p_seg[g_seg == u] > thr).mean() for u in subs])
    out["majority_vote"] = (vote > 0.5).astype(int), vote      # paper 3's rule
    mp = np.array([p_seg[g_seg == u].mean() for u in subs])
    out["mean_prob"] = (mp > thr).astype(int), mp
    lg = np.array([np.log(np.clip(p_seg[g_seg == u], 1e-6, 1 - 1e-6) /
                          (1 - np.clip(p_seg[g_seg == u], 1e-6, 1 - 1e-6))).mean()
                   for u in subs])
    out["mean_logit"] = (lg > 0).astype(int), lg
    mx = np.array([p_seg[g_seg == u].max() for u in subs])
    out["max_prob"] = (mx > thr).astype(int), mx
    return ys, out


def fit_predict(Xtr, ytr, Xte, cols=None):
    if cols is not None:
        Xtr, Xte = Xtr[:, cols], Xte[:, cols]
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(Xtr), ytr)
    return m.predict_proba(sc.transform(Xte))[:, 1]


def main():
    X, y, g, names = load()
    print(f"{X.shape[0]} segments, {len(set(g))} subjects, {X.shape[1]} features, "
          f"segment prevalence {y.mean():.3f}\n")
    res = {}

    # ---------------------------------------------------- 1. aggregation rules
    print("1. Subject-level aggregation rules, same segment scores throughout")
    acc = collections.defaultdict(list); auc = collections.defaultdict(list)
    seg_auc = []
    for seed in range(SEEDS):
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr, te in cv.split(X, y, groups=g):
            p = fit_predict(X[tr], y[tr], X[te])
            seg_auc.append(rope.auroc(y[te], p))
            ys, rules = aggregate(p, y[te], g[te])
            for rname, (yh, sc) in rules.items():
                acc[rname].append(rope.bacc(ys, yh))
                auc[rname].append(rope.auroc(ys, sc))
    print(f"   segment-level AUROC {np.mean(seg_auc):.3f}+-{np.std(seg_auc):.3f}")
    base = "majority_vote"
    for rname in ("majority_vote", "mean_prob", "mean_logit", "max_prob"):
        d = np.array(acc[rname]) - np.array(acc[base])
        tag = "  <- paper 3's rule" if rname == base else \
              f"   vs vote {d.mean():+.3f}"
        print(f"   {rname:14} subject BACC {np.mean(acc[rname]):.3f}+-{np.std(acc[rname]):.3f}"
              f"   AUROC {np.mean(auc[rname]):.3f}{tag}")
        res[f"agg_{rname}"] = {"bacc": [float(np.mean(acc[rname])), float(np.std(acc[rname]))],
                               "auroc": [float(np.mean(auc[rname])), float(np.std(auc[rname]))],
                               "delta_vs_vote": float(d.mean())}
    # paired bootstrap on the best non-vote rule
    best = max(("mean_prob", "mean_logit", "max_prob"), key=lambda r: np.mean(acc[r]))
    d = np.array(acc[best]) - np.array(acc[base])
    rng = np.random.default_rng(0)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
    lo, hi = np.quantile(bs, [0.025, 0.975])
    print(f"   paired: {best} - majority_vote = {d.mean():+.4f} "
          f"[{lo:+.4f},{hi:+.4f}] over {len(d)} folds"
          f"{'  SIGNIFICANT' if (lo>0 or hi<0) else '  not significant'}")
    res["agg_paired_best"] = {"rule": best, "delta": float(d.mean()),
                              "ci95": [float(lo), float(hi)], "n_folds": len(d)}

    # --------------------------------------------- 2. feature subsets / pipeline
    print("\n2. Feature subsets (selection inside the training fold)")
    subsets = {"all_71": None}
    for gname, pred in GROUPS.items():
        cols = [i for i, n in enumerate(names) if pred(n)]
        if cols:
            subsets[f"{gname}_{len(cols)}"] = cols
    rows_out = {}
    for sname, cols in subsets.items():
        a = []
        for seed in range(SEEDS):
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            for tr, te in cv.split(X, y, groups=g):
                p = fit_predict(X[tr], y[tr], X[te], cols)
                ys, rules = aggregate(p, y[te], g[te])
                a.append(rope.auroc(ys, rules["mean_logit"][1]))
        rows_out[sname] = [float(np.mean(a)), float(np.std(a))]
        print(f"   {sname:22} subject AUROC {np.mean(a):.3f}+-{np.std(a):.3f}")
    # top-k by univariate AUROC, chosen within each training fold
    for k in (5, 10, 20, 40):
        a = []
        for seed in range(SEEDS):
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            for tr, te in cv.split(X, y, groups=g):
                sc = [abs(rope.auroc(y[tr], X[tr, j]) - 0.5) for j in range(X.shape[1])]
                cols = list(np.argsort(sc)[::-1][:k])
                p = fit_predict(X[tr], y[tr], X[te], cols)
                ys, rules = aggregate(p, y[te], g[te])
                a.append(rope.auroc(ys, rules["mean_logit"][1]))
        rows_out[f"top{k}_in_fold"] = [float(np.mean(a)), float(np.std(a))]
        print(f"   {'top'+str(k)+'_in_fold':22} subject AUROC {np.mean(a):.3f}+-{np.std(a):.3f}")
    res["feature_subsets"] = rows_out

    # ------------------------------------------------------- 3. the leakage cost
    print("\n3. What a row-wise split would have reported (the leak, quantified)")
    grouped, leaky = [], []
    for seed in range(SEEDS):
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr, te in cv.split(X, y, groups=g):
            grouped.append(rope.auroc(y[te], fit_predict(X[tr], y[tr], X[te])))
        cv2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr, te in cv2.split(X, y):
            leaky.append(rope.auroc(y[te], fit_predict(X[tr], y[tr], X[te])))
    print(f"   subject-grouped segment AUROC {np.mean(grouped):.3f}+-{np.std(grouped):.3f}")
    print(f"   row-wise     segment AUROC {np.mean(leaky):.3f}+-{np.std(leaky):.3f}"
          f"   inflated by {np.mean(leaky)-np.mean(grouped):+.3f}")
    res["leakage"] = {"grouped": [float(np.mean(grouped)), float(np.std(grouped))],
                      "row_wise": [float(np.mean(leaky)), float(np.std(leaky))],
                      "inflation": float(np.mean(leaky) - np.mean(grouped))}

    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/bensparx_aggregation.json", "w"), indent=1)
    print(f"\nwrote {OUT}/bensparx_aggregation.json")


if __name__ == "__main__":
    main()
