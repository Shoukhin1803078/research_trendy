"""
Supervised reference on BenSParX's 71 handcrafted features, subject-grouped.

This is the context any zero-shot number needs: how much of the PD signal is in
these features at all, when a model is allowed to train on them. No LLM.

Grouping matters: the 900 rows are 7-8 segments from each of 120 subjects, so a
row-wise split leaks a subject across train and test. We split by SUBJECT.
"""
import csv, json, os, sys, warnings
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rope

warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold

CSV = "/mnt/extra-storage/data/bensparx/BenSParX.csv"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def load():
    rows = list(csv.reader(open(CSV)))
    hdr, data = rows[0], rows[1:]
    X = np.array([[float(v) for v in r[1:-1]] for r in data])
    y = np.array([int(r[-1]) for r in data])
    g = np.array([r[0].rsplit("_", 1)[0] for r in data])
    return X, y, g, hdr[1:-1]


def subject_level(y, g, s):
    """Aggregate segment scores to one score per subject by MEAN, and one label."""
    subs = sorted(set(g))
    ys = np.array([y[g == u][0] for u in subs])
    ss = np.array([s[g == u].mean() for u in subs])
    return ys, ss, subs


def main():
    X, y, g, names = load()
    print(f"{X.shape[0]} segments, {len(set(g))} subjects, {X.shape[1]} features, "
          f"segment prevalence {y.mean():.3f}")

    res = {}
    for mname, mk in (("logreg", lambda: LogisticRegression(C=1.0, max_iter=5000)),
                      ("random_forest", lambda: RandomForestClassifier(
                          n_estimators=300, random_state=0, n_jobs=2))):
        seg_auc, sub_auc, sub_bacc = [], [], []
        # 5-fold, stratified on label, grouped by subject, repeated over 5 seeds
        for seed in range(5):
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            for tr, te in cv.split(X, y, groups=g):
                sc = StandardScaler().fit(X[tr])
                m = mk().fit(sc.transform(X[tr]), y[tr])
                p = m.predict_proba(sc.transform(X[te]))[:, 1]
                seg_auc.append(rope.auroc(y[te], p))
                ys, ss, _ = subject_level(y[te], g[te], p)
                sub_auc.append(rope.auroc(ys, ss))
                sub_bacc.append(rope.bacc(ys, (ss > 0.5).astype(int)))
        res[mname] = {"segment_auroc": [float(np.mean(seg_auc)), float(np.std(seg_auc))],
                      "subject_auroc": [float(np.mean(sub_auc)), float(np.std(sub_auc))],
                      "subject_bacc": [float(np.mean(sub_bacc)), float(np.std(sub_bacc))]}
        r = res[mname]
        print(f"  {mname:14} segment AUROC {r['segment_auroc'][0]:.3f}+-{r['segment_auroc'][1]:.3f}"
              f"   subject AUROC {r['subject_auroc'][0]:.3f}+-{r['subject_auroc'][1]:.3f}"
              f"   subject BACC {r['subject_bacc'][0]:.3f}+-{r['subject_bacc'][1]:.3f}")

    # The leakage this guards against, quantified: same model, UNGROUPED split.
    print("\n  Why grouping matters - the same logreg with a row-wise (leaky) split:")
    from sklearn.model_selection import StratifiedKFold
    leaky = []
    for seed in range(5):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr, te in cv.split(X, y):
            sc = StandardScaler().fit(X[tr])
            m = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(X[tr]), y[tr])
            p = m.predict_proba(sc.transform(X[te]))[:, 1]
            leaky.append(rope.auroc(y[te], p))
    res["logreg_row_wise_LEAKY"] = [float(np.mean(leaky)), float(np.std(leaky))]
    print(f"  {'logreg (leaky)':14} segment AUROC {np.mean(leaky):.3f}+-{np.std(leaky):.3f}"
          f"   <- inflated by {np.mean(leaky)-res['logreg']['segment_auroc'][0]:+.3f}")

    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/bensparx_reference.json", "w"), indent=1)
    print(f"\nwrote {OUT}/bensparx_reference.json")


if __name__ == "__main__":
    main()
