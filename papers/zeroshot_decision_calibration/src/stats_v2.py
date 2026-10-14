"""
P4 - correct the uncertainty estimates.

Two problems the audit found:

(1) The aggregation test bootstrapped 25 fold-results as independent. They are
    5 repeats x 5 folds over the same 120 subjects. Within a repeat the folds
    partition one dataset; across repeats the same subjects recur. The correct
    unit of resampling is the REPEAT.

(2) The BenSParX k-shot numbers report the SD across 50 draws, which is a
    spread, not a confidence interval on the mean, and the draws overlap.
    A SUBJECT-level bootstrap gives an interval that accounts for the fact that
    there are only 40 subjects.

Writes new files; originals untouched.
"""
import csv, json, os, sys, warnings, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rope

warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(HERE, "results")
CSVF = "/mnt/extra-storage/data/bensparx/BenSParX.csv"
SEEDS = 5


def load():
    rows = list(csv.reader(open(CSVF)))
    X = np.array([[float(v) for v in r[1:-1]] for r in rows[1:]])
    y = np.array([int(r[-1]) for r in rows[1:]])
    g = np.array([r[0].rsplit("_", 1)[0] for r in rows[1:]])
    return X, y, g


def aggregate(p, y_seg, g_seg):
    subs = sorted(set(g_seg))
    ys = np.array([y_seg[g_seg == u][0] for u in subs])
    vote = np.array([(p[g_seg == u] > 0.5).mean() for u in subs])
    mp = np.array([p[g_seg == u].mean() for u in subs])
    return ys, (vote > 0.5).astype(int), (mp > 0.5).astype(int)


def part1():
    print("=== P4.1 aggregation: repeat-level vs fold-level bootstrap ===")
    X, y, g = load()
    per_repeat = collections.defaultdict(list)       # repeat -> list of deltas
    flat = []
    for seed in range(SEEDS):
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr, te in cv.split(X, y, groups=g):
            sc = StandardScaler().fit(X[tr])
            m = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(X[tr]), y[tr])
            p = m.predict_proba(sc.transform(X[te]))[:, 1]
            ys, yv, ym = aggregate(p, y[te], g[te])
            d = rope.bacc(ys, ym) - rope.bacc(ys, yv)
            per_repeat[seed].append(d)
            flat.append(d)
    flat = np.array(flat)
    rng = np.random.default_rng(0)

    # WRONG: resample the 25 folds as if independent
    naive = [flat[rng.integers(0, len(flat), len(flat))].mean() for _ in range(20000)]
    # RIGHT: resample the 5 repeats (each repeat contributes its 5 folds together)
    reps = sorted(per_repeat)
    rmeans = np.array([np.mean(per_repeat[r]) for r in reps])
    clust = [rmeans[rng.integers(0, len(rmeans), len(rmeans))].mean() for _ in range(20000)]

    print(f"  mean difference (mean_prob - majority_vote): {flat.mean():+.4f}")
    print(f"  per-repeat means: {', '.join(f'{v:+.4f}' for v in rmeans)}")
    print(f"  NAIVE  fold-level bootstrap (n=25, wrong): "
          f"[{np.quantile(naive,0.025):+.4f}, {np.quantile(naive,0.975):+.4f}]")
    print(f"  CORRECT repeat-level bootstrap (n=5):      "
          f"[{np.quantile(clust,0.025):+.4f}, {np.quantile(clust,0.975):+.4f}]")
    out = {"delta_mean": float(flat.mean()),
           "per_repeat_means": [float(v) for v in rmeans],
           "ci95_fold_level_NAIVE": [float(np.quantile(naive, .025)),
                                     float(np.quantile(naive, .975))],
           "ci95_repeat_level_CORRECT": [float(np.quantile(clust, .025)),
                                         float(np.quantile(clust, .975))],
           "n_repeats": len(reps), "n_folds": int(len(flat))}
    excl = all(np.sign(v) == np.sign(rmeans[0]) for v in rmeans)
    print(f"  all {len(reps)} repeats agree in sign: {excl}")
    out["all_repeats_same_sign"] = bool(excl)
    return out


def part2():
    print("\n=== P4.2 BenSParX k-shot: subject-level bootstrap ===")
    rows = [json.loads(l) for l in
            open(f"{R}/llm_bensparx_llama32_3b_list_f10.jsonl") if l.strip()]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["group"]].append(r)
    subs = sorted(by)
    y = np.array([by[u][0]["y"] for u in subs])
    s = np.array([np.mean([r["s"] for r in by[u]]) for u in subs])

    def fit_thresh(ss, yy):
        if len(np.unique(yy)) < 2:
            return 0.0
        c = np.unique(ss)
        c = np.concatenate([[c[0] - 1], (c[:-1] + c[1:]) / 2, [c[-1] + 1]])
        best, bt = -1, 0.0
        for t in c:
            b = rope.bacc(yy, (ss > t).astype(int))
            if b > best:
                best, bt = b, t
        return bt

    rng = np.random.default_rng(0)
    out = {}
    print(f"  {'k':>3} {'mean':>7} {'SD over draws':>14} {'subject-level 95% CI':>26}")
    for k in (4, 8, 16):
        # point estimate: same protocol as before
        accs = []
        for sd in range(50):
            r = np.random.default_rng(777 * sd + k)
            idx = r.choice(len(y), size=k, replace=False)
            m = np.ones(len(y), bool); m[idx] = False
            accs.append(rope.bacc(y[m], (s[m] > fit_thresh(s[idx], y[idx])).astype(int)))
        pt, sd_draws = float(np.mean(accs)), float(np.std(accs))
        # bootstrap the SUBJECTS, then repeat the whole procedure inside each resample
        boot = []
        for _ in range(400):
            bi = rng.integers(0, len(y), len(y))
            yb, sb = y[bi], s[bi]
            if len(np.unique(yb)) < 2:
                continue
            inner = []
            for sd2 in range(10):
                r2 = np.random.default_rng(31 * sd2 + k)
                idx = r2.choice(len(yb), size=k, replace=False)
                m = np.ones(len(yb), bool); m[idx] = False
                if len(np.unique(yb[m])) < 2 or len(np.unique(yb[idx])) < 2:
                    continue
                inner.append(rope.bacc(yb[m],
                             (sb[m] > fit_thresh(sb[idx], yb[idx])).astype(int)))
            if inner:
                boot.append(np.mean(inner))
        lo, hi = float(np.quantile(boot, .025)), float(np.quantile(boot, .975))
        out[k] = {"mean": pt, "sd_over_draws": sd_draws, "ci95_subject_boot": [lo, hi],
                  "n_boot": len(boot)}
        print(f"  {k:>3} {pt:7.3f} {sd_draws:14.3f}   [{lo:+.3f}, {hi:+.3f}]")
    print("  The subject-level interval is the one to quote: it carries the fact")
    print("  that there are only 40 subjects, which the SD over draws does not.")
    return out


if __name__ == "__main__":
    res = {"aggregation": part1(), "bensparx_kshot": part2()}
    json.dump(res, open(f"{R}/stats_v2.json", "w"), indent=1)
    print(f"\nwrote {R}/stats_v2.json")
