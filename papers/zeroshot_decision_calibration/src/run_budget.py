"""
The label-budget experiment.

Question: given k labels and a frozen zero-shot model, where should the labels
go? Standard practice spends them fitting a decision boundary (linear probe on
512-d features, or Platt scaling on the score). We test spending them on ONE
number instead - the cohort prevalence - and letting the model's own ranking
supply the ordering.

Protocol, copied from the few-shot convention used in this repo:
  support   k labelled examples drawn from the TRAIN split
  cohort    the TEST split, unlabelled, available at deployment (transductive;
            this is an advantage ROPE has over the inductive baselines and is
            reported as such, not hidden)
  evaluate  balanced accuracy on the TEST split
  seeds     50, paired across methods (same support set for every method)
"""
import json, os, sys, warnings
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rope
from run_arm_a import load, zero_shot_scores, ENDPOINTS

warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
SHOTS = (2, 4, 8, 16, 32, 64)
SEEDS = 50


def platt(s_sup, y_sup, s_eval):
    """2-parameter logistic recalibration of the zero-shot score."""
    if len(np.unique(y_sup)) < 2:
        return (s_eval > 0).astype(int)
    lr = LogisticRegression(C=1e3, max_iter=1000)
    lr.fit(np.asarray(s_sup).reshape(-1, 1), y_sup)
    return (lr.predict_proba(np.asarray(s_eval).reshape(-1, 1))[:, 1] > 0.5).astype(int)


def thresh_on_support(s_sup, y_sup, s_eval):
    """1 parameter, but fitted directly: the threshold that maximises balanced
    accuracy on the k labels. The natural 1-parameter competitor to ROPE-Q."""
    if len(np.unique(y_sup)) < 2:
        return (s_eval > 0).astype(int)
    cand = np.unique(s_sup)
    cand = np.concatenate([[cand[0] - 1], (cand[:-1] + cand[1:]) / 2, [cand[-1] + 1]])
    best, bt = -1, 0.0
    for t in cand:
        b = rope.bacc(y_sup, (s_sup > t).astype(int))
        if b > best:
            best, bt = b, t
    return (s_eval > bt).astype(int)


def linear_probe(X_sup, y_sup, X_eval, tuned=True):
    """The standard few-shot baseline: logistic regression on the 512-d frozen
    features, i.e. k labels spent on a 512-parameter boundary.

    `tuned=True` is the fair version and the one to report: C is selected by
    leave-one-out balanced accuracy on the support set, and class_weight is
    balanced. The untuned version (fixed C=1.0, no class weighting) is kept
    because the first run of this experiment used it and reported a degeneracy
    finding; `run_budget.py` now scores both so that finding can be checked
    rather than trusted. On a skewed support set, unweighted logistic
    regression predicts the majority class by construction, which is a
    property of the baseline's configuration and not of few-shot learning."""
    if len(np.unique(y_sup)) < 2:
        return np.zeros(len(X_eval), int)
    if not tuned:
        lr = LogisticRegression(C=1.0, max_iter=2000)
        lr.fit(X_sup, y_sup)
        return lr.predict(X_eval)
    # The fair minimum fix, and the one that isolates the suspected cause:
    # same C, but class_weight balanced. An unweighted fit on a skewed support
    # set minimises error by predicting the majority class, which is what
    # produced the "degenerates at exactly 0.500" result in the first run.
    # Tuning C as well was tried and is far too slow here (a leave-one-out
    # grid over 512-d features at k=64 x 50 seeds x 9 endpoints), so C is held
    # fixed and only the weighting changes.
    lr = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    lr.fit(X_sup, y_sup)
    return lr.predict(X_eval)


def run(ds, label_key, pos_idx, name):
    img, txt = load(ds)
    y_all = img[f"labels__{label_key}"]
    if y_all.ndim > 1:
        return None
    f = img["features"].astype(np.float64)
    f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-12)
    s_all = zero_shot_scores(img, txt, label_key, pos_idx)
    yb_all = np.isin(y_all, pos_idx).astype(int)

    tr, te = img["split__train"], img["split__test"]
    y_te, s_te, X_te = yb_all[te], s_all[te], f[te]
    y_tr, s_tr, X_tr = yb_all[tr], s_all[tr], f[tr]
    if y_te.sum() in (0, len(y_te)):
        return None

    a = rope.auroc(y_te, s_te)
    diag = rope.headroom(y_te, s_te)
    r_cohort = rope.ranks01(s_te)                  # transductive, labels unused
    nat = rope.bacc(y_te, (s_te > 0).astype(int))
    oracle_pi = rope.bacc(y_te, rope.predict_quantile(s_te, y_te.mean(), r=r_cohort))

    res = {"endpoint": name, "dataset": ds, "task": label_key, "pos_classes": pos_idx,
           "n_test": int(len(y_te)), "prevalence": float(y_te.mean()),
           "auroc": a, "inverted": bool(a < 0.5), "H": diag["H"],
           "bacc_native": nat, "bacc_rope_q_pi_known": oracle_pi, "shots": {}}

    for k in SHOTS:
        acc = {m: [] for m in ("rope_q_pik", "thresh_k", "platt",
                               "linear_probe", "linear_probe_untuned")}
        pis = []
        for sd in range(SEEDS):
            r = np.random.default_rng(10_000 * sd + k)
            idx = r.choice(len(y_tr), size=min(k, len(y_tr)), replace=False)
            ys, ss, Xs = y_tr[idx], s_tr[idx], X_tr[idx]

            pik = float(ys.mean())                 # the one number ROPE needs
            pis.append(pik)
            acc["rope_q_pik"].append(rope.bacc(
                y_te, rope.predict_quantile(s_te, pik, r=r_cohort)))
            acc["thresh_k"].append(rope.bacc(y_te, thresh_on_support(ss, ys, s_te)))
            acc["platt"].append(rope.bacc(y_te, platt(ss, ys, s_te)))
            acc["linear_probe"].append(
                rope.bacc(y_te, linear_probe(Xs, ys, X_te, tuned=True)))
            acc["linear_probe_untuned"].append(
                rope.bacc(y_te, linear_probe(Xs, ys, X_te, tuned=False)))

        res["shots"][k] = {m: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                               "per_seed": [float(x) for x in v]}
                           for m, v in acc.items()}
        res["shots"][k]["pi_hat_mean"] = float(np.mean(pis))
        res["shots"][k]["pi_hat_std"] = float(np.std(pis))
    return res


def paired(a, b, n=5000, seed=0):
    """Mean paired difference a-b with a percentile bootstrap CI over seeds."""
    a = np.asarray(a, float); b = np.asarray(b, float); d = a - b
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


if __name__ == "__main__":
    out = []
    for ds, lk, pos, nm in ENDPOINTS:
        r = run(ds, lk, pos, nm)
        if r is None:
            print(f"SKIP {nm}", flush=True); continue
        out.append(r)
        tag = " [INVERTED]" if r["inverted"] else ""
        print(f"\n=== {nm}{tag}  n={r['n_test']} pi={r['prevalence']:.3f} "
              f"AUROC={r['auroc']:.3f} H={r['H']:+.3f}", flush=True)
        print(f"    zero labels: native {r['bacc_native']:.3f}   "
              f"ROPE-Q with pi known {r['bacc_rope_q_pi_known']:.3f}", flush=True)
        print(f"    {'k':>3} {'ROPE-Q(pi_k)':>13} {'thresh_k':>13} {'Platt':>13} "
              f"{'probe(tuned)':>13} {'probe(untuned)':>14}   "
              f"{'ROPE-Q - best baseline':>24}", flush=True)
        for k in SHOTS:
            sh = r["shots"][k]
            cells = []
            for m in ("rope_q_pik", "thresh_k", "platt", "linear_probe",
                      "linear_probe_untuned"):
                cells.append(f"{sh[m]['mean']:.3f}+-{sh[m]['std']:.3f}")
            base = max(("thresh_k", "platt", "linear_probe"),
                       key=lambda m: sh[m]["mean"])
            d, lo, hi = paired(sh["rope_q_pik"]["per_seed"], sh[base]["per_seed"])
            star = "*" if (lo > 0 or hi < 0) else " "
            print(f"    {k:>3} " + " ".join(f"{c:>13}" for c in cells) +
                  f"   {d:+.3f} [{lo:+.3f},{hi:+.3f}]{star} vs {base}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/budget.json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"\nwrote {OUT}/budget.json ({len(out)} endpoints)", flush=True)
