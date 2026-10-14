"""
The BenSParX experiment: paper 3's limitation, tested.

arXiv:2605.24806 scores a zero-shot LLM on BenSParX by forcing one token per
10-second segment and taking a MAJORITY VOTE of those hard 0/1 decisions to get
a subject-level call. Two things follow, and this script measures both.

(1) The score is thrown away twice over. Once per segment, when logP("1") -
    logP("0") is collapsed to its sign; and once per subject, when 7-8 signs are
    counted instead of averaged. Both discard magnitude.
(2) The reported subject BACC is therefore one arbitrary operating point. The
    paper concludes "zero-shot capability alone is insufficient"; that does not
    follow unless the score itself lacks signal, which AUROC answers separately.

Aggregations compared, all on the same forced-token logprobs:
    vote_native   sign per segment, majority vote      <- paper 3's protocol
    mean_native   mean score per subject, sign          (keeps magnitude once)
    mean_fitted   mean score per subject, threshold fitted on k labelled subjects
Reference point: the paper reports LLaMA-3 (8B) at BACC 83.33, AUROC 0.901 here.
"""
import glob, json, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rope

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
PUBLISHED = {"BenSParX LLaMA-3 8B (paper 3)": {"bacc": 0.8333, "auroc": 0.901,
                                               "sens": 0.8667, "spec": 0.8000}}
SHOTS = (4, 8, 16)
SEEDS = 50


def load_cell(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if not rows:
        return None
    by = collections.defaultdict(list)
    for r in rows:
        by[r["group"]].append(r)
    subs = sorted(by)
    y = np.array([by[u][0]["y"] for u in subs])
    s_mean = np.array([np.mean([r["s"] for r in by[u]]) for u in subs])
    s_vote = np.array([np.mean([1.0 if r["s"] > 0 else 0.0 for r in by[u]]) for u in subs])
    seg_y = np.array([r["y"] for r in rows])
    seg_s = np.array([r["s"] for r in rows])
    seg_group = np.array([r["group"] for r in rows])
    topk = float(np.mean([r.get("both_in_topk", True) for r in rows]))
    return {"path": path, "model": rows[0]["model"], "prompt": rows[0]["prompt"],
            "n_seg": len(rows), "n_sub": len(subs), "subs": subs,
            "y": y, "s_mean": s_mean, "s_vote": s_vote,
            "seg_y": seg_y, "seg_s": seg_s, "seg_group": seg_group,
            "both_in_topk_rate": topk}


def fit_threshold(s_sup, y_sup):
    if len(np.unique(y_sup)) < 2:
        return 0.0
    c = np.unique(s_sup)
    c = np.concatenate([[c[0] - 1], (c[:-1] + c[1:]) / 2, [c[-1] + 1]])
    best, bt = -1, 0.0
    for t in c:
        b = rope.bacc(y_sup, (s_sup > t).astype(int))
        if b > best:
            best, bt = b, t
    return bt


def analyse(c):
    y = c["y"]
    out = {k: c[k] for k in ("model", "prompt", "n_seg", "n_sub", "both_in_topk_rate")}
    out["prevalence"] = float(y.mean())

    # segment level - is there signal at all, before any aggregation?
    out["segment"] = {"auroc": rope.auroc(c["seg_y"], c["seg_s"]),
                      "bacc_native": rope.bacc(c["seg_y"], (c["seg_s"] > 0).astype(int))}

    # subject level, three rules
    out["subject"] = {}
    # paper 3: majority vote of hard decisions (ties -> positive at exactly 0.5)
    yv = (c["s_vote"] > 0.5).astype(int)
    se, sp = rope.sens_spec(y, yv)
    out["subject"]["vote_native"] = {"bacc": rope.bacc(y, yv), "sens": se, "spec": sp,
                                     "auroc": rope.auroc(y, c["s_vote"]),
                                     "pred_pos_rate": float(yv.mean())}
    # mean score, model's own sign
    ym = (c["s_mean"] > 0).astype(int)
    se, sp = rope.sens_spec(y, ym)
    d = rope.headroom(y, c["s_mean"])
    out["subject"]["mean_native"] = {"bacc": rope.bacc(y, ym), "sens": se, "spec": sp,
                                     "auroc": d["auroc"], "pred_pos_rate": float(ym.mean())}
    out["diagnostic"] = d

    # Aggregation compared AFTER removing the model's bias.
    # This cell's model emits "1" for nearly every segment, so at its native
    # threshold voting and averaging agree trivially. Fitting a segment-level
    # threshold on k labelled subjects first separates the two questions:
    # where the operating point sits (bias) and how segments are pooled
    # (aggregation). Both rules get the same k labels and the same scores.
    out["subject"]["vote_fitted"] = {}
    for k in SHOTS:
        if k > len(y) // 2:
            continue
        accs_v, accs_m = [], []
        for sd in range(SEEDS):
            r = np.random.default_rng(999 * sd + k)
            idx = r.choice(len(y), size=k, replace=False)
            sup = set(np.array(c["subs"])[idx])
            m_sub = np.array([u not in sup for u in c["subs"]])
            seg_in_sup = np.array([g in sup for g in c["seg_group"]])
            # vote: threshold the SEGMENTS, then majority-vote per subject
            t_seg = fit_threshold(c["seg_s"][seg_in_sup], c["seg_y"][seg_in_sup])
            vote = np.array([(c["seg_s"][c["seg_group"] == u] > t_seg).mean()
                             for u in c["subs"]])
            accs_v.append(rope.bacc(y[m_sub], (vote[m_sub] > 0.5).astype(int)))
            # mean: average the SCORES per subject, then threshold
            t_sub = fit_threshold(c["s_mean"][idx], y[idx])
            accs_m.append(rope.bacc(y[m_sub], (c["s_mean"][m_sub] > t_sub).astype(int)))
        out["subject"]["vote_fitted"][k] = {
            "bacc_mean": float(np.mean(accs_v)), "bacc_std": float(np.std(accs_v)),
            "per_seed": [float(x) for x in accs_v]}
        out["subject"].setdefault("mean_fitted_paired", {})[k] = {
            "bacc_mean": float(np.mean(accs_m)), "bacc_std": float(np.std(accs_m)),
            "per_seed": [float(x) for x in accs_m]}

    # mean score, threshold fitted on k labelled subjects, evaluated on the rest
    out["subject"]["mean_fitted"] = {}
    for k in SHOTS:
        if k > len(y) // 2:        # keep at least half the cohort for evaluation
            continue
        accs = []
        for sd in range(SEEDS):
            r = np.random.default_rng(777 * sd + k)
            idx = r.choice(len(y), size=k, replace=False)
            m = np.ones(len(y), bool); m[idx] = False
            t = fit_threshold(c["s_mean"][idx], y[idx])
            accs.append(rope.bacc(y[m], (c["s_mean"][m] > t).astype(int)))
        out["subject"]["mean_fitted"][k] = {"bacc_mean": float(np.mean(accs)),
                                           "bacc_std": float(np.std(accs))}
    out["auroc_ci95"] = rope.bootstrap_ci(rope.auroc, y, c["s_mean"], n=2000)
    return out


def main():
    paths = sorted(glob.glob(f"{OUT}/llm_bensparx_*.jsonl"))
    if not paths:
        print(f"no score files yet in {OUT} (run score_llm.py --task bensparx)")
        return
    cells = []
    for p in paths:
        c = load_cell(p)
        if c is None or c["n_sub"] < 10:
            print(f"skip {os.path.basename(p)}: too few subjects"); continue
        cells.append(analyse(c))

    print(f"{'model':22} {'prompt':9} {'nsub':>4} {'segAUC':>7} {'subAUC':>7} "
          f"{'vote':>6} {'mean':>6} {'reach':>6} {'H':>7} {'fit16':>6}")
    print("-" * 96)
    for a in cells:
        f16 = a["subject"]["mean_fitted"].get(16, {}).get("bacc_mean", float("nan"))
        print(f"{a['model']:22} {a['prompt']:9} {a['n_sub']:4} "
              f"{a['segment']['auroc']:7.3f} {a['subject']['mean_native']['auroc']:7.3f} "
              f"{a['subject']['vote_native']['bacc']:6.3f} "
              f"{a['subject']['mean_native']['bacc']:6.3f} "
              f"{a['diagnostic']['bacc_binormal']:6.3f} {a['diagnostic']['H']:+7.3f} "
              f"{f16:6.3f}")

    print("\nSensitivity / specificity, to compare with the paper's own failure mode")
    for a in cells:
        v, m = a["subject"]["vote_native"], a["subject"]["mean_native"]
        print(f"  {a['model']:22} {a['prompt']:9} "
              f"vote sens {v['sens']:.3f} spec {v['spec']:.3f} (pred+ {v['pred_pos_rate']:.3f}) | "
              f"mean sens {m['sens']:.3f} spec {m['spec']:.3f} (pred+ {m['pred_pos_rate']:.3f})")

    print("\nDoes averaging scores beat voting on hard decisions? (same logprobs)")
    diffs = [a["subject"]["mean_native"]["bacc"] - a["subject"]["vote_native"]["bacc"]
             for a in cells]
    print(f"  mean_native - vote_native: " +
          ", ".join(f"{d:+.3f}" for d in diffs) +
          f"   | mean {np.mean(diffs):+.3f} over {len(diffs)} cells")

    print("\nAggregation at a FITTED threshold (bias removed, same k labels, same scores)")
    for a in cells:
        vf = a["subject"].get("vote_fitted", {})
        mf = a["subject"].get("mean_fitted_paired", {})
        for k in sorted(vf, key=int):
            v, m = vf[k], mf[k]
            d = np.array(m["per_seed"]) - np.array(v["per_seed"])
            rng = np.random.default_rng(0)
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
            lo, hi = np.quantile(bs, [0.025, 0.975])
            sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "ns"
            print(f"  {a['model']:18} k={k:>2}  vote {v['bacc_mean']:.3f}+-{v['bacc_std']:.3f}"
                  f"   mean {m['bacc_mean']:.3f}+-{m['bacc_std']:.3f}"
                  f"   mean-vote {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}] {sig}")

    print("\nReference, as published")
    for k, v in PUBLISHED.items():
        print(f"  {k}: BACC {v['bacc']:.3f}  AUROC {v['auroc']:.3f}  "
              f"sens {v['sens']:.3f}  spec {v['spec']:.3f}  -> "
              f"reachable {rope.bacc_binormal(v['auroc']):.3f}, "
              f"H {rope.bacc_binormal(v['auroc'])-v['bacc']:+.3f}")

    json.dump(cells, open(f"{OUT}/bensparx_llm.json", "w"), indent=1, default=float)
    print(f"\nwrote {OUT}/bensparx_llm.json ({len(cells)} cells)")


if __name__ == "__main__":
    main()
