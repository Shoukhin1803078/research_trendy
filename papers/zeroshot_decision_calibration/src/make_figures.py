"""
Paper figures. Static PDF+PNG for print, so no hover layer and no dark mode:
the relief rule for the two sub-3:1 slots is met by direct labels, and the
paper's own tables serve as the table view.

Palette: the validated reference instance. Scatter uses slots 1-3 only, which
is the all-pairs cap, and the audit happens to have exactly three failure modes.
Lines use slots 1-3 on the adjacent pairlist. Diverging uses blue<->red with a
gray midpoint.
"""
import json, os, sys
from math import sqrt
from statistics import NormalDist
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

N = NormalDist()
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R, F = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
os.makedirs(F, exist_ok=True)

# ---- validated palette (references/palette.md), light surface
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
DIV_POS, DIV_NEG, DIV_MID = "#2a78d6", "#e34948", "#f0efec"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
    "lines.linewidth": 2.0, "savefig.bbox": "tight", "savefig.facecolor": SURFACE,
})
MODE_COLOR = {"A_threshold_limited": S1, "B_signal_limited": S3, "C_inverted": S2}
MODE_LABEL = {"A_threshold_limited": "A  decision-limited",
              "B_signal_limited": "B  signal-limited",
              "C_inverted": "C  sign-inverted"}


def reachable(a):
    a = min(max(a, 1e-6), 1 - 1e-6)
    return N.cdf(sqrt(2) * N.inv_cdf(a) / 2.0)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(F, f"{name}.{ext}"), dpi=200)
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf / .png")


# ------------------------------------------------------- Fig 1: the audit
def fig1(cells):
    """Headroom is shown as the vertical distance from a point up to the curve.
    That is what H *is*, so drawing it beats labelling every point - which
    collided badly in the first render."""
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    xs = np.linspace(0.02, 0.98, 400)
    ax.plot(xs, [reachable(x) for x in xs], color=INK, lw=1.5, ls="--", zorder=3)
    ax.axhline(0.5, color=AXIS, lw=0.9, zorder=1)
    ax.axvline(0.5, color=AXIS, lw=0.9, zorder=1)

    # the gap itself, drawn
    for c in cells:
        if c["mode"] == "A_threshold_limited" and c["H"] > 0.02:
            ax.plot([c["auroc"], c["auroc"]], [c["bacc"], c["reachable"]],
                    color=S1, lw=1.0, alpha=0.45, zorder=2)

    for mode in ("B_signal_limited", "C_inverted", "A_threshold_limited"):
        pts = [c for c in cells if c["mode"] == mode]
        ax.scatter([c["auroc"] for c in pts], [c["bacc"] for c in pts],
                   s=46, color=MODE_COLOR[mode], edgecolor=SURFACE, linewidth=1.2,
                   zorder=5, label=MODE_LABEL[mode])

    ax.annotate("balanced accuracy a well-placed\ndecision point would reach",
                xy=(0.33, reachable(0.33)), xytext=(0.115, 0.64), fontsize=8.5,
                color=INK2, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9,
                                connectionstyle="arc3,rad=-0.15"))
    worst = max(cells, key=lambda c: c["H"])
    ax.annotate(f"{worst['model']} on {worst['dataset']}\n"
                f"AUROC {worst['auroc']:.3f}, reported {worst['bacc']:.3f}\n"
                f"H = +{worst['H']:.3f} unused",
                xy=(worst["auroc"], worst["bacc"] - 0.004),
                xytext=(0.695, 0.275), fontsize=8.5, color=S1, ha="left",
                arrowprops=dict(arrowstyle="->", color=S1, lw=1.1,
                                connectionstyle="arc3,rad=0.2"))
    for c in cells:                       # only the two clearly separated outliers
        if c["auroc"] < 0.30:
            ax.annotate(f"{c['model']} / {c['dataset'].split()[0]}",
                        xy=(c["auroc"], c["bacc"]), xytext=(8, 4),
                        textcoords="offset points", fontsize=7.5, color=INK2)

    ax.set_xlabel("AUROC as reported  (how well the model ranks patients)")
    ax.set_ylabel("Balanced accuracy as reported\n(the yes / no call)")
    ax.set_title("Reported decisions fall short of what the same scores allow\n"
                 f"{len(cells)} cells copied from published tables; "
                 "blue lines are the unused accuracy", loc="left")
    ax.set_xlim(0.08, 1.0); ax.set_ylim(0.22, 1.0)
    h, l = ax.get_legend_handles_labels()
    order = [l.index(MODE_LABEL[m]) for m in
             ("A_threshold_limited", "B_signal_limited", "C_inverted")]
    ax.legend([h[i] for i in order], [l[i] for i in order],
              loc="upper left", fontsize=8.5)
    save(fig, "fig1_audit_scatter")


# --------------------------------------------- Fig 2: headroom per published cell
def fig2(cells):
    llm = [c for c in cells if "non-LLM baseline" not in c.get("note", "")
           and c["paper"] != "ours" and not c["is_subgroup"]]
    llm.sort(key=lambda c: c["H"])
    fig, ax = plt.subplots(figsize=(6.4, 7.6))
    y = np.arange(len(llm))
    # Colour by failure mode, not by the sign of H alone. One cell (Pengi on
    # MDVR-KCL, AUROC 0.268) has H > +0.02 while ranking patients backwards:
    # moving its threshold does not recover it, flipping its sign would. Using
    # H > 0.02 here painted it blue and made the figure show 13 recoverable
    # cells against the 12 mode-A cells reported in the text.
    def bar_colour(c):
        if c["mode"] == "C_inverted":
            return MODE_COLOR["C_inverted"]
        return DIV_POS if c["H"] > 0.02 else (DIV_NEG if c["H"] < -0.02 else MUTED)
    cols = [bar_colour(c) for c in llm]
    ax.barh(y, [c["H"] for c in llm], color=cols, height=0.68,
            edgecolor=SURFACE, linewidth=1.0)
    ax.axvline(0, color=AXIS, lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c['model']} · {c['dataset']} · {c['setting']}" for c in llm],
                       fontsize=7)
    for i, c in enumerate(llm):
        if abs(c["H"]) > 0.10:
            ax.text(c["H"] + (0.008 if c["H"] > 0 else -0.008), i, f"{c['H']:+.3f}",
                    va="center", ha="left" if c["H"] > 0 else "right",
                    fontsize=7, color=INK2)
    ax.set_xlim(-0.17, 0.40)
    ax.set_xlabel("Headroom H  (balanced accuracy left unused by the decision point)")
    ax.set_title(f"{len(llm)} published LLM cells, sorted by unused accuracy",
                 loc="left")
    nA = sum(c["mode"] == "A_threshold_limited" for c in llm)
    ax.legend(handles=[Line2D([0], [0], marker="s", ls="", ms=7, color=DIV_POS,
                              label=f"A  decision-limited, recoverable ({nA})"),
                       Line2D([0], [0], marker="s", ls="", ms=7, color=DIV_NEG,
                              label="B  nothing to recover"),
                       Line2D([0], [0], marker="s", ls="", ms=7,
                              color=MODE_COLOR["C_inverted"],
                              label="C  ranks backwards (needs a sign flip)")],
              loc="lower right", fontsize=8)
    save(fig, "fig2_headroom_bars")


# ----------------------------------------- Fig 3: forward test on BenSParX
def fig3():
    d = json.load(open(os.path.join(R, "bensparx_llm.json")))[0]
    mf = d["subject"]["mean_fitted"]
    ks = [0] + sorted(int(k) for k in mf)
    vals = [d["subject"]["mean_native"]["bacc"]] + [mf[str(k)]["bacc_mean"] for k in ks[1:]]
    errs = [0.0] + [mf[str(k)]["bacc_std"] for k in ks[1:]]
    reach = d["diagnostic"]["bacc_binormal"]

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.axhline(reach, color=INK, ls="--", lw=1.4)
    ax.annotate(f"predicted reachable  {reach:.3f}", xy=(0.15, reach),
                xytext=(0.2, reach + 0.016), fontsize=8, color=INK2)
    ax.axhline(0.5, color=AXIS, lw=1.0)
    ax.annotate("chance", xy=(0.1, 0.5), xytext=(0.15, 0.506), fontsize=8, color=MUTED)
    x = np.arange(len(ks))
    ax.errorbar(x, vals, yerr=errs, color=S1, marker="o", ms=7, capsize=3,
                elinewidth=1.2, mec=SURFACE, mew=1.2, zorder=4)
    for xi, v in zip(x, vals):
        ax.annotate(f"{v:.3f}", xy=(xi, v), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("labelled subjects used to place the decision point")
    ax.set_ylabel("subject balanced accuracy")
    ax.set_title("Forward test: the fix lands where the diagnostic said\n"
                 "BenSParX, llama3.2:3b (predicted +0.228, measured +0.232)", loc="left")
    ax.set_ylim(0.44, 0.80)
    save(fig, "fig3_forward_test")


# ------------------------------------- Fig 4: where a label budget should go
def fig4():
    rows = json.load(open(os.path.join(R, "budget.json")))
    ok = [r for r in rows if not r["inverted"]]
    ks = [2, 4, 8, 16, 32, 64]
    series = [("thresh_k", "decision point (1 parameter)", S1),
              ("linear_probe", "linear probe (512 parameters)", S2),
              ("platt", "Platt scaling (2 parameters)", S3)]
    native = float(np.mean([r["bacc_native"] for r in ok]))

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.axhline(native, color=INK, ls="--", lw=1.4)
    ax.annotate(f"model's own decision, 0 labels  {native:.3f}",
                xy=(2, native), xytext=(2.1, native + 0.006), fontsize=8, color=INK2)
    for key, lab, col in series:
        ys = [float(np.mean([r["shots"][str(k)][key]["mean"] for r in ok])) for k in ks]
        ax.plot(ks, ys, color=col, marker="o", ms=6, mec=SURFACE, mew=1.2, label=lab)
        ax.annotate(f"{ys[-1]:.3f}", xy=(ks[-1], ys[-1]), xytext=(6, -3),
                    textcoords="offset points", fontsize=8, color=col)
    ax.set_xscale("log", base=2); ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("number of labelled examples")
    ax.set_ylabel("mean balanced accuracy\n(6 endpoints, AUROC > 0.5)")
    ax.set_title("Below ~8 labels nothing beats the model's own decision;\n"
                 "above it, a properly configured probe wins", loc="left")
    ax.axvspan(1.8, 8, color=GRID, alpha=0.5, zorder=0)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(1.8, 85)
    save(fig, "fig4_label_budget")


# ------------------------------- Fig 5: averaging scores vs voting on decisions
def fig5():
    d = json.load(open(os.path.join(R, "bensparx_llm.json")))[0]["subject"]
    agg = json.load(open(os.path.join(R, "bensparx_aggregation.json")))
    ks = sorted(int(k) for k in d["vote_fitted"])
    vote = [d["vote_fitted"][str(k)]["bacc_mean"] for k in ks]
    mean = [d["mean_fitted_paired"][str(k)]["bacc_mean"] for k in ks]
    vote_e = [d["vote_fitted"][str(k)]["bacc_std"] for k in ks]
    mean_e = [d["mean_fitted_paired"][str(k)]["bacc_std"] for k in ks]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.8, 3.8),
                                   gridspec_kw={"width_ratios": [1.3, 1]})
    x = np.arange(len(ks)); w = 0.36
    axL.bar(x - w / 2 - 0.012, vote, w, yerr=vote_e, capsize=3, color=S1,
            edgecolor=SURFACE, linewidth=1.5, label="majority vote on hard calls",
            error_kw=dict(ecolor=INK2, elinewidth=1.1))
    axL.bar(x + w / 2 + 0.012, mean, w, yerr=mean_e, capsize=3, color=S2,
            edgecolor=SURFACE, linewidth=1.5, label="average the scores",
            error_kw=dict(ecolor=INK2, elinewidth=1.1))
    for xi, (v, m, ve, me) in enumerate(zip(vote, mean, vote_e, mean_e)):
        top = max(v + ve, m + me)
        axL.text(xi, top + 0.045, f"+{m - v:.3f}", ha="center",
                 fontsize=8.5, color=INK, fontweight="bold")
    axL.set_xticks(x); axL.set_xticklabels([f"k = {k}" for k in ks])
    axL.set_xlabel("labelled subjects used to place the decision point")
    axL.set_ylabel("subject balanced accuracy")
    axL.set_ylim(0, 1.02)
    axL.set_title("On real LLM log-probabilities\nall three gaps significant", loc="left")
    axL.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), fontsize=8.5, ncol=1)

    rules = [("agg_majority_vote", "majority vote", S1),
             ("agg_mean_prob", "mean probability", S2),
             ("agg_mean_logit", "mean logit", S3),
             ("agg_max_prob", "maximum (control)", S4)]
    vals = [agg[k]["bacc"][0] for k, _, _ in rules]
    errs = [agg[k]["bacc"][1] for k, _, _ in rules]
    cols = [c for _, _, c in rules]
    y = np.arange(len(rules))
    axR.barh(y, vals, xerr=errs, capsize=3, color=cols, edgecolor=SURFACE,
             linewidth=1.5, height=0.64, error_kw=dict(ecolor=INK2, elinewidth=1.1))
    axR.set_yticks(y); axR.set_yticklabels([l for _, l, _ in rules], fontsize=8.5)
    axR.invert_yaxis()
    for yi, (v, er) in enumerate(zip(vals, errs)):
        axR.text(v + er + 0.025, yi, f"{v:.3f}", va="center", fontsize=8.5, color=INK2)
    axR.set_xlim(0, 1.30); axR.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axR.set_xlabel("subject balanced accuracy")
    ci = agg["agg_paired_best"]["ci95"]
    axR.set_title(f"On supervised scores\npaired +{agg['agg_paired_best']['delta']:.3f} "
                  f"[{ci[0]:+.3f}, {ci[1]:+.3f}]", loc="left")
    fig.tight_layout()
    save(fig, "fig5_aggregation")


# --------------------------------------------- Fig 6: feature set saturation
def fig6():
    fs = json.load(open(os.path.join(R, "bensparx_aggregation.json")))["feature_subsets"]
    order = ["top5_in_fold", "top10_in_fold", "top20_in_fold", "top40_in_fold", "all_71",
             "mfcc_39", "pitch_intensity_9", "energy_4", "jitter_shimmer_11",
             "pulses_4", "voice_quality_4"]
    order = [k for k in order if k in fs]
    nice = {"top5_in_fold": "top 5 (in fold)", "top10_in_fold": "top 10 (in fold)",
            "top20_in_fold": "top 20 (in fold)", "top40_in_fold": "top 40 (in fold)",
            "all_71": "all 71 features", "mfcc_39": "MFCC only (39)",
            "pitch_intensity_9": "pitch / intensity (9)", "energy_4": "energy (4)",
            "jitter_shimmer_11": "jitter + shimmer (11)", "pulses_4": "pulses (4)",
            "voice_quality_4": "voice quality (4)"}
    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    y = np.arange(len(order))
    vals = [fs[k][0] for k in order]
    errs = [fs[k][1] for k in order]
    cols = [S1 if k.startswith("top") or k == "all_71" else S3 for k in order]
    ax.barh(y, vals, xerr=errs, capsize=3, color=cols, height=0.66,
            edgecolor=SURFACE, linewidth=1.5)
    ax.set_yticks(y); ax.set_yticklabels([nice[k] for k in order], fontsize=8)
    ax.invert_yaxis()
    for yi, v in zip(y, vals):
        ax.text(v + 0.004, yi, f"{v:.3f}", va="center", fontsize=8, color=INK2)
    ax.set_xlim(0.80, 1.045)
    ax.set_xlabel("subject AUROC")
    ax.set_title("Five features do the work of seventy-one\n"
                 "BenSParX, selection inside each training fold", loc="left")
    ax.legend(handles=[Line2D([0], [0], marker="s", ls="", ms=7, color=S1,
                              label="selected by rank / all"),
                       Line2D([0], [0], marker="s", ls="", ms=7, color=S3,
                              label="one feature family")],
              loc="lower right", fontsize=8)
    save(fig, "fig6_feature_saturation")




# ------------------------- Fig 7: predicted vs realised gain, expanded set
def fig7():
    """RQ1. Plots the LEVEL form (estimated reachable vs achieved), which shares
    no term with the baseline, alongside the difference form."""
    d = json.load(open(os.path.join(R, "endpoints_v2.json")))
    eps = [e for e in d["endpoints"] if e["auroc"] > 0.5]
    clusters = sorted(set(e["cluster"] for e in eps))
    cmap = {c: col for c, col in zip(clusters, [S1, S2, S3, S4])}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.2, 4.1))
    k = "64"
    # left: difference form
    H = [e["H"] for e in eps]
    g = [e["shots"][k]["mean"] - e["bacc_native"] for e in eps]
    lim = [min(min(H), min(g)) - 0.03, max(max(H), max(g)) + 0.04]
    axL.plot(lim, lim, color=MUTED, lw=1.2, ls="--", zorder=1)
    axL.axhline(0, color=AXIS, lw=0.9); axL.axvline(0, color=AXIS, lw=0.9)
    for e in eps:
        axL.scatter(e["H"], e["shots"][k]["mean"] - e["bacc_native"], s=60,
                    color=cmap[e["cluster"]], edgecolor=SURFACE, lw=1.2, zorder=4)
    axL.set_xlim(lim); axL.set_ylim(lim)
    axL.set_xlabel("estimated headroom H")
    axL.set_ylabel("realised gain from fitting\nthe decision point (k = 64)")
    axL.set_title("Difference form\n(shares the term -BACC_native)", loc="left")

    # right: level form, no shared term
    rc = [e["bacc_binormal"] for e in eps]
    ac = [e["shots"][k]["mean"] for e in eps]
    lim2 = [min(min(rc), min(ac)) - 0.03, max(max(rc), max(ac)) + 0.04]
    axR.plot(lim2, lim2, color=MUTED, lw=1.2, ls="--", zorder=1)
    for e in eps:
        axR.scatter(e["bacc_binormal"], e["shots"][k]["mean"], s=60,
                    color=cmap[e["cluster"]], edgecolor=SURFACE, lw=1.2, zorder=4)
    axR.set_xlim(lim2); axR.set_ylim(lim2)
    axR.set_xlabel("estimated reachable BACC")
    axR.set_ylabel("achieved BACC (k = 64)")
    axR.set_title("Level form\n(no shared term; Spearman 0.905, p = 0.005)", loc="left")

    handles = [Line2D([0], [0], marker="o", ls="", ms=7, color=cmap[c],
                      label=c.replace("::", " / ")) for c in clusters]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Estimated headroom predicts the realised gain across 8 endpoints",
                 x=0.005, ha="left", fontsize=10.5)
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    save(fig, "fig7_validation_expanded")


# ------------------------------- Fig 8: how robust is the headline statistic?
def fig8():
    hs = json.load(open(os.path.join(R, "harvest_sensitivity.json")))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.0, 3.6),
                                   gridspec_kw={"width_ratios": [1.1, 1]})
    cuts = sorted(float(k) for k in hs["sensitivity_H_cut"])
    shares = [100 * hs["sensitivity_H_cut"][str(c)][0] / hs["n_cells"] for c in cuts]
    cols = [S2 if abs(c - 0.02) < 1e-9 else S1 for c in cuts]
    axL.bar(range(len(cuts)), shares, color=cols, edgecolor=SURFACE, lw=1.5, width=0.62)
    for i, (c, s) in enumerate(zip(cuts, shares)):
        axL.text(i, s + 1.4, f"{s:.0f}%", ha="center", fontsize=8.5, color=INK2)
    axL.set_xticks(range(len(cuts)))
    axL.set_xticklabels([f"{c:.2f}" for c in cuts])
    axL.set_xlabel("cut-off on H used to call a cell decision-limited")
    axL.set_ylabel("share of the 36 audited cells")
    axL.set_ylim(0, 60)
    axL.set_title("The headline moves with an arbitrary cut-off\n"
                  "(orange = the value used in the paper)", loc="left")

    pp = hs["per_paper"]
    y = np.arange(len(pp) + 1)
    vals = [100 * p["A"] / p["n"] for p in pp] + [100 * hs["pooled_rate"]]
    labs = [f"{p['paper']}  (n={p['n']})" for p in pp] + ["pooled  (n=36)"]
    cols2 = [S1] * len(pp) + [S2]
    axR.barh(y, vals, color=cols2, edgecolor=SURFACE, lw=1.5, height=0.6)
    lo, hi = hs["cluster_boot_ci95"]
    axR.errorbar([100 * hs["pooled_rate"]], [len(pp)],
                 xerr=[[100 * hs["pooled_rate"] - 100 * lo],
                       [100 * hi - 100 * hs["pooled_rate"]]],
                 fmt="none", ecolor=INK, elinewidth=1.6, capsize=4, zorder=5)
    axR.set_yticks(y); axR.set_yticklabels(labs, fontsize=8.5); axR.invert_yaxis()
    for yi, v in zip(y, vals):
        axR.text(v + 1.5, yi, f"{v:.1f}%", va="center", fontsize=8.5, color=INK2)
    axR.set_xlim(0, 72); axR.set_xlabel("decision-limited cells (%)")
    axR.set_title("Rates differ by paper; the bar is the\n"
                  "paper-level bootstrap interval", loc="left")
    fig.tight_layout()
    save(fig, "fig8_sensitivity")


if __name__ == "__main__":
    # The expanded (phase-2) audit is the one the manuscript reports. Subgroup
    # rows are dropped here for the same reason they are kept out of the primary
    # statistic: the 14 rows partition a single cohort, so plotting them beside
    # the independent cells would show one dataset 14 times.
    hv2 = json.load(open(os.path.join(R, "harvest_v2.json")))
    cells = [c for c in hv2["cells"] if not c["is_subgroup"]]
    print("rendering figures:")
    fig1(cells); fig2(cells); fig3(); fig4(); fig5(); fig6(); fig7(); fig8()
    print("done")
