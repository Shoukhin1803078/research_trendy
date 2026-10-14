"""
Meta-analysis: how much of the reported zero-shot clinical-LLM "failure" is a
decision-rule artifact?

Every cell below is transcribed from a PUBLISHED table. Nothing is re-run. For
each cell we compute, from the authors' own AUROC, the balanced accuracy a
well-placed threshold on the SAME scores would reach:

    d' = sqrt(2) Phi^-1(AUROC)        reachable = Phi(d'/2)
    H  = reachable - BACC(reported)

and classify the cell:

    A  threshold-limited   AUROC > 0.5 and H > +0.02   (rescuable, no new model)
    B  signal-limited      AUROC > 0.5 and H <= +0.02  (threshold already fine)
    C  sign-inverted       AUROC < 0.5                 (ranks backwards)

The binormal step assumes equal-variance Gaussian scores. It is an estimate,
not a bound: `bacc_oracle` in rope.py shows the empirical ceiling can sit a
little above or below it. Cells where the reported BACC already exceeds
`reachable` are reported as H < 0 rather than clipped.

PROVENANCE: `src` names the paper and table for every row. Metrics are copied
as printed; where a paper reports sensitivity and specificity but not balanced
accuracy, BACC = (sens + spec)/2 and that is flagged in `bacc_derived`.
"""
import json, os, sys
from math import sqrt
from statistics import NormalDist
N = NormalDist()

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# ---------------------------------------------------------------- the cells
# fields: paper, model, dataset, setting, auroc, bacc, sens, spec, note
CELLS = [
 # --- arXiv:2605.24806 "Zero-Shot Parkinson's Disease Detection from Speech",
 #     subject-level zero-shot table. BACC printed; sens/spec printed too.
 ("2605.24806", "LLaMA-3 8B",      "BenSParX (bn)", "handcrafted feats", 0.901, 0.8333, 0.8667, 0.8000, ""),
 ("2605.24806", "Qwen2-Audio 7B",  "BenSParX (bn)", "raw audio",         0.536, 0.5000, 0.1667, 0.8333, ""),
 ("2605.24806", "Pengi",           "BenSParX (bn)", "raw audio",         0.617, 0.5833, 0.4500, 0.7167, ""),
 ("2605.24806", "Audio-Reasoner",  "BenSParX (bn)", "raw audio",         0.534, 0.5083, 0.9167, 0.1000, ""),
 ("2605.24806", "LLaMA-3 8B",      "MDVR-KCL (en)", "handcrafted feats", 0.702, 0.5074, 0.0625, 0.9524, ""),
 ("2605.24806", "Qwen2-Audio 7B",  "MDVR-KCL (en)", "raw audio",         0.506, 0.5298, 0.2500, 0.8095, ""),
 ("2605.24806", "Pengi",           "MDVR-KCL (en)", "raw audio",         0.268, 0.2976, 0.5000, 0.0952, ""),
 ("2605.24806", "Audio-Reasoner",  "MDVR-KCL (en)", "raw audio",         0.609, 0.6949, 0.4375, 0.9524, ""),
 ("2605.24806", "LLaMA-3 8B",      "IPVS (it)",     "handcrafted feats", 0.805, 0.5179, 0.0357, 1.0000, ""),
 ("2605.24806", "Qwen2-Audio 7B",  "IPVS (it)",     "raw audio",         0.472, 0.5487, 0.6429, 0.4545, ""),
 ("2605.24806", "Pengi",           "IPVS (it)",     "raw audio",         0.188, 0.3263, 0.1071, 0.5455, ""),
 ("2605.24806", "Audio-Reasoner",  "IPVS (it)",     "raw audio",         0.395, 0.5195, 0.8571, 0.1818, ""),
 ("2605.24806", "LLaMA-3 8B",      "NeuroVoz (es)", "handcrafted feats", 0.486, 0.5258, 0.3913, 0.6604, ""),
 ("2605.24806", "Qwen2-Audio 7B",  "NeuroVoz (es)", "raw audio",         0.519, 0.6304, 0.2609, 1.0000, ""),
 ("2605.24806", "Pengi",           "NeuroVoz (es)", "raw audio",         0.517, 0.5304, 0.1739, 0.8868, ""),
 ("2605.24806", "Audio-Reasoner",  "NeuroVoz (es)", "raw audio",         0.676, 0.6390, 0.3913, 0.8868, ""),

 # --- arXiv:2607.12260 "From Many to Meaningful", Table 3, zero-shot CKD.
 #     BACC and macro AUROC printed; no sens/spec. Dataset-3 has no AUROC
 #     (single-class) so those six cells are excluded by construction.
 ("2607.12260", "Qwen-3 8B",   "CKD-1 Bangladesh", "all feats",      0.735, 0.6204, None, None, ""),
 ("2607.12260", "Qwen-3 8B",   "CKD-1 Bangladesh", "selected feats", 0.836, 0.7832, None, None, ""),
 ("2607.12260", "LLaMa-3 8B",  "CKD-1 Bangladesh", "all feats",      0.749, 0.5000, None, None, "BACC CI 50.00-50.00"),
 ("2607.12260", "LLaMa-3 8B",  "CKD-1 Bangladesh", "selected feats", 0.816, 0.5426, None, None, ""),
 ("2607.12260", "GPT-4o",      "CKD-1 Bangladesh", "all feats",      0.803, 0.6969, None, None, ""),
 ("2607.12260", "GPT-4o",      "CKD-1 Bangladesh", "selected feats", 0.851, 0.7711, None, None, ""),
 ("2607.12260", "Mistral 7B",  "CKD-1 Bangladesh", "all feats",      0.794, 0.5553, None, None, ""),
 ("2607.12260", "Mistral 7B",  "CKD-1 Bangladesh", "selected feats", 0.838, 0.7727, None, None, ""),
 ("2607.12260", "GPT-4o",      "CKD-2 India",      "all feats",      0.925, 0.8557, None, None, ""),
 ("2607.12260", "GPT-4o",      "CKD-2 India",      "selected feats", 0.943, 0.8800, None, None, ""),
 ("2607.12260", "Qwen-3 8B",   "CKD-2 India",      "all feats",      0.901, 0.8161, None, None, ""),
 ("2607.12260", "Qwen-3 8B",   "CKD-2 India",      "selected feats", 0.843, 0.8124, None, None, ""),
 ("2607.12260", "Mistral 7B",  "CKD-2 India",      "all feats",      0.822, 0.5347, None, None, ""),
 ("2607.12260", "Mistral 7B",  "CKD-2 India",      "selected feats", 0.932, 0.8522, None, None, ""),
 ("2607.12260", "LLaMa-3 8B",  "CKD-2 India",      "all feats",      0.923, 0.5020, None, None, "macro F1 27.68"),
 ("2607.12260", "LLaMa-3 8B",  "CKD-2 India",      "selected feats", 0.849, 0.7645, None, None, ""),

 # --- arXiv:2609.04013 "LLM4CKD", Table II, few-shot, selected feats, Dataset-1.
 ("2609.04013", "Qwen-3 (LLM)", "CKD-1 Bangladesh", "4-shot",  0.867, 0.814, 0.873, None, ""),
 ("2609.04013", "Qwen-3 (LLM)", "CKD-1 Bangladesh", "8-shot",  0.877, 0.712, 0.509, None, ""),
 ("2609.04013", "Qwen-3 (LLM)", "CKD-1 Bangladesh", "16-shot", 0.878, 0.782, 0.736, None, ""),
 ("2609.04013", "Qwen-3 (LLM)", "CKD-1 Bangladesh", "32-shot", 0.874, 0.764, 0.636, None, ""),
 ("2609.04013", "TabPFN",       "CKD-1 Bangladesh", "4-shot",  0.813, 0.658, 0.436, None, "non-LLM baseline"),
 ("2609.04013", "TabPFN",       "CKD-1 Bangladesh", "32-shot", 0.871, 0.820, 0.736, None, "non-LLM baseline"),
 ("2609.04013", "NODE",         "CKD-1 Bangladesh", "4-shot",  0.863, 0.768, 0.718, None, "non-LLM baseline"),
 ("2609.04013", "NODE",         "CKD-1 Bangladesh", "32-shot", 0.871, 0.797, 0.709, None, "non-LLM baseline"),
 ("2609.04013", "MLP",          "CKD-1 Bangladesh", "4-shot",  0.827, 0.754, 0.645, None, "non-LLM baseline"),
 ("2609.04013", "MLP",          "CKD-1 Bangladesh", "32-shot", 0.814, 0.771, 0.736, None, "non-LLM baseline"),

 # --- ours, measured (not published elsewhere)
 ("ours", "llama3.2:3b", "BenSParX (bn)", "10 feats, subject-level", 0.805, 0.5000, 1.0000, 0.0000, "our run"),
]


def reachable(a):
    """Balanced accuracy at the optimal threshold, binormal, SAME sign as the
    model. No flip: an inverted score cannot be rescued by a monotone rule."""
    a = min(max(a, 1e-6), 1 - 1e-6)
    return float(N.cdf(sqrt(2) * N.inv_cdf(a) / 2.0))


def classify(a, h):
    if a < 0.5:
        return "C_inverted"
    return "A_threshold_limited" if h > 0.02 else "B_signal_limited"


def main():
    rows = []
    for paper, model, ds, setting, a, bacc, sens, spec, note in CELLS:
        derived = False
        if bacc is None and sens is not None and spec is not None:
            bacc, derived = 0.5 * (sens + spec), True
        r = reachable(a)
        h = r - bacc
        rows.append({"paper": paper, "model": model, "dataset": ds,
                     "setting": setting, "auroc": a, "bacc": bacc,
                     "sens": sens, "spec": spec, "reachable": r, "H": h,
                     "mode": classify(a, h), "bacc_derived": derived,
                     "note": note})

    print(f"{len(rows)} published/measured cells harvested\n")
    print(f"{'paper':12} {'model':17} {'dataset':19} {'setting':24} "
          f"{'AUROC':>6} {'BACC':>6} {'reach':>6} {'H':>7}  mode")
    print("-" * 120)
    for r in sorted(rows, key=lambda x: -x["H"]):
        print(f"{r['paper']:12} {r['model']:17} {r['dataset']:19} {r['setting']:24} "
              f"{r['auroc']:6.3f} {r['bacc']:6.3f} {r['reachable']:6.3f} "
              f"{r['H']:+7.3f}  {r['mode']}")

    print("\n--- distribution of failure modes ---")
    import collections
    c = collections.Counter(r["mode"] for r in rows)
    for m in ("A_threshold_limited", "B_signal_limited", "C_inverted"):
        print(f"  {m:22} {c[m]:3}/{len(rows)}  ({100*c[m]/len(rows):.0f}%)")

    llm = [r for r in rows if "non-LLM baseline" not in r["note"] and r["paper"] != "ours"]
    cl = collections.Counter(r["mode"] for r in llm)
    print(f"\n  restricted to published LLM cells (n={len(llm)}):")
    for m in ("A_threshold_limited", "B_signal_limited", "C_inverted"):
        print(f"  {m:22} {cl[m]:3}/{len(llm)}  ({100*cl[m]/len(llm):.0f}%)")

    a_cells = [r for r in llm if r["mode"] == "A_threshold_limited"]
    if a_cells:
        tot = sum(r["H"] for r in a_cells) / len(a_cells)
        print(f"\n  mean recoverable BACC in threshold-limited LLM cells: {tot:+.3f}")
        print(f"  largest: " + ", ".join(
            f"{r['model']}/{r['dataset']} {r['H']:+.3f}"
            for r in sorted(a_cells, key=lambda x: -x["H"])[:4]))

    os.makedirs(OUT, exist_ok=True)
    json.dump(rows, open(f"{OUT}/harvest.json", "w"), indent=1)
    print(f"\nwrote {OUT}/harvest.json")


if __name__ == "__main__":
    main()


# ------------------------------------------------------------------ analysis 2
def feature_selection_confound():
    """arXiv:2607.12260's central claim is that a compact feature subset enables
    "clinically meaningful zero-shot CKD screening". Its evidence is balanced
    accuracy, which is threshold-dependent. Its own table also reports macro
    AUROC, which is not.

    For each (model, dataset) we therefore ask two separate questions:
      does selection improve the RANKING?   (delta AUROC)
      does selection improve the DECISION?  (delta BACC)
    and compare the reported selected-feature BACC against what a threshold fix
    on the ALL-feature scores would already have reached.
    """
    rows = [c for c in CELLS if c[0] == "2607.12260"]
    bykey = {}
    for paper, model, ds, setting, a, bacc, sens, spec, note in rows:
        bykey.setdefault((model, ds), {})[setting] = (a, bacc)

    print("\n\n=== Does feature selection fix the ranking or just the threshold? ===")
    print("    (arXiv:2607.12260 Table 3; 'reach(all)' = BACC a threshold fix on the")
    print("     all-feature scores would reach, from the authors' own AUROC)\n")
    print(f"{'model':13} {'dataset':19} {'AUROC all':>9} {'AUROC sel':>9} {'dAUROC':>7} "
          f"| {'BACC all':>8} {'BACC sel':>8} {'dBACC':>7} | {'reach(all)':>10} {'verdict':>22}")
    print("-" * 125)
    rank_worse = thresh_beats_sel = 0
    for (model, ds), d in sorted(bykey.items()):
        if "all feats" not in d or "selected feats" not in d:
            continue
        a_all, b_all = d["all feats"]
        a_sel, b_sel = d["selected feats"]
        da, db = a_sel - a_all, b_sel - b_all
        r_all = reachable(a_all)
        if da < 0:
            rank_worse += 1
            verdict = "ranking got WORSE"
        elif da < 0.02:
            verdict = "ranking ~unchanged"
        else:
            verdict = "ranking improved"
        if r_all > b_sel:
            thresh_beats_sel += 1
            verdict += " *"
        print(f"{model:13} {ds:19} {a_all:9.3f} {a_sel:9.3f} {da:+7.3f} "
              f"| {b_all:8.3f} {b_sel:8.3f} {db:+7.3f} | {r_all:10.3f} {verdict:>22}")
    n = sum(1 for d in bykey.values() if "all feats" in d and "selected feats" in d)
    print(f"\n  selection made the RANKING worse in {rank_worse}/{n} (model, dataset) pairs")
    print(f"  * = a threshold fix on the ALL-feature scores already beats the reported")
    print(f"      selected-feature BACC: {thresh_beats_sel}/{n} pairs")
    print("\n  Reading: where dAUROC < 0 but dBACC > 0, selection did not add information.")
    print("  It moved the operating point. That is a decision-rule effect reported as a")
    print("  feature-engineering result.")


if __name__ == "__main__":
    feature_selection_confound()
