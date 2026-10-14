"""
AUDIT A3 - the published-result audit: data integrity, clustering, robustness.
"""
import json, os, sys, collections
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
import rope

cells = json.load(open(os.path.join(HERE, "results", "harvest.json")))
llm = [c for c in cells if "non-LLM baseline" not in c["note"] and c["paper"] != "ours"]

print("=== A3.1 provenance and composition ===")
bypaper = collections.Counter(c["paper"] for c in cells)
for p, n in bypaper.items():
    nl = sum(1 for c in llm if c["paper"] == p)
    print(f"  {p:12} {n:3} cells total, {nl:3} in the LLM analysis")
print(f"  TOTAL {len(cells)} cells / {len(bypaper)} sources; LLM analysis uses "
      f"{len(llm)} cells from {len(set(c['paper'] for c in llm))} papers")
print(f"  derived BACC (from sens+spec rather than printed): "
      f"{sum(1 for c in cells if c['bacc_derived'])}")

print("\n=== A3.2 NON-INDEPENDENCE: cells cluster inside papers ===")
print("  The headline '33% of cells' treats 36 cells as 36 observations.")
print("  They are not: they come from 3 papers, and cells share models/datasets.")
cl = collections.Counter((c["paper"], c["mode"]) for c in llm)
for p in sorted(set(c["paper"] for c in llm)):
    tot = sum(1 for c in llm if c["paper"] == p)
    a = cl[(p, "A_threshold_limited")]
    print(f"  {p:12} A = {a:2}/{tot:2}  ({100*a/tot:5.1f}%)")
a_all = sum(1 for c in llm if c["mode"] == "A_threshold_limited")
print(f"  pooled       A = {a_all:2}/{len(llm):2}  ({100*a_all/len(llm):5.1f}%)")

# cluster bootstrap: resample PAPERS, not cells
rng = np.random.default_rng(0)
papers = sorted(set(c["paper"] for c in llm))
bycell = {p: [c for c in llm if c["paper"] == p] for p in papers}
props = []
for _ in range(20000):
    pick = rng.choice(len(papers), len(papers), replace=True)
    pool = [c for i in pick for c in bycell[papers[i]]]
    props.append(np.mean([c["mode"] == "A_threshold_limited" for c in pool]))
props = np.array(props)
print(f"  CLUSTER bootstrap over the 3 papers: "
      f"{100*props.mean():.1f}% [{100*np.quantile(props,0.025):.1f}, "
      f"{100*np.quantile(props,0.975):.1f}]")
# naive cell-level bootstrap for comparison
nv = [np.mean(rng.choice([c["mode"] == "A_threshold_limited" for c in llm],
                         len(llm), replace=True)) for _ in range(20000)]
print(f"  NAIVE cell bootstrap (wrong, ignores clustering): "
      f"{100*np.mean(nv):.1f}% [{100*np.quantile(nv,0.025):.1f}, "
      f"{100*np.quantile(nv,0.975):.1f}]")

print("\n=== A3.3 ROBUSTNESS of the A/B/C cut-offs ===")
print("  Mode A requires H > +0.02. That 0.02 is arbitrary. Vary it:")
for thr in (0.00, 0.01, 0.02, 0.05, 0.10):
    a = sum(1 for c in llm if c["auroc"] > 0.5 and c["H"] > thr)
    print(f"    H > {thr:.2f}:  A = {a:2}/{len(llm)}  ({100*a/len(llm):5.1f}%)")
print("  Mode C requires AUROC < 0.5. Vary that too:")
for thr in (0.45, 0.50, 0.55):
    c_ = sum(1 for c in llm if c["auroc"] < thr)
    print(f"    AUROC < {thr:.2f}: C = {c_:2}/{len(llm)}")

print("\n=== A3.4 cells where the REPORTED BACC exceeds the ESTIMATE ===")
over = [c for c in cells if c["H"] < 0]
print(f"  {len(over)}/{len(cells)} cells have H < 0 (reported above the binormal estimate)")
worst = sorted(over, key=lambda c: c["H"])[:5]
for c in worst:
    print(f"    {c['model']:16} {c['dataset']:18} AUROC {c['auroc']:.3f} "
          f"reported {c['bacc']:.3f} est {c['reachable']:.3f}  H {c['H']:+.3f}")
print("  A2.2 showed the binormal form UNDER-estimates when variances are unequal,")
print("  so H < 0 is consistent with non-Gaussian scores, not with an error.")
print("  CONSEQUENCE: cells classed 'signal-limited' may have real headroom that")
print("  the estimate misses. The 33% figure is therefore a LOWER bound on A,")
print("  not a point estimate - and that must be stated.")

print("\n=== A3.5 sanity: do reported sens/spec reproduce the reported BACC? ===")
bad = 0
for c in cells:
    if c["sens"] is not None and c["spec"] is not None and not c["bacc_derived"]:
        implied = 0.5 * (c["sens"] + c["spec"])
        if abs(implied - c["bacc"]) > 0.005:
            bad += 1
            print(f"    MISMATCH {c['paper']} {c['model']:16} {c['dataset']:16} "
                  f"printed BACC {c['bacc']:.4f} vs (sens+spec)/2 {implied:.4f}")
print(f"  {bad} mismatches found among cells with all three printed")
