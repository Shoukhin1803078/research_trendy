"""
T4 - recompute the audit statistics on the expanded set.

Pre-committed in audit/PHASE2_PLAN.md section D: if the expansion lowers the
rate or widens heterogeneity, that is the result and the thesis is narrowed.
The H > 0.02 cut-off is NOT re-tuned after seeing the new data.
"""
import json, os, collections
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(HERE, "results")
d = json.load(open(f"{R}/harvest_v2.json"))
cells = d["cells"]

prim = [c for c in cells if "non-LLM baseline" not in c.get("note", "")
        and c["paper"] != "ours" and not c["is_subgroup"]]
papers = sorted(set(c["paper"] for c in prim))
by = {p: [c for c in prim if c["paper"] == p] for p in papers}

print("=== expanded audit: primary LLM cells ===")
print(f"  {len(prim)} cells from {len(papers)} papers (phase 1: 36 from 3)\n")
print(f"  {'paper':14} {'cells':>6} {'mode A':>7} {'rate':>7}")
for p in papers:
    a = sum(1 for c in by[p] if c["mode"] == "A_threshold_limited")
    print(f"  {p:14} {len(by[p]):6} {a:7} {100*a/len(by[p]):6.1f}%")
a_all = sum(1 for c in prim if c["mode"] == "A_threshold_limited")
print(f"  {'POOLED':14} {len(prim):6} {a_all:7} {100*a_all/len(prim):6.1f}%")
print(f"  phase-1 pooled was 12/36 = 33.3%")

rng = np.random.default_rng(0)
props = []
for _ in range(20000):
    pick = rng.integers(0, len(papers), len(papers))
    pool = [c for i in pick for c in by[papers[i]]]
    props.append(np.mean([c["mode"] == "A_threshold_limited" for c in pool]))
props = np.array(props)
lo, hi = np.quantile(props, [.025, .975])
print(f"\n  paper-level bootstrap: {100*props.mean():.1f}% "
      f"95% CI [{100*lo:.1f}, {100*hi:.1f}]   (phase 1: 34.3% [18.8, 50.0])")

print("\n=== heterogeneity across papers ===")
rates = [100 * sum(1 for c in by[p] if c["mode"] == "A_threshold_limited") / len(by[p])
         for p in papers]
print(f"  per-paper rates: {', '.join(f'{r:.1f}%' for r in rates)}")
print(f"  range {min(rates):.1f}% to {max(rates):.1f}%  (phase 1: 18.8% to 50.0%)")
print("  The spread WIDENED. The audited papers do not share a common rate, so a")
print("  single pooled percentage is a weak summary and is reported as such.")

print("\n=== sensitivity to the H cut-off (NOT re-tuned) ===")
sens = {}
for thr in (0.00, 0.01, 0.02, 0.05, 0.10):
    a = sum(1 for c in prim if c["auroc"] > 0.5 and c["H"] > thr)
    sens[thr] = a
    mark = "  <- primary" if abs(thr - 0.02) < 1e-9 else ""
    print(f"  H > {thr:.2f}: {a:2}/{len(prim)} = {100*a/len(prim):5.1f}%{mark}")

print("\n=== the new paper's primary cell ===")
nc = [c for c in prim if c["paper"] == "2607.29082"][0]
print(f"  {nc['endpoint']}: AUROC {nc['auroc']:.3f}, BACC {nc['bacc']:.3f}, "
      f"reachable {nc['reachable']:.3f}, H {nc['H']:+.3f} -> {nc['mode']}")
print("  It is NOT decision-limited, so the expansion slightly LOWERS the rate.")

print("\n=== within-paper heterogeneity: the subgroup rows (not counted as cells) ===")
sub = [c for c in cells if c["is_subgroup"]]
sa = [c for c in sub if c["mode"] == "A_threshold_limited"]
print(f"  {len(sub)} subgroup rows partition the same 17,106 evaluations.")
print(f"  {len(sa)} of them meet the decision-limited criterion:")
for c in sorted(sub, key=lambda x: -x["H"])[:5]:
    flag = " <- all-positive predictions" if c["sens"] > 0.98 else ""
    print(f"    {c['endpoint']:22} AUROC {c['auroc']:.3f} BACC {c['bacc']:.3f} "
          f"sens {c['sens']:.3f} spec {c['spec']:.3f} H {c['H']:+.3f}{flag}")
print("  The degenerate cells are the two poorest wealth quintiles. Within one")
print("  evaluation, the decision artifact is concentrated in the subgroups with")
print("  the fewest resources. Reported as an observation about THIS paper, not")
print("  as independent evidence, and not pooled into the primary statistic.")

print("\n=== eligibility: what the field reports ===")
elig = d["eligibility"]
p2 = [e for e in elig if e["phase"] == 2]
ineligible = [e for e in p2 if e["eligible"] is False]
print(f"  Papers examined in phase 2: {len(p2)}")
print(f"  Reporting NO threshold-dependent metric: {len(ineligible)} "
      f"({', '.join(e['paper'] for e in ineligible)})")
print(f"  Inaccessible: {sum(1 for e in p2 if e['eligible'] is None)}")
print("  CONSEQUENCE - a threat to validity, now explicit: our audit can only")
print("  include papers that report both metric families, and most do not.")
print("  The audited sample is therefore not representative of the literature.")

out = {"n_cells": len(prim), "n_papers": len(papers),
       "per_paper": [{"paper": p,
                      "A": sum(1 for c in by[p] if c["mode"] == "A_threshold_limited"),
                      "n": len(by[p])} for p in papers],
       "pooled_rate": float(a_all / len(prim)),
       "cluster_boot_mean": float(props.mean()),
       "cluster_boot_ci95": [float(lo), float(hi)],
       "per_paper_rate_range": [float(min(rates)), float(max(rates))],
       "sensitivity_H_cut": {str(k): [v, len(prim)] for k, v in sens.items()},
       "n_subgroup_rows": len(sub), "n_subgroup_mode_A": len(sa),
       "papers_examined_phase2": len(p2),
       "papers_no_threshold_metric": len(ineligible)}
json.dump(out, open(f"{R}/harvest_v2_stats.json", "w"), indent=1)
print(f"\nwrote {R}/harvest_v2_stats.json")
