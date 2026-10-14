"""
P2 - the audited-sample statistic, with the uncertainty it requires.

The headline "33% of published LLM cells are decision-limited" needs three
things it did not have:
  1. an interval that respects CLUSTERING (36 cells come from 3 papers),
  2. a SENSITIVITY analysis on the arbitrary H > 0.02 cut-off,
  3. an explicit statement that it is a property of the AUDITED SAMPLE.
"""
import json, os, sys, collections
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(HERE, "results")
cells = json.load(open(f"{R}/harvest.json"))
llm = [c for c in cells if "non-LLM baseline" not in c["note"] and c["paper"] != "ours"]
papers = sorted(set(c["paper"] for c in llm))
by = {p: [c for c in llm if c["paper"] == p] for p in papers}

print(f"Audited sample: {len(llm)} result cells from {len(papers)} papers.")
print("This is a statement about THESE papers, not about clinical LLM research.\n")

print("=== per-paper rates (they differ a lot) ===")
rows = []
for p in papers:
    a = sum(1 for c in by[p] if c["mode"] == "A_threshold_limited")
    rows.append((p, a, len(by[p])))
    print(f"  {p:12} {a:2}/{len(by[p]):2} = {100*a/len(by[p]):5.1f}%")
a_all = sum(1 for c in llm if c["mode"] == "A_threshold_limited")
print(f"  pooled       {a_all:2}/{len(llm):2} = {100*a_all/len(llm):5.1f}%")

print("\n=== cluster bootstrap over papers (the correct unit) ===")
rng = np.random.default_rng(0)
props = []
for _ in range(20000):
    pick = rng.integers(0, len(papers), len(papers))
    pool = [c for i in pick for c in by[papers[i]]]
    props.append(np.mean([c["mode"] == "A_threshold_limited" for c in pool]))
props = np.array(props)
lo, hi = np.quantile(props, [.025, .975])
print(f"  {100*props.mean():.1f}%  95% CI [{100*lo:.1f}, {100*hi:.1f}]")
print(f"  With only {len(papers)} clusters this interval can only take a few")
print(f"  values; it is a floor on the uncertainty, not a precise interval.")

print("\n=== sensitivity to the H cut-off ===")
sens = {}
for thr in (0.00, 0.01, 0.02, 0.05, 0.10):
    a = sum(1 for c in llm if c["auroc"] > 0.5 and c["H"] > thr)
    sens[thr] = (a, len(llm))
    mark = "  <- used in the paper" if abs(thr - 0.02) < 1e-9 else ""
    print(f"  H > {thr:.2f}:  {a:2}/{len(llm)} = {100*a/len(llm):5.1f}%{mark}")
print("  The headline moves from 50% to 19% across reasonable cut-offs, so the")
print("  cut-off must be reported with the number.")

print("\n=== sensitivity to the AUROC cut for sign-inversion ===")
for thr in (0.45, 0.50, 0.55):
    c_ = sum(1 for c in llm if c["auroc"] < thr)
    print(f"  AUROC < {thr:.2f}:  C = {c_:2}/{len(llm)} = {100*c_/len(llm):5.1f}%")

print("\n=== is the estimate conservative? (direction of the error) ===")
neg = [c for c in cells if c["H"] < 0]
print(f"  {len(neg)}/{len(cells)} cells have reported BACC ABOVE the estimate.")
print("  audit/a2_formula.py shows the binormal form under-estimates when the")
print("  two score distributions have different spreads, so this is expected.")
print("  Therefore the A-count is a LOWER BOUND: cells we do not flag may still")
print("  have headroom the estimate cannot see.")

out = {"n_cells": len(llm), "n_papers": len(papers),
       "per_paper": [{"paper": p, "A": a, "n": n} for p, a, n in rows],
       "pooled_rate": float(a_all / len(llm)),
       "cluster_boot_mean": float(props.mean()),
       "cluster_boot_ci95": [float(lo), float(hi)],
       "sensitivity_H_cut": {str(k): list(v) for k, v in sens.items()},
       "n_cells_H_negative": len(neg), "n_cells_total": len(cells),
       "interpretation": "lower bound on decision-limited prevalence in the "
                         "audited sample; not a population estimate"}
json.dump(out, open(f"{R}/harvest_sensitivity.json", "w"), indent=1)
print(f"\nwrote {R}/harvest_sensitivity.json")
