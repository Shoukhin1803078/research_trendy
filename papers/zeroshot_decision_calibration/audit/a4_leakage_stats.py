"""
AUDIT A4 - leakage scan and statistical-design check.
"""
import json, os, sys, re, glob
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
import rope

print("=== A4.1 where do TEST labels touch a threshold or a parameter? ===")
findings = [
 ("run_arm_a.py", "pi_true = float(y.mean())  on the TEST split",
  "ORACLE, correctly labelled 'ROPE-Q (pi known)'. Used only to REFUTE the "
  "method. Not a reported method. ACCEPTABLE, but must stay labelled oracle."),
 ("rope.py bacc_oracle", "picks the best threshold on the evaluation data",
  "OPTIMISTIC by construction. Documented in the docstring. Must never be "
  "quoted as an achievable result. A2.5 measured the bias: +0.054 at n=40."),
 ("run_arm_a.py", "rope.headroom(y, s) on the TEST split",
  "H is computed from test AUROC and test native BACC. The gain it is "
  "correlated against is also measured on test. A1 quantified the overlap."),
 ("run_budget.py", "support from split__train, evaluation on split__test",
  "CLEAN. Threshold and probe fitted on train-drawn support only."),
 ("analyze_bensparx.py", "threshold fitted on k subjects, scored on the rest",
  "CLEAN. Held-out mask excludes the support subjects."),
 ("bensparx_aggregation.py", "StratifiedGroupKFold(groups=subject)",
  "CLEAN. Feature selection is inside the training fold."),
]
for f, what, verdict in findings:
    print(f"  [{f}]\n     {what}\n     -> {verdict}")

print("\n=== A4.2 are the 25 'folds' in the aggregation test independent? ===")
print("  Design: 5 repeats x 5-fold CV = 25 fold-results, then a paired")
print("  bootstrap RESAMPLES THOSE 25 as if independent.")
print("  They are not: within one repeat the 5 folds partition the same 120")
print("  subjects, and the 5 repeats reuse all of the same subjects.")
agg = json.load(open(os.path.join(HERE, "results", "bensparx_aggregation.json")))
d = agg["agg_paired_best"]
print(f"  reported: delta {d['delta']:+.4f} CI {d['ci95']} over n={d['n_folds']} folds")
print("  The effective sample size is closer to the number of REPEATS (5) than")
print("  to 25, so that CI is too narrow. A repeat-level bootstrap is the fix.")

print("\n=== A4.3 BenSParX LLM arm: is the k-shot CI reported? ===")
b = json.load(open(os.path.join(HERE, "results", "bensparx_llm.json")))[0]
mf = b["subject"]["mean_fitted"]
for k in sorted(mf, key=int):
    v = mf[k]
    n_eval = b["n_sub"] - int(k)
    print(f"  k={k:>2}: BACC {v['bacc_mean']:.3f} +- {v['bacc_std']:.3f} (sd over 50 draws), "
          f"evaluated on {n_eval} held-out subjects")
print("  The +- is the SD across draws, not a CI on the mean, and the draws")
print("  overlap heavily (50 draws of k from 40 subjects). Needs a stated")
print("  interpretation or a subject-level bootstrap.")
print(f"  diagnostic bacc_oracle = {b['diagnostic']['bacc_oracle']:.3f} vs "
      f"binormal {b['diagnostic']['bacc_binormal']:.3f}")
print("  -> on our own data the EMPIRICAL ceiling exceeds the binormal estimate,")
print("     the same direction A2.2 predicts. Evidence the estimate is conservative.")

print("\n=== A4.4 claims in the manuscript vs numbers on disk ===")
checks = []
cells = json.load(open(os.path.join(HERE, "results", "harvest.json")))
n_neg = sum(1 for c in cells if c["H"] < 0)
checks.append(("draft_paper 8.1 'In 6 of our 43 cells the reported BACC is "
                "already above reachable'", f"actual count = {n_neg}", n_neg == 6))
llm = [c for c in cells if "non-LLM baseline" not in c["note"] and c["paper"] != "ours"]
a = sum(1 for c in llm if c["mode"] == "A_threshold_limited")
checks.append(("'33% of published LLM cells'", f"{a}/{len(llm)} = {100*a/len(llm):.1f}%",
               abs(100*a/len(llm) - 33.3) < 0.5))
mean_h = np.mean([c["H"] for c in llm if c["mode"] == "A_threshold_limited"])
checks.append(("'mean recoverable +0.138'", f"{mean_h:+.3f}", abs(mean_h - 0.138) < 0.001))
rows = json.load(open(os.path.join(HERE, "results", "budget.json")))
ok = [r for r in rows if not r["inverted"]]
p64 = np.mean([r["shots"]["64"]["linear_probe"]["mean"] for r in ok])
checks.append(("'probe reaches 0.814 at k=64'", f"{p64:.3f}", abs(p64 - 0.814) < 0.001))
for claim, actual, good in checks:
    print(f"  [{'OK ' if good else 'FAIL'}] {claim}\n         -> {actual}")
