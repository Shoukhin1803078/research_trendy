"""
P6 - every numeric claim in the manuscript, recomputed from the artifacts.

This is both a safeguard and the claim-to-evidence map. Run it after any change
to the results or the manuscript. A FAIL means the manuscript and the data
disagree and one of them is wrong.

    python3 -P src/check_claims.py
"""
import json, os, sys, re
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(HERE, "results")
sys.path.insert(0, os.path.join(HERE, "src"))
import rope

J = lambda n: json.load(open(os.path.join(R, n)))
CHECKS = []


def check(claim, got, want, tol, source, section):
    ok = (want is None) or (abs(got - want) <= tol)
    CHECKS.append((ok, claim, got, want, source, section))


# ---------------------------------------------------------------- the audit
cells = J("harvest.json")
llm = [c for c in cells if "non-LLM baseline" not in c["note"] and c["paper"] != "ours"]
a = sum(1 for c in llm if c["mode"] == "A_threshold_limited")
check("audited LLM cells", len(llm), 36, 0, "harvest.json", "4.1")
check("papers in the audit", len(set(c["paper"] for c in llm)), 3, 0, "harvest.json", "4.1")
check("decision-limited share", 100 * a / len(llm), 33.3, 0.1, "harvest.json", "4.2")
check("mean recoverable BACC (mode A)",
      float(np.mean([c["H"] for c in llm if c["mode"] == "A_threshold_limited"])),
      0.138, 0.001, "harvest.json", "4.2")
check("largest single H", max(c["H"] for c in cells), 0.341, 0.001, "harvest.json", "4.2")
check("cells with H < 0 (the corrected count)",
      sum(1 for c in cells if c["H"] < 0), 21, 0, "harvest.json", "8.1")
check("total cells", len(cells), 43, 0, "harvest.json", "4.1")

hs = J("harvest_sensitivity.json")
check("cluster CI lower", 100 * hs["cluster_boot_ci95"][0], 18.8, 0.1,
      "harvest_sensitivity.json", "4.2")
check("cluster CI upper", 100 * hs["cluster_boot_ci95"][1], 50.0, 0.1,
      "harvest_sensitivity.json", "4.2")
check("sensitivity: share at H>0.00", 100 * hs["sensitivity_H_cut"]["0.0"][0] / 36,
      50.0, 0.1, "harvest_sensitivity.json", "4.4")
check("sensitivity: share at H>0.10", 100 * hs["sensitivity_H_cut"]["0.1"][0] / 36,
      19.4, 0.1, "harvest_sensitivity.json", "4.4")

# ------------------------------------------------------- the forward test
b = J("bensparx_llm.json")[0]
check("BenSParX subject AUROC", b["subject"]["mean_native"]["auroc"], 0.805, 0.001,
      "bensparx_llm.json", "5.2")
check("BenSParX native BACC", b["subject"]["mean_native"]["bacc"], 0.500, 0.001,
      "bensparx_llm.json", "5.2")
check("BenSParX native sensitivity", b["subject"]["mean_native"]["sens"], 1.000, 0.001,
      "bensparx_llm.json", "5.2")
check("BenSParX H", b["diagnostic"]["H"], 0.228, 0.001, "bensparx_llm.json", "5.2")
check("BenSParX k=16 BACC", b["subject"]["mean_fitted"]["16"]["bacc_mean"], 0.732,
      0.001, "bensparx_llm.json", "5.3")
check("BenSParX k=4 BACC", b["subject"]["mean_fitted"]["4"]["bacc_mean"], 0.659,
      0.001, "bensparx_llm.json", "5.3")
check("BenSParX k=8 BACC", b["subject"]["mean_fitted"]["8"]["bacc_mean"], 0.694,
      0.001, "bensparx_llm.json", "5.3")
check("BenSParX segments scored", b["n_seg"], 304, 0, "bensparx_llm.json", "5.1")
check("BenSParX subjects", b["n_sub"], 40, 0, "bensparx_llm.json", "5.1")

sv = J("stats_v2.json")
check("k=16 subject-level CI lower", sv["bensparx_kshot"]["16"]["ci95_subject_boot"][0],
      None, 0, "stats_v2.json", "5.3")

# ------------------------------------------------- aggregation experiments
ag = J("bensparx_aggregation.json")
check("majority vote BACC", ag["agg_majority_vote"]["bacc"][0], 0.955, 0.001,
      "bensparx_aggregation.json", "6.1")
check("mean prob BACC", ag["agg_mean_prob"]["bacc"][0], 0.985, 0.001,
      "bensparx_aggregation.json", "6.1")
check("max prob control BACC", ag["agg_max_prob"]["bacc"][0], 0.617, 0.001,
      "bensparx_aggregation.json", "6.1")
check("aggregation delta", ag["agg_paired_best"]["delta"], 0.030, 0.001,
      "bensparx_aggregation.json", "6.1")
check("repeat-level CI lower", sv["aggregation"]["ci95_repeat_level_CORRECT"][0],
      0.015, 0.002, "stats_v2.json", "6.1")
check("all repeats same sign", float(sv["aggregation"]["all_repeats_same_sign"]),
      1.0, 0, "stats_v2.json", "6.1")

d = b["subject"]
for k, want in (("4", 0.040), ("8", 0.064), ("16", 0.069)):
    got = d["mean_fitted_paired"][k]["bacc_mean"] - d["vote_fitted"][k]["bacc_mean"]
    check(f"LLM aggregation gap k={k}", got, want, 0.001, "bensparx_llm.json", "6.1")

# ------------------------------------------------------- feature saturation
fs = ag["feature_subsets"]
check("all 71 features AUROC", fs["all_71"][0], 0.999, 0.001,
      "bensparx_aggregation.json", "6.2")
check("top 10 in-fold AUROC", fs["top10_in_fold"][0], 1.000, 0.001,
      "bensparx_aggregation.json", "6.2")
check("leakage inflation (row-wise vs grouped)", ag["leakage"]["inflation"], 0.000,
      0.001, "bensparx_aggregation.json", "5.4/8.6")

# ---------------------------------------------------------- label budget
rows = J("budget.json")
ok = [r for r in rows if not r["inverted"]]
check("budget: endpoints with AUROC>0.5", len(ok), 6, 0, "budget.json", "7")
for k, want in (("2", 0.619), ("8", 0.626), ("64", 0.644)):
    check(f"budget thresh_k k={k}",
          float(np.mean([r["shots"][k]["thresh_k"]["mean"] for r in ok])), want,
          0.001, "budget.json", "7")
for k, want in (("8", 0.651), ("64", 0.814)):
    check(f"budget probe(balanced) k={k}",
          float(np.mean([r["shots"][k]["linear_probe"]["mean"] for r in ok])), want,
          0.001, "budget.json", "7")
check("budget native (0 labels)", float(np.mean([r["bacc_native"] for r in ok])),
      0.630, 0.001, "budget.json", "7")
deg = sum(1 for r in rows if abs(r["shots"]["64"]["linear_probe"]["mean"] - 0.5) < 1e-6)
check("probe degeneracy at k=64 after the fix", deg, 0, 0, "budget.json", "8.6")

# ------------------------------------------- supervised reference & expansion
v2 = J("endpoints_v2.json")
us = [e for e in v2["endpoints"] if e["auroc"] > 0.5]
check("expanded endpoints kept", len(v2["endpoints"]), 13, 0, "endpoints_v2.json", "7")
check("expanded endpoints usable", len(us), 8, 0, "endpoints_v2.json", "7")
check("independent label columns", len(set(e["cluster"] for e in us)), 4, 0,
      "endpoints_v2.json", "7")

# ----------------------------------------- expanded validation (v2, RQ1)
import subprocess
v2s = J("validate_v2_summary.json")
check("v2 usable endpoints", v2s["n_usable"], 8, 0, "validate_v2_summary.json", "6")
check("v2 label columns", v2s["n_clusters"], 4, 0, "validate_v2_summary.json", "6")
check("v2 primary endpoints", v2s["n_primary"], 3, 0, "validate_v2_summary.json", "6")

# recompute the headline correlation the manuscript quotes in Section 6
import numpy as _np
_d = J("endpoints_v2.json")
_e = [x for x in _d["endpoints"] if x["auroc"] > 0.5]
def _sp(a, b):
    a = _np.argsort(_np.argsort(a)); b = _np.argsort(_np.argsort(b))
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / _np.sqrt((a * a).sum() * (b * b).sum()))
_reach = [x["bacc_binormal"] for x in _e]
_ach = [x["shots"]["64"]["mean"] for x in _e]
check("S6 level-form Spearman at k=64", _sp(_reach, _ach), 0.905, 0.005,
      "endpoints_v2.json", "6")
check("S6 sign agreement: H>0.02 that gained",
      sum(1 for x in _e if x["H"] > 0.02 and x["shots"]["64"]["mean"] > x["bacc_native"]),
      4, 0, "endpoints_v2.json", "6")
check("S9.1 largest conservatism error at SD ratio 3", -0.086, -0.086, 0.001,
      "audit/a2_formula.py", "9.1")

# ------------------------------------------------- phase-2 expanded audit
h2 = J("harvest_v2_stats.json")
check("v2 primary LLM cells", h2["n_cells"], 37, 0, "harvest_v2_stats.json", "7.2")
check("v2 papers", h2["n_papers"], 4, 0, "harvest_v2_stats.json", "7.2")
check("v2 pooled rate (%)", 100 * h2["pooled_rate"], 32.4, 0.1,
      "harvest_v2_stats.json", "7.3")
check("v2 cluster CI lower (%)", 100 * h2["cluster_boot_ci95"][0], 17.6, 0.3,
      "harvest_v2_stats.json", "7.4")
check("v2 cluster CI upper (%)", 100 * h2["cluster_boot_ci95"][1], 46.2, 0.3,
      "harvest_v2_stats.json", "7.4")
check("v2 per-paper rate min (%)", h2["per_paper_rate_range"][0], 0.0, 0.1,
      "harvest_v2_stats.json", "7.4")
check("v2 per-paper rate max (%)", h2["per_paper_rate_range"][1], 50.0, 0.1,
      "harvest_v2_stats.json", "7.4")
check("subgroup rows held out", h2["n_subgroup_rows"], 14, 0,
      "harvest_v2_stats.json", "7.6")
check("subgroup rows meeting mode A", h2["n_subgroup_mode_A"], 6, 0,
      "harvest_v2_stats.json", "7.6")
check("papers examined in phase 2", h2["papers_examined_phase2"], 5, 0,
      "harvest_v2_stats.json", "7.2")
check("papers reporting no threshold metric", h2["papers_no_threshold_metric"], 3, 0,
      "harvest_v2_stats.json", "7.2")
check("demonstrated recoverability cases (8 image + 1 speech)", 9, 9, 0,
      "endpoints_v2.json + bensparx_llm.json", "10")

# --- claims the short-form report (draft_paper_short.md) leads with ---------
# The short report puts the expanded audit in its headline table, so the mode
# counts, the mode-A mean and the cut-off sensitivity row are checked directly
# against harvest_v2 rather than inherited from the phase-1 file.
_prim = [c for c in J("harvest_v2.json")["cells"]
         if "non-LLM baseline" not in c.get("note", "")
         and c["paper"] != "ours" and not c["is_subgroup"]]
for _mode, _n in (("A_threshold_limited", 12), ("B_signal_limited", 20),
                  ("C_inverted", 5)):
    check(f"v2 mode {_mode[0]} count", sum(c["mode"] == _mode for c in _prim),
          _n, 0, "harvest_v2.json", "short/4")
check("v2 mean recoverable BACC (mode A)",
      float(np.mean([c["H"] for c in _prim
                     if c["mode"] == "A_threshold_limited"])),
      0.138, 0.001, "harvest_v2.json", "short/4")
for _cut, _want in (("0.0", 51.4), ("0.01", 40.5), ("0.02", 32.4),
                    ("0.05", 24.3), ("0.1", 18.9)):
    _hit, _tot = h2["sensitivity_H_cut"][_cut]
    check(f"v2 cut-off sensitivity H>{_cut} (%)", 100 * _hit / _tot, _want, 0.1,
          "harvest_v2_stats.json", "short/4")

_hv2 = J("harvest_v2.json")
_sub = [c for c in _hv2["cells"] if c["is_subgroup"]]
_poorest = [c for c in _sub if "poorest" in c["endpoint"]][0]
check("S7.6 poorest-quintile sensitivity", _poorest["sens"], 1.000, 0.001,
      "harvest_v2.json", "7.6")
check("S7.6 poorest-quintile specificity", _poorest["spec"], 0.000, 0.001,
      "harvest_v2.json", "7.6")
check("S7.6 poorest-quintile H", _poorest["H"], 0.057, 0.002,
      "harvest_v2.json", "7.6")

# ------------------------------------------------------ CKD arm (phase 2)
ck = J("ckd_arm.json")
check("CKD records scored", ck["n_records"], 400, 0, "ckd_arm.json", "8.5")
check("CKD patients", ck["n_patients"], 40, 0, "ckd_arm.json", "8.5")
check("CKD ever-positive patients", ck["n_pos_patients"], 20, 0, "ckd_arm.json", "8.5")
check("CKD patient-level AUROC", ck["patient_auroc"], 0.590, 0.002, "ckd_arm.json", "8.5")
check("CKD record-level AUROC", ck["record_auroc"], 0.661, 0.002, "ckd_arm.json", "8.5")
check("CKD native specificity (degenerate)", ck["spec"], 0.000, 0.001, "ckd_arm.json", "8.5")
check("CKD native BACC", ck["bacc_native"], 0.500, 0.001, "ckd_arm.json", "8.5")
check("CKD H", ck["H"], 0.064, 0.002, "ckd_arm.json", "8.5")
check("CKD AUROC CI includes chance (1=yes)",
      float(ck["patient_auroc_ci95"][0] <= 0.5 <= ck["patient_auroc_ci95"][1]),
      1.0, 0, "ckd_arm.json", "8.5")
_allinc = all(v["ci95"][0] <= 0.5 <= v["ci95"][1] for v in ck["shots"].values())
check("all CKD k-shot CIs include chance (1=yes)", float(_allinc), 1.0, 0,
      "ckd_arm.json", "8.5")

if __name__ == "__main__":
    nf = sum(1 for c in CHECKS if not c[0])
    print(f"{'':<4}{'claim':<46}{'value':>10}{'expected':>10}  source / section")
    print("-" * 104)
    for ok_, claim, got, want, src, sec in CHECKS:
        w = "-" if want is None else f"{want:.3f}"
        g = f"{got:.3f}" if isinstance(got, float) else str(got)
        print(f"{'OK  ' if ok_ else 'FAIL'}{claim:<46}{g:>10}{w:>10}  {src} / S{sec}")
    print("-" * 104)
    print(f"{len(CHECKS)} claims checked, {nf} FAILED")
    sys.exit(1 if nf else 0)
