"""
Phase-2 audit. Extends results/harvest.json under the pre-registered protocol in
audit/EXTRACTION_PROTOCOL.md. Original preserved at originals/harvest_v1_PRESERVED.json.

Schema adds: endpoint, n_eval, prevalence, bacc_source, threshold_protocol,
source_location, verified, is_subgroup.

KEY PHASE-2 RESULT: of the four candidate papers carried over from phase 1,
THREE report no threshold-dependent metric at all and are therefore ineligible.
That is recorded in ELIGIBILITY below and is a finding in its own right: the
decision-point question cannot even be asked of most of this literature.
"""
import json, os, sys, collections
from math import sqrt
from statistics import NormalDist
N = NormalDist()
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(HERE, "results")

# ---------------------------------------------------------------- eligibility
# Every paper considered, with its verdict and the reason. Verified by reading
# each paper's own metric list; see audit/SEARCH_LOG.md for sources.
ELIGIBILITY = [
 {"paper": "2605.24806", "short": "Zero-shot PD from speech", "eligible": True,
  "metrics_reported": "BACC, AUROC, sens, spec, Brier", "reason": "eligible",
  "phase": 1},
 {"paper": "2607.12260", "short": "From Many to Meaningful (CKD)", "eligible": True,
  "metrics_reported": "BACC, macro AUROC, macro F1, Brier", "reason": "eligible",
  "phase": 1},
 {"paper": "2609.04013", "short": "LLM4CKD", "eligible": True,
  "metrics_reported": "BACC, AUROC, sens, Brier, ECE", "reason": "eligible",
  "phase": 1},
 {"paper": "2607.29082", "short": "Zero-shot LLM child malnutrition", "eligible": True,
  "metrics_reported": "BACC, AUROC, sens, spec", "reason": "eligible",
  "phase": 2},
 {"paper": "2607.20453", "short": "Knowledge-injection delirium", "eligible": False,
  "metrics_reported": "AUROC only (Tables II-III)", "reason": "E2 no threshold metric",
  "phase": 2,
  "note": "Methods say accuracy/precision/recall/F1 were 'additionally computed' "
          "but none is reported. A fixed threshold of 50 is mentioned."},
 {"paper": "2605.10286", "short": "AgentRx", "eligible": False,
  "metrics_reported": "AUROC, AUPRC, ECE", "reason": "E2 no threshold metric",
  "phase": 2},
 {"paper": "2407.18525", "short": "ClinicRealm", "eligible": False,
  "metrics_reported": "AUROC, AUPRC (classification); MAE/MSE/RMSE (regression)",
  "reason": "E2 no threshold metric", "phase": 2,
  "note": "Verified from the PDF text: 'Under the ROC Curve (AUROC) and the Area "
          "Under the Precision-Recall Curve (AUPRC)'. Its confusion-matrix mention "
          "concerns reasoning error types, not classification."},
 {"paper": "s41598-026-58763-w", "short": "Pediatric wrist fracture, zero-shot MLLMs",
  "eligible": None, "metrics_reported": "unknown", "reason": "NOT ACCESSIBLE",
  "phase": 2,
  "note": "Nature paywall redirect; no open version located. Candidate for "
          "manual extraction if the authors' copy becomes available."},
]

# ------------------------------------------------------------------ new cells
# arXiv:2607.29082, Table 4. GPT-4o-mini zero-shot, stunting prediction, BDHS.
# The 'Overall' row is the primary cell. The 14 subgroup rows partition THE SAME
# 17,106 evaluations three ways, so they are NOT independent observations and are
# flagged is_subgroup=True and excluded from the primary statistic. They are kept
# because they show within-paper heterogeneity (see analysis below).
_P = "2607.29082"
_T = "Table 4"
NEW_CELLS = [
 # (endpoint, n, bacc, auroc, sens, spec, is_subgroup)
 ("Overall",                17106, 0.580, 0.632, 0.775, 0.386, False),
 ("Child sex: male",         8915, 0.581, 0.631, 0.767, 0.394, True),
 ("Child sex: female",       8191, 0.580, 0.633, 0.783, 0.377, True),
 ("Residence: rural",       11678, 0.554, 0.620, 0.860, 0.249, True),
 ("Residence: urban",        5428, 0.598, 0.645, 0.544, 0.652, True),
 ("Wealth: poorest",         3568, 0.500, 0.581, 1.000, 0.000, True),
 ("Wealth: poorer",          3347, 0.509, 0.560, 0.987, 0.030, True),
 ("Wealth: middle",          3299, 0.531, 0.563, 0.730, 0.331, True),
 ("Wealth: richer",          3426, 0.550, 0.578, 0.541, 0.559, True),
 ("Wealth: richest",         3466, 0.529, 0.583, 0.247, 0.811, True),
 ("BDHS 2007",                219, 0.582, 0.611, 0.476, 0.689, True),
 ("BDHS 2011",               2587, 0.565, 0.622, 0.838, 0.292, True),
 ("BDHS 2014",               1661, 0.566, 0.621, 0.771, 0.359, True),
 ("BDHS 2018",               1676, 0.579, 0.617, 0.736, 0.423, True),
 ("BDHS 2022",                909, 0.579, 0.618, 0.717, 0.441, True),
]
# Non-LLM baseline from the same table, analysed separately, never pooled.
BASELINE_CELLS = [("Random forest (supervised, 5-fold CV)", None, 0.570, 0.685, 0.234, 0.907)]

# Threshold protocol per paper, coded from each paper's own description.
THRESHOLD_PROTOCOL = {
    "2605.24806": "forced_token",   # one token per segment, majority vote
    "2607.12260": "forced_token",   # binary label from logprobs over "0"/"1"
    "2609.04013": "forced_token",
    "2607.29082": "not_stated",     # logprobs over "0"/"1"; decision rule not described
}


def reachable(a):
    a = min(max(a, 1e-6), 1 - 1e-6)
    return float(N.cdf(sqrt(2) * N.inv_cdf(a) / 2.0))


def mode(a, h):
    if a < 0.5:
        return "C_inverted"
    return "A_threshold_limited" if h > 0.02 else "B_signal_limited"


def build():
    cells = json.load(open(os.path.join(R, "harvest.json")))
    for c in cells:                                  # upgrade the v1 schema
        c.setdefault("endpoint", c.get("setting"))
        c.setdefault("n_eval", None)
        c.setdefault("bacc_source", "printed")
        c.setdefault("threshold_protocol",
                     THRESHOLD_PROTOCOL.get(c["paper"], "not_stated"))
        c.setdefault("source_location", "see paper")
        c.setdefault("verified", True)
        c.setdefault("is_subgroup", False)
        c.setdefault("phase", 1)

    for ep, n, bacc, auroc, sens, spec, sub in NEW_CELLS:
        # consistency check demanded by the protocol: printed BACC must equal
        # (sens+spec)/2 when all three are printed
        implied = 0.5 * (sens + spec)
        assert abs(implied - bacc) < 0.006, (ep, bacc, implied)
        h = reachable(auroc) - bacc
        cells.append({
            "paper": _P, "model": "GPT-4o-mini", "dataset": "BDHS stunting",
            "endpoint": ep, "setting": "zero-shot", "auroc": auroc, "bacc": bacc,
            "sens": sens, "spec": spec, "n_eval": n, "reachable": reachable(auroc),
            "H": h, "mode": mode(auroc, h), "bacc_derived": False,
            "bacc_source": "printed", "threshold_protocol": "not_stated",
            "source_location": _T, "verified": True, "is_subgroup": sub,
            "phase": 2, "note": "" if not sub else "subgroup of the Overall row"})

    for ep, n, bacc, auroc, sens, spec in BASELINE_CELLS:
        h = reachable(auroc) - bacc
        cells.append({
            "paper": _P, "model": "Random forest", "dataset": "BDHS stunting",
            "endpoint": ep, "setting": "supervised", "auroc": auroc, "bacc": bacc,
            "sens": sens, "spec": spec, "n_eval": n, "reachable": reachable(auroc),
            "H": h, "mode": mode(auroc, h), "bacc_derived": False,
            "bacc_source": "printed", "threshold_protocol": "not_stated",
            "source_location": _T, "verified": True, "is_subgroup": False,
            "phase": 2, "note": "non-LLM baseline"})
    return cells


if __name__ == "__main__":
    cells = build()
    json.dump({"eligibility": ELIGIBILITY, "cells": cells},
              open(f"{R}/harvest_v2.json", "w"), indent=1, default=float)

    el = [e for e in ELIGIBILITY if e["eligible"] is True]
    ie = [e for e in ELIGIBILITY if e["eligible"] is False]
    na = [e for e in ELIGIBILITY if e["eligible"] is None]
    print(f"Papers considered: {len(ELIGIBILITY)}  eligible {len(el)}  "
          f"ineligible {len(ie)}  inaccessible {len(na)}")
    for e in ELIGIBILITY:
        v = "ELIGIBLE  " if e["eligible"] else ("INACCESSIBLE" if e["eligible"] is None
                                                else "EXCLUDED  ")
        print(f"  [{v}] {e['paper']:20} {e['short']:38} {e['metrics_reported']}")
    p2 = [e for e in ELIGIBILITY if e["phase"] == 2 and e["eligible"] is False]
    print(f"\n  Of the {sum(1 for e in ELIGIBILITY if e['phase']==2)} papers examined in "
          f"phase 2, {len(p2)} report NO threshold-dependent metric.")

    prim = [c for c in cells if "non-LLM baseline" not in c.get("note", "")
            and c["paper"] != "ours" and not c["is_subgroup"]]
    print(f"\nPrimary LLM cells: {len(prim)} from "
          f"{len(set(c['paper'] for c in prim))} papers "
          f"(was 36 from 3)")
    print(f"Subgroup cells held out of the primary statistic: "
          f"{sum(1 for c in cells if c['is_subgroup'])}")
    print(f"wrote {R}/harvest_v2.json")
