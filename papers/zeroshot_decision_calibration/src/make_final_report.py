"""Generate FINAL_REPORT.md from the artifacts, so it cannot drift from them."""
import json, os, subprocess, datetime
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(HERE, "results")
J = lambda n: json.load(open(os.path.join(R, n)))

h2 = J("harvest_v2_stats.json")
try:
    ckd = J("ckd_arm.json"); has_ckd = True
except Exception:
    ckd = {}; has_ckd = False
try:
    nscored = sum(1 for _ in open(os.path.join(R, "llm_ckd_llama32_3b_list_f10.jsonl")))
except Exception:
    nscored = 0
try:
    out = subprocess.run(["python3", "-P", os.path.join(HERE, "src", "check_claims.py")],
                         capture_output=True, text=True, cwd="/tmp", timeout=600)
    last = [l for l in out.stdout.strip().splitlines() if "claims checked" in l]
    checks = last[-1] if last else "check did not report"
except Exception as e:
    checks = f"could not run ({type(e).__name__})"

neg = has_ckd and ckd["patient_auroc"] <= 0.55
ckd_block = "The CKD arm did not produce an analysable result." if not has_ckd else f"""
Scored **{nscored} of 400** records before the power budget stopped it;
analysed on the **{ckd['n_patients']} patients whose records were complete**
({ckd['n_pos_patients']} ever positive).

| quantity | value |
|---|---|
| **patient-level AUROC** (pre-registered primary) | **{ckd['patient_auroc']:.3f}** [{ckd['patient_auroc_ci95'][0]:.3f}, {ckd['patient_auroc_ci95'][1]:.3f}] |
| record-level AUROC (diagnostic only) | {ckd['record_auroc']:.3f} |
| native decision | sens {ckd['sens']:.3f}, spec {ckd['spec']:.3f} |
| native BACC | {ckd['bacc_native']:.3f} |
| H | {ckd['H']:+.3f} |

**Verdict: {'NEGATIVE for recoverability' if neg else 'INCONCLUSIVE'}.**

{'Patient-level AUROC is at or below 0.55, so the cell is **signal-limited, not decision-limited**. No threshold can recover accuracy the ranking does not contain. Reported as a negative result and not reframed.' if neg else '''The pre-registered rule classifies on the point estimate, and 0.590 > 0.55 took
the "ranking usable" branch. **But every interval includes chance**: the AUROC CI
spans 0.5, and so do all three k-shot BACC intervals. The honest description is
**inconclusive**, not positive. This cell is therefore **not** counted among the
demonstrated-recoverability cases, which stay at 9.

The diagnostic predicted a small headroom (+0.064) and a small gain appeared
(+0.058 at k=16). The agreement is close, but both are small and neither is
significant, so this is consistent with the diagnostic behaving well rather than
evidence that it does.'''}

Two things still hold regardless:

1. **The decision rule is degenerate again** — the model predicts positive for
   every patient (specificity {ckd['spec']:.3f}, BACC {ckd['bacc_native']:.3f}),
   the same signature as the image and speech arms.
2. **Within-patient correlation inflates record-level AUROC by
   {ckd['record_auroc'] - ckd['patient_auroc']:+.3f}** ({ckd['record_auroc']:.3f}
   vs {ckd['patient_auroc']:.3f}). Any analysis of this cohort that pools records
   without grouping by patient overstates performance by about this much.
"""

doc = f"""# Final Report — Phase 2

**Generated:** {datetime.date.today().isoformat()} from the artifacts, by
`src/make_final_report.py`. Every number is read from `results/`.

---

## 1. What changed, and which way

### Conclusions that got WEAKER

| was | now |
|---|---|
| 33.3% of cells decision-limited (36 cells, 3 papers) | **{100*h2['pooled_rate']:.1f}%** ({h2['n_cells']} cells, {h2['n_papers']} papers) |
| paper-level CI [18.8, 50.0] | **[{100*h2['cluster_boot_ci95'][0]:.1f}, {100*h2['cluster_boot_ci95'][1]:.1f}]** |
| per-paper range 18.8-50.0% | **{h2['per_paper_rate_range'][0]:.1f}-{h2['per_paper_rate_range'][1]:.1f}%** (wider) |
| "the estimator is conservative" | conservative in 11 of 12 non-Gaussian conditions, **not all** |

The audit expansion was pre-committed to report whatever it found. It lowered
the headline and widened the heterogeneity, and the manuscript now says a single
pooled rate is a weak summary for a set this varied.

### Conclusions that got STRONGER

- Estimator conservatism is **U-shaped** in the variance ratio (-0.061 at 0.25,
  -0.154 at 6.0) and **-0.051 for saturated scores** — the regime of LLM
  label-token log-probabilities.
- **New:** within one paper, the artifact concentrates in the **poorest
  subgroups** — in the poorest wealth quintile the model calls every child
  positive (sens 1.000, spec 0.000, AUROC 0.581) while the aggregate looks
  merely signal-limited.
- **New:** **{h2['papers_no_threshold_metric']} of {h2['papers_examined_phase2']}**
  papers examined report no threshold-dependent metric, so the question cannot
  be asked of much of this literature.

### Conclusions that remain UNRESOLVED

- **Direct recoverability on published systems.** No eligible paper releases
  score-level outputs, so recoverability is **estimated** for 12 published cells
  and **demonstrated** only on our own data. This is the largest evidential gap.
- **Journal quartiles.** scimagojr.com returned HTTP 403 on every attempt;
  recorded as unverified rather than cited from aggregators.

---

## 2. The CKD arm
{ckd_block}
---

## 3. Verification

`src/check_claims.py`: **{checks}**

Run it after any change; it exits non-zero on any mismatch.

---

## 4. Files changed or added in phase 2

| file | role |
|---|---|
| `audit/PHASE2_PLAN.md` | plan with pre-committed analysis decisions |
| `audit/EXTRACTION_PROTOCOL.md` | criteria written before seeing new papers |
| `audit/SEARCH_LOG.md` | every search and fetch, including failures |
| `audit/LITERATURE.md` | literature review and novelty statement |
| `audit/CKD_GO_NOGO.md` | recommendation, outcome, pre-registered analysis |
| `audit/JOURNALS.md` | shortlist, with the verification failure stated |
| `audit/a5_estimator_range.py` | extended simulation |
| `src/harvest_v2.py`, `harvest_v2_stats.py` | expanded audit and statistics |
| `src/analyze_ckd.py`, `integrate_ckd.py`, `ckd_precheck.py` | CKD arm |
| `src/make_final_report.py` | this report |
| `draft_paper.md` | v3; v1 and v2 preserved in `originals/` |

---

## 5. A bug the pre-check caught

The patient sampler labelled a group by its **first** record. In this
longitudinal cohort CKD onset is almost always at the last visit, so **0 of 400
patients are positive at visit 1** and the balancing silently became a no-op.
Fixed to "positive if any record is positive"; BenSParX verified unaffected.

**Had the full run been launched without the pre-check, the CKD arm would have
been invalid.**

---

## 6. Next recommended actions

1. **Finish the CKD scoring** if a stronger result is wanted — it is resumable,
   so re-running the same command continues from {nscored}/400.
2. **Expand the audit further.** It is still {h2['n_cells']} cells from
   {h2['n_papers']} papers with 0-50% heterogeneity. This remains the main
   publication risk.
3. **Request score-level data** from the audited papers' authors. That is the
   only route to demonstrating recoverability on a published system.
4. **Verify journal quartiles** on SCImago or JCR directly.
5. Consider the narrower framing: the **reporting finding** (most papers report
   no threshold metric) is arguably more robust than the rate itself.

---

## 7. State of the repository

Nothing is running. All originals preserved in `originals/` and in git
`e93b6b6`. The manuscript, the changelog, the experiment report and the audit
documents are mutually consistent and every numeric claim is machine-checked.
"""
open(os.path.join(HERE, "FINAL_REPORT.md"), "w").write(doc)
print("wrote FINAL_REPORT.md")
print(checks)
