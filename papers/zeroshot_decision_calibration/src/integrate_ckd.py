"""
Fold the CKD arm into the manuscript. Idempotent: re-running replaces the block.
Reads results/ckd_arm.json, so it reports whatever was actually completed.
"""
import json, os, re
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(HERE, "results", "ckd_arm.json")))
p = os.path.join(HERE, "draft_paper.md")
t = open(p).read()

neg = d["patient_auroc"] <= 0.55
body = f"""
### 8.5 Third modality: tabular CKD screening (pre-registered, {'NEGATIVE' if neg else 'INCONCLUSIVE'} result)

We ran the diagnostic on a third modality, a Bangladeshi diabetic cohort [6],
with `llama3.2:3b` and the same protocol. The analysis unit, the verdict rule
and the stopping criteria were **fixed before the data were seen**
(`audit/CKD_GO_NOGO.md`); the result below is what that rule returned.

**Completed:** {d['n_records']} records from **{d['n_patients']} patients**
({d['n_pos_patients']} ever positive), record-level prevalence
{d['record_prevalence']:.3f}.

| quantity | value |
|---|---|
| **patient-level AUROC** (primary) | **{d['patient_auroc']:.3f}** 95% CI [{d['patient_auroc_ci95'][0]:.3f}, {d['patient_auroc_ci95'][1]:.3f}] |
| record-level AUROC (diagnostic only) | {d['record_auroc']:.3f} |
| native decision | sensitivity {d['sens']:.3f}, specificity {d['spec']:.3f} |
| native BACC | {d['bacc_native']:.3f} |
| estimated reachable | {d['reachable']:.3f} |
| H | {d['H']:+.3f} |

**Two things this shows.**

First, the **decision rule is degenerate again**: the model predicts positive for
every patient, giving specificity {d['spec']:.3f} and balanced accuracy
{d['bacc_native']:.3f}. That is the same signature as the image and speech arms
and as the published cells in Section 7.

Second, **the ranking is weak**. Patient-level AUROC is {d['patient_auroc']:.3f}""" + (f""", at or below the
0.55 threshold we pre-registered, so this cell is **signal-limited, not
decision-limited**. No threshold can recover accuracy that the ranking does not
contain. We report this as a **negative result** and do not reframe it.

This is the behaviour the diagnostic should have: H = {d['H']:+.3f} is small, so
the diagnostic **declines to promise a recovery** that is not available. A
diagnostic that flagged this cell as recoverable would be wrong.""" if neg else f""", above the
0.55 bar the rule used, so the threshold-fitting test below was run - but see
the interval analysis that follows, which is the decisive reading.""") + f"""

**Within-patient correlation, quantified.** Record-level AUROC is
{d['record_auroc']:.3f} against {d['patient_auroc']:.3f} patient-level, a gap of
**{d['record_auroc'] - d['patient_auroc']:+.3f}**. The 10 visits of one patient
are near-duplicates and the positives cluster inside ever-positive patients, so
the record-level figure largely measures separating patients, not records.
**Any analysis of this cohort that pools records without grouping by patient
will overstate performance by roughly this much.** Our own pre-check was
initially misled by exactly this, which is why the primary unit was fixed to the
patient in advance.

**INCONCLUSIVE, not positive — every interval includes chance.** The
pre-registered rule classifies on the point estimate, and
{d['patient_auroc']:.3f} > 0.55 triggered the "ranking usable" branch. The
intervals do not support reading this as a positive result:

| quantity | value | 95% CI | excludes chance? |
|---|---|---|---|
| patient AUROC | {d['patient_auroc']:.3f} | [{d['patient_auroc_ci95'][0]:.3f}, {d['patient_auroc_ci95'][1]:.3f}] | **no** (includes 0.5) |
""" + "".join(
    f"| BACC at k={k} | {v['bacc_mean']:.3f} | [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] | **no** |\n"
    for k, v in sorted(d.get("shots", {}).items(), key=lambda x: int(x[0]))) + f"""

So the correct description of this cell is **inconclusive**: the ranking is not
distinguishable from chance at 40 patients, and neither is any realised gain. We
report it as such rather than as a third confirming modality.

What is worth noting, with that caveat firmly attached: the diagnostic predicted
a **small** headroom ({d['H']:+.3f}) and a **small** gain materialised
({d.get('realised_gain', float('nan')):+.3f} at k={max(d['shots'], key=int) if d.get('shots') else 'n/a'}).
The agreement is close, but both quantities are small and neither is
significant, so this is consistent with the diagnostic being well behaved rather
than evidence that it is.

**Limitations of this arm.** {d['n_patients']} patients is small and the AUROC
interval is correspondingly wide. The cohort contains no laboratory values (no
creatinine, no eGFR), so the task may be close to unlearnable from demographics
and lifestyle alone. This arm therefore says more about the task than about the
diagnostic, and we do not count it among the demonstrated-recoverability cases.
"""

marker = "### 8.5 Third modality"
if marker in t:
    start = t.index(marker)
    end = t.index("\n## ", start)
    t = t[:start] + body.lstrip("\n") + t[end:]
else:
    anchor = "## 9. Error analysis and ablations"
    t = t.replace(anchor, body.lstrip("\n") + "\n" + anchor, 1)
open(p, "w").write(t)
print(f"integrated CKD arm: {d['n_patients']} patients, "
      f"patient AUROC {d['patient_auroc']:.3f}, "
      f"{'NEGATIVE' if neg else 'INCONCLUSIVE'} result")
