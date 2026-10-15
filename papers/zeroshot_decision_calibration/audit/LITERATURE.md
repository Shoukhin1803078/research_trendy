# Literature Review and Novelty Statement

**Date:** 2026-10-14. Covers `feedback2.md` Priority 4.
**Verification policy:** each entry is marked **[verified]** (read directly) or
**[snippet]** (seen only in a search result and not retrievable). Nothing
marked [snippet] is used to support a numeric claim in the manuscript.

---

## 1. Signal detection theory

**Green & Swets, *Signal Detection Theory and Psychophysics*, 1966.** [standard]
The binormal model and the AUROC-to-d' relationship. Our estimator is this
result applied unchanged. We claim no novelty for it, and the manuscript says so
in §2 and §9.

## 2. Clinical prediction-model evaluation and reporting

**TRIPOD+AI. Collins et al., BMJ 2024;385:e078378** (April 2024; erratum
BMJ 2024;385:q902). [verified]
27-item reporting checklist for clinical prediction models using regression or
machine learning; supersedes TRIPOD 2015.

*Why it matters here.* Our phase-2 finding that **3 of 5 recent zero-shot LLM
papers report no threshold-dependent metric** is a reporting-adherence problem
of exactly the kind TRIPOD+AI exists to address. It gives the audit a normative
anchor: the information we need is information such papers are already expected
to report.

*Caveat.* We did not retrieve the checklist items themselves, so the manuscript
does not claim which specific item is violated.

**"Beyond AUROC: evaluating predictive performance metrics under class imbalance
in real-world clinical data." JMIR Formative Research 2026;e86379.** [snippet]
Reported to find AUROC 0.928 alongside recall 0.372 for a minority class in
17,018 hospitalised COVID-19 patients, and to argue AUROC alone is insufficient
under imbalance.

*This is the closest prior work we found.* It makes the same core observation -
strong ranking can coexist with a poor decision - in conventional clinical ML.
**We could not retrieve the full text** (JMIR and the ScienceDirect mirror both
returned errors), so it is cited as related work only and no number from it is
used.

**Decision curve analysis.** [snippet] Quantifies net benefit at a given
threshold probability and is motivated by the observation that AUROC weights
sensitivity and specificity equally, ignoring misclassification costs.

## 3. Threshold optimisation and calibration

Youden's J, prevalence-adjusted and cost-weighted cut-offs are long-established.
Calibration (whether probabilities match frequencies) is a distinct property
from both discrimination and the decision rule; a model can be well calibrated
and still decide badly if the cut-off is wrong, and vice versa. The manuscript
keeps the three separate in §10.

## 4. Conformal prediction and selective classification

Searched because both set operating points. [snippet, plus one preprint]

- Abstention is commonly implemented by **thresholding a nonconformity or
  confidence score**, and sweeping that threshold traces a **risk-coverage
  curve**. This is the closest mechanical analogue to what we do.
- Cost-sensitive conformal prediction with human-in-the-loop abstention
  (arXiv:2607.27143, preprint) reports that **marginal conformal prediction can
  badly under-cover rare minority classes**, which is the regime of most of our
  audited tasks.

**How our work differs, precisely:**

| | conformal / selective classification | this work |
|---|---|---|
| intervention | **abstain** on uncertain cases | **relocate** the decision point |
| requires | a calibration set with labels | nothing, for the diagnostic |
| output | a prediction set or a coverage guarantee | an estimate of recoverable BACC |
| applies to | a system you control | a **published result** you only read |

The decisive difference is the last row. Conformal methods need the model and a
calibration set. Our diagnostic runs on two numbers already printed in a paper,
which is what makes a retrospective audit possible at all.

**We did not find** a review comparing threshold-selection strategies for
clinical deployment, nor any application of conformal or selective methods to
audit published zero-shot LLM evaluations.

## 5. Clinical LLM benchmarking

The audited papers themselves ([1], [2], [3], [9]) plus the three that report no
threshold metric ([10], [11], [12]). ClinicRealm [12] is the largest benchmark
examined and reports AUROC and AUPRC only.

---

## 6. Novelty statement

**Not novel, and stated as such in the manuscript:**
- the binormal AUROC-to-reachable-accuracy relation (Green & Swets, 1966);
- operating-point selection in clinical ML (Youden, cost-weighted cut-offs);
- the observation that AUROC alone misleads under imbalance (JMIR 2026 [snippet],
  decision curve analysis literature);
- soft versus hard voting in ensembles.

**What we believe is new:**
1. **A retrospective audit of the zero-shot clinical LLM literature** for
   decision-point artifacts, with a quantified rate (12/37 cells, 4 papers),
   paper-level uncertainty and cut-off sensitivity.
2. **A label-free diagnostic computable from already-published numbers**, which
   is what permits an audit of papers whose scores are unavailable.
3. **Characterisation of the estimator's error direction**, including that it is
   U-shaped in the variance ratio and conservative for saturated scores - the
   regime of LLM label-token log-probabilities.
4. **The reporting finding**: 3 of 5 recent papers report no threshold-dependent
   metric, so the question cannot be asked of much of the field.
5. **The subgroup observation** (§7.6): within one evaluation the artifact
   concentrates in the poorest quintiles, where the model predicts positive for
   every child.

**Competing explanations we must not dismiss:**
- *Our estimator is simply wrong for these papers' score shapes.* Partly
  addressed by the simulation, but we never see their scores. This is why the
  estimated/demonstrated distinction is enforced throughout.
- *Authors chose their operating point deliberately* for a clinical reason
  (e.g. favouring sensitivity). Three of four audited papers use a forced token,
  which is not a deliberate choice; but for any individual cell we cannot
  exclude it.
- *Selection effect.* Papers reporting both metric families may differ
  systematically from those that do not, in an unknown direction.
- *Publication and reporting incentives* may make low BACC more likely to appear
  in papers that also report AUROC, inflating our rate.

---

## 7. Verification failures, recorded

| source | problem |
|---|---|
| JMIR Formative Res 2026;e86379 | JMIR returned empty content; ScienceDirect mirror HTTP 403 |
| TRIPOD+AI checklist items | not retrieved; only the statement's metadata verified |
| scimagojr.com | HTTP 403 on all attempts (journal quartiles) |
| Nature s41746-026-02539-z, s41598-026-58763-w | paywall redirects |
