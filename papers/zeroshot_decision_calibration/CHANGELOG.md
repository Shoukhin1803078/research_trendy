# Changelog

Changes made during the audit of 2026-10-14, driven by `feedback.md`.
Original state: git `e93b6b6`. Manuscript v1 preserved at
`originals/draft_paper_v1_PRESERVED.md`; original results at
`originals/results_v1_PRESERVED/`. No original artifact was overwritten.

---

## Corrections to published claims

| # | What was wrong | What it is now | Evidence |
|---|---|---|---|
| C1 | "**6 of 43 cells** have reported BACC above the estimate" | **21 of 43 (49%)** | `audit/a4_leakage_stats.py`; fixed in `draft_paper.md`, `FINDINGS.md`, `EXPERIMENTS.md` |
| C2 | "Spearman **+1.000**" as the validation headline | reported with the shared-term form (Spearman 0.762 at k=8, 0.905 at k=64), permutation p, and cluster CIs | `src/validate_diagnostic_v2.py` |
| C3 | "33% of published LLM cells" with no uncertainty | 33.3% with paper-level CI [18.8, 50.0] and a cut-off sensitivity table (50% to 19%) | `src/harvest_sensitivity.py` |
| C4 | k-shot results reported as "0.732" | 0.732 with subject-level 95% CI [0.612, 0.879], and an explicit warning not to read the 0.004 agreement as precision | `src/stats_v2.py` |
| C5 | Aggregation CI bootstrapped 25 folds as independent | repeat-level bootstrap [+0.0150, +0.0450]; all 5 repeats agree in sign | `src/stats_v2.py` |

## New analyses

| # | What | Why | File |
|---|---|---|---|
| N1 | Independent verification of the binormal formula by simulation | `feedback.md` §3 asks for it | `audit/a2_formula.py` |
| N2 | Error-direction analysis: the estimator **under-estimates** under unequal variance | explains C1 and makes the 33% a lower bound | `audit/a2_formula.py`, §9.1 |
| N3 | Circularity analysis of the validation | H and gain share `-BACC_native` | `audit/a1_circularity.py` |
| N4 | Endpoint set expanded 9 -> 13 (8 usable, 4 label columns), with a minimum of 10 positives | `feedback.md` §3 asks whether 9 endpoints suffice | `src/run_endpoints_v2.py` |
| N5 | Cluster-aware validation (bootstrap by label column; `primary` subset) | endpoints are nested cut-points, not independent | `src/validate_diagnostic_v2.py` |
| N6 | Cut-off sensitivity for modes A and C | the 0.02 cut was arbitrary and undisclosed | `src/harvest_sensitivity.py` |
| N7 | Small-sample bias of the empirical ceiling (+0.054 at n=40) | guards against quoting `bacc_oracle` | `audit/a2_formula.py`, §9.3 |
| N8 | Automated claim-to-evidence verification, 44 checks | `feedback.md` §10 | `src/check_claims.py` |
| N9 | Two new figures: expanded validation, sensitivity | support RQ1 and §7.4 | `src/make_figures.py` |

## Manuscript

- Restructured to the 17-section specification in `feedback.md` §9: added
  Research Questions and Hypotheses (§3), Error Analysis and Ablations (§9),
  Threats to Validity (§13), Ethical Considerations (§14), Journal Fit (§16).
- Every use of `reachable` now says **estimated**, and §4.2 states it is not a
  bound.
- §7.3 states explicitly that the audit cannot demonstrate that fitting a line
  on those published systems would reach the estimate, only that the reported
  result is consistent with a decision-point artifact.
- §7.4 frames the headline as a property of the audited sample.
- §13 documents the incomplete CKD experiment with the remaining steps.
- §16 lists candidate journals and states that quartile status is **not
  verified**.
- §17 adds a citation-status note.

## Preserved unchanged

- All retracted and corrected findings (§11), including the unweighted-probe
  results, the sign-agnostic diagnostic bug, the miscount, the reporting slip,
  the refuted prevalence-matching method, and the unsupported leakage hypothesis.
- The negative result that row-wise splitting did not leak.
- The statement that the work contains no new method.

## Not done

- The CKD LLM arm (Experiment 13). See `draft_paper.md` §13 for why and for the
  remaining steps.
- Extraction of four further audit papers.
- A survey of conformal prediction and selective classification.
- Verification of journal quartile status.

---

# Phase 2 (2026-10-14) — driven by `feedback2.md`

Manuscript v2 preserved at `originals/draft_paper_v2_PRESERVED.md`; audit v1
preserved at `originals/harvest_v1_PRESERVED.json`. Plan: `audit/PHASE2_PLAN.md`.

## Conclusions that became WEAKER

| # | Was | Now | Evidence |
|---|---|---|---|
| W1 | 33.3% of cells decision-limited | **32.4%** (12/37) | `harvest_v2_stats.json` |
| W2 | per-paper rates 18.8-50.0% | **0-50%**; the expansion **widened** heterogeneity | same |
| W3 | paper-level CI [18.8, 50.0] | [17.6, 46.2] | same |
| W4 | the estimator is conservative | conservative in 11 of 12 non-Gaussian conditions, but **mildly anti-conservative in 1** (bimodal negatives, +0.0029 +/- 0.0009) | `a5_estimator_range.py` |

The expansion was pre-committed to report whatever it found
(`PHASE2_PLAN.md` §D). It lowered the headline and widened the spread, and the
manuscript now says a single pooled rate is the wrong summary.

## Conclusions that became STRONGER

| # | What | Evidence |
|---|---|---|
| S1 | The estimator's conservatism is **U-shaped** in the SD ratio, so it is strict in both directions (-0.061 at 0.25, -0.154 at 6.0) | `a5_estimator_range.py` |
| S2 | **Saturated/clipped scores give -0.051** — directly relevant, since LLM label-token logprobs are often saturated | same |
| S3 | New finding: within one paper, the artifact **concentrates in the poorest subgroups** (poorest quintile: sens 1.000, spec 0.000, AUROC 0.581) | `harvest_v2.json`, §7.6 |
| S4 | New finding: **3 of 5 papers examined report no threshold-dependent metric**, so the question is unanswerable for much of the field | `SEARCH_LOG.md`, §7.2 |

## Newly explicit limitations

- **Selection effect quantified** (§13): eligibility depends on a reporting
  property most papers lack, so the sample is a minority of the literature and
  the bias direction is unknown.
- **No direct recoverability on published systems** (§10, §12.7): estimated for
  12 published cells, demonstrated only on our own 9. No eligible paper releases
  score-level outputs; the repository search failed.
- **Journal quartiles unverified** (§16): scimagojr.com returned HTTP 403 on
  every attempt.

## New files

| file | content |
|---|---|
| `audit/PHASE2_PLAN.md` | prioritized plan with pre-committed analysis decisions |
| `audit/EXTRACTION_PROTOCOL.md` | inclusion/exclusion criteria, written **before** seeing the new papers |
| `audit/SEARCH_LOG.md` | every search and fetch, including failures |
| `audit/CKD_GO_NOGO.md` | conditional-go recommendation; **not launched** |
| `audit/JOURNALS.md` | shortlist with explicit verification failure |
| `audit/a5_estimator_range.py` | extended simulation |
| `src/harvest_v2.py` | expanded audit, richer schema, eligibility log |
| `src/harvest_v2_stats.py` | recomputed statistics |
| `results/harvest_v2.json`, `harvest_v2_stats.json`, `a5_estimator_range.log` | artifacts |

## Verification

`src/check_claims.py` extended from 50 to **65 checks, 0 failures**.

## Preserved unchanged

All phase-1 retractions and corrections, the refuted prevalence-matching method,
the unsupported leakage hypothesis, and the statement that the work contains no
new method.

## Not done in phase 2

- CKD experiment: **not launched**, awaiting approval of a 15-minute pre-check.
- Systematic (as opposed to targeted) literature search.
- Conformal prediction and selective classification survey — still open.
- Direct quartile verification.

## CKD arm completed (2026-10-14, 23:02)

Scoring finished all **400 records / 40 patients** inside the power budget
(throughput improved to 6.9 s/item as the machine freed up).

| quantity | value |
|---|---|
| patient-level AUROC (pre-registered primary) | **0.590** [0.404, 0.767] |
| record-level AUROC (diagnostic only) | 0.661 |
| native decision | sens 1.000, spec **0.000** |
| native BACC | **0.500** |
| H | **+0.064** |
| realised gain at k=16 | **+0.058** |

**Reported as INCONCLUSIVE, not as a confirming result.** The pre-registered
rule classifies on the point estimate and 0.590 > 0.55 took the "ranking usable"
branch, but **every interval includes chance**: the AUROC CI spans 0.5, and so
do all three k-shot BACC intervals. The cell is therefore **not** counted among
the demonstrated-recoverability cases, which stay at 9.

What holds regardless: the decision rule is **degenerate again** (positive for
every patient, specificity 0.000), the same signature as the image and speech
arms.

Two corrections made while integrating:
- The section initially read "the ranking does not support recovery here"
  followed by the positive branch - a template bug producing a self-contradictory
  paragraph. Fixed.
- My earlier statement that within-patient correlation inflates record-level
  AUROC "by about 0.27" came from the 6-patient pre-check. **At 40 patients the
  gap is +0.071.** The manuscript carries the correct figure.

`src/check_claims.py` extended to **75 checks, 0 failures**.

---

## 2026-10-15 — short-form reframing (Option B) and figure/text reconciliation

The user chose the fallback framing recorded in `audit/JOURNALS.md`: lead with
the **reporting finding** rather than the 32.4% rate, as a short report / methods
note. The rate is small-sample (37 cells, 4 papers, per-paper range 0-50%); the
reporting finding does not depend on the estimator being right.

| # | Change | Why |
|---|---|---|
| S1 | New `draft_paper_short.md` (~3,300 words), leading with "most of this literature cannot be audited" | The submission version. `draft_paper.md` is retained in full as supplementary/preprint. v3 preserved at `originals/draft_paper_v3_PRESERVED.md` |
| S2 | `src/make_figures.py` read the **phase-1** `harvest.json` while the text reported phase-2 numbers | Figures 1-2 showed 43/36 cells against 45/37 in the text. Now reads `harvest_v2.json`, subgroup rows excluded, counts computed from the data so they cannot go stale again |
| S3 | Figure 2 coloured bars by `H > 0.02` alone | That painted **13** bars "recoverable" against the **12** mode-A cells in the text. One cell (Pengi/MDVR-KCL, AUROC 0.268) ranks backwards: a sign flip helps it, a threshold shift does not. Bars now coloured by failure mode, with a third legend entry |
| S4 | Stale counts in `draft_paper.md` §1.3, the dataset table and the Figure 3 caption; citation `[17]` did not exist | Carried over from phase 1; corrected to 37/4 and `[9]` |
| S5 | §7.2 said the 14 subgroup rows "partition one evaluation three ways" | It is **four** (sex, residence, wealth quintile, survey year) |
| S6 | `src/check_claims.py` extended with 9 checks | The short report puts the mode counts, the mode-A mean and the cut-off sensitivity row in its headline tables; these are now verified against `harvest_v2.json` directly rather than inherited from the phase-1 file. **84 claims, 0 failures** |

**Not changed:** no result, no experiment and no finding was altered. The
retractions, corrections and the INCONCLUSIVE CKD verdict all carry over
verbatim into the short form.
