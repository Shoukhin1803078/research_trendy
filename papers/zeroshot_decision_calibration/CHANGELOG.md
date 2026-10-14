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
