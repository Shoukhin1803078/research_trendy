# Phase 2 Plan

**Date:** 2026-10-14. **Spec:** `feedback2.md`. **Prior phase:** `audit/AUDIT_REPORT.md`, `CHANGELOG.md`.
**Preservation:** phase-1 manuscript at `originals/draft_paper_v1_PRESERVED.md`; git `e93b6b6` holds the pre-audit tree. Phase-2 work writes new files; `results/harvest.json` is copied to `originals/` before any schema change.

---

## A. Reconciliation: what is new, done, or open

| Spec item | State entering phase 2 | Classification |
|---|---|---|
| P1 extract 4 identified papers | identified only (2607.20453, 2607.29082, 2407.18525, 2605.10286); none extracted | **NEW** |
| P1 inclusion/exclusion criteria | stated informally in `draft_paper.md` §7.1 | **partial** - needs a written, pre-registered protocol |
| P1 richer extraction schema | current schema has no endpoint, threshold protocol, n, or source location | **NEW** |
| P1 recompute statistics after expansion | pipeline exists (`harvest_sensitivity.py`) | **reusable** |
| P2 direct recoverability | not attempted | **NEW** - the highest-value item |
| P3 verify estimator simulation | done for SD ratios 0.5-3.0, 2 non-Gaussian shapes (`audit/a2_formula.py`) | **partial** - extend range and shapes, add uncertainty |
| P4 literature review | 2 searches only; conformal prediction and selective classification unsurveyed | **NEW** |
| P5 CKD go/no-go | pipeline written, never run | **NEW** - recommendation only, no launch |
| P6 journal verification | candidates listed, quartile explicitly unverified | **NEW** |

Already complete, not to be repeated: the binormal formula verification, the leakage scan, cluster-aware intervals, cut-off sensitivity, the 21/43 correction, the endpoint expansion to 8 usable, and `src/check_claims.py` (50 checks).

---

## B. Tasks, in execution order

### T1 - Pre-register the extraction protocol *(no compute; blocks T2)*
Write inclusion/exclusion criteria, the extraction schema, and the
missing-data policy **before** looking at the new papers, so eligibility cannot
be decided by whether a cell helps the thesis.
**Value:** removes the strongest reviewer objection to a hand-built audit.
**Risk:** none. **Output:** `audit/EXTRACTION_PROTOCOL.md`.

### T2 - Extract the four identified papers *(web reads; ~1h)*
Fetch each, apply the protocol, record every cell with its source location, and
record exclusions with reasons.
**Value:** directly addresses the main publication threat (36 cells, 3 papers).
**Risk:** papers may report AUROC without any threshold metric, making them
ineligible. **That outcome is itself a finding** and will be reported as the
count of papers that cannot be audited.
**Dependency:** T1. **Output:** `src/harvest_v2.py`, `results/harvest_v2.json`.

### T3 - Search for further eligible studies *(web; ~45m)*
Targeted searches on zero-shot clinical LLM evaluation reporting both metric
families. Log every search string and its yield, including failures.
**Value:** moves the audit from a convenience sample toward a documented one.
**Risk:** low yield; documented either way. **Output:** `audit/SEARCH_LOG.md`.

### T4 - Recompute audit statistics *(minutes)*
Re-run per-paper rates, cluster bootstrap, and cut-off sensitivity on the
expanded set. **Pre-committed:** if the expanded audit lowers the
decision-limited rate or widens heterogeneity, the manuscript reports that and
the thesis is revised, not the data.
**Dependency:** T2, T3.

### T5 - Direct recoverability *(variable)*
Identify published studies supplying score-level data, confusion matrices, or
per-case outputs. Where found: refit a threshold on validation data only,
evaluate on an independent test split, and report the paired improvement as
**measured**, separate from binormal estimates.
**Value:** the single biggest upgrade - converts an estimate into a
demonstration. **Risk:** most papers publish neither scores nor code; if none
is usable, the gap is preserved as an explicit limitation with a statement of
exactly what evidence is missing.
**Fallback (no extra claim):** we already hold one directly measured case - our
own BenSParX cell, where fitting on 16 labelled subjects moved BACC 0.500 ->
0.732 with the estimate at 0.728. That remains the only *demonstrated*
recoverability in the paper if T5 finds nothing.

### T6 - Extend the estimator simulation *(minutes)*
Widen variance ratios, add skewed/heavy-tailed/mixture families, and attach
Monte-Carlo uncertainty. State when the binormal estimate is informative and
when misleading. **Constraint from the spec: do not change the estimator to get
a better result.** Any alternative must be justified and separately evaluated.
**Output:** `audit/a5_estimator_range.py`.

### T7 - Literature review *(web; ~1h)*
Signal detection theory, clinical prediction-model evaluation (including
reporting standards such as TRIPOD), threshold optimisation and calibration,
conformal prediction, selective classification, clinical LLM benchmarking.
Verify every citation against the original. Write a precise novelty statement
and name competing explanations for our findings.
**Output:** `audit/LITERATURE.md`, manuscript §2 rewrite.

### T8 - CKD go/no-go *(analysis only; NO LAUNCH)*
Inspect the implementation, confirm resource estimates, judge whether the result
would change the central conclusion, define outputs and stopping criteria, note
data-use constraints. **Recommendation only. The experiment will not be launched
without explicit approval.**
**Output:** `audit/CKD_GO_NOGO.md`.

### T9 - Journal verification *(web; ~45m)*
For each candidate: current quartile with ranking source and year, subject
category, comparable recent papers, article type, OA options and fees where
verifiable. **No quartile claim without an authoritative current source.**
**Output:** `audit/JOURNALS.md`.

### T10 - Manuscript and changelog *(after evidence verified)*
Update only claims backed by new verified artifacts. Extend
`src/check_claims.py` to cover every new number. Preserve v2 before writing v3.

---

## C. Cost and risk summary

| task | compute | wall time | main risk |
|---|---|---|---|
| T1 | none | 15m | none |
| T2 | none (web) | ~1h | papers ineligible |
| T3 | none (web) | ~45m | low yield |
| T4 | trivial CPU | 5m | **expanded audit may weaken the headline** |
| T5 | low-moderate | 1-3h | data almost certainly unavailable |
| T6 | ~5 min CPU | 20m | none |
| T7 | none (web) | ~1h | none |
| T8 | none | 20m | none |
| T9 | none (web) | ~45m | quartile sources paywalled |
| T10 | trivial | 1h | none |

**No task in this plan launches the CKD experiment.** T8 produces a
recommendation only.

---

## D. Pre-committed analysis decisions

Fixed now, before the new data are seen:

1. Eligibility is decided by the T1 protocol alone, never by whether a cell
   supports the thesis.
2. The primary statistic stays `H > 0.02` with the full sensitivity table; the
   cut-off will not be re-tuned after seeing new cells.
3. Papers remain the clustering unit. Cells within a paper are never treated as
   independent papers.
4. If the expanded audit lowers the rate or shows strong heterogeneity, the
   manuscript reports it and the thesis is narrowed.
5. Any directly measured recoverability is reported separately from binormal
   estimates and never pooled with them.
6. Failed searches, unavailable data, and ineligible papers are reported with
   counts.
