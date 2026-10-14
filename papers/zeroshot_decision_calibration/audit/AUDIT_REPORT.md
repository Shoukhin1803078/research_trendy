# Audit Report and Implementation Plan

**Date:** 2026-10-14
**Scope:** read-only audit of the whole project against `feedback.md`.
**Originals:** every file is committed in git at `e93b6b6`. Nothing below has
been modified yet. Audit scripts live in `audit/` and touch no original file.

---

## 1. Publication readiness, in one paragraph

The project has one defensible, interesting thesis and real supporting evidence,
but it is **not submittable today**. There is one outright factual error in the
manuscript, the headline statistic is presented without the uncertainty and
sensitivity analyses it requires, and the central validation rests on six
endpoints with a shared baseline term. None of these is fatal. All but one are
fixable with the data already on disk. Realistic assessment: **a strong
workshop/short-paper now; a credible Q1 submission after the P1-P6 items below.**

---

## 2. Problems, ranked by severity

### S1 - FACTUAL ERROR in the manuscript (critical)

`draft_paper.md` §8.1 and `FINDINGS.md` state:

> "In **6 of our 43 cells** the reported balanced accuracy is already above
> `reachable`, the most extreme by 0.117."

**The actual count is 21 of 43 (49%).** Verified by `audit/a4_leakage_stats.py`.
This understates by a factor of 3.5 the frequency with which the paper's own
estimator is exceeded. It must be corrected before anything else.

### S2 - The headline "33%" is reported without its uncertainty (critical)

Three separate problems, all measured in `audit/a3_harvest.py`:

1. **Clustering.** 36 cells come from 3 papers, and the rate differs sharply
   between them: 18.8% / 43.8% / 50.0%. A **cluster bootstrap over papers**
   gives **34.3% [18.8, 50.0]** - a CI spanning nearly three-fold. The naive
   cell-level bootstrap, which the paper implicitly assumes, gives a
   misleadingly similar interval for the wrong reason.
2. **An arbitrary cut-off drives the number.** Mode A is `H > 0.02`. Varying it:

   | cut | A cells | share |
   |---|---|---|
   | H > 0.00 | 18/36 | 50.0% |
   | H > 0.01 | 14/36 | 38.9% |
   | **H > 0.02** | **12/36** | **33.3%** |
   | H > 0.05 | 9/36 | 25.0% |
   | H > 0.10 | 7/36 | 19.4% |

3. **No sensitivity analysis is reported for either.**

### S3 - The central validation is under-powered (critical)

`n = 6` endpoints with AUROC > 0.5. Measured in `audit/a1_circularity.py`:

- H and the measured gain **share the term `-BACC_native`** and are computed on
  the same test set. This induces correlation before the diagnostic does work.
- Removing the shared term (correlating `reachable` against `achieved` directly):
  Pearson drops +0.925 -> +0.904 at k=8, Spearman **+1.000 -> +0.829**, and the
  permutation p-value moves from 0.0028 to **0.0583 - no longer significant**.
- A null built from the baseline spread alone reaches +0.830 at the 95th
  percentile, so the observed value sits at about p = 0.012 against it.
- Bootstrap CI on Pearson at k=8: **[+0.584, +1.000]**.

The finding survives, but "Spearman +1.000" as a headline is not defensible.

### S4 - Confidence intervals assume independence they do not have (major)

The aggregation test (`+0.0300 [+0.0150, +0.0450]`) bootstraps **25 fold-results
as if independent**. They are 5 repeats x 5 folds over the same 120 subjects:
within a repeat the folds partition the same data, and the repeats reuse all of
it. Effective sample size is nearer 5 than 25, so the interval is too narrow.

The BenSParX k-shot `+-` values are the **SD across 50 overlapping draws**, not a
CI on the mean, and are not labelled as such.

### S5 - The binormal estimate is conservative, and the paper does not say so (major)

`audit/a2_formula.py` verifies the formula is **exact** under its own assumption
(error < 0.0015 across d' from 0 to 3). Under **unequal variance it systematically
under-estimates**:

| ratio of SDs | formula | empirical best | error |
|---|---|---|---|
| 1.0 | 0.7735 | 0.7744 | -0.001 |
| 1.5 | 0.7222 | 0.7363 | **-0.014** |
| 2.0 | 0.6824 | 0.7215 | **-0.039** |
| 3.0 | 0.6308 | 0.7163 | **-0.086** |

Bimodal positive scores: -0.050. This explains S1's 21 cells with H < 0, and it
is **favourable** to the thesis: the 33% is a **lower bound** on decision-limited
cells, not a point estimate. Our own BenSParX cell shows the same direction
(empirical ceiling 0.775 vs binormal estimate 0.728).

### S6 - `bacc_oracle` is optimistically biased and lives in the artifacts (moderate)

Measured bias: **+0.054 at n=40**, +0.034 at n=100, +0.013 at n=400. It is
correctly documented in the code and is **not currently quoted in any claim**,
but it sits in `bensparx_llm.json` where it could be picked up. Needs an explicit
warning.

### S7 - Experiment 13 (CKD LLM arm) incomplete (moderate)

Already documented as not-run in `EXPERIMENTS.md` §9. Compliant with the spec.
The question is whether it is essential; see §4.

### S8 - Manuscript structure (minor)

`draft_paper.md` has 10 sections. `feedback.md` §9 specifies 17, including
Research Questions, Threats to Validity, and Ethical Considerations.

---

## 3. What the audit confirmed as sound

Recording these so they are not re-litigated:

- **The formula is correct.** Independently verified by simulation.
- **No test-label leakage** in `run_budget.py` (support from train, eval on
  test), `analyze_bensparx.py` (held-out mask), or `bensparx_aggregation.py`
  (subject-grouped, selection inside the fold).
- The one place test labels touch a parameter is `ROPE-Q (pi known)`, which is
  an **oracle used only to refute our own method** and is labelled as such.
- **Harvest data integrity is clean:** 0 of 43 cells needed BACC derived from
  sens/spec, and 0 mismatches where a paper printed all three.
- 3 of 4 spot-checked manuscript numbers reproduce exactly from the artifacts
  (33.3%, +0.138, 0.814). The fourth is S1.
- Edge cases behave correctly: AUROC 0.5 -> 0.5, ties -> 0.5, all-one-class ->
  NaN, no-flip handling of AUROC < 0.5.

---

## 4. Prioritized implementation plan

Ordered by scientific importance per unit of compute. P1-P6 use data already on
disk and need no LLM.

| # | Action | Fixes | Cost | Status |
|---|---|---|---|---|
| **P1** | Correct the 6 -> 21 error everywhere | S1 | minutes | required |
| **P2** | Add cluster bootstrap + cut-off sensitivity to the audit and manuscript | S2 | minutes | required |
| **P3** | Expand the endpoint set from 9 to ~20 for the validation | S3 | ~10 min CPU | high value |
| **P4** | Repeat-level bootstrap for the aggregation CI; subject-level bootstrap for the k-shot results | S4 | minutes | required |
| **P5** | Add the conservatism simulation to the paper; reframe 33% as a lower bound | S5 | minutes | high value |
| **P6** | Warning on `bacc_oracle`; automated consistency checks claim-vs-artifact | S6 | minutes | required |
| **P7** | Restructure the manuscript to the 17-section specification | S8 | moderate | required |
| **P8** | Run the CKD LLM arm | S7 | hours, needs a free machine | optional |

### On P3, the only item with real compute cost

The validation currently uses 9 endpoints (6 usable) built from two MedIMeta
datasets. More endpoints are available from the **same cached features at no
download cost**, by using label columns and cut-points not yet exploited:
`Artifact` (6 classes, unused entirely), further `DR level` cut-points, and
one-vs-rest splits of `case category`, `Clarity` and `Field definition`.

Expected yield ~20 endpoints, which roughly triples the degrees of freedom in
the correlation test. Expected runtime under 10 minutes on CPU. **Purpose:**
directly answers `feedback.md` §3's question about whether nine endpoints
suffice. **Risk:** endpoints from one dataset are themselves correlated, so the
analysis must cluster by dataset, not treat endpoints as independent.

### On P8, which I am not starting without a decision

The CKD arm needs roughly 800 LLM calls. At the throughput measured yesterday
(~15 s/item with a 10-feature prompt, when the machine is free) that is about
3.5 hours of exclusive machine use. It would add a second modality to the
forward validation. It is **not required** for the central claim, which rests on
the published-result audit. I will not launch it without you saying so.

---

## 5. What I will not do

- Not revive prevalence matching (refuted; `feedback.md` §6 Experiment 3).
- Not reintroduce the unweighted-probe conclusions (retracted).
- Not treat `reachable` as an upper bound.
- Not claim the 33% generalises beyond the audited sample.
- Not describe the 3B run as a replication of the published 8B evaluation.
