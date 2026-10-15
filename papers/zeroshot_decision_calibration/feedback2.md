# Next Research Phase: Strengthen the Evidence for Q1 Submission

The initial audit and improvement phase is complete. Read `audit/AUDIT_REPORT.md`, `CHANGELOG.md`, `REPRODUCE.md`, and the current `draft_paper.md` before proceeding.

The goal of this phase is to address the remaining limitations that most threaten publication: the small published-result audit, uncertainty in paper-level estimates, and the gap between estimated headroom and demonstrated recoverability.

## Priority 1: Expand the Published-Result Audit

This is the highest-priority task.

1. Identify and extract the four additional papers already identified in the project.
2. Verify their reported results directly from the original papers and supplementary materials.
3. Search for additional relevant studies on zero-shot clinical LLM evaluation, clinical classification, threshold selection, and decision metrics.
4. Define and document inclusion and exclusion criteria before adding new results.
5. Extract eligible result cells with paper ID, model, dataset, endpoint, AUROC, BACC, sensitivity, specificity, threshold-selection protocol, and source location where available.
6. Track missing information explicitly. Never estimate or invent unreported metrics.
7. Avoid treating correlated result cells from the same paper as independent observations.
8. Recalculate the audit statistics, paper-level rates, uncertainty intervals, and sensitivity analyses after the audit expands.

The current audit covers 36 result cells from three papers. The reported decision-limited proportion varies substantially by paper, and the paper-level confidence interval is wide. Determine whether a larger audit supports a stable conclusion or reveals substantial heterogeneity.

## Priority 2: Validate the Main Claim More Directly

The paper currently estimates potential recoverability from AUROC and a binormal assumption. This does not establish that threshold tuning on the published systems would actually recover the estimated performance.

Investigate which published studies provide enough score-level data, confusion matrices, or supplementary results to test recoverability directly.

Where data are available:
- Reproduce the original evaluation protocol.
- Fit thresholds using validation data only.
- Evaluate the selected threshold on an independent test set.
- Compare native and tuned decisions using the same test observations.
- Report paired uncertainty estimates and the size of the improvement.
- Distinguish direct empirical evidence from model-based estimates.

If direct replication is impossible, preserve this as a limitation and explain exactly what evidence is missing.

## Priority 3: Verify the Estimator's Limitations

Review the simulation showing that the headroom estimator underestimates reachable performance under unequal variances.

1. Verify the simulation code and assumptions.
2. Test a justified range of class-conditional distributions and variance ratios.
3. Report bias, uncertainty, and failure cases.
4. Explain when the binormal estimate is informative and when it may be misleading.
5. Ensure the manuscript does not describe the estimate as a guaranteed upper bound or interpret the negative-headroom count as conclusive proof without supporting assumptions.

Do not modify the estimator merely to produce a stronger result. Any proposed alternative must be justified and independently evaluated.

## Priority 4: Complete the Literature Review

Review literature on:
- Signal detection theory and binormal models.
- Clinical prediction-model evaluation.
- Threshold optimization and calibration.
- Conformal prediction and selective classification.
- Clinical LLM benchmarking and reliability.
- Reporting standards for clinical AI and prediction models.

Identify the exact distinction between existing work and this study's contribution. Add verified references and discuss competing explanations for the observed findings.

## Priority 5: Decide Whether to Run the CKD Experiment

The CKD arm requires approximately 800 calls and 3.5 hours on an exclusive machine according to the current project report.

Before running it:
- Inspect the implementation and estimated resource requirements.
- Determine whether its result would materially change the central conclusions.
- Verify that the dataset and evaluation protocol are appropriate.
- Define the expected outputs and stopping criteria.
- Document any access, privacy, or data-use constraints.

Provide a short go/no-go recommendation before launching this expensive experiment. Do not run it automatically.

## Priority 6: Journal Selection and Submission Strategy

Identify journals whose scope matches an empirical audit of clinical LLM evaluation.

For each candidate, verify:
- Current quartile and ranking year.
- Relevant subject category and ranking source.
- Recent papers with similar methodology and scope.
- Article type and manuscript requirements.
- Open-access options and publication fees, where verifiable.

Do not claim Q1 status without a current authoritative source. Rank journals by scope fit and realistic suitability, not prestige alone.

## Required Deliverables

Produce:
1. An updated audit dataset with traceable source references.
2. A reproducible extraction and analysis pipeline.
3. Updated audit statistics and uncertainty estimates.
4. A direct-recoverability analysis where evidence permits.
5. A revised literature review and novelty statement.
6. A CKD experiment go/no-go recommendation.
7. A verified journal shortlist.
8. An updated manuscript and changelog.
9. A concise report explaining which conclusions became stronger, weaker, or remained unresolved.

Preserve the original and current manuscript versions. Keep every new claim linked to reproducible evidence. If the expanded audit contradicts the current thesis, revise the thesis rather than forcing the results to support it.

Start by reviewing the existing audit artifacts, then proceed with the literature audit and extraction. Do not launch expensive experiments or overwrite the current manuscript before documenting the plan.