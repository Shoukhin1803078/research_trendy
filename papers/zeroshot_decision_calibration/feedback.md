# Research Project Review and Improvement for Q1 Journal Submission

## Objective

Review my entire research project and improve it toward submission to a high-quality Q1 journal in medical informatics, clinical AI, biomedical AI, or trustworthy AI.

**Research title:** *Are Reported Zero-Shot Clinical LLM Failures Real, or Are They Decision-Point Artifacts?*

The central hypothesis is that some reported failures of zero-shot clinical LLMs may arise from inappropriate classification thresholds rather than poor ranking ability. The project distinguishes ranking performance, measured by AUROC, from decision performance, measured by balanced accuracy (BACC).

Do not merely improve the writing. Conduct a rigorous methodological, statistical, experimental, and reproducibility audit.

## 1. Audit the Entire Project

Inspect all project files, including:
- `EXPERIMENTS.md`
- `FINDINGS.md`
- `draft_paper.md`
- All experiment scripts, notebooks, configurations, result tables, and generated artifacts.

Before modifying anything, understand the existing pipeline and map each experiment to its corresponding code, results, and claims in the manuscript.

Identify:
- Missing experiments and incomplete analyses.
- Inconsistencies between code, reported results, and manuscript claims.
- Unsupported conclusions or overstated novelty.
- Data leakage risks and incorrect evaluation procedures.
- Statistical weaknesses and reproducibility issues.
- Results that were previously corrected or retracted.

Preserve a record of the original results and document all proposed changes.

## 2. Strengthen the Main Research Contribution

Evaluate whether the current contribution is sufficiently novel and rigorous for a Q1 journal.

The main contribution should focus on distinguishing decision-point artifacts from genuine ranking failures in published zero-shot clinical LLM evaluations.

Clearly distinguish three categories:

1. **Decision-limited:** AUROC indicates useful ranking ability, but the reported BACC may improve through appropriate threshold selection.
2. **Signal-limited:** Ranking is better than random, but the estimated decision headroom is small.
3. **Sign-inverted:** AUROC is below 0.5, indicating that the score direction may be reversed.

Check that these categories are implemented consistently and that all thresholds and classification rules are justified.

Do not claim that the headroom estimate is a guaranteed upper bound. Explain its assumptions and limitations.

## 3. Validate the Headroom Diagnostic

The current headroom diagnostic uses:

`d' = sqrt(2) * Phi^-1(AUROC)`

`reachable_BACC = Phi(d'/2)`

`headroom = reachable_BACC - reported_BACC`

Review the mathematical implementation, especially the assumptions behind the binormal model and AUROC values below 0.5.

Required improvements:
- Verify the formula independently.
- Test numerical edge cases and score-direction handling.
- Evaluate the relationship between predicted headroom and observed threshold-tuning improvement.
- Report Pearson and Spearman correlations with uncertainty estimates where statistically appropriate.
- Investigate whether the current validation on only nine endpoints is sufficient.
- Perform additional validation on independent endpoints or datasets if the available data and compute budget permit.
- Avoid data reuse that could produce optimistic estimates of predictive performance.

Clearly distinguish estimated reachable BACC from empirically achieved BACC.

## 4. Expand and Improve the Published-Result Audit

The current audit includes 36 published LLM result cells from three papers.

Assess whether this sample adequately supports the manuscript's field-level conclusions.

If additional relevant papers are available:
- Define explicit inclusion and exclusion criteria.
- Extract reported AUROC, BACC, model, dataset, endpoint, threshold-selection procedure, and evaluation protocol.
- Verify extracted numbers against the original papers.
- Track unavailable metrics and ambiguous reporting.
- Avoid treating multiple correlated results from the same paper as independent observations.
- Report both the number of papers and the number of result cells.
- Perform sensitivity analyses when assumptions affect the classification of results.

Do not fabricate missing metrics or infer results that cannot be verified.

The claim that approximately one-third of audited cells exhibit decision-point artifacts must be presented as a finding from the audited sample, not as a universal statement about all clinical LLM research.

## 5. Improve Statistical Rigor and Evaluation Protocols

Review every experiment for:
- Appropriate train, validation, and test separation.
- Patient-level grouping wherever multiple observations belong to one patient.
- Threshold selection using training or validation data only.
- Independent final evaluation after threshold selection.
- Class imbalance and class-weighting issues.
- Confidence intervals, paired comparisons, and uncertainty reporting.
- Appropriate treatment of repeated runs and correlated samples.
- Reproducible random seeds and documented configurations.

For small label budgets, such as 2, 4, 8, 16, 32, and 64 labels:
- Repeat sampling with multiple random draws.
- Report the mean, standard deviation, and suitable confidence intervals.
- Compare methods under the same data splits and label budgets.
- Avoid selecting the best-performing run and presenting it as the expected performance.

Investigate whether the statistical tests and confidence intervals used in the existing experiments are appropriate for the sampling design.

## 6. Revisit the Main Experiments

Review and improve the following experiments where justified.

### Experiment 1: Headroom validation
Validate whether estimated headroom predicts actual improvement from threshold tuning. Investigate the limited sample size and the risk of overinterpreting correlations from nine endpoints.

### Experiment 2: Failure-mode classification
Verify the decision-limited, signal-limited, and sign-inverted classifications. Test their robustness to reasonable changes in the classification criteria.

### Experiment 3: Prevalence matching
The existing prevalence-matching approach did not perform well. Preserve this negative result, verify the analysis, and explain why the approach failed. Do not revive it without new evidence.

### Experiment 4: Label-budget comparison
Compare threshold fitting, linear probes, Platt scaling, and native decisions. Use the corrected class-weighted results. Do not reintroduce the previously retracted unweighted LogisticRegression conclusions.

### Experiments 5–8: BenSParX analysis
Review the supervised reference, aggregation methods, feature selection, and row-wise leakage analysis. Preserve the finding that the investigated row-wise leakage hypothesis was not supported. Ensure subject-level evaluation remains the primary protocol where appropriate.

Treat very high supervised performance cautiously and discuss potential dataset-specific effects, subject overlap, confounding, and external generalization.

### Experiments 9–10: Local LLM and aggregation
Clearly state the differences between the local Llama 3.2 3B experiment and the published Llama 3 8B evaluation. Do not describe the local experiment as a replication of the published setup.

Verify score extraction, threshold fitting, aggregation procedures, label budgets, repeated draws, and confidence intervals.

### Experiments 11–12: Published audit and feature-selection analysis
Strengthen the evidence supporting the main thesis. Distinguish observed improvements from model-based estimates of reachable performance. Do not interpret an estimated reachable BACC as proof that threshold tuning would achieve that score.

### Experiment 13: CKD LLM analysis
This experiment is currently incomplete. Determine whether it is essential to the main contribution. If feasible, complete it using the existing reproducible pipeline and an appropriate evaluation protocol. Otherwise, explicitly document it as incomplete and exclude its results from completed-experiment claims.

Do not report planned or partially executed experiments as completed.

## 7. Add Stronger Baselines and Ablation Studies

Identify the minimum additional experiments needed to substantiate the paper's main contribution.

Where data and compute permit, consider:
- Native model decisions versus tuned thresholds.
- Threshold fitting versus calibration methods.
- Ranking metrics versus decision metrics.
- Different label budgets.
- Score averaging versus majority voting.
- Sensitivity to class prevalence and score direction.
- Alternative reasonable definitions of decision headroom.
- Validation on independent clinical datasets or published evaluations.

Prioritize experiments that directly test the central hypothesis rather than adding unrelated models or analyses.

Do not add experiments solely to increase the number of results.

## 8. Strengthen the Clinical and Methodological Interpretation

Discuss the implications of AUROC and BACC differences for clinical screening, while avoiding claims of clinical effectiveness that the experiments do not establish.

Explicitly discuss:
- The difference between discrimination, calibration, and classification decisions.
- The dependence of BACC on the chosen threshold.
- The assumptions behind estimated reachable performance.
- The effects of small datasets and limited label budgets.
- Patient-level dependence and dataset-specific characteristics.
- Whether the findings generalize beyond the audited papers and endpoints.

Do not present a statistical diagnostic as a clinically validated decision-support system.

## 9. Improve the Manuscript

Revise `draft_paper.md` to follow a strong empirical research-paper structure:

1. Title
2. Abstract
3. Introduction
4. Related Work
5. Research Questions and Hypotheses
6. Methods
7. Experimental Setup
8. Results
9. Error Analysis and Ablation Studies
10. Discussion
11. Limitations
12. Threats to Validity
13. Ethical Considerations
14. Conclusion
15. Data and Code Availability
16. References
17. Supplementary Material

Requirements:
- State the research gap precisely.
- Explain why the problem matters for clinical LLM evaluation.
- Make research questions explicit and testable.
- Describe every formula and evaluation protocol.
- Separate measured results from estimated quantities.
- Report negative and inconclusive findings.
- Align every numerical claim with verified experimental output.
- Include clear tables and figures with captions and uncertainty estimates.
- Avoid exaggerated novelty and causal claims.
- Do not invent citations, references, experimental outcomes, or statistical significance.

Use concise academic language and maintain a consistent distinction between ranking performance and decision performance.

## 10. Reproducibility and Code Quality

Make the project reproducible from a clean environment where possible.

Check:
- Dependency and model versions.
- Dataset provenance and licensing.
- Data preprocessing and feature extraction.
- Patient identifiers and split generation.
- Prompt templates and score extraction.
- Configuration files and random seeds.
- Experiment commands and output locations.
- Result tables generated directly from verified outputs.

Add automated checks for formula correctness, metric consistency, and data-split integrity where practical.

Never overwrite original experimental outputs without preserving them. Record changes in a changelog.

## 11. Literature Review and Journal Fit

Identify relevant recent work on:
- Clinical LLM evaluation.
- Threshold optimization and decision theory.
- Signal detection theory.
- Calibration and selective classification.
- Clinical prediction-model evaluation.
- LLM reliability and benchmarking.

Verify every citation against the original publication. Clearly distinguish established methods from this paper's contribution.

Recommend suitable journals based on actual scope fit and recent published work. Verify current Q1 status against the relevant subject category and ranking year. Do not guarantee acceptance or describe a journal as Q1 without verification.

## 12. Required Deliverables

After completing the audit, provide:

1. A concise assessment of current publication readiness.
2. A list of methodological and statistical problems, ranked by severity.
3. A table mapping each research claim to its evidence, supporting code, and limitations.
4. A prioritized list of additional experiments.
5. Updated manuscript files and supplementary material, while preserving originals.
6. Reproducible commands for running each completed experiment.
7. A list of unresolved issues and experiments that could not be completed.
8. A journal shortlist with verified scope fit and quartile status.
9. A final checklist for submission readiness.

## Important Constraints

- Work incrementally and inspect the repository before making changes.
- Do not fabricate results, references, or data.
- Do not silently alter the study's hypotheses or reinterpret negative findings.
- Do not use test-set labels to select thresholds or tune models.
- Do not claim causal conclusions from observational comparisons.
- Do not treat estimated reachable BACC as a guaranteed upper bound.
- Do not claim that all clinical LLM failures are threshold artifacts.
- Preserve all corrected findings and previously retracted claims.
- If compute, data, or dependencies prevent an experiment, document the limitation instead of inventing an outcome.
- Prioritize scientific validity and reproducibility over producing impressive-looking results.

Begin with a read-only audit and an actionable plan. Do not launch expensive experiments or modify the original manuscript until the audit and proposed changes have been documented.