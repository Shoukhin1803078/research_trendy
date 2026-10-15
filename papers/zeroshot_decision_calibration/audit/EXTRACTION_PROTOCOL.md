# Extraction Protocol

**Written 2026-10-14, BEFORE inspecting the candidate papers**, so that
eligibility cannot be decided by whether a cell supports the thesis.

## 1. Eligibility

A **paper** is eligible if it evaluates a language model, or an audio/vision
language model, on a clinical classification task in a zero-shot or few-shot
setting without task-specific weight updates.

A **result cell** within an eligible paper is extracted if and only if the paper
reports, for one (model, dataset, endpoint, setting) combination, **both**:

- **R.** a ranking metric: AUROC (or AUC, C-statistic, macro-AUROC), **and**
- **D.** a threshold-dependent metric from which balanced accuracy can be
  obtained **exactly**, meaning one of:
  - balanced accuracy printed directly, or
  - sensitivity **and** specificity printed (BACC = (sens+spec)/2), or
  - a confusion matrix, or
  - accuracy **plus** class prevalence **plus** enough structure to recover
    sens and spec without assumption.

## 2. Exclusions, each recorded with its reason

- **E1** no ranking metric reported.
- **E2** no threshold-dependent metric reported.
- **E3** single-class evaluation set, so AUROC is undefined.
- **E4** metrics reported only in a figure, not readable to the printed
  precision.
- **E5** the task is not clinical classification.
- **E6** the model was fine-tuned on the task.
- **E7** values ambiguous or internally inconsistent.

## 3. Never do

- Never estimate, impute, or back-calculate an unreported metric.
- Never read values off a plot.
- Never convert accuracy to balanced accuracy without exact prevalence and
  per-class counts.
- Never treat cells from one paper as independent papers.

## 4. Schema

Each cell records: `paper_id`, `paper_short`, `model`, `dataset`, `endpoint`,
`setting`, `n_eval` (if stated), `prevalence` (if stated), `auroc`, `bacc`,
`sens`, `spec`, `bacc_source` (printed | from_sens_spec | from_confusion),
`threshold_protocol` (how the paper chose its operating point, or
`not_stated`), `source_location` (table/figure number), `verified` (bool),
`notes`.

Missing values are recorded as `null` and **counted**, never filled.

## 5. Threshold protocol coding

How the paper obtained its decision, which is the variable our thesis concerns:

- `forced_token` - one generated token taken as the decision.
- `fixed_0.5` - a fixed probability cut-off.
- `tuned_on_val` - threshold selected on validation data.
- `tuned_on_test` - threshold selected on the evaluation data (optimistic).
- `argmax` - highest-scoring class.
- `not_stated`.

## 6. Verification

Every number is read from the paper's own table and re-read once against the
HTML or PDF. `verified=true` only after that second read. Any cell failing
re-read is dropped and recorded under E7.

## 7. Analysis rules, fixed in advance

- Clustering unit is the **paper**.
- Primary statistic: proportion of eligible LLM cells with AUROC > 0.5 and
  H > 0.02, reported with a paper-level bootstrap interval and the full
  sensitivity table over cut-offs 0.00-0.10.
- The cut-off is **not** re-tuned after seeing new data.
- Non-LLM baseline rows are extracted where available but analysed separately.
- If the expanded audit lowers the rate or shows strong heterogeneity, that is
  the result.
