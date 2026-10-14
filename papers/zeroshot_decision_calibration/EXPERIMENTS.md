# Experiment Report

**Project:** Are reported zero-shot clinical LLM failures real, or are they
decision-point artifacts?
**Prepared:** 2026-10-14
**Scope:** every experiment run, including those that failed and those whose
results were withdrawn.

---

## 1. The question in one paragraph

A screening model does two separate jobs. First it **ranks** patients, putting
the sick ones higher. Second it **makes a call**, drawing a line and saying
"this one is positive". These can fail separately. A model can rank well and
still draw the line in a useless place, which makes it look like a total
failure. Ranking is measured by AUROC. The call is measured by balanced
accuracy. We check what happens when a paper reports a high AUROC and a
near-chance balanced accuracy at the same time.

We measure this with one number, the **headroom H**:

```
d'        = sqrt(2) * Phi^-1(AUROC)
reachable = Phi(d' / 2)
H         = reachable - balanced accuracy reported
```

`reachable` is the balanced accuracy a well-placed line would give on the same
scores. H is how much is being left unused. It needs only numbers that papers
already print.

---

## 2. All experiments at a glance

| # | Experiment | Data | Status | Main result |
|---|---|---|---|---|
| 1 | Headroom diagnostic, does it predict? | MedIMeta, 9 endpoints | **done** | Spearman +1.000 at k=8 |
| 2 | Three failure modes | MedIMeta, 9 endpoints | **done** | splits 3 / 3 / 3 |
| 3 | Prevalence matching (our proposed method) | MedIMeta, 9 endpoints | **done, refuted** | fails 3 separate ways |
| 4 | Label budget: where should labels go? | MedIMeta, 9 endpoints | **done, corrected** | crossover at k ~ 8 |
| 5 | Supervised reference on speech features | BenSParX | **done** (partial) | subject BACC 0.985 |
| 6 | Aggregation rule: vote vs average | BenSParX | **done** | +0.030, significant |
| 7 | How many features are needed? | BenSParX | **done** | 5-10 of 71 are enough |
| 8 | Does row-wise splitting leak? | BenSParX | **done, refuted** | no, -0.000 |
| 9 | LLM replication | BenSParX, llama3.2:3b | **done** | H +0.228, realised +0.232 |
| 10 | Aggregation on real LLM scores | BenSParX, llama3.2:3b | **done** | +0.064 to +0.069, significant |
| 11 | Audit of published results | 3 published papers | **done** | 33% of cells are decision-limited |
| 12 | Is a published conclusion confounded? | arXiv:2607.12260 | **done** | yes, partly |
| 13 | CKD LLM arm | Diabetic CKD cohort | **not run** | blocked on compute |

---

## 3. Data used

| dataset | what it is | size | licence / source |
|---|---|---|---|
| **BenSParX** | Bengali conversational speech for Parkinson's, released feature table | 900 segments, 120 subjects (60 / 60), 71 features | MIT, arXiv:2505.12192 |
| **MedIMeta** `bus`, `dr_regular` | breast ultrasound and retinal fundus images, frozen BiomedCLIP features | 780 and 2000 images | public |
| **Diabetic CKD cohort** | demographic and lifestyle records, BIRDEM Dhaka | 400 patients x 10 visits | Mendeley 10.17632/hjkzgbxgv5.2 |
| **Published tables** | result cells copied from 3 papers | 43 cells | arXiv:2605.24806, 2607.12260, 2609.04013 |

**Two data problems we found and should report:**

1. The **Diabetic CKD** file does not match its own description. The page says
   400 individuals. The CSV has 4000 rows, because it is 400 patients with 10
   yearly visits each. **Rows are not independent.** Any split must be grouped by
   patient, or results will be inflated.
2. The **WoundTissue** dataset publishes only **13 of its 147 images**, so the
   wound-segmentation papers cannot be reproduced from the public release.

---

## 4. Experiments on images (MedIMeta, 9 screening endpoints)

Setup: frozen BiomedCLIP, nine binary screening endpoints (referable diabetic
retinopathy, sight-threatening DR, malignancy, and so on). Support sets drawn
from the training split, scoring on the test split, 50 paired random draws,
metric is balanced accuracy.

### Experiment 1 — Does the headroom diagnostic predict the real gain?

We compute H from AUROC alone, then separately measure how much a fitted
decision point actually gains. If H is a good diagnostic, the two should agree.

| labels used | Pearson r | Spearman |
|---|---|---|
| 8 | **+0.925** | **+1.000** |
| 32 | +0.914 | +0.943 |
| 64 | +0.934 | +0.943 |

Agreement on direction is exact: of the 3 endpoints with H above +0.02, all 3
gained. Of the 3 with H at or below +0.02, none did.

### Experiment 2 — Three failure modes

| mode | test | endpoints |
|---|---|---|
| **A** decision-limited | AUROC > 0.5, H > +0.02 | 3 of 9 |
| **B** signal-limited | AUROC > 0.5, H <= +0.02 | 3 of 9 |
| **C** sign-inverted | AUROC < 0.5 | 3 of 9 |

Mode C matters. A model with AUROC below 0.5 ranks backwards. It has real signal
with the wrong sign, and no simple threshold rule can fix it. Published example:
one audio model is reported at AUROC 0.188 and read as a weak model.

### Experiment 3 — Our proposed method, and why we dropped it

We proposed setting the line by calling the top `pi` fraction of the cohort
positive. It failed three separate ways:

| test | result |
|---|---|
| estimate `pi` from the score distribution (no labels) | within 0.10 in **0 of 9** endpoints, mean error 0.341 |
| estimate `pi` from k labels | loses to simply fitting the line at **every** budget |
| give it the **true** `pi` (oracle) | wins only **4 of 9** endpoints |

Handed the right answer it is a coin flip. We abandoned the method. Fitting the
line directly on the labels is simpler and better.

### Experiment 4 — Where should a small label budget go?

Mean balanced accuracy over the 6 endpoints with AUROC above 0.5:

| labels | decision point (1 parameter) | linear probe (512 parameters) | Platt scaling | model's own call (0 labels) |
|---|---|---|---|---|
| 2 | 0.619 | 0.539 | 0.602 | **0.630** |
| 4 | 0.617 | 0.589 | 0.583 | **0.630** |
| 8 | 0.626 | **0.651** | 0.573 | 0.630 |
| 16 | 0.629 | **0.728** | 0.560 | 0.630 |
| 32 | 0.638 | **0.777** | 0.565 | 0.630 |
| 64 | 0.644 | **0.814** | 0.559 | 0.630 |

Two readings:

1. **Below about 8 labels, nothing beats the model's own decision.**
2. **Above about 8 labels, a properly set up supervised probe wins clearly** and
   keeps improving.

So the decision-point fix belongs in the zero-label and very-few-label setting.
It is not a replacement for supervised training once labels exist.

Platt scaling gets **worse** as labels increase (0.602 down to 0.559).

> **Note:** an earlier version of this table was wrong. See Section 8.

---

## 5. Experiments on speech (BenSParX)

Setup for 5-8: subject-grouped 5-fold cross-validation, 5 repeats (25 folds),
logistic regression on standardised features. Splits are by **subject**, because
the 900 rows are 7-8 segments from each of 120 subjects.

### Experiment 5 — Supervised reference

How much Parkinson's signal is in these 71 features at all?

| | segment AUROC | subject AUROC | subject balanced accuracy |
|---|---|---|---|
| logistic regression | 0.879 | 0.998 | **0.985** |
| published zero-shot LLaMA-3 8B | - | 0.901 | 0.833 |

A plain supervised model on the same features the LLM is shown reaches 0.985.
The zero-shot LLM reaches 0.833. For this dataset H = -0.015, so the line is
already fine and the gap is in the **ranking**.

This matters: without a supervised reference you cannot tell a bad ranking from
a badly placed line. The published paper never ran one.

*Partial:* a random-forest arm was also written but was **killed mid-run**
because it competed with other jobs for CPU. Only the logistic-regression
numbers above are measured.

### Experiment 6 — Aggregation rule: vote or average?

The published method forces one 0/1 answer per 10-second segment, then takes a
**majority vote** across a subject's 7-8 segments. This throws away how strong
each segment's evidence was. We hold the scores fixed and change only the rule.

| rule | balanced accuracy | AUROC |
|---|---|---|
| majority vote (the published habit) | 0.955 +/- 0.046 | 0.989 |
| mean probability | **0.985 +/- 0.020** | 0.998 |
| mean logit | **0.985 +/- 0.020** | 0.999 |
| maximum (control) | 0.617 +/- 0.063 | 0.995 |

Paired over 25 folds: **+0.0300, 95% CI [+0.0150, +0.0450]**. Averaging also
halves the spread. The maximum rule falls to 0.617, which shows the gain comes
from averaging, not from merely avoiding a vote.

**The size of the effect is predictable.** Segment AUROC 0.879 gives d' = 1.655.
Averaging 7.5 near-independent segments gives d' x sqrt(7.5) = 4.531:

| quantity | predicted | observed |
|---|---|---|
| subject AUROC after averaging | 0.9993 | **0.999** |
| accuracy, majority vote of 7 | 0.964 | **0.955** |
| accuracy, averaging 7 | 0.986 | **0.985** |

So this is not a quirk of one dataset.

### Experiment 7 — How many features are needed?

Selection done **inside** each training fold, so the numbers are not inflated.

| feature set | subject AUROC |
|---|---|
| all 71 | 0.999 +/- 0.003 |
| top 5 (in fold) | 0.999 +/- 0.003 |
| top 10 (in fold) | **1.000 +/- 0.001** |
| MFCC only (39) | 0.994 +/- 0.009 |
| pitch, intensity, formants (9) | 0.979 +/- 0.033 |
| energy (4) | 0.974 +/- 0.026 |
| jitter and shimmer (11) | 0.970 +/- 0.030 |
| pulses (4) | 0.903 +/- 0.048 |
| voice quality (4) | 0.873 +/- 0.078 |

The information saturates by 5 to 10 features. The published pipeline is about
**7 times larger than the task needs**. This also shortened our prompt from about
650 tokens to about 120, which is what made Experiment 9 possible at all.

### Experiment 8 — Does row-wise splitting leak? (our hypothesis was wrong)

We expected that splitting the 900 rows without grouping by subject would
inflate results. It did not.

| split | segment AUROC |
|---|---|
| grouped by subject (correct) | 0.879 +/- 0.024 |
| row-wise (expected to leak) | 0.879 +/- 0.023 |
| **inflation** | **-0.000** |

Grouping is still the correct protocol. But this null is useful: it is the
evidence that segments are near-independent given the subject, which is the
assumption Experiment 6's prediction needs.

---

## 6. LLM experiments (BenSParX, llama3.2:3b)

Setup: 40 label-balanced subjects (20 Parkinson's, 20 control), 304 segments,
10 features, one forced output token with the top-20 log-probabilities kept, so
the score `s = log P("1") - log P("0")` is exact. All 304 items scored.

**This is not a reproduction of the published work.** They used LLaMA-3 **8B**,
all **71** features and **120** subjects. We used a 3B model, 10 features and 40
subjects, because that is what our machine could run. Their published row is a
reference point, not a target.

### Experiment 9 — The fault appears, and the diagnostic predicts the fix

| quantity | value |
|---|---|
| segment AUROC | 0.628 |
| subject AUROC | **0.805** |
| the model's own call | sensitivity **1.000**, specificity **0.000** |
| its balanced accuracy | **0.500** |
| reachable | 0.728 |
| **H** | **+0.228** |

The model answers "1" for all 40 subjects. Balanced accuracy is exactly chance,
while the ranking is clearly useful. Fitting the line:

| labelled subjects | balanced accuracy |
|---|---|
| 0 | 0.500 |
| 4 | 0.659 |
| 8 | 0.694 |
| 16 | **0.732** |

**Predicted +0.228. Measured +0.232.**

The diagnostic also separates two models inside the same dataset: the published
8B cell has H = -0.015 (nothing to gain), our 3B cell has H = +0.228. Neither
label needs the answer known in advance.

### Experiment 10 — Aggregation, now on real LLM scores

At the model's own line every segment is positive, so vote and average agree and
the question cannot be asked. We fit the line first, which separates the bias
from the pooling rule.

| labelled subjects | majority vote | score averaging | difference |
|---|---|---|---|
| 4 | 0.611 +/- 0.073 | 0.652 +/- 0.102 | **+0.040 [+0.011, +0.069]** |
| 8 | 0.655 +/- 0.057 | 0.719 +/- 0.062 | **+0.064 [+0.040, +0.087]** |
| 16 | 0.658 +/- 0.057 | 0.727 +/- 0.069 | **+0.069 [+0.047, +0.091]** |

All three significant. The effect is **larger on LLM scores than on supervised
scores** (+0.064 to +0.069 against +0.030), which fits the mechanism: LLM segment
scores are noisier, so throwing away their size costs more.

---

## 7. Audit of published results

43 result cells copied from published tables. **Nothing was re-run.** For each,
the paper's own AUROC gives `reachable`, and H follows.

Sources: arXiv:2605.24806 (16 cells), arXiv:2607.12260 (16 cells with AUROC),
arXiv:2609.04013 (10 cells), plus our own cell.

### Experiment 11 — How common is the problem?

Restricted to the 36 **published LLM** cells:

| mode | count | share |
|---|---|---|
| **A decision-limited** | **12 / 36** | **33%** |
| B signal-limited | 19 / 36 | 53% |
| C sign-inverted | 5 / 36 | 14% |

**Mean unused balanced accuracy in the decision-limited cells: +0.138.**

| paper | model | dataset | AUROC | reported | reachable | H |
|---|---|---|---|---|---|---|
| 2607.12260 | LLaMa-3 8B | CKD India | 0.923 | **0.502** | 0.843 | **+0.341** |
| 2605.24806 | LLaMA-3 8B | IPVS (Italian) | 0.805 | 0.518 | 0.728 | +0.210 |
| 2607.12260 | Mistral 7B | CKD India | 0.822 | 0.535 | 0.743 | +0.208 |
| 2607.12260 | LLaMa-3 8B | CKD Bangladesh | 0.816 | 0.543 | 0.738 | +0.195 |
| 2607.12260 | LLaMa-3 8B | CKD Bangladesh | 0.749 | 0.500 | 0.683 | +0.183 |
| 2605.24806 | LLaMA-3 8B | MDVR-KCL (English) | 0.702 | 0.507 | 0.646 | +0.139 |

The first row is the clearest case. That model ranks CKD patients at AUROC 0.923,
close to the best model in the same table, and is reported at 0.502 balanced
accuracy with macro F1 of 27.68, meaning it put nearly everything in one class.
The paper prints both numbers side by side and does not comment.

### Experiment 12 — Is a published conclusion confounded?

Paper arXiv:2607.12260 claims a compact feature set makes zero-shot CKD screening
work. Its evidence is balanced accuracy, which depends on the line. Its own table
also reports AUROC, which does not. So the two effects can be separated.

| model | dataset | change in AUROC | change in BACC | reachable with all features | reported with selected |
|---|---|---|---|---|---|
| LLaMa-3 8B | CKD India | **-0.074** | **+0.262** | **0.843** | 0.764 |
| Qwen-3 8B | CKD India | **-0.058** | -0.004 | 0.819 | 0.812 |
| Mistral 7B | CKD India | +0.110 | +0.318 | 0.743 | 0.852 |
| Qwen-3 8B | CKD Bangladesh | +0.101 | +0.163 | 0.672 | 0.783 |
| LLaMa-3 8B | CKD Bangladesh | +0.067 | +0.043 | 0.683 | 0.543 |
| GPT-4o | CKD Bangladesh | +0.048 | +0.074 | 0.727 | 0.771 |
| Mistral 7B | CKD Bangladesh | +0.044 | +0.217 | 0.719 | 0.773 |
| GPT-4o | CKD India | +0.018 | +0.024 | 0.846 | 0.880 |

1. In **2 of 8** pairs selection made the **ranking worse** while balanced
   accuracy rose or held. There it added no information; it moved the line.
2. In **3 of 8** pairs, fixing the line on the all-feature scores already beats
   the reported selected-feature result. Clearest case: **0.843 against 0.764**.
3. In the other **5 of 8** the ranking genuinely improved.

So the claim is **not wrong everywhere**. It is confounded, and the paper's own
AUROC column is enough to separate the two effects.

---

## 8. Results we withdrew

These were measured, reported, and then found to be wrong. They are listed so
they are not reused.

### 8.1 Two false results from a badly configured baseline

Our first linear probe used `LogisticRegression(C=1.0)` with **no class
weighting**. On a skewed support set an unweighted fit minimises error by
predicting the majority class. Two claims came from this and are **withdrawn**:

| withdrawn claim | what we reported | corrected |
|---|---|---|
| "the probe collapses as labels increase" | degenerate on 5 of 9 endpoints at k=64 | **0 of 9** at every k >= 4 |
| "few labels are nearly worthless" | 64 labels buy +0.014 | 64 labels buy **+0.184** |

The corrected numbers are in Experiment 4. **Lesson:** a baseline sitting at
exactly 0.500 is a misconfiguration, not a finding. It should be checked before
it is written down.

### 8.2 A bug in our own diagnostic

An early version computed `reachable` from `max(AUROC, 1 - AUROC)`. That silently
assumes you already know which way the score runs, which costs labels. It
reported +0.237 and +0.248 of headroom on two endpoints whose AUROC was about
0.32, where no simple rule can recover anything. Fixed. All results above use the
corrected form.

### 8.3 A reporting slip

An earlier version of the Experiment 9 table listed 0.611 and 0.655 at 4 and 8
labels. Those were the **majority-vote** values, not the fitted-line values. The
correct values are 0.659 and 0.694. The headline numbers were unaffected.

### 8.4 Our proposed method

Prevalence matching, refuted three ways (Experiment 3).

### 8.5 Our leakage hypothesis

Row-wise splitting did not leak (Experiment 8).

---

## 9. What has not been done

- **The CKD LLM arm has not been run.** The code is written and resumable
  (`RUN_WHEN_FREE.sh`). It is blocked on machine capacity, not on method.
- **The audit covers 3 papers (36 published LLM cells).** Four more have been
  identified but not extracted: delirium (arXiv:2607.20453), child malnutrition
  (arXiv:2607.29082), ClinicRealm (arXiv:2407.18525), AgentRx (arXiv:2605.10286).
  This is the main gap before submission.
- **Many papers report AUROC with no threshold metric at all**, which blocks the
  audit. How many is itself worth counting and reporting.
- **The conformal prediction and selective classification literature has not been
  surveyed.** Both deal with operating points in medical AI. Any novelty claim
  should be checked against them first.
- **The binormal step is an estimate, not a bound.** In **6 of 43 cells** the
  reported balanced accuracy is already above `reachable`, the worst by 0.117.
  We report these as negative H rather than hiding them.

---

## 10. An honest note on novelty

There is **no new method** in this work. The headroom formula is signal detection
theory from the 1960s. Choosing an operating point is standard practice in
clinical machine learning. Averaging scores instead of voting on hard calls is
textbook. One of our findings (fewer features work better) was published
independently in arXiv:2607.12260 before we measured it.

The contribution is at the level of the field, not the method: the recent
zero-shot clinical LLM literature reports decision-point artifacts as model
failures, at a measurable rate of about one third of cells, and the tool needed
to notice it was already printed in those papers' own tables.

---

## 11. Figures

Six figures in `figures/`, as PDF and PNG, built directly from the result files.

| figure | shows |
|---|---|
| `fig1_audit_scatter` | all 43 cells; reported AUROC against reported balanced accuracy, with the reachable curve. The gap is drawn. |
| `fig2_headroom_bars` | the 36 published LLM cells sorted by H |
| `fig3_forward_test` | Experiment 9: the predicted line drawn against the measured points |
| `fig4_label_budget` | Experiment 4, corrected |
| `fig5_aggregation` | Experiments 6 and 10 side by side |
| `fig6_feature_saturation` | Experiment 7 |

Colours are from a colour-blind-safe palette, checked with a validator rather
than by eye.

---

## 12. Files

| file | what it is |
|---|---|
| `EXPERIMENTS.md` | this report |
| `draft_paper.md` | the paper draft |
| `FINDINGS.md` | the full working log, including every withdrawn claim in place |
| `src/` | 10 scripts, one per experiment group |
| `results/` | raw outputs as JSON and logs; `harvest.json` carries the paper and table for every audited cell |
| `figures/` | the six figures |
| `RUN_WHEN_FREE.sh` | the queued CKD arm |

Every number in this report traces to a file in `results/`.
