# When a Zero-Shot Model Looks Like It Failed But Did Not: A Decision-Rule Audit of Clinical LLM Screening

**Authors:** (to be filled)
**Status:** draft, 2026-10-14
**Target:** Q1 journal in medical AI (Artificial Intelligence in Medicine / Computers in Biology and Medicine / IEEE JBHI)

---

## Abstract

Many recent papers test large language models on clinical screening tasks without
any training. They ask the model a question, read one word of the answer, and
score it. When the score is low, they conclude the model cannot do the task.

We show this conclusion is often wrong. A model can order patients correctly and
still make bad yes/no calls, because the yes/no call depends on where the
decision point sits. The ordering is measured by AUROC. The yes/no call is
measured by balanced accuracy. When the first is high and the second is near
chance, the problem is the decision point, not the model.

We use a simple formula from signal detection theory to estimate, from the AUROC
a paper already reports, how good its yes/no calls could be if the decision
point were placed well. We call the gap *headroom* (H). We then audit 43 result
cells copied from three published papers. In 33% of the published LLM cells the
reported failure is a decision-point problem. On average these cells leave 0.138
balanced accuracy unused. The worst case leaves 0.341: a model that orders
kidney-disease patients at AUROC 0.923 is reported at 0.502 balanced accuracy,
and the paper does not mention it.

We then test the idea forward on Bengali Parkinson's speech data. Our own model
shows the same fault. Headroom predicted a gain of 0.228, and fitting a decision
point on 16 labelled subjects gave 0.232. We also show that one common habit,
voting on hard per-segment decisions instead of averaging the scores, costs 0.064
to 0.069 balanced accuracy.

We add no new method. The tool we use is old. The point is that the field has
this tool in its own result tables and does not use it.

---

## 1. Introduction

### 1.1 The problem

A screening model does two separate jobs.

1. **It ranks patients.** Given many patients, it puts the sick ones higher.
2. **It makes a call.** It draws a line and says "this patient is positive".

These jobs can fail separately. A model can rank well and still draw the line in
a useless place. If the line is too high, it calls everyone healthy. If the line
is too low, it calls everyone sick. Either way the balanced accuracy is near 0.5,
which looks like total failure.

Zero-shot LLM screening makes this very likely. The usual recipe is:

- put the patient data into a prompt,
- force the model to answer with one token, "0" or "1",
- treat that token as the decision.

That token is one fixed decision point. Nobody chose it. It comes from whatever
the model happens to prefer. There is no way to move it, because it is not a
number you can tune.

### 1.2 What we do

We separate the two jobs using the numbers papers already publish.

From the reported AUROC we compute the balanced accuracy a well-placed line would
reach on the *same scores*. We subtract the reported balanced accuracy. The
difference is the headroom H. If H is large and positive, the paper's model ranks
well and decides badly, and the reported failure is recoverable without a better
model.

We do three things with this.

1. **An audit.** We copy 43 result cells from three published papers and compute
   H for each. We sort every cell into three failure modes.
2. **A forward test.** We run the audit's prediction on new data and check
   whether the predicted gain appears.
3. **A protocol check.** We test two habits in this literature that affect the
   decision but are not usually examined: how per-segment decisions are combined,
   and how many input features are used.

### 1.3 Contributions

- A simple, label-free diagnostic (H) that says, before any new labelling,
  whether a reported zero-shot failure is a decision problem or a model problem.
- A three-way failure taxonomy: decision-limited, signal-limited, sign-inverted.
- An audit of 43 published cells. 33% of published LLM cells are
  decision-limited, with mean 0.138 balanced accuracy unused.
- Evidence that one published conclusion about feature selection is confounded
  with a decision-point effect, shown from that paper's own table.
- A forward test where H predicted 0.228 and the measured gain was 0.232.
- Measured cost of a common aggregation habit: 0.064-0.069 balanced accuracy.
- Honest negative results, including three of our own hypotheses that failed.

---

## 2. Related work

**Zero-shot LLMs for clinical screening.** Several recent papers prompt an LLM
with patient data and score the answer. Examples include chronic kidney disease
screening [1, 2] and Parkinson's disease detection from speech [3]. These papers
report AUROC and at least one threshold-dependent metric. None of them studies
where the decision point sits.

**Signal detection theory.** The link from AUROC to the best reachable accuracy
under a Gaussian model is standard and old [8]. We use it unchanged. We claim no
novelty for the formula.

**Threshold selection in clinical machine learning.** Choosing an operating point
is a mature topic. Youden's J, cost-weighted cutoffs and prevalence-adjusted
cutoffs are all standard practice. Our point is not that these methods are new.
Our point is that the zero-shot LLM literature does not apply them, and that this
has a measurable cost.

**Calibration of model confidence.** Work on neural network calibration and on
LLM confidence shows that reported probabilities are often wrong. Calibration and
operating-point choice are related but not the same. Calibration fixes the
numbers. We are asking a narrower question: given the scores as they are, how
much is lost by the single fixed decision point.

**Gap we fill.** No prior work, as far as we found, audits this literature for
decision-point artifacts, or reports what fraction of published failures are
recoverable. We also note our search was limited (Section 8.4).

---

## 3. Method

### 3.1 One score per patient

Every model we study is reduced to one number per patient, where a higher number
means "more likely positive".

For an LLM we force one output token and read the log-probabilities:

```
s = log P("1") - log P("0")
```

The model's own decision is `s > 0`. That is exactly the token it emitted. So the
token is not extra information. It is one operating point on `s`.

For a vision-language model we use the difference of the prompt logits:

```
s = logsumexp(positive class prompts) - logsumexp(negative class prompts)
```

Again the native decision is `s > 0`.

### 3.2 The headroom diagnostic

Assume the scores of the positive and negative groups are Gaussian with the same
spread. This is the binormal model. Then AUROC fixes the separation `d'`, and the
best balanced accuracy at the best line follows:

```
d'        = sqrt(2) * Phi^-1(AUROC)
reachable = Phi(d' / 2)
H         = reachable - BACC(reported)
```

`Phi` is the normal CDF. H needs only two numbers that papers already print.

**One detail matters.** We use AUROC in the direction the model already points.
We do **not** use `max(AUROC, 1 - AUROC)`. Using the maximum assumes you already
know the score runs the right way, and knowing that costs labels. An AUROC below
0.5 means the model ranks backwards, and no simple threshold rule can fix that.

We found this the hard way. Our first version used the maximum and reported
0.237 and 0.248 of headroom on two endpoints whose AUROC was about 0.32. That
headroom was not real. The corrected version is in all results below.

### 3.3 The three failure modes

| mode | test | meaning |
|---|---|---|
| **A: decision-limited** | AUROC > 0.5 and H > +0.02 | ranks well, decides badly. Recoverable. |
| **B: signal-limited** | AUROC > 0.5 and H <= +0.02 | the line is already fine. Needs a better model. |
| **C: sign-inverted** | AUROC < 0.5 | ranks backwards. No monotone rule helps. |

Mode C is easy to miss. A paper may report a low accuracy and call the model
weak, when in fact the model has strong but reversed signal.

### 3.4 The fix we test

For mode A cells the fix is one number: pick the line that gives the best
balanced accuracy on `k` labelled cases, then apply it to everyone else. One
parameter, no training, no gradients.

---

## 4. Audit of published results

### 4.1 How we built the table

We copied 43 result cells from published tables. We ran nothing. Sources:

- **[3]** Parkinson's speech, zero-shot table. 16 cells. Reports balanced
  accuracy, AUROC, sensitivity and specificity.
- **[2]** Zero-shot CKD screening, Table 3. 16 cells that have AUROC. Its six
  Dataset-3 cells have no AUROC, because that dataset has only positive cases,
  so we exclude them.
- **[1]** LLM4CKD, Table II. 4 LLM cells and 6 non-LLM baseline cells.
- **Ours.** 1 cell from Section 5.

Where a paper gives sensitivity and specificity but not balanced accuracy, we use
`BACC = (sens + spec) / 2` and flag the row. Every row records its paper and
table.

### 4.2 One third of published LLM results are decision-limited

Restricted to the 36 published **LLM** cells (our own cell and the non-LLM
baselines removed):

| mode | count | share |
|---|---|---|
| **A: decision-limited** | **12/36** | **33%** |
| B: signal-limited | 19/36 | 53% |
| C: sign-inverted | 5/36 | 14% |

**Mean unused balanced accuracy in the decision-limited cells: +0.138.**

The largest cases, all taken from the papers' own tables:

| paper | model | dataset | AUROC | reported BACC | reachable | H |
|---|---|---|---|---|---|---|
| [2] | LLaMa-3 8B | CKD India | 0.923 | **0.502** | 0.843 | **+0.341** |
| [3] | LLaMA-3 8B | IPVS (Italian) | 0.805 | 0.518 | 0.728 | +0.210 |
| [2] | Mistral 7B | CKD India | 0.822 | 0.535 | 0.743 | +0.208 |
| [2] | LLaMa-3 8B | CKD Bangladesh | 0.816 | 0.543 | 0.738 | +0.195 |
| [2] | LLaMa-3 8B | CKD Bangladesh | 0.749 | 0.500 | 0.683 | +0.183 |
| [3] | LLaMA-3 8B | MDVR-KCL (English) | 0.702 | 0.507 | 0.646 | +0.139 |

The first row is the clearest case we found. That model ranks CKD patients at
AUROC 0.923, which is close to the best model in the same table. It is reported
at 0.502 balanced accuracy, with macro F1 of 27.68, which means it put almost
everything in one class. The paper prints both numbers next to each other and
does not comment.

![Figure 1](figures/fig1_audit_scatter.png)

**Figure 1.** Every one of the 43 audited cells. The x axis is the AUROC the
paper reported, which measures ranking. The y axis is the balanced accuracy it
reported, which measures the yes/no call. The dashed curve is the balanced
accuracy a well-placed decision point would reach on the same scores. Blue
vertical lines are the unused accuracy, H. Cells on or above the curve have
nothing to recover. Cells left of 0.5 rank backwards.

![Figure 2](figures/fig2_headroom_bars.png)

**Figure 2.** The 36 published LLM cells sorted by H. Blue bars are recoverable
without a better model. Red bars have nothing to recover; there the reported
balanced accuracy already sits at or above the binormal estimate, which is
discussed in Section 8.1.


The same pattern appears in [3] with the opposite sign. On IPVS, LLaMA-3 has
sensitivity 3.57% and specificity 100%: it calls almost everyone healthy. That
paper concludes "zero-shot capability alone is insufficient". For this cell the
evidence does not support that. The ranking is good.

### 4.3 A published conclusion is confounded

Paper [2] argues that a small, chosen feature set makes zero-shot CKD screening
work. Its evidence is balanced accuracy, which depends on the decision point. Its
own table also reports AUROC, which does not. So we can separate the two effects.

| model | dataset | change in AUROC | change in BACC | reachable with all features | reported BACC with selected |
|---|---|---|---|---|---|
| LLaMa-3 8B | CKD India | **-0.074** | **+0.262** | **0.843** | 0.764 |
| Qwen-3 8B | CKD India | **-0.058** | -0.004 | 0.819 | 0.812 |
| Mistral 7B | CKD India | +0.110 | +0.318 | 0.743 | 0.852 |
| Qwen-3 8B | CKD Bangladesh | +0.101 | +0.163 | 0.672 | 0.783 |
| LLaMa-3 8B | CKD Bangladesh | +0.067 | +0.043 | 0.683 | 0.543 |
| GPT-4o | CKD Bangladesh | +0.048 | +0.074 | 0.727 | 0.771 |
| Mistral 7B | CKD Bangladesh | +0.044 | +0.217 | 0.719 | 0.773 |
| GPT-4o | CKD India | +0.018 | +0.024 | 0.846 | 0.880 |

Three readings:

1. In **2 of 8** pairs feature selection made the **ranking worse** while balanced
   accuracy stayed flat or rose. There it added no information. It moved the line.
2. In **3 of 8** pairs, just fixing the line on the all-feature scores already
   beats the reported selected-feature result. The clearest case is LLaMa-3 on
   CKD India: **0.843 reachable with all features against 0.764 reported with
   selected features**, so doing nothing but placing the line is 0.079 better.
3. In the other **5 of 8** pairs the ranking really did improve. So the claim is
   **not wrong everywhere**. It is confounded. And the paper's own AUROC column
   is enough to tell the two effects apart.

![Figure 6](figures/fig6_feature_saturation.png)

**Figure 6.** Our own measurement of the same question on speech data. The
information saturates by 5 to 10 features, so a 71-feature pipeline is about
seven times larger than the task needs. Selection is done inside each training
fold.


---

## 5. Forward test on Bengali Parkinson's speech

The audit looks backwards. A diagnostic is only useful if it also works forwards.

### 5.1 Data and setup

We use **BenSParX** [4], a public Bengali conversational speech corpus for
Parkinson's disease, released under the MIT licence. It has 900 segments from 120
subjects, 60 with Parkinson's and 60 controls, and 71 released acoustic features.

Because the 900 rows are 7 to 8 segments from each subject, the rows are not
independent. All our splits are by **subject**, never by row.

We run `llama3.2:3b` on 40 label-balanced subjects (20 and 20), 304 segments. We
force one token and keep the top-20 log-probabilities, so `s` is exact.

**This is not a reproduction of [3].** They used LLaMA-3 **8B**, all **71**
features and **120** subjects. We use a 3B model, 10 features and 40 subjects.
The 3B model is what fits in our memory. The 10 features follow our own result in
Section 6.2. We use their published row as a reference point, not as a target.

### 5.2 The same fault appears

| quantity | value |
|---|---|
| segment AUROC | 0.628 |
| subject AUROC | **0.805** |
| native decision | sensitivity **1.000**, specificity **0.000** |
| native balanced accuracy | **0.500** |
| reachable | 0.728 |
| **H** | **+0.228** |

The model answers "1" for all 40 subjects. Balanced accuracy is exactly chance.
The ranking is clearly useful, at subject AUROC 0.805. This is the mirror image of
the IPVS cell in [3], which calls everyone healthy.

### 5.3 The diagnostic predicted the gain

We fit the line on `k` labelled subjects and score the rest, 50 random draws:

| labelled subjects | balanced accuracy |
|---|---|
| 0 (native) | 0.500 |
| 4 | 0.659 |
| 8 | 0.694 |
| 16 | **0.732** |

**Predicted H = +0.228. Measured gain at k = 16 = +0.232.**

![Figure 3](figures/fig3_forward_test.png)

**Figure 3.** The forward test. With no labels the model answers "1" for every
subject, so balanced accuracy is exactly chance. Fitting the decision point on a
few labelled subjects moves it to the level the diagnostic predicted from AUROC
alone. The dashed line is that prediction, made before any of these points were
measured. Bars are one standard deviation over 50 random draws.


Within the same dataset the diagnostic also separates two models correctly. The
LLaMA-3 8B cell in [3] has **H = -0.015**, so there is nothing to gain there, and
indeed that cell already performs near its ceiling. Our 3B cell has **H = +0.228**.
Neither label needs the answer known in advance.

### 5.4 Why the deficit is in the ranking, not the line, for the 8B model

We also trained a plain logistic regression on the same 71 features, with
subject-grouped 5-fold cross-validation, 5 repeats:

| | segment AUROC | subject AUROC | subject BACC |
|---|---|---|---|
| logistic regression | 0.879 | 0.998 | **0.985** |
| zero-shot LLaMA-3 8B [3] | - | 0.901 | 0.833 |

A simple supervised model on the features the LLM is shown reaches 0.985. The
zero-shot LLM reaches 0.833. Together with H = -0.015 this says the 8B deficit is
in the ordering, not the line.

This matters for how [3] states its conclusion. On BenSParX their sentence is
right. But they could not have known it, because they never ran a supervised
reference on their own features. Without that reference you cannot tell a bad
ranking from a badly placed line. On their IPVS and MDVR-KCL cells the pattern is
the opposite, and the same sentence does not hold.

---

## 6. Two protocol habits that change the decision

### 6.1 Voting on hard decisions costs accuracy

Paper [3] forces one 0/1 token per 10-second segment, then takes a **majority
vote** over a subject's 7 to 8 segments. This throws away the size of the evidence
twice: once when the score becomes a sign, again when signs are counted instead
of averaged.

We hold the scores fixed and change only how they are combined.

**On supervised scores** (subject-grouped 5-fold, 5 repeats, 25 folds):

| rule | balanced accuracy | AUROC |
|---|---|---|
| majority vote (the habit in [3]) | 0.955 +- 0.046 | 0.989 |
| mean probability | **0.985 +- 0.020** | 0.998 |
| mean logit | **0.985 +- 0.020** | 0.999 |
| maximum probability (control) | 0.617 +- 0.063 | 0.995 |

Paired over 25 folds: **+0.0300, 95% CI [+0.0150, +0.0450]**. Averaging also cuts
the spread in half. The maximum-probability control falls to 0.617, which shows
the gain is about averaging and not about "anything other than voting".

**On real LLM log-probabilities.** At the model's own line every segment is
positive, so voting and averaging agree and the question cannot be asked. We
therefore fit the line first, which separates the bias from the pooling. Both
rules get the same labels and the same scores:

| labelled subjects | majority vote | score averaging | difference |
|---|---|---|---|
| 4 | 0.611 +- 0.073 | 0.652 +- 0.102 | **+0.040 [+0.011, +0.069]** |
| 8 | 0.655 +- 0.057 | 0.719 +- 0.062 | **+0.064 [+0.040, +0.087]** |
| 16 | 0.658 +- 0.057 | 0.727 +- 0.069 | **+0.069 [+0.047, +0.091]** |

All three are significant. The effect is **larger on LLM scores than on
supervised scores**, which fits the mechanism: the LLM's segment scores are
noisier, so throwing away their size costs more.

**The size of the effect is predictable.** Segment AUROC 0.879 gives `d' = 1.655`.
If a subject's segments are near-independent, averaging 7.5 of them gives
`d' * sqrt(7.5) = 4.531`:

| quantity | predicted | observed |
|---|---|---|
| subject AUROC after averaging | 0.9993 | 0.999 |
| accuracy, majority vote of 7 | 0.964 | 0.955 |
| accuracy, averaging 7 | 0.986 | 0.985 |

So this is not a quirk of one corpus. It is the expected cost of discarding
magnitude before pooling.

![Figure 5](figures/fig5_aggregation.png)

**Figure 5.** Left: on real LLM log-probabilities, with the decision point fitted
first so that the bias and the pooling rule are separated. All three gaps are
significant. Right: the same comparison on supervised scores, where the maximum
rule is included as a control to show the gain comes from averaging and not
merely from avoiding a vote.


### 6.2 Five features do the work of seventy-one

Paper [3] lists its fixed feature pipeline as a limitation. We measured what it
costs. Selection is done **inside** each training fold, so the numbers are not
inflated.

| feature set | subject AUROC |
|---|---|
| all 71 | 0.999 +- 0.003 |
| top 5 (chosen in fold) | 0.999 +- 0.003 |
| top 10 (chosen in fold) | **1.000 +- 0.001** |
| MFCC only (39) | 0.994 +- 0.009 |
| pitch, intensity, formants (9) | 0.979 +- 0.033 |
| energy (4) | 0.974 +- 0.026 |
| jitter and shimmer (11) | 0.970 +- 0.030 |
| pulses (4) | 0.903 +- 0.048 |
| voice quality (4) | 0.873 +- 0.078 |

The information saturates by 5 to 10 features. The pipeline is about 7 times
larger than the task needs. This also has a practical effect: it shortened our
prompt from about 650 tokens to about 120, which is what made our run possible at
all.

We note that a compact-feature result for zero-shot CKD was published
independently in [2]. Our contribution here is the measurement on speech, and the
observation in Section 4.3 that part of their gain is a decision-point effect.

---

## 7. How many labels are worth spending, and on what

We also asked where a small label budget should go. We used 9 binary screening
endpoints built from two public MedIMeta [5] datasets with a frozen BiomedCLIP
encoder, 50 paired random draws, support from the training split, scoring on the
test split.

Mean balanced accuracy over the 6 endpoints whose AUROC is above 0.5:

| labels | 1-parameter line | linear probe (512 parameters) | Platt scaling | native (0 labels) |
|---|---|---|---|---|
| 2 | 0.619 | 0.539 | 0.602 | **0.630** |
| 4 | 0.617 | 0.589 | 0.583 | **0.630** |
| 8 | 0.626 | **0.651** | 0.573 | 0.630 |
| 16 | 0.629 | **0.728** | 0.560 | 0.630 |
| 32 | 0.638 | **0.777** | 0.565 | 0.630 |
| 64 | 0.644 | **0.814** | 0.559 | 0.630 |

Two readings:

1. **Below about 8 labels, nothing beats the model's own decision** or a
   one-parameter line on it.
2. **Above about 8 labels, a properly set up supervised probe wins clearly** and
   keeps improving, reaching 0.814 at 64 labels.

So the decision-point fix belongs to the zero-label and very-few-label setting,
which is exactly where the zero-shot screening literature works. It is **not** a
replacement for supervised training once labels exist.

Platt scaling gets **worse** as labels increase, from 0.602 to 0.559. Two
parameters fitted on a handful of skewed labels is a worse rule than the model's
own.

![Figure 4](figures/fig4_label_budget.png)

**Figure 4.** Where a small label budget should go. The shaded band is the region
where no method beats the model's own zero-label decision. The linear probe is
configured with balanced class weights; an earlier version of this experiment
used an unweighted probe and produced a false result, which we withdraw in
Section 8.6.


On the diagnostic itself, H predicts the gain from fitting the line across these
endpoints: Pearson +0.925 and Spearman +1.000 at k = 8, and Pearson +0.914 to
+0.934 at k = 32 and 64. The three failure modes split the 9 endpoints 3 / 3 / 3.

---

## 8. Limitations

We list the ones that could change our conclusions.

### 8.1 The binormal step is an estimate, not a bound

`reachable` assumes equal-spread Gaussian scores. Real scores are not always
Gaussian. In **6 of our 43 cells the reported balanced accuracy is already above
`reachable`**, the most extreme by 0.117. We report these as negative H rather
than hiding them. So H should be read as an estimate of recoverable accuracy, not
as a hard ceiling. Where a paper publishes an ROC curve, the empirical ceiling
should be reported next to it.

### 8.2 The audit is small

36 published LLM cells from 3 papers. We have identified four further candidate
papers but have not yet extracted them. Many papers in this area report AUROC
with **no** threshold-dependent metric at all, which blocks the audit and is
itself worth reporting about the field.

### 8.3 Our forward test is one cell

We validated the prediction on one model and one dataset, with a 3B model
standing in for an 8B one, 10 features instead of 71, and 40 subjects instead of
120. The queued CKD arm is written but not run (see Section 10).

### 8.4 Our literature search was limited

We ran a small number of searches. We did not survey conformal prediction or
selective classification, which also deal with operating points in medical AI.
Any novelty claim in this paper should be checked against those areas first.

### 8.5 BenSParX is close to saturated

A plain logistic regression reaches subject AUROC 0.998 and balanced accuracy
0.985. This is an easy dataset at subject level, so it is a weak test bed for
hard screening questions. The jump from segment AUROC 0.879 to 0.998 is fully
explained by averaging near-independent segment noise, so we do not claim a
hidden confound.

### 8.6 Our own failed hypotheses

We report these so they are not repeated.

- **Prevalence matching does not work.** We tried setting the line by calling the
  top `pi` fraction positive. Estimating `pi` from the score distribution with a
  two-component Gaussian mixture was within 0.10 in **0 of 9** endpoints, mean
  error 0.341. Estimating `pi` from `k` labels lost to simply fitting the line at
  every budget. Even given the **true** prevalence, it won only **4 of 9**
  endpoints. We abandoned it.
- **Row-wise splitting did not leak here.** We expected inflation from splitting
  the 900 BenSParX rows without grouping by subject. Measured inflation was
  **-0.000**. Grouping is still the right protocol; it just does not change this
  number. The null does support the independence assumption used in Section 6.1.
- **Two earlier findings of ours were artifacts and are withdrawn.** Our first
  linear-probe baseline was fitted without class weighting, so on skewed support
  sets it predicted the majority class. That produced a false "the probe collapses
  as labels increase" result (0 of 9 after the fix, not 5 of 9) and a false "few
  labels are nearly worthless" result (a correct probe reaches 0.814 at 64 labels,
  +0.184 over zero labels, not +0.014). Section 7 reports the corrected numbers.

---

## 9. Conclusion

We add no new method. The formula we use is decades old, and choosing an operating
point is standard practice in clinical machine learning. That is the point of the
paper.

The recent zero-shot clinical LLM literature reports decision-point artifacts as
model failures. We measured the rate: about one third of published LLM result
cells, leaving 0.138 balanced accuracy unused on average and up to 0.341 in the
worst case. The tool needed to notice this was already printed in those papers'
own tables, next to the number being misread.

For practice we suggest three things, all cheap:

1. Report AUROC and a threshold metric together, and say which one your
   conclusion rests on.
2. Check the sign. An AUROC below 0.5 is reversed signal, not a weak model.
3. Average scores before deciding. Do not vote on hard per-item calls.

---

## 10. Reproducibility and artifacts

- All figures: `figures/` (PDF and PNG), produced by `src/make_figures.py`
  directly from the result files. Colours come from a validated colour-blind-safe
  palette; the scatter uses only the three slots that pass all-pairs separation.
- All code: `src/`. `rope.py` (diagnostic and metrics), `harvest.py` (audit),
  `score_llm.py` (LLM scoring, resumable), `bensparx_*.py`, `run_budget.py`.
- All raw results: `results/`, including `harvest.json` with the paper and table
  of every audited cell.
- `FINDINGS.md` records every measured result and every withdrawn claim in place.
- `RUN_WHEN_FREE.sh` runs the queued CKD arm when a machine is free.

**Data note.** The public diabetic CKD dataset [6] does not match its own
description. Its page states 400 individuals, but the CSV has 4000 rows: 400
patients with 10 yearly visits each. Record-level prevalence is 9.65%, and 185 of
400 patients ever become positive. **Rows are not independent and any split must
be grouped by patient.** We report this because a row-wise split would leak.

**Data note.** The WoundTissue release [7] publishes 13 of its 147 images, so
results on that corpus cannot be reproduced from the public release.

---

## References

[1] LLM4CKD: Large Language Models for Early Stage Chronic Kidney Disease
    Screening. arXiv:2609.04013.
[2] From Many to Meaningful: Feature-Guided Zero-Shot Chronic Kidney Disease
    Screening using Large Language Models. arXiv:2607.12260.
[3] M. A. Kabir and S. Munira. Zero-Shot Parkinson's Disease Detection from
    Speech: Comparing Large Audio and Language Models. arXiv:2605.24806.
[4] R. Hossain, M. A. Kabir, A. I. G. Mowla, A. C. Roy, R. K. Ghosh. BenSParX: A
    Robust Explainable Machine Learning Framework for Parkinson's Disease
    Detection from Bengali Conversational Speech. arXiv:2505.12192.
[5] S. Woerner, C. F. Baumgartner. MedIMeta: A Multi-Domain Multi-Task Medical
    Imaging Meta-Dataset. arXiv:2404.16000 / arXiv:2408.08058.
[6] A Dataset on Demographic and Lifestyle Factors for Prognosticating Chronic
    Kidney Disease Progression in Diabetic Patients. Mendeley Data,
    doi:10.17632/hjkzgbxgv5.2.
[7] Deep Learning for Wound Tissue Segmentation: A Comprehensive Evaluation.
    arXiv:2502.10652. Dataset: github.com/akabircs/WoundTissue.
[8] D. M. Green, J. A. Swets. Signal Detection Theory and Psychophysics. 1966.

---

## Appendix A. Sections still to write

- Extend the audit to arXiv:2607.20453 (delirium), arXiv:2607.29082 (child
  malnutrition), ClinicRealm (arXiv:2407.18525), AgentRx (arXiv:2605.10286).
- Count how many papers report AUROC with no threshold metric.
- Run the queued CKD arm for a second forward test in a different modality.
- Survey conformal prediction and selective classification (Section 8.4).
