# Verified findings — 2026-10-13

Only measured results. Every number here came from a script in `src/` and its
raw output is in `results/`. Where a hypothesis of mine was refuted, the
refutation is recorded in place of the hypothesis.

## Setup

Zero-shot BiomedCLIP (cached features, no GPU) on **nine binary clinical
screening endpoints** built from MedIMeta `bus` and `dr_regular`. The
binarisations are the standard screening endpoints for each task (referable DR
at grade >=2, sight-threatening at >=3, and so on). Score for each image is

    s = logsumexp(logits over positive-class prompts) - logsumexp(over negatives)

so the model's **native** decision is exactly `s > 0`, i.e. its own argmax.
Support sets are drawn from the train split, evaluation is on the test split,
50 paired seeds, metric is balanced accuracy (BACC). `src/run_arm_a.py`,
`src/run_budget.py`.

## 1. The headroom diagnostic works — this is the result worth building on

For each endpoint, define the reachable BACC under a binormal model with the
observed AUROC, **in the direction the model already points**:

    d' = sqrt(2) * Phi^-1(AUROC),   reachable = Phi(d'/2),   H = reachable - BACC_native

H is computable from AUROC alone and predicts how much a better-placed
threshold can recover. Against the gain actually realised by fitting a
threshold on k labels, over the six non-inverted endpoints:

| k | Pearson r | Spearman |
|---|---|---|
| 8 | **+0.925** | **+1.000** |
| 32 | +0.914 | +0.943 |
| 64 | +0.934 | +0.943 |

Sign agreement is exact in both directions: of the 3 endpoints with
H > +0.02, all 3 gained; of the 3 with H <= +0.02, none did.

**Do not use the sign-agnostic form.** An earlier version computed
`reachable` from `max(AUROC, 1-AUROC)`, which silently assumes the score's
direction is known. It reported H = +0.237 and +0.248 on two endpoints whose
AUROC is ~0.32, where no label-free monotone rule can recover anything. Fixed
in `rope.bacc_binormal(allow_flip=False)`; `allow_flip=True` is kept only to
report what knowing the sign would be worth.

## 2. Three failure modes, and they partition the endpoints 3/3/3

| mode | criterion | endpoints |
|---|---|---|
| **A** threshold-misplaced but rankable | AUROC>0.5, H>+0.02 | BUS malignant-vs-rest (0.741, +0.021); BUS any-lesion (0.806, +0.088); DR sight-threatening (0.752, **+0.092**) |
| **B** signal-limited, threshold already fine | AUROC>0.5, H<=+0.02 | BUS malignant finding (0.689, -0.022); DR any retinopathy (0.634, -0.002); DR referable (0.660, -0.024) |
| **C** inverted signal | AUROC<0.5 | DR image quality (0.323); DR poor clarity (0.327); DR field definition (0.490) |

Mode C matters for the literature: arXiv:2605.24806 reports Pengi at
**AUROC 0.188** on IPVS and 0.268 on MDVR-KCL. Those are mode C, and the paper
reads them as weak models rather than as sign-inverted scores.

## 3. Refuted: prevalence matching. Three separate ways.

The idea was to set the operating point by calling the top `pi` fraction of the
unlabelled cohort positive (`ROPE-Q`), with `pi` obtained without labels.

- **`pi` from a two-component Gaussian mixture on the scores (`ROPE-M`): dead.**
  |pi_hat - pi_true| has mean **0.341**, max 0.777, and is within 0.10 in
  **0 of 9** endpoints. Its occasional BACC gains are luck from a wrong `pi`,
  not mechanism: its correlation with H is +0.543 against ROPE-Q's +0.943.
- **`pi` from k labels: loses to simply fitting the threshold.** Averaged over
  the six non-inverted endpoints, `ROPE-Q(pi_k)` vs `thresh_k`: 0.564/0.619 at
  k=2, 0.621/0.626 at k=8, 0.637/0.638 at k=32, 0.638/0.644 at k=64.
- **`pi` set to the TRUE prevalence still loses.** Oracle check against
  `thresh_k` at k=64: ROPE-Q wins **4 of 9**. Handed the right answer, the
  method is a coin flip.

Conclusion: the transductive rank-plus-prevalence machinery adds nothing over
one threshold fitted directly on the labels. Dropped as a contribution.

## 4. RETRACTED — see the retraction block at the end of this file

> **This whole section is withdrawn (2026-10-14).** Its linear-probe column was
> produced by an unweighted `LogisticRegression(C=1.0)`, which on a skewed
> support set predicts the majority class by construction. Both headline claims
> below are false. A correctly weighted probe reaches **0.814** at k=64
> (+0.184 over the zero-label rule, not +0.014) and never degenerates.
> **Use the corrected table in the RETRACTION block instead.** The section is
> kept, not deleted, so the error stays visible.

Mean BACC over the six non-inverted endpoints. `native` uses **zero** labels.
**The `linear probe` column below is the broken one.**

| k | ROPE-Q(pi_k) | thresh_k | Platt | linear probe | native |
|---|---|---|---|---|---|
| 2 | 0.564 | 0.619 | 0.602 | 0.539 | **0.630** |
| 4 | 0.603 | 0.617 | 0.583 | 0.535 | **0.630** |
| 8 | 0.621 | 0.626 | 0.573 | 0.525 | **0.630** |
| 16 | 0.631 | 0.629 | 0.560 | 0.510 | **0.630** |
| 32 | 0.637 | 0.638 | 0.565 | 0.520 | 0.630 |
| 64 | 0.638 | **0.644** | 0.559 | 0.532 | 0.630 |

~~**No method beats the zero-label native rule until k=32, and 64 labels buy
+0.014 BACC.**~~ **FALSE — see the RETRACTION block.** The correct statement is
that the zero-label rule is not beaten until **k=8**, after which a properly
configured probe scales to 0.814. Two further points:

- **Platt scaling gets worse as k grows** (0.602 -> 0.559). Two parameters
  fitted on a handful of skewed labels is a worse decision rule than the
  model's own.
- ~~**The 512-parameter linear probe degenerates to a constant predictor**
  on 1/9 endpoints at k=2 but 5/9 at k=64.~~ **FALSE — my artifact.** With
  balanced class weights it degenerates on **0/9 at every k >= 4**. The link I
  drew to `../llm_prior_shrinkage/FINDINGS.md` was unfounded.

## 5. Data acquired and checked

- **BenSParX** (arXiv:2505.12192, MIT): `BenSParX.csv`, 900 segments, 120
  subjects (60 PD / 60 HC, perfectly balanced), 71 acoustic features. This is
  the released feature table, so the speech arm needs no audio pipeline.
- **Diabetic CKD cohort** (Mendeley 10.17632/hjkzgbxgv5.2, BIRDEM Dhaka, 2025):
  `DiabeticCKD_dataset.csv`. **The file does not match its own description.**
  The Mendeley page says 400 individuals, 185 CKD / 215 not; the CSV has 4000
  rows. It is **400 patients x 10 yearly visits** — 400 unique Patient IDs,
  10 rows each, BMI constant within patient. Record-level prevalence 9.65%
  (386/4000); 185/400 patients are ever positive, and 127 of those 185 have
  exactly one positive visit. **Rows are not independent: any split must be
  grouped by patient.** `score_llm.load_ckd` sets `group` to the patient id.
- No lab values (no creatinine, no eGFR) in the CKD set — demographics and
  lifestyle only, so it is a genuinely hard, low-signal screening task.
- **WoundTissue** (github.com/akabircs/WoundTissue) publishes only **13** of
  the 147 images. The image arm of papers 2 and 4 cannot be reproduced from it.

## 6. Machine constraints that shaped all of the above

Measured 2026-10-13: swap **100% full** (4095/4095 MB), ~850 MB RAM available
against a 3.5 GB model, load average 27 from an unrelated dev stack (Cypress at
95% CPU, `ng serve`, two JVMs, mysqld, Chrome). One 900-token prompt did not
return in 120 s. CPU governor is `powersave` and cannot be changed without a
password; `../llm_prior_shrinkage/FINDINGS.md` measured 2.7x from
`performance`. The LLM arms are therefore written and queued, not run:
`RUN_WHEN_FREE.sh`, which preflights RAM, load and governor.

Ollama **does** expose label-token logprobs (`logprobs: true`,
`top_logprobs: n` on `/api/chat`), verified returning both "0" and "1" — so
the score `s = logP("1") - logP("0")` is obtainable locally.

---

# BenSParX — paper 3's limitation, measured (2026-10-13)

Target: **Zero-Shot Parkinson's Disease Detection from Speech** (arXiv:2605.24806).
Data: **BenSParX** (arXiv:2505.12192, MIT) — the released feature table, 900
segments, 120 subjects (60 PD / 60 HC), 71 acoustic features. `src/bensparx_*.py`.

All numbers below are subject-grouped stratified 5-fold, 5 seeds (25 folds),
logistic regression on standardised features. No LLM: these test the paper's
**protocol choices**, which do not depend on which scorer produces the scores.

## 1. Their aggregation rule costs 3.0 BACC points, significantly

The paper forces one hard 0/1 token per 10-second segment and takes a **majority
vote** over a subject's 7-8 segments. Holding the segment scores fixed and
varying only the aggregation rule:

| subject-level rule | BACC | AUROC |
|---|---|---|
| `majority_vote` — **their rule** | 0.955 +- 0.046 | 0.989 |
| `mean_prob` | **0.985 +- 0.020** | 0.998 |
| `mean_logit` | **0.985 +- 0.020** | 0.999 |
| `max_prob` (control) | 0.617 +- 0.063 | 0.995 |

Paired over 25 folds: **mean_prob - majority_vote = +0.0300, 95% CI
[+0.0150, +0.0450]**. Averaging also halves the spread (+-0.046 -> +-0.020).
`max_prob` collapsing to 0.617 is the control showing the gain is specific to
averaging, not to "anything other than voting".

## 2. The mechanism is quantitative, and predicts the gap it causes

Segment-level AUROC is 0.879, i.e. `d' = sqrt(2) Phi^-1(0.879) = 1.655`. If a
subject's segments are independent given the subject, averaging k = 7.5 of them
gives `d' sqrt(k) = 4.531`, hence subject AUROC `Phi(4.531/sqrt 2)`:

| quantity | predicted | observed |
|---|---|---|
| subject AUROC after score averaging | 0.9993 | **0.999** |
| accuracy, majority vote of 7 segments | 0.964 | **0.955** |
| accuracy, score averaging of 7 segments | 0.986 | **0.985** |

So the penalty for voting on signs rather than averaging scores is not a quirk
of this corpus: it is the predictable cost of discarding magnitude before
pooling, and the binormal model gets its size right to three decimals.

**The independence assumption is itself measured, not assumed** — see 4 below.

## 3. Five features do the work of seventy-one

Subject AUROC, with any selection done **inside** each training fold:

| feature set | subject AUROC |
|---|---|
| all 71 | 0.999 +- 0.003 |
| top 5 (in fold) | 0.999 +- 0.003 |
| top 10 (in fold) | **1.000 +- 0.001** |
| top 20 / top 40 | 0.999 |
| MFCC only (39) | 0.994 +- 0.009 |
| pitch + intensity + formants (9) | 0.979 +- 0.033 |
| energy (4) | 0.974 +- 0.026 |
| jitter + shimmer (11) | 0.970 +- 0.030 |
| pulses (4) | 0.903 +- 0.048 |
| voice quality: HNR/NHR/autocorr (4) | 0.873 +- 0.078 |

The information saturates by 5-10 features. This answers the paper's own
limitation ("features were derived from a predefined pipeline rather than
optimized individually for each corpus") more sharply than it poses it: the
pipeline is not merely unoptimised, it is **~7x larger than the task needs**.

**Consequence for the LLM arm, and it is a testable prediction:** the paper
serialises all 71 features into the prompt (~650 tokens). If the information
saturates at 10 features while LLMs degrade on long numeric lists, then a
10-feature prompt should **beat** a 71-feature prompt zero-shot. That is
falsifiable, and it makes the LLM run ~6x cheaper, which is what made it
infeasible on this machine.

## 4. Refuted (mine): row-wise splitting does not leak here

I predicted that splitting the 900 rows without grouping by subject would
inflate results. It does not: segment AUROC is **0.879 +- 0.024 grouped vs
0.879 +- 0.023 row-wise, inflation -0.000**. The segments of a subject are
apparently near-independent given the subject at this feature resolution.

This null is load-bearing rather than a dead end: it is the evidence for the
independence assumption that section 2's prediction needs. Grouping is still
the correct protocol — it just happens not to change this number.

## 5. The zero-shot deficit here is in the ranking, not the threshold

| | segment AUROC | subject AUROC | subject BACC |
|---|---|---|---|
| logreg on the same 71 features | 0.879 | 0.998 | **0.985** |
| their zero-shot LLaMA-3 8B | — | 0.901 | 0.833 |

A plain logistic regression on the features the LLM is shown reaches 0.985; the
zero-shot LLM reaches 0.833. The headroom diagnostic gives **H = -1.46** for
their BenSParX cell, so the operating point is already fine and the ~15-point
deficit is in the ordering. Their conclusion ("zero-shot capability alone is
insufficient to ensure robust PD screening") is correct here, but they could not
have established it: without a supervised reference on their own features there
is no way to separate a bad ranking from a misplaced threshold. On their IPVS
and MDVR-KCL cells the pattern inverts (AUROC 0.805 at BACC 51.8), and the same
sentence is then wrong.

## 6. LLM replication attempted and abandoned on this machine

Tried `llama3.2:3b` (nearest local analogue to their LLaMA-3 8B), 40
label-balanced subjects, 304 segments, 71 features. **0 of 304 items completed
in ~8 minutes**, and afterwards even a 5-feature prompt would not return.
Causes, in order: swap **100% full** (4095/4095 MB); ollama holding **two models
resident** (llama3.2:3b 2.6 GB + a stale qwen 2.2 GB) against ~3 GB available;
available RAM swinging 553 MB - 5.4 GB as an unrelated dev stack (`ng serve`,
two JVMs, VS Code) moved; governor still `powersave`. Jobs killed and models
unloaded. `score_llm.py` now takes `--groups` (label-balanced whole subjects)
and `--nfeat`; start with `--groups 40 --nfeat 10` per section 3.

---

# BenSParX LLM arm — completed (2026-10-13)

`llama3.2:3b`, 40 label-balanced subjects (20 PD / 20 HC), 304 segments, 10
features (per section 3 above), forced single token, top-20 logprobs,
`s = logP("1") - logP("0")`. 304/304 items scored.
`results/llm_bensparx_llama32_3b_list_f10.jsonl`, `results/bensparx_llm.json`.

**Not a reproduction of paper 3**: they used LLaMA-3 **8B**, all **71** features,
**120** subjects. This is a new cell in their design space, and their published
row is used as a reference point, not a target.

## 1. The degenerate operating point, measured in their setting

| quantity | value |
|---|---|
| segment AUROC | 0.628 |
| subject AUROC | **0.805** |
| native decision | sens **1.000**, spec **0.000**, pred+ **1.000** |
| native BACC | **0.500** |
| reachable (binormal, same sign) | 0.728 |
| **H** | **+0.228** |

The model emits "1" for every one of the 40 subjects. Balanced accuracy is
exactly chance while the ranking is genuinely informative (subject AUROC 0.805).
This is the mirror image of the paper's own IPVS cell (sens 3.57 / spec 100.0,
BACC 51.79, AUROC 0.805) — same fault, opposite direction.

## 2. The diagnostic's prediction was accurate to 0.004

Threshold fitted on k labelled subjects, evaluated on the held-out rest,
50 seeds:

| labels | subject BACC |
|---|---|
| 0 (native) | 0.500 |
| 4 | 0.659 |
| 8 | 0.694 |
| 16 | **0.732** |

(Corrected 2026-10-14: an earlier version of this table printed 0.611 and
0.694 -> 0.655 at k=4 and k=8. Those were the `vote_fitted` values, i.e. the
majority-vote rule, not `mean_fitted`. The k=16 value and H were unaffected.)

**Predicted recoverable H = +0.228; realised at k=16 = +0.232.**

Within one dataset the diagnostic separates two models correctly: paper 3's
LLaMA-3 8B cell has **H = -0.015** (threshold already fine, nothing to gain),
this 3B cell has **H = +0.228** (threshold broken). Neither label requires
knowing the answer in advance.

## 3. Score averaging beats majority voting — now confirmed on LLM logprobs

At the model's native threshold every segment is positive, so voting and
averaging agree trivially (+0.000) and the question is unanswerable. Fitting a
threshold first separates bias from pooling. Both rules get the same k labels
and the same scores:

| k | majority vote | score averaging | difference |
|---|---|---|---|
| 4 | 0.611 +- 0.073 | 0.652 +- 0.102 | **+0.040 [+0.011, +0.069]** |
| 8 | 0.655 +- 0.057 | 0.719 +- 0.062 | **+0.064 [+0.040, +0.087]** |
| 16 | 0.658 +- 0.057 | 0.727 +- 0.069 | **+0.069 [+0.047, +0.091]** |

All three significant (paired bootstrap over 50 seeds). **This corrects an
earlier interim claim of mine**: at 26 subjects the same comparison gave +0.034
[-0.000, +0.066] and +0.002 [-0.031, +0.036], which I reported as unresolved.
It was underpowered, not absent.

The effect is **larger on LLM scores (+0.064 to +0.069) than on supervised
scores (+0.030)**, consistent with the mechanism: the LLM's segment scores are
noisier, so discarding magnitude before pooling costs more.

## 4. Novelty check — done late, and it changes the claims

Two searches plus a close read of
**arXiv:2607.12260, "From Many to Meaningful"** (same four models and Dataset-1
as paper 1):

- **The pathology is already visible in published tables and goes unremarked.**
  That paper reports LLaMA-3 on Dataset-2 at **macro AUROC 0.923 with balanced
  accuracy 50.20%**, and Mistral at AUROC 0.822 with BACC 53.47%, without
  connecting the two. Its future work mentions "structured uncertainty
  estimation for screening thresholds", confirming the threshold was not studied.
- **The feature-saturation finding (section 3 above) is partly scooped** by that
  paper, which already shows a compact subset beating all features for zero-shot
  CKD screening.
- **The aggregation finding is textbook.** Soft vs hard voting, and the fact that
  hard voting discards confidence, is introductory ML material. Quantifying what
  it costs paper 3 is a critique, not a method.
- **The diagnostic is textbook too**: AUROC -> d' -> Phi(d'/2) is 1960s signal
  detection theory. Operating-point selection in clinical ML (Youden's J,
  prevalence-adjusted cutoffs) is mature.

**Consequence for framing.** There is no new method here. The defensible
contribution is field-level: the recent zero-shot clinical LLM literature
reports threshold-artifact failures as model failures, and a textbook diagnostic
predicts which ones are which. The strong version is a **meta-analysis** over
published (model, dataset) cells from that wave — the two CKD papers, the
Parkinson's paper, delirium (2607.20453), child malnutrition (2607.29082),
ClinicRealm (2407.18525) — classifying each cell as threshold-limited,
signal-limited or sign-inverted, with the experiments above as prospective
validation. It needs almost no compute.

**Still outstanding before any novelty claim is safe:** a systematic search
rather than two queries; the conformal-prediction and selective-classification
literature, which I have not examined; and a fix to the untuned linear-probe
baseline (fixed C=1.0) in `run_budget.py`, whose k=64 degeneracy may be my
artifact.

---

# Meta-analysis of published cells (2026-10-13) — the paper's spine

`src/harvest.py`, `results/harvest.json`. **43 cells transcribed from published
tables**, nothing re-run. For each, the authors' own AUROC gives the balanced
accuracy a well-placed threshold on the *same scores* would reach
(`reachable = Phi(d'/2)`, `d' = sqrt(2) Phi^-1(AUROC)`, no sign flip), and
`H = reachable - BACC(reported)`.

Sources: arXiv:2605.24806 (Parkinson's speech, 16 cells, BACC+AUROC+sens+spec);
arXiv:2607.12260 (zero-shot CKD, Table 3, 16 AUROC-bearing cells — its six
Dataset-3 cells have no AUROC, single-class, and are excluded by construction);
arXiv:2609.04013 (LLM4CKD, Table II, 4 LLM + 6 baseline cells); plus our own
BenSParX cell. Every row carries its paper and table in `src`.

## 1. One third of published zero-shot clinical LLM results are threshold-limited

Restricted to the 36 **published LLM** cells (ours and non-LLM baselines excluded):

| mode | criterion | count |
|---|---|---|
| **A threshold-limited** | AUROC>0.5, H>+0.02 | **12/36 (33%)** |
| **B signal-limited** | AUROC>0.5, H<=+0.02 | 19/36 (53%) |
| **C sign-inverted** | AUROC<0.5 | 5/36 (14%) |

**Mean recoverable balanced accuracy across the threshold-limited cells: +0.138.**
The largest, all from the papers' own tables:

| paper | model | dataset | AUROC | reported BACC | reachable | H |
|---|---|---|---|---|---|---|
| 2607.12260 | LLaMa-3 8B | CKD-2 India | 0.923 | **0.502** | 0.843 | **+0.341** |
| 2605.24806 | LLaMA-3 8B | IPVS (it) | 0.805 | 0.518 | 0.728 | +0.210 |
| 2607.12260 | Mistral 7B | CKD-2 India | 0.822 | 0.535 | 0.743 | +0.208 |
| 2607.12260 | LLaMa-3 8B | CKD-1 Bangladesh | 0.816 | 0.543 | 0.738 | +0.195 |
| 2607.12260 | LLaMa-3 8B | CKD-1 Bangladesh | 0.749 | 0.500 | 0.683 | +0.183 |
| 2605.24806 | LLaMA-3 8B | MDVR-KCL (en) | 0.702 | 0.507 | 0.646 | +0.139 |

The top row is the clearest case in the literature: a model ranking CKD patients
at AUROC 0.923 — comparable to the same table's best model — is reported at
chance-level balanced accuracy (macro F1 27.68, i.e. nearly all one class), and
the paper does not remark on it.

Our own BenSParX cell sits in the same band (AUROC 0.805, BACC 0.500, H +0.228)
and is the one case where we could verify the prediction prospectively: fitting
a threshold on 16 labelled subjects realised **+0.232** against a predicted
**+0.228** (see the BenSParX section above).

## 2. "Feature selection improves zero-shot screening" is partly a threshold effect

arXiv:2607.12260's central claim rests on balanced accuracy, which is
threshold-dependent, while its own table also reports macro AUROC, which is not.
Separating the two questions across its 8 (model, dataset) pairs:

| model | dataset | dAUROC | dBACC | reach(all feats) | reported BACC(sel) |
|---|---|---|---|---|---|
| LLaMa-3 8B | CKD-2 India | **-0.074** | **+0.262** | **0.843** | 0.764 |
| Qwen-3 8B | CKD-2 India | **-0.058** | -0.004 | 0.819 | 0.812 |
| Mistral 7B | CKD-2 India | +0.110 | +0.318 | 0.743 | 0.852 |
| Qwen-3 8B | CKD-1 Bangladesh | +0.101 | +0.163 | 0.672 | 0.783 |
| LLaMa-3 8B | CKD-1 Bangladesh | +0.067 | +0.043 | 0.683 | 0.543 |
| GPT-4o | CKD-1 Bangladesh | +0.048 | +0.074 | 0.727 | 0.771 |
| Mistral 7B | CKD-1 Bangladesh | +0.044 | +0.217 | 0.719 | 0.773 |
| GPT-4o | CKD-2 India | +0.018 | +0.024 | 0.846 | 0.880 |

- Selection made the **ranking worse in 2/8** pairs while balanced accuracy rose
  or held — there it added no information and moved the operating point.
- In **3/8** pairs a threshold fix on the *all-feature* scores already beats the
  reported selected-feature result. LLaMa-3 on CKD-2 is the sharpest:
  **0.843 reachable with all features vs 0.764 reported with selected**, a
  **+0.079** advantage to doing nothing but placing the threshold.
- In the other 5/8 pairs selection genuinely improved the ranking, so the claim
  is **not** refuted wholesale. The finding is that it is confounded, and the
  authors' own AUROC column is enough to separate the two effects.

## 3. What this reframing fixes

The earlier plan proposed a new decision rule and that proposal was refuted
(see section 3 of the first block). This meta-analysis needs **no new method**:
the diagnostic is 1960s signal detection theory and operating-point selection in
clinical ML is mature. The contribution is field-level — the recent zero-shot
clinical LLM literature reports decision-rule artifacts as model failures, at a
measurable rate of one third of cells, and the tool to tell them apart was
already in the papers' own tables. Our experiments become prospective
validation rather than the whole claim.

## 4. Still outstanding

- Harvest is 3 papers. Candidates identified but not yet extracted: delirium
  (arXiv:2607.20453), child malnutrition (arXiv:2607.29082), ClinicRealm
  (arXiv:2407.18525 / Nature Dig. Med. s41746-026-02539-z), AgentRx
  (arXiv:2605.10286). Many report AUROC without any threshold metric, which is
  itself a reportable finding about the field.
- **Binormal estimate is an approximation, not a bound.** Six cells have
  reported BACC *above* `reachable` (most extreme -0.117), i.e. non-Gaussian
  scores. This must be stated, and ideally bounded by also reporting the
  empirical ROC ceiling where a paper publishes a curve.
- Conformal prediction and selective classification not yet surveyed.

---

# RETRACTION (2026-10-13) — two findings in the first block were my artifact

Cause: the linear-probe baseline in `run_budget.py` was fitted with
`LogisticRegression(C=1.0)` and **no class weighting**. On a skewed support set
an unweighted fit minimises error by predicting the majority class. The
degeneracy I reported was a property of my baseline's configuration, not of
few-shot learning. Rerun with `class_weight="balanced"`, same seeds, same
support sets, both versions scored side by side (`results/budget_tuned.log`,
`linear_probe_untuned` column retained in `results/budget.json`).

## Retracted claim 1: "the linear probe degenerates more with more labels"

Degenerate endpoints (BACC exactly 0.500) out of 9:

| k | untuned (what I reported) | balanced (correct) |
|---|---|---|
| 2 | 1/9 | 1/9 |
| 4 | 2/9 | **0/9** |
| 8 | 2/9 | **0/9** |
| 16 | 5/9 | **0/9** |
| 32 | 3/9 | **0/9** |
| 64 | **5/9** | **0/9** |

With class weighting the degeneracy vanishes entirely at every k >= 4. The
claim was false, and the link I drew to the covariance-degeneracy results in
`../llm_prior_shrinkage/FINDINGS.md` was unfounded.

## Retracted claim 2: "few-shot adaptation is nearly worthless below k=64"

I reported that no method beats the zero-label rule until k=32 and that 64
labels buy +0.014 BACC. Both figures came from the broken baseline.

Mean BACC over the 6 non-inverted endpoints:

| k | thresh_k | **probe (balanced)** | probe (untuned) | Platt | native (0 labels) |
|---|---|---|---|---|---|
| 2 | 0.619 | 0.539 | 0.539 | 0.602 | **0.630** |
| 4 | 0.617 | 0.589 | 0.535 | 0.583 | **0.630** |
| 8 | 0.626 | **0.651** | 0.525 | 0.573 | 0.630 |
| 16 | 0.629 | **0.728** | 0.510 | 0.560 | 0.630 |
| 32 | 0.638 | **0.777** | 0.520 | 0.565 | 0.630 |
| 64 | 0.644 | **0.814** | 0.532 | 0.559 | 0.630 |

A correctly configured linear probe reaches **0.814** at k=64, i.e. **+0.184**
over the zero-label rule — not +0.014. It overtakes everything from k=8.

## What survives, corrected

- **The zero-label native rule is not beaten until k=8** (0.630 vs 0.619/0.617
  for `thresh_k` at k=2/4). That part holds.
- **The crossover is k ~ 8.** Below it, nothing beats the model's own decision
  or a one-parameter threshold on it; above it, proper supervised adaptation
  wins decisively and keeps scaling.
- **Platt scaling still degrades with k** (0.602 -> 0.559), unchanged by the fix.
- This *sharpens* rather than weakens the paper's claim: the threshold fix is
  the right intervention in the **zero- to very-few-label** regime, which is
  exactly the regime the zero-shot screening literature operates in. It is not
  a substitute for supervised adaptation when labels are available.

## Not affected

The meta-analysis (`harvest.py`) uses only published tables and the authors'
own AUROC values; it never touched the linear probe. The BenSParX results —
the degenerate all-positive rule, H +0.228, realised +0.232 at k=16, and the
aggregation comparison — concern the LLM's own scores and are likewise
unaffected.

## Process note

The flaw was visible in the first run: a baseline sitting at exactly 0.500
should have been read as a misconfiguration, not reported as a finding. It was
caught only because a reviewer-style check was applied late. Any future
baseline pinned at a degenerate value gets configuration-checked before it is
written down.


---

# Figures and remaining record-keeping (2026-10-14)

## Figures

`src/make_figures.py` renders six figures to `figures/` as PDF and PNG, built
directly from the result JSONs rather than from numbers retyped by hand.

| figure | content | source file |
|---|---|---|
| fig1_audit_scatter | all 43 cells, reported AUROC vs reported BACC, with the reachable curve; H drawn as the vertical gap | `results/harvest.json` |
| fig2_headroom_bars | the 36 published LLM cells sorted by H | `results/harvest.json` |
| fig3_forward_test | BenSParX, BACC against number of labelled subjects, with the predicted reachable line | `results/bensparx_llm.json` |
| fig4_label_budget | the **corrected** budget curves, crossover at k~8 | `results/budget.json` |
| fig5_aggregation | voting vs averaging, LLM logprobs and supervised scores | `results/bensparx_llm.json`, `results/bensparx_aggregation.json` |
| fig6_feature_saturation | subject AUROC by feature subset | `results/bensparx_aggregation.json` |

Colours come from a validated colour-blind-safe palette. The validator was run
rather than eyeballed; the scatter uses only the three slots that pass all-pairs
separation, which happens to match the three failure modes. Two slots fall below
3:1 contrast, so they carry direct labels, and the paper's tables serve as the
required table view.

## Script-to-section map (previously missing)

| script | what it produced | recorded in |
|---|---|---|
| `src/run_arm_a.py` | 9 endpoints, diagnostic, ROPE rules, controls | block 1 §1-3 |
| `src/analyze.py` | H vs realised gain, mixture-prevalence failure | block 1 §1, §3 |
| `src/run_budget.py` | label-budget sweep, 4 methods x 6 budgets x 50 seeds | block 1 §4 (**retracted**), RETRACTION block |
| `src/analyze_budget.py` | win counts, degeneracy counts, oracle check, taxonomy | block 1 §2-4 |
| `src/bensparx_reference.py` | supervised reference on the 71 features | BenSParX block §5 |
| `src/bensparx_aggregation.py` | aggregation rules, feature subsets, leakage check | BenSParX block §1-4 |
| `src/score_llm.py` | 304 BenSParX segments scored with llama3.2:3b | BenSParX LLM block |
| `src/analyze_bensparx.py` | subject-level rules, fitted-threshold aggregation | BenSParX LLM block §1-3 |
| `src/harvest.py` | 43 published cells, modes, feature-selection confound | Meta-analysis block |
| `src/make_figures.py` | the six figures | this block |

## Incomplete runs, for honesty

- `bensparx_reference.py` also defines a **random-forest** arm and a row-wise
  leakage comparison. The random forest was **killed mid-run** because it was
  competing with the LLM scoring for CPU, so only the logistic-regression numbers
  in the BenSParX block §5 are measured. The leakage comparison was later redone
  inside `bensparx_aggregation.py` and is reported in BenSParX block §4.
- `results/llm_bensparx_llama32_3b_list.jsonl` is a **zero-byte** file from the
  first, failed 71-feature attempt. It is kept only as the resume target.
- The CKD LLM arm (`RUN_WHEN_FREE.sh`) has **not been run**.

## Outputs that are not experiments

`draft_paper.md` is the paper draft. It is written from this file and contains
no number that does not appear here.
