# Plan — the shared limitation in five 2025/26 clinical-AI papers, and one method that attacks it

## The five papers

| # | paper | what it does | stated limitations |
|---|---|---|---|
| 1 | **LLM4CKD**, arXiv:2609.04013 | zero-/few-shot LLM CKD screening on serialized tabular features (Gemma-2-9B, Llama-3-8B, Qwen-3-8B, Mistral-7B, GPT-4o-mini) vs ML/DL/TabPFN and 5 clinical tools | small primary cohort (284, Mirzapur); "LLM reliability depends on prompt design, feature serialization, model version"; feature subset inherited from prior ML work and "may not be optimal"; cross-dataset results "only preliminary evidence"; "we did not directly evaluate robustness to patient-level missing features" |
| 2 | **WoundFormer**, arXiv:2605.19868 | SegFormer MiT-B5 encoder + spatially-preserving multi-scale aggregation decoder, 6-class wound tissue segmentation | **no limitations section**. Only "larger multi-center validation" in the conclusion |
| 3 | **Zero-shot PD from speech**, arXiv:2605.24806 | 71 handcrafted acoustic features -> LLaMA-3 vs raw waveform -> Qwen2-Audio / Pengi / Audio-Reasoner, 4 languages | zero-shot only, no fine-tuning; "dataset sizes remain modest"; features from "a predefined pipeline rather than optimized individually for each corpus"; not clinically validated |
| 4 | **Wound tissue segmentation benchmark**, arXiv:2502.10652 | the WoundTissue dataset (147 images, 6 tissues, 3 labelling formats) + 82 derived segmentation models; FPN+VGG16 best at 82.25 Dice | field-level only: no public labelled data, no standardised evaluation. No study-specific limitations |
| 5 | **BeliN**, NLP Journal 11 (2025) 100138 / arXiv:2501.01069 | BeliN Bengali religious-news corpus + MultiGen: article [SEP] category [SEP] aspect [SEP] sentiment -> BanglaT5/mBART/mT5/mT0 | annotated-data scarcity; "hardware limitations also restricted fine-tuning large-scale generative models"; wider domain coverage |

## Limitations the papers do not state, found in their own numbers

- **(1) LLM4CKD: performance is non-monotone in the number of labels.** Qwen-3 balanced
  accuracy 0.814 at 4 shots falls to 0.764 at 32, while TabPFN rises 0.658 -> 0.820 and
  overtakes it. The authors note "their performance does not consistently improve as the
  number of shots increases" but treat it as a property of LLMs. It is a property of the
  *mechanism*: few-shot prompting is an uncontrolled mixture of the model's prior and the
  support set, with no coefficient anyone can set or fit.
- **(3) PD paper: the ranking is good and the decision is broken.** LLaMA-3 on IPVS scores
  sensitivity 3.57 / specificity 100.0 / BACC 51.79 with **AUROC 0.805**. On MDVR-KCL:
  sens 6.25 / spec 95.24 / BACC 50.74 with AUROC 0.702. Pengi on IPVS has AUROC **0.188** --
  strongly *anti*-correlated, i.e. real signal with an inverted sign. The paper's conclusion,
  "zero-shot capability alone is insufficient to ensure robust PD screening", does not follow
  from these numbers: what is insufficient is the hard one-token decision rule.
- **(2,4) The wound papers: the claimed margin is smaller than the visible variance.**
  147 images split 118/14/**15**, single split, single seed, no confidence intervals.
  WoundFormer claims "up to 4.3 Dice points" over baselines, but its own augmentation ablation
  moves Maceration 81.0 -> 92.2 (+11.2) and the average 81.9 -> 85.5 (+3.6). The worst classes
  (Bone 71.7, Tendon 75.8; Fibrin 69.3 on DFUTissue) are the rarest ones. Paper 4 then selects
  among **82 models** by weighted Dice on the same tiny split -- selection on 15 images.
- **(5) BeliN: possible gold-feature leakage.** Category, aspect and sentiment are *manually
  annotated* corpus fields. The paper never says whether they are predicted by a classifier at
  inference or taken from the gold annotation at test time. If gold, the +2.53 BLEU
  (16.08 -> 18.61) is not obtainable on an unseen article.

## The convergent limitation (4 of 5 papers)

> These systems are scored with **threshold-dependent metrics on a single tiny split**, with
> **no fitted operating point** and **no mechanism that trades the model's prior against the
> few labels actually available**. Discriminative signal is reported as a modelling failure
> when it is a decision-rule failure.

Three coupled defects: (a) no tunable decision point -- a token output "0"/"1" has none;
(b) non-monotone shot scaling -- nothing mediates prior vs support; (c) no reliability
estimate at n = 15 images / 37-120 subjects / 284 patients.

This repo has already measured the same pathology independently: in
`papers/llm_prior_shrinkage/FINDINGS.md`, `zero_shot` on MedIMeta `bus` scores accuracy
0.647, *below* the 0.718 majority baseline, while its AUROC is 0.689 -- "genuine signal,
badly placed threshold" -- and `zero_shot` is last on QWK at every k on `dr_regular` while
top-three on accuracy.

## How much is recoverable -- computed from the PD paper's own table

Binormal estimate of the balanced accuracy available at the *optimal* threshold for a given
AUROC: `d' = sqrt(2) * Phi^-1(max(AUROC, 1-AUROC))`, `BACC* = Phi(d'/2)`.

| dataset | model | AUROC | BACC reported | BACC* achievable | headroom |
|---|---|---|---|---|---|
| BenSParX | LLaMA-3 | 0.901 | 83.33 | 81.87 | -1.46 |
| MDVR-KCL | LLaMA-3 | 0.702 | 50.74 | 64.61 | **+13.87** |
| IPVS | LLaMA-3 | 0.805 | 51.79 | 72.84 | **+21.05** |
| NeuroVoz | LLaMA-3 | 0.486 | 52.58 | 50.99 | -1.59 |
| NeuroVoz | Audio-Reasoner | 0.676 | 63.90 | 62.66 | -1.24 |
| MDVR-KCL | Audio-Reasoner | 0.609 | 69.49 | 57.76 | -11.73 |
| IPVS | Pengi | 0.188 | 32.63 | 73.43 | **+40.80** |
| MDVR-KCL | Pengi | 0.268 | 29.76 | 66.92 | **+37.16** |

The diagnostic separates cleanly, which is what makes it a hypothesis rather than a hope:
rows at or above the ceiling (BenSParX, NeuroVoz, Audio-Reasoner/MDVR) have a threshold that
is already fine and are limited by signal -- no method can help them. Rows with +14 to +41
headroom are threshold artifacts and nothing else. Negative headroom (-11.73) means the
reported operating point is *better* than binormal, i.e. the score distribution is not
Gaussian there; treat BACC* as an estimate, not a bound.

## Proposed methodology — PACS: Prior-Anchored Calibrated Shrinkage

Closed-form, no gradients, CPU-only, four steps. Each step is tied to one defect above.

**S1. Replace the token decision with a score.** For each subject, `s(x)` = normalised
log-prob ratio over the "0"/"1" label tokens, averaged over `T` prompt variants
(serialisation style x instruction/chat x feature-order shuffle). LLM4CKD already extracts
these log-probs for Brier/ECE; the PD paper does not and that is why its operating point is
unfittable. The prompt ensemble answers LLM4CKD's own "depends on prompt design" limitation --
and `FINDINGS.md` already measured that a 5-variant majority vote turns 3/5-shuffle-unstable
rank elicitation into 3/3 exact.

**S2. Fit a 1-D monotone calibration map on the k labels you already have.**
`g: s -> p_hat`, two-parameter Platt (`sigma(a*s + b)`) at small k, isotonic once k is large
enough. This provably cannot change AUROC, which is the point: it isolates defect (a) and
makes the claim falsifiable. If the papers' failures are threshold failures, two parameters
recover most of the headroom column; if not, nothing moves.

**S3. Shrink the map toward the zero-shot prior, with lambda chosen analytically.**

    g_lambda = argmin_g  sum_i loss(g(s_i), y_i) + lambda * ||g - g_0||^2

`g_0` = identity (keep the zero-shot decision). `lambda` by **analytic leave-one-out**, one
extra hat-matrix diagonal, the same route already costed in `llm_prior_shrinkage/PLAN.md`.
`lambda -> inf` at k=0 recovers pure zero-shot; `lambda -> 0` as k grows recovers plain
calibration. This is the fix for defect (b): the prior/data weight becomes an estimated
quantity instead of an emergent property of prompt concatenation, so BACC is monotone in k up
to LOO error -- the property LLM4CKD empirically lacks. It strictly generalises both endpoints,
the same design discipline as the TCLA Phase-B proposal in this repo.

**S4. Place the threshold at the deployment prevalence, not the support prevalence.**
Balanced k-shot support against a skewed test set is the exact mismatch that generates
defect (a) -- already quantified in `FINDINGS.md`. Choose `tau` to maximise expected BACC under
the stated deployment prevalence, and report BACC, AUROC, Brier **and** ECE together, as
LLM4CKD does (the best-instrumented of the five) and as papers 2-5 do not.

**S5. Reliability protocol, treated as a contribution rather than boilerplate.**
50 seeds; BCa 95% CIs on *paired* model-vs-model differences; a **shuffled-label control** and
a **majority-class baseline** on every table (repo rule, carried over). Applied to WoundFormer's
15-image test set this is expected to produce a CI wide enough to swallow the +4.3 Dice claim --
a legitimate negative result, and the honest answer to papers 2 and 4.

## Validation plan — one method, three modalities, all public data

| modality | data | source | tests |
|---|---|---|---|
| tabular | UCI CKD (400 records) | public; paper 1's Dataset-2 | S1-S4 vs paper 1's own LLM and TabPFN numbers; does monotonicity in k hold? |
| speech | BenSParX (Bengali, 60/60) | public, Kabir group | S1-S4 vs paper 3's Table; does the IPVS/MDVR headroom materialise? |
| image | WoundTissue (147 img, 6 tissues) | github.com/akabircs/WoundTissue | S5 only: re-estimate papers 2 and 4 with CIs |

Paper 1's Mirzapur cohort is not public, so the tabular arm reproduces against Dataset-2 and
cites their primary-cohort numbers without re-deriving them.

## Predictions, stated before running anything

1. Platt on 8 labels recovers >= 60% of the headroom column for LLaMA-3 on IPVS and MDVR-KCL.
2. It recovers **nothing** on NeuroVoz/LLaMA-3 (AUROC 0.486) and BenSParX (already at ceiling).
3. Pengi's AUROC 0.188 is sign-inverted; a monotone-increasing map cannot exploit it, and
   fitting the *direction* on k<=8 labels is close to a coin flip. Expect high variance, and
   report it rather than restricting the map class after seeing the result.
4. Shot-scaling becomes monotone under S3 where LLM4CKD's is not (0.814@4 -> 0.764@32).
5. WoundFormer's +4.3 Dice does not survive a paired CI on 15 test images.

## Risks / honest accounting

- Calibrating LLM scores is not itself novel. The claim worth defending is narrower:
  **(i)** analytic-LOO prior-shrinkage of the calibration map, giving monotone shot-scaling;
  **(ii)** the cross-modal demonstration that reported zero-shot "failure" in clinical
  screening is predominantly a decision-rule artifact, with the AUROC-headroom diagnostic that
  says in advance which cases are rescuable.
- If prediction 1 fails, the deliverable is the diagnostic plus the reliability re-analysis of
  papers 2-4 -- still a paper, at a workshop rather than an A* venue. Pre-registered, as in
  `llm_prior_shrinkage/PLAN.md`.
- Fitting a threshold on the same k labels used elsewhere risks optimistic bias; LOO and the
  shuffled control are what keep it honest. Both must be in the first results table, not added
  later.
