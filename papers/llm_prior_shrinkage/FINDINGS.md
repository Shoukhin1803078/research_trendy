# Verified findings — 2026-10-08

Only measured results. Everything here was produced by a script in `src/` or `phase0/`
and has its raw numbers in `results/`.

## Hardware / feasibility
- i7-8665U, 8 threads, no GPU. BiomedCLIP ViT-B/16 on CPU: **4.84 img/s** (was 1.81 until
  the cpufreq governor was set to `performance`; it does NOT survive reboot, re-apply before
  long runs). 512-d features. Feature extraction is one-time and cached.
- Grayscale datasets (ultrasound, mammography) are stored (N,H,W) and must be replicated to
  3 channels; RGB ones are (N,H,W,3).

## The local LLM: reliable at rank order, unreliable at magnitude and at detection
Measured three separate times, same pattern each time.

| what we asked | result |
|---|---|
| sort clinical severity scales (Gleason, ICDR, BI-RADS) | **3/3 exact** under 5-shuffle majority vote, Kendall tau=1.0. Single calls are shuffle-sensitive (3/5, 4/5, 5/5) |
| decide *whether* a label set is ordinal | **useless** — 0/3 negative controls under every prompt design tried (v1 simple, v2 complex, v3 decomposed). It answers whatever the prompt framing primes |
| P(malignant \| BI-RADS **mass** descriptor) | **works** — Spearman +0.894 (margins), **+0.913 (shape), p=0.010** vs empirical train-split rates it never saw. SPICULATED 0.80 vs true 0.93; CIRCUMSCRIBED 0.05 vs true 0.13; LYMPH_NODE 0.00 vs true 0.00 |
| P(malignant \| BI-RADS **calcification** descriptor) | **fails** — Spearman −0.269 (type), **−0.707 (distribution)**. Outputs 0.200 for nearly everything; inverts LINEAR (said 0.05, true 0.71) and FINE_LINEAR_BRANCHING (said 0.20, true 0.77) |

Interpretation: common medical vocabulary (spiculated, circumscribed, oval, irregular) is
learned; specialist mammography vocabulary (fine linear branching, amorphous, dystrophic,
eggshell) is not. Consequence: use the LLM rule for masses, empirical rates for calcifications.
Magnitudes are compressed throughout (0.200 for both ILL_DEFINED true=0.68 and OBSCURED
true=0.21) — **ranking is usable, calibration is not.**

## The local generative VLM cannot do the image task
`qwen2.5vl:3b` zero-shot on breast ultrasound malignancy, n=10 (indicative, not a measurement):
accuracy 0.700 = **exactly** the majority baseline, because it predicted benign for **10/10**
at a 30% true malignant rate. **188 seconds per image** on CPU, so the 156-image test set
would take ~8 h. Matches published results (Qwen2.5-VL-3B ~35.6% on medical diagnosis;
Claude 3.5 Sonnet 59% on breast ultrasound).
Conclusion: frozen features for perception, LLM text-only for knowledge, generative VLM for
neither.

## The ordinal line: one strong diagnostic, one dead method

**Diagnostic (holds).** MedIMeta declares `task_target: ORDINAL_REGRESSION` in its per-task
YAML (4 of 5 tasks in `dr_regular`). The official benchmark `medimeta-fsl-benchmark`
@757d7d8, which is the code behind Woerner & Baumgartner arXiv 2408.08058 by the *same
authors*, contains **0 occurrences of "ordinal"** across 15 Python files (control: `AUROC`
appears 10 times). Its only metrics are `Accuracy` and `AUROC`, both order-blind, and it
model-selects on `geometric_average_AUROC`.
Consequence, measured on `dr_regular`/"DR level": **the best method flips depending on the
metric at every k.** `tip_adapter` tops accuracy at all k (0.504–0.541) while sitting
near-last on QWK; `zero_shot` is top-three on accuracy and **dead last on QWK at every k**.
Both exploit the 50%-class-0 test split.

**Method (dead).** Shrinking the covariance toward an LLM-derived label-geometry target
Psi(G) does not work. Oracle sweep (gamma x graph-mode, tuned *on the test set*): mean delta
QWK **+0.0000**; the **no-information `uniform` control wins at every k**; delta is negative
at the larger k. Honest tuning can only be worse. Cause: BiomedCLIP's class text embeddings
sit at cosine 0.909–0.994 — a near-collinear cone — so Psi's difference vectors carry no
usable signal even when the rank order is perfect (rho=1.000). Across 4 ordinal tasks x 6
templates, order fidelity is template-dependent: mean rho +0.617, monotonic in only 9/24 cells.

**Partial rescue (real but modest).** Using the same LLM ordering at the *decision* level
instead — Frank & Hall monotone reduction, C-1 binary "grade > j" discriminants — beats the
isotropic GDA baseline on QWK at k=2 (+0.084, p=0.037) and k=8 (+0.066, p=0.018), not at
k=4/16. It does **not** significantly beat `linear_probe` anywhere and loses at k=16.
The ablation is the strongest result: correct order beats **shuffled** order by +0.08 to
+0.18 QWK, significant at 3 of 4 k. Mechanism: binary splits pool classes, giving 2.5x more
samples per estimated mean (8 -> 20 at k=8), which attacks the degeneracy directly.

## Accuracy is the wrong headline metric on these datasets
Majority-class baselines, checked before running anything:

| dataset / task | majority baseline | best method achieved |
|---|---|---|
| dr_regular / DR level (5-class) | 0.500 | 0.540 (**+0.040**) |
| bus / malignancy (binary) | 0.718 | 0.760 (**+0.042**), **AUROC 0.819** |
| bus / case category (3-class) | 0.545 | not yet run |
| mammo_mass / pathology | 0.611 | not yet run |
| mammo_calc / pathology | 0.604 | not yet run |

`majority_class` was missing from the first results table — a real omission, now a standard
baseline in `src/estimators.py`. On `bus`, `zero_shot` scores accuracy 0.647 (**below** the
0.718 baseline) while its AUROC is 0.689: genuine signal, badly placed threshold. **AUROC is
the metric to report on these skewed binary tasks; 0.819 at 16-shot with no training is the
number worth building on.**

## Degeneracy, quantified
Balanced k-shot (as `task_presampling.py` samples it) makes the within-class covariance
singular in every configuration; empirical rank equals the theoretical C(k-1) in all 20
tested. Fraction of the 512 feature directions fixed by the **prior** rather than the data:
Gleason C=4 — 100%/99.2%/97.7%/94.5%/88.3% at k=1/2/4/8/16; DR C=5 — 100%/99.0%/97.1%/
93.2%/85.4%. Widest case tested (C=15,k=16) still 56%.
Two bugs this exposed, both fixed: at k=1 the scatter is exactly zero, and Ledoit-Wolf then
returns alpha=0, so Sigma became the zero matrix and the classifier silently degenerated to a
constant predictor (QWK exactly 0.0000). Guarded in `_scale` and `_alpha`.

## Data on disk
`/mnt/extra-storage/features/*.npz` — cached BiomedCLIP image+text features.
`/mnt/extra-storage/data/medimeta_extracted/` — dr_regular, dr_uwf, bus, mammo_mass, mammo_calc.
`mammo_mass` carries the BI-RADS lexicon as labels: mass shape (8), mass margins (5);
`mammo_calc`: calc type (14), calc distribution (5).
Not yet fetched/extracted: skinl_derm, skinl_photo (zips present), SICAPv2 (2.1 GB, prostate
Gleason, 4 ordinal classes), derm (download was truncated and deleted).

---

# Phase A — TCLA reproduction (arXiv:2607.09562), BUSI

Target dataset verified identical: MedIMeta `bus` class counts [133, 437, 210] == BUSI's
published composition. 3 classes (normal/benign/malignant), chance BACC = 33.3.
Protocol copied from their `utils/data.py`: k/class support sampled from ALL data, test =
every remaining image, 50 seeds, balanced accuracy. Method transcribed from their code
(`src/tcla.py`), constants and provenance documented there.

## Reproduction succeeded

| shots | their TCLA | ours | gap |
|---|---|---|---|
| 1 | 52.63 | 52.46 | **-0.17** |
| 2 | 56.12 | 54.96 | -1.16 |
| 4 | 60.80 | 59.71 | -1.09 |
| 8 | 65.32 | 65.17 | **-0.15** |
| 16 | 71.41 | 70.94 | **-0.47** |

Two honest gaps:
- **zero-shot 41.27 vs their 44.46** (-3.19). We used their two `ultrasound` templates with a
  plain mean ensemble; they likely use different class names or the PDA path. Prompt-level
  difference, not a method difference.
- **our Tip-Adapter is NOT faithful**: 47.36 vs their 62.43 at 16-shot, because we used
  untuned beta=5.5/alpha=1.0. So we reproduced their *method*, not their *baselines*, and
  must not cite our Tip-Adapter numbers as a check on their comparison.

## Main finding: removing TCLA's titular component improves it

`proto_only` = their multi-layer Mahalanobis prototype kernel used directly, `argmax(K)`,
with the Residual Logit Correction (RLC) deleted entirely.

| shots | proto_only | TCLA | delta |
|---|---|---|---|
| 1 | 47.86 | **52.46** | TCLA +4.60 |
| 2 | **55.58** | 54.96 | proto +0.62 |
| 4 | **63.43** | 59.71 | **proto +3.72** |
| 8 | **69.68** | 65.17 | **proto +4.51** |
| 16 | **73.60** | 70.94 | **proto +2.66** |

proto_only at 16-shot (73.60) also **exceeds their published TCLA (71.41)**.
So on BUSI the reported gains appear to come from the multi-layer prototype construction,
not from the closed-form residual correction the paper is named after. RLC helps only at
1-shot and hurts from 2-shot upward.
**Status: needs a paired test before this is believed** (std 2.5-7.0 across 50 seeds).
Per-seed values were not saved in the first run; re-running to get them.

## Mechanism hypothesis (to test, not yet established)

TCLA computes `Z_final = Z_ZS + mu*eta*K W`, fitting W to the residual `R = Y - Z_ZS`.
This **anchors every prediction to the zero-shot logits**, which on BUSI are weak
(BACC 41.27) while the prototype signal is strong (73.60). At 1-shot the prototypes are
noisy so the anchor helps; from 2-shot the anchor is drag. The additive form hard-codes a
coefficient of 1.0 on `Z_ZS`, so there is no way for the fit to discount it.
Secondary issue: `lambda_reg = 1.0` is fixed while `K^T K` scales with support size, so the
effective regularisation silently varies with k (~33% of total at 1-shot binary, ~3% at 16).

## Phase B proposal — let the data choose the anchor

Replace the hard-coded residual offset with a joint ridge that includes the zero-shot logits
as *features* rather than as a fixed offset, and choose lambda analytically:

    TCLA :  Z = Z_ZS + eta * K W,         W = argmin ||K W - (Y - Z_ZS)||^2 + 1.0*||W||^2
    ours :  Z = [K | Z_ZS | 1] V,         V by ridge, lambda chosen by analytic leave-one-out

This strictly generalises both baselines: the fit can recover TCLA (coefficient 1 on Z_ZS)
or proto_only (coefficient 0), and picks per task/shot instead of being told. Closed-form,
no gradients, CPU-only. Analytic LOO for ridge costs one extra hat-matrix diagonal.
Controls required: shuffled-label control, and report against proto_only (the real baseline
to beat, not TCLA).
