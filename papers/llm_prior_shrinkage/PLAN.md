# Plan — reproduce TCLA (2026), then beat it

## Target paper
**TCLA: Training-Free Class-wise Logit Adaptation for Medical Vision-Language Models**
Tianyou Jiang, Ziyu Zhou. arXiv:2607.09562, submitted 10 July 2026.
Code: https://github.com/SkyCol/TCLA

Why this paper: it is 2026, disease-focused, **purely training-free** (so no GPU is needed —
only frozen feature extraction, which we do on CPU at 4.84 img/s), it has released code, and
it reports concrete numbers we can check ourselves.

## What TCLA does (from the paper)
Backbone: **BiomedCLIP** (PMC-15M) — the same one we already have cached.
Four components, all closed-form:
- Mahalanobis-weighted class prototypes: `p_c^(l) = sum_i pi_i^(l) f_i^(l)`,
  `pi_i^(l) = softmax(-d_i^(l))`, `d` = Mahalanobis distance to the class centre
- Layer selection by inter-class separability: `s_bar^(l) = mean_{i<j} cos(p_i^(l), p_j^(l))`,
  `score(l) = 1 - s_bar^(l)`, keep top-K layers, weights `omega^(l) = score(l)/sum score`
- Sharp prototype basis: `K_sharp^(l)(x) = exp[beta (A^(l)(x) - 1)]`
- Prompt distribution alignment: per-layer text prototypes
  `t_c^(l) = sum_j w_{c,j}^(l) t_{c,j}`, `w = softmax(beta_t s_{c,j}^(l))`,
  then `t_c = sum_l omega^(l) t_c^(l)`
- **Closed-form residual logit correction (the core):** ridge regression
  `W_m = [(K_S^m)^T K_S^m + lambda I]^{-1} (K_S^m)^T R_S`, **lambda = 1.0**,
  magnitude `eta_m = <R_S, R_hat_S^m>_F / (||R_hat_S^m||_F^2 + eps)`,
  final `Z_TCLA(x) = Z_ZS(x) + mu_m(s) eta_m K_m(x) W_m`

Setup: 9 datasets, shots 1/2/4/8/16, **50 seeds**, metric **balanced accuracy (BACC)**.
Reported: COVID 55.64 -> 61.34 (1-shot); CXRP 73.62 -> 77.52; LC25k 49.49 -> 80.77;
COVID 72.21 and CXRP 93.48 at 16-shot.

## Phase A — reproduce exactly  (do this first, change nothing)
- [ ] A1 Clone https://github.com/SkyCol/TCLA; read the code, not just the paper. Note every
      place the code and the paper disagree.
- [ ] A2 Identify which of the 9 datasets we can obtain. **BUSI we already have** as MedIMeta
      `bus`; BTMRI overlaps the brain-tumour work already in this repo. Get CXRP/COVID if
      they are small enough.
- [ ] A3 Implement BACC (balanced accuracy) in `src/metrics.py` — TCLA's metric, which we do
      not currently compute.
- [ ] A4 Reproduce TCLA's own numbers on at least one shared dataset at 1 and 16 shots, with
      **50 seeds** as the paper uses. Record the gap between our number and theirs.
- [ ] A5 Also reproduce their baselines (zero-shot, Tip-Adapter, APE) so the comparison is
      ours end-to-end, not copied from their table.
**Gate:** if we cannot land within a reasonable margin of a published TCLA number, stop and
find out why before proposing anything. A reproduction that does not reproduce is worthless
as a baseline.

## Phase B — propose something better
Only after A4 passes. Candidate improvements, each grounded in something we already measured
(see FINDINGS.md) rather than guessed:

1. **BACC vs the skew.** TCLA reports balanced accuracy, but its correction is fitted on a
   *balanced* support set. We measured that balanced k-shot makes the covariance singular
   (85–99% of directions fixed by the prior) and that `zero_shot` scores *below* the
   majority baseline on `bus` while having real AUROC signal — i.e. the operating point, not
   the ranking, is what is broken. A class-prior-corrected residual target may beat TCLA's
   on skewed data.
2. **Ridge lambda is fixed at 1.0** in TCLA. We have an analytic leave-one-out route for
   ridge that costs nothing and adapts per task. Cheap, principled, testable.
3. **Decision-level ordinal structure.** Our Frank-Hall reduction beat the isotropic GDA
   baseline on QWK at k=2 (p=0.037) and k=8 (p=0.018), and beat a *shuffled* ordering by
   +0.08 to +0.18 QWK. TCLA treats all classes as unordered. On its ordinal datasets this is
   a real gap.
4. **BI-RADS concept bottleneck** for the breast datasets, using the LLM mass-descriptor
   rule we validated at Spearman +0.913, p=0.010 — and explicitly NOT for calcifications,
   where the same elicitation gave −0.707.

Pick based on Phase A: whichever of TCLA's assumptions the reproduction shows to be weakest.

## Rules carried over from the last attempt
- Report every accuracy against the **majority-class baseline**. On these datasets it is
  0.50–0.72, and our best DR result beat it by only 4 points.
- Any claimed gain needs a **shuffled/no-information control** before it is believed. The
  Psi(G) idea died because the uniform control won.
- Pre-register the fallback: if Phase B finds nothing, the deliverable is the reproduction
  plus the ordinal diagnostic from FINDINGS.md, at MICCAI rather than an A* venue.
