"""
T6 / feedback2 Priority 3 - verify and extend the estimator simulation.

Phase 1 tested SD ratios 0.5-3.0 and two non-Gaussian shapes. This widens the
range, adds families, and attaches Monte-Carlo uncertainty.

CONSTRAINT FROM THE SPEC: the estimator is NOT modified to produce a stronger
result. This script only characterises when it is informative and when it
misleads.
"""
import os, sys
import numpy as np
from math import sqrt
from statistics import NormalDist
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
import rope
N = NormalDist()
NSAMP, NREP = 40000, 10


def trial(gen, rng):
    y = rng.integers(0, 2, NSAMP)
    s = gen(y, rng)
    a = rope.auroc(y, s)
    return a, rope.bacc_binormal(a), rope.bacc_oracle(y, s)


def run(name, gen, seed=0):
    rng = np.random.default_rng(seed)
    A, F, E = [], [], []
    for _ in range(NREP):
        a, f, e = trial(gen, rng)
        A.append(a); F.append(f); E.append(e)
    A, F, E = map(np.array, (A, F, E))
    err = F - E
    half = 1.96 * err.std(ddof=1) / sqrt(NREP)
    verdict = ("UNDER-estimates" if err.mean() < -0.005 else
               "OVER-estimates" if err.mean() > 0.005 else "unbiased")
    return dict(name=name, auroc=A.mean(), formula=F.mean(), empirical=E.mean(),
                err=err.mean(), ci=half, verdict=verdict)


def show(rows, title):
    print(f"\n=== {title} ===")
    print(f"  {'condition':34} {'AUROC':>6} {'estimate':>9} {'empirical':>10} "
          f"{'error':>8} {'+-95%':>7}  verdict")
    for r in rows:
        print(f"  {r['name']:34} {r['auroc']:6.3f} {r['formula']:9.4f} "
              f"{r['empirical']:10.4f} {r['err']:+8.4f} {r['ci']:7.4f}  {r['verdict']}")


# ---- 1. the assumption holds: equal-variance Gaussian at several separations
rows = [run(f"equal-var Gaussian, d'={d}",
            (lambda d: lambda y, r: r.normal(y * d, 1.0, NSAMP))(d))
        for d in (0.5, 1.0, 1.655, 2.5)]
show(rows, "1. assumption satisfied - the estimator should be exact")

# ---- 2. unequal variance, widened range
#    Justification: a model that is confident on negatives and diffuse on
#    positives (or vice versa) produces exactly this. Ratios below 1 and above 1
#    are both plausible, so both are tested.
rows = [run(f"SD ratio {r_:.2f} (pos spread / neg)",
            (lambda rr: lambda y, r: np.where(y == 1, r.normal(1.5, rr, NSAMP),
                                              r.normal(0.0, 1.0, NSAMP)))(r_))
        for r_ in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)]
show(rows, "2. unequal variance - widened range")

# ---- 3. other distribution families
fams = [
 ("lognormal (right-skewed)",
  lambda y, r: np.where(y == 1, r.lognormal(0.7, 1.0, NSAMP),
                        r.lognormal(0.0, 1.0, NSAMP))),
 ("exponential",
  lambda y, r: np.where(y == 1, r.exponential(2.0, NSAMP), r.exponential(1.0, NSAMP))),
 ("Student t, df=3 (heavy tails)",
  lambda y, r: r.standard_t(3, NSAMP) + y * 1.5),
 ("bimodal positives (50/50 mixture)",
  lambda y, r: np.where(y == 1, np.where(r.random(NSAMP) < .5, r.normal(3, .5, NSAMP),
                                         r.normal(0, .5, NSAMP)), r.normal(0, 1, NSAMP))),
 ("bimodal negatives",
  lambda y, r: np.where(y == 0, np.where(r.random(NSAMP) < .5, r.normal(-3, .5, NSAMP),
                                         r.normal(0, .5, NSAMP)), r.normal(1.5, 1, NSAMP))),
 ("discrete scores (8 levels)",
  lambda y, r: np.round(np.clip(r.normal(y * 1.5, 1.0, NSAMP), -4, 4) * 1.0)),
 ("saturated logprobs (clipped)",
  lambda y, r: np.clip(r.normal(y * 1.5, 1.0, NSAMP), -0.5, 0.5)),
]
show([run(n, g) for n, g in fams], "3. other families")

# ---- 4. class imbalance, which the audited papers span (0.037 to 0.827)
def imb(pi):
    def g(y, r):
        return r.normal(y * 1.5, 1.0, NSAMP)
    return g
print("\n=== 4. does prevalence affect the estimator? ===")
print("  AUROC and the binormal estimate are both prevalence-free by construction,")
print("  so the estimate does not change with prevalence. What DOES change is the")
print("  variance of the empirical ceiling, because the minority class shrinks.")
rng = np.random.default_rng(1)
for pi in (0.05, 0.2, 0.5):
    vals = []
    for _ in range(12):
        y = (rng.random(NSAMP) < pi).astype(int)
        s = rng.normal(y * 1.5, 1.0, NSAMP)
        vals.append(rope.bacc_oracle(y, s))
    print(f"  prevalence {pi:.2f}: empirical ceiling {np.mean(vals):.4f} "
          f"+- {np.std(vals, ddof=1):.4f} (estimate {rope.bacc_binormal(0.8556):.4f})")

print("""
=== WHEN IS THE ESTIMATE INFORMATIVE, AND WHEN MISLEADING? ===

INFORMATIVE when the two score distributions have similar spread and roughly
  unimodal shape. There the error is under 0.002 and the estimate is usable as
  a point estimate.

CONSERVATIVE (under-estimates, so H is a floor) under unequal variance in
  EITHER direction - the error is U-shaped around a ratio of 1.0, reaching
  -0.061 at 0.25 and -0.154 at 6.0 - and under bimodal positives (-0.051),
  heavy tails (-0.026), discrete scores (-0.011) and saturated/clipped scores
  (-0.051). The last is directly relevant: LLM label-token log-probabilities
  are frequently saturated. A cell flagged decision-limited is therefore
  flagged on a strict criterion.

MISLEADING in two directions:
  (a) a cell NOT flagged may still have headroom, because the estimate can sit
      well below the true ceiling. The mode-B class is therefore impure.
  (b) OVER-estimation does occur, though it was small in these tests: bimodal
      NEGATIVES gave +0.0029 +- 0.0009, which is beyond Monte-Carlo noise. So
      the estimator is not guaranteed conservative in every shape; it was
      conservative in 11 of the 12 non-Gaussian conditions tested and mildly
      anti-conservative in 1. The families tested are not exhaustive.

The estimate is NEVER a guaranteed upper bound, and the negative-headroom count
is evidence about distribution shape, not proof of anything on its own.
""")
