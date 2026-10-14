"""
AUDIT A2 - is the binormal headroom formula correct, and where does it break?

Independent check by simulation: draw scores from known distributions, compute
the EMPIRICAL best-threshold BACC, and compare with the formula's prediction.
"""
import os, sys
import numpy as np
from math import sqrt
from statistics import NormalDist
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
import rope
N = NormalDist()

print("=== A2.1 equal-variance Gaussian: formula should be exact ===")
print(f"{'d_true':>7} {'AUROC':>7} {'formula':>8} {'empirical':>10} {'err':>8}")
rng = np.random.default_rng(0)
for d in (0.0, 0.5, 1.0, 1.655, 2.0, 3.0):
    n = 400000
    y = rng.integers(0, 2, n)
    s = rng.normal(y * d, 1.0)
    a = rope.auroc(y, s)
    f = rope.bacc_binormal(a)
    e = rope.bacc_oracle(y, s)
    print(f"{d:7.3f} {a:7.4f} {f:8.4f} {e:10.4f} {f-e:+8.4f}")

print("\n=== A2.2 UNEQUAL variance: the assumption the paper leans on ===")
print("  (positives have sd = r; formula assumes r = 1)")
print(f"{'r':>5} {'AUROC':>7} {'formula':>8} {'empirical':>10} {'err':>8}  note")
for r in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
    n = 400000
    y = rng.integers(0, 2, n)
    s = np.where(y == 1, rng.normal(1.5, r, n), rng.normal(0.0, 1.0, n))
    a = rope.auroc(y, s)
    f = rope.bacc_binormal(a)
    e = rope.bacc_oracle(y, s)
    note = "formula UNDER-estimates" if f < e - 0.005 else (
           "formula OVER-estimates" if f > e + 0.005 else "ok")
    print(f"{r:5.2f} {a:7.4f} {f:8.4f} {e:10.4f} {f-e:+8.4f}  {note}")

print("\n=== A2.3 heavy-tailed / skewed scores ===")
for name, gen in (("lognormal", lambda y, n: np.where(y == 1, rng.lognormal(0.7, 1.0, n),
                                                      rng.lognormal(0.0, 1.0, n))),
                  ("bimodal pos", lambda y, n: np.where(
                      y == 1, np.where(rng.random(n) < 0.5, rng.normal(3, .5, n),
                                       rng.normal(0, .5, n)), rng.normal(0, 1, n)))):
    n = 400000
    y = rng.integers(0, 2, n)
    s = gen(y, n)
    a = rope.auroc(y, s); f = rope.bacc_binormal(a); e = rope.bacc_oracle(y, s)
    print(f"  {name:12} AUROC {a:.4f}  formula {f:.4f}  empirical {e:.4f}  err {f-e:+.4f}")

print("\n=== A2.4 edge cases ===")
cases = [("AUROC = 0.5", 0.5), ("AUROC = 0.0", 0.0), ("AUROC = 1.0", 1.0),
         ("AUROC = 0.188 (Pengi/IPVS)", 0.188)]
for nm, a in cases:
    print(f"  {nm:28} reachable(no flip) {rope.bacc_binormal(a):.4f}   "
          f"if sign known {rope.bacc_binormal(a, allow_flip=True):.4f}")
y = np.array([1, 1, 1]); s = np.array([1.0, 2.0, 3.0])
print(f"  all-positive labels -> auroc {rope.auroc(y, s)} (nan expected)")
y = np.array([0, 1]); s = np.array([1.0, 1.0])
print(f"  tied scores         -> auroc {rope.auroc(y, s):.3f} (0.5 expected)")
print(f"  bacc_oracle floors at 0.5: {rope.bacc_oracle(np.array([0,1,0,1]), np.array([4.,3.,2.,1.])):.3f}")

print("\n=== A2.5 small-sample bias in the EMPIRICAL ceiling ===")
print("  bacc_oracle picks the threshold on the same data, so it is optimistic.")
print(f"{'n':>6} {'true BACC*':>11} {'mean oracle':>12} {'bias':>8}")
true = rope.bacc_binormal(rope.auroc(*(lambda n: (rng.integers(0,2,n), None))(2)) if False else 0.76)
for n in (40, 100, 400, 2000):
    vals = []
    for _ in range(400):
        y = rng.integers(0, 2, n)
        s = rng.normal(y * 1.0, 1.0)
        vals.append(rope.bacc_oracle(y, s))
    t = rope.bacc_binormal(N.cdf(1.0 / sqrt(2)))
    print(f"{n:6} {t:11.4f} {np.mean(vals):12.4f} {np.mean(vals)-t:+8.4f}")
