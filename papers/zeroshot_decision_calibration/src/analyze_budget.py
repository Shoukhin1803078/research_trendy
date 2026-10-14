"""What survived the budget experiment, stated plainly."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
rows = json.load(open(f"{R}/budget.json"))
M = ("rope_q_pik", "thresh_k", "platt", "linear_probe")


def pearson(a, b):
    a = np.asarray(a, float) - np.mean(a); b = np.asarray(b, float) - np.mean(b)
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


def spearman(a, b):
    return pearson(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))


print("1. Which method wins at each budget, counted over all 9 endpoints")
print(f"   {'k':>3} " + " ".join(f"{m:>13}" for m in M))
for k in ("2", "4", "8", "16", "32", "64"):
    wins = {m: 0 for m in M}
    for r in rows:
        best = max(M, key=lambda m: r["shots"][k][m]["mean"])
        wins[best] += 1
    print(f"   {k:>3} " + " ".join(f"{wins[m]:>13}" for m in M))

print("\n2. Mean BACC across the 6 non-inverted endpoints")
ok = [r for r in rows if not r["inverted"]]
print(f"   {'k':>3} " + " ".join(f"{m:>13}" for m in M) + f"{'native':>13}")
for k in ("2", "4", "8", "16", "32", "64"):
    cells = [np.mean([r["shots"][k][m]["mean"] for r in ok]) for m in M]
    nat = np.mean([r["bacc_native"] for r in ok])
    print(f"   {k:>3} " + " ".join(f"{c:13.3f}" for c in cells) + f"{nat:13.3f}")

print("\n3. Linear probe degeneracy (BACC exactly 0.500 = constant predictor)")
for k in ("2", "8", "32", "64"):
    deg = sum(1 for r in rows
              if abs(r["shots"][k]["linear_probe"]["mean"] - 0.5) < 1e-6)
    print(f"   k={k:>2}: degenerate on {deg}/9 endpoints")

print("\n4. Does H predict the gain of the method that actually works (thresh_k)?")
H = [r["H"] for r in ok]
for k in ("8", "32", "64"):
    g = [r["shots"][k]["thresh_k"]["mean"] - r["bacc_native"] for r in ok]
    print(f"   k={k:>2}: thresh_k - native vs H   pearson {pearson(H,g):+.3f}  "
          f"spearman {spearman(H,g):+.3f}")
    gq = [r["shots"][k]["rope_q_pik"]["mean"] - r["bacc_native"] for r in ok]
    print(f"         ROPE-Q  - native vs H   pearson {pearson(H,gq):+.3f}  "
          f"spearman {spearman(H,gq):+.3f}")

print("\n5. Is prevalence-matching ever better than fitting the threshold directly?")
print("   (ROPE-Q with the TRUE prevalence vs thresh_k at k=64 - an oracle check)")
w = 0
for r in rows:
    a, b = r["bacc_rope_q_pi_known"], r["shots"]["64"]["thresh_k"]["mean"]
    tag = "ROPE" if a > b else "thresh_k"
    if a > b:
        w += 1
    print(f"     {r['endpoint']:32} oracle-pi {a:.3f}  thresh_k(64) {b:.3f}  -> {tag}")
print(f"   ROPE-Q wins {w}/9 even when handed the true prevalence.")

print("\n6. The three failure modes, by endpoint")
for r in rows:
    if r["auroc"] < 0.5:
        mode = "C: inverted signal (no monotone rule can help)"
    elif r["H"] > 0.02:
        mode = "A: threshold-misplaced but rankable (rescuable)"
    else:
        mode = "B: signal-limited (threshold already fine)"
    print(f"   {r['endpoint']:32} AUROC {r['auroc']:.3f}  H {r['H']:+.3f}  {mode}")
