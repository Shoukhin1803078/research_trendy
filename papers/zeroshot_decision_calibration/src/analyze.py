"""Does the headroom diagnostic H predict the gain a zero-label rank rule gets?

That is the paper's central claim, so it gets its own table. Cells where the
score is inverted (AUROC < 0.5) are reported separately: no monotone label-free
rule can help there, and pooling them would flatter the correlation.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rope

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
rows = json.load(open(f"{R}/arm_a_medimeta.json"))

print(f"{'endpoint':32} {'n':>4} {'pi':>5} {'AUROC':>6} {'nat':>5} {'reach':>6} "
      f"{'H':>6} | {'Q':>5} {'gainQ':>6} | {'M':>5} {'gainM':>6} {'pi_M':>5}")
print("-" * 112)
recs = []
for r in rows:
    d = r["diagnostic"]; rl = {x["rule"]: x for x in r["rules"]}
    nat = rl["native (s>0)"]["bacc"]
    q = rl["ROPE-Q (pi known)"]["bacc"]; m = rl["ROPE-M (pi from scores)"]["bacc"]
    rec = dict(name=r["endpoint"], n=r["n_test"], pi=r["prevalence"],
               auroc=d["auroc"], inv=d["auroc"] < 0.5, nat=nat,
               reach=d["bacc_binormal"], H=d["bacc_binormal"] - nat,
               q=q, gq=q - nat, m=m, gm=m - nat, pim=r["pi_mixture"],
               s32=r["rope_s"]["32"]["bacc_mean"] if "32" in r["rope_s"] else r["rope_s"][32]["bacc_mean"])
    recs.append(rec)
    flag = " INV" if rec["inv"] else ""
    print(f"{rec['name']:32} {rec['n']:4} {rec['pi']:5.3f} {rec['auroc']:6.3f} "
          f"{nat:5.3f} {rec['reach']:6.3f} {rec['H']:+6.3f} | {q:5.3f} {rec['gq']:+6.3f} "
          f"| {m:5.3f} {rec['gm']:+6.3f} {rec['pim']:5.3f}{flag}")

ok = [r for r in recs if not r["inv"]]
inv = [r for r in recs if r["inv"]]
print(f"\n{len(ok)} usable cells (AUROC>0.5), {len(inv)} inverted cells excluded from the fit")

def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))

def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return pearson(ra, rb)

H = [r["H"] for r in ok]
print("\nH vs realised gain, over the usable cells:")
for lbl, g in (("ROPE-Q (pi known)", [r["gq"] for r in ok]),
               ("ROPE-M (pi est.)  ", [r["gm"] for r in ok])):
    print(f"  {lbl}  pearson r = {pearson(H, g):+.3f}   spearman = {spearman(H, g):+.3f}")

print("\nSign agreement (does H>0 predict gain>0?), usable cells:")
for lbl, key in (("ROPE-Q", "gq"), ("ROPE-M", "gm")):
    tp = sum(1 for r in ok if r["H"] > 0.02 and r[key] > 0)
    tn = sum(1 for r in ok if r["H"] <= 0.02 and r[key] <= 0.02)
    print(f"  {lbl}: H>+0.02 and gained: {tp}/{sum(1 for r in ok if r['H']>0.02)}   "
          f"H<=+0.02 and did not: {tn}/{sum(1 for r in ok if r['H']<=0.02)}")

print("\nProblem the numbers expose - the mixture prevalence estimator:")
err = [abs(r["pim"] - r["pi"]) for r in recs]
print(f"  |pi_mixture - pi_true|: mean {np.mean(err):.3f}  max {np.max(err):.3f}  "
      f"within 0.10 in {sum(1 for e in err if e <= 0.10)}/{len(err)} cells")
print("  per-cell (true -> estimated):")
for r in recs:
    print(f"    {r['name']:32} {r['pi']:.3f} -> {r['pim']:.3f}   err {abs(r['pim']-r['pi']):+.3f}")
