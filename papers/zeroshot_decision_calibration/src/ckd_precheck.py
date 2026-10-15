"""
CKD pre-check. Applies the rule fixed in advance in audit/CKD_GO_NOGO.md §5:

    pre-check AUROC >= 0.60  -> GO at --groups 40  (100 min)
    0.55 <= AUROC < 0.60     -> GO at --groups 20  (50 min)
    AUROC < 0.55             -> NO-GO; record as attempted, insufficient signal

The rule is not revised after seeing the number.
"""
import json, os, sys, collections
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
import rope

F = os.path.join(HERE, "results", "llm_ckd_PRECHECK.jsonl")
rows = [json.loads(l) for l in open(F) if l.strip()]
y = np.array([r["y"] for r in rows])
s = np.array([r["s"] for r in rows])
g = np.array([r["group"] for r in rows])

print(f"records scored: {len(rows)}  patients: {len(set(g))}  "
      f"positive records: {int(y.sum())}  prevalence: {y.mean():.3f}")
topk = np.mean([r.get("both_in_topk", True) for r in rows])
print(f"both label tokens in top-k: {100*topk:.0f}% (score exact when 100%)")

if len(np.unique(y)) < 2:
    print("\nONLY ONE CLASS PRESENT - AUROC undefined. NO-GO.")
    sys.exit(0)

a = rope.auroc(y, s)
lo, hi = rope.bootstrap_ci(rope.auroc, y, s, n=4000)
nat = rope.bacc(y, (s > 0).astype(int))
se, sp = rope.sens_spec(y, (s > 0).astype(int))
d = rope.headroom(y, s)

print(f"\nrecord-level AUROC {a:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
print(f"native decision: sens {se:.3f}  spec {sp:.3f}  BACC {nat:.3f}  "
      f"predicted-positive rate {float((s > 0).mean()):.3f}")
print(f"estimated reachable {d['bacc_binormal']:.3f}   H {d['H']:+.3f}")

# patient level, mean score per patient, label = ever positive
subs = sorted(set(g))
yp = np.array([int(y[g == u].max()) for u in subs])
sp_ = np.array([s[g == u].mean() for u in subs])
if len(np.unique(yp)) > 1:
    print(f"patient-level AUROC {rope.auroc(yp, sp_):.3f} "
          f"({len(subs)} patients, {int(yp.sum())} ever-positive)")

print("\n--- pre-committed decision rule (audit/CKD_GO_NOGO.md) ---")
if a >= 0.60:
    v = "GO at --groups 40 (400 items, ~100 min)"
elif a >= 0.55:
    v = "GO at --groups 20 (200 items, ~50 min)"
else:
    v = ("NO-GO. Record as attempted with insufficient ranking signal to test "
         "recoverability, and keep as a documented limitation.")
print(f"  AUROC {a:.3f}  ->  {v}")
print("\nNOTE: n is small, so this AUROC is imprecise; the CI above is the")
print("honest read. The rule is applied to the point estimate as pre-registered.")

json.dump({"n_records": len(rows), "n_patients": len(subs),
           "prevalence": float(y.mean()), "auroc": float(a),
           "auroc_ci95": [float(lo), float(hi)], "bacc_native": float(nat),
           "sens": float(se), "spec": float(sp), "H": float(d["H"]),
           "decision": v},
          open(os.path.join(HERE, "results", "ckd_precheck.json"), "w"), indent=1)
