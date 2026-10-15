"""
Full CKD arm, analysed PATIENT-LEVEL as pre-registered in audit/CKD_GO_NOGO.md.

Pre-registered before seeing the data:
  - primary unit is the PATIENT, not the record;
  - if patient-level AUROC <= 0.55 the cell is reported as signal-limited or
    inverted, i.e. a NEGATIVE result for recoverability, and is not reframed;
  - thresholds are fitted on support patients only and evaluated on held-out
    patients;
  - record-level figures are reported only as a diagnostic of within-patient
    correlation, never as the headline.
"""
import json, os, sys, collections
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
import rope

R = os.path.join(HERE, "results")
F = os.path.join(R, "llm_ckd_llama32_3b_list_f10.jsonl")
SHOTS = (4, 8, 16)
SEEDS = 50


def fit_thresh(ss, yy):
    if len(np.unique(yy)) < 2:
        return 0.0
    c = np.unique(ss)
    c = np.concatenate([[c[0] - 1], (c[:-1] + c[1:]) / 2, [c[-1] + 1]])
    best, bt = -1, 0.0
    for t in c:
        b = rope.bacc(yy, (ss > t).astype(int))
        if b > best:
            best, bt = b, t
    return bt


def main():
    rows = [json.loads(l) for l in open(F) if l.strip()]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["group"]].append(r)
    # Use only patients whose 10 records are ALL scored. A partially scored
    # patient would have a mean score over a different number of visits, which
    # is not comparable. This also makes the analysis valid on a run that was
    # interrupted: it simply uses fewer, complete patients.
    EXPECTED = 10
    incomplete = [u for u, v in by.items() if len(v) != EXPECTED]
    for u in incomplete:
        del by[u]
    pats = sorted(by)
    if incomplete:
        print(f"NOTE: {len(incomplete)} partially scored patient(s) excluded; "
              f"analysis uses {len(pats)} complete patients.")
    # patient label = ever positive; patient score = mean over that patient's records
    y = np.array([max(r["y"] for r in by[u]) for u in pats])
    s = np.array([np.mean([r["s"] for r in by[u]]) for u in pats])
    ry = np.array([r["y"] for r in rows])
    rs = np.array([r["s"] for r in rows])

    print(f"records {len(rows)}  patients {len(pats)}  "
          f"ever-positive patients {int(y.sum())}  "
          f"record prevalence {ry.mean():.3f}")
    topk = np.mean([r.get("both_in_topk", True) for r in rows])
    print(f"both label tokens in top-k: {100*topk:.0f}%")

    if len(np.unique(y)) < 2:
        print("only one patient class present - cannot evaluate"); return

    # --- the pre-registered primary analysis
    a = rope.auroc(y, s)
    lo, hi = rope.bootstrap_ci(rope.auroc, y, s, n=4000)
    d = rope.headroom(y, s)
    se, sp = rope.sens_spec(y, (s > 0).astype(int))
    print(f"\n=== PATIENT LEVEL (primary, pre-registered) ===")
    print(f"  AUROC {a:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  native: sens {se:.3f}  spec {sp:.3f}  BACC {d['bacc_native']:.3f}  "
          f"predicted-positive rate {float((s > 0).mean()):.3f}")
    print(f"  estimated reachable {d['bacc_binormal']:.3f}   H {d['H']:+.3f}")

    # --- record level, diagnostic only
    ra = rope.auroc(ry, rs)
    print(f"\n=== record level (diagnostic only, NOT the headline) ===")
    print(f"  AUROC {ra:.3f} vs patient-level {a:.3f}  "
          f"difference {ra - a:+.3f}")
    print("  A large positive difference indicates within-patient correlation")
    print("  inflating the record-level figure.")

    # --- the pre-registered verdict
    print(f"\n=== PRE-REGISTERED VERDICT ===")
    if a <= 0.55:
        verdict = ("NEGATIVE for recoverability: patient-level AUROC <= 0.55, so "
                   "this cell is signal-limited or inverted. Reported as a "
                   "negative result, not reframed.")
    else:
        verdict = ("Patient-level ranking is usable (AUROC > 0.55), so the "
                   "threshold-fitting test below is meaningful.")
    print(f"  patient AUROC {a:.3f} -> {verdict}")

    out = {"n_records": len(rows), "n_patients": len(pats),
           "n_pos_patients": int(y.sum()), "record_prevalence": float(ry.mean()),
           "patient_auroc": float(a), "patient_auroc_ci95": [float(lo), float(hi)],
           "record_auroc": float(ra), "bacc_native": float(d["bacc_native"]),
           "sens": float(se), "spec": float(sp),
           "reachable": float(d["bacc_binormal"]), "H": float(d["H"]),
           "verdict": verdict, "shots": {}}

    if a > 0.55:
        print(f"\n=== threshold fitted on k labelled PATIENTS, scored on the rest ===")
        print(f"  {'k':>3} {'BACC':>7} {'sd':>7}  {'patient-level 95% CI':>24}")
        rng = np.random.default_rng(0)
        for k in SHOTS:
            if k > len(y) // 2:
                continue
            accs = []
            for sd in range(SEEDS):
                r = np.random.default_rng(555 * sd + k)
                idx = r.choice(len(y), size=k, replace=False)
                if len(np.unique(y[idx])) < 2:
                    continue
                m = np.ones(len(y), bool); m[idx] = False
                accs.append(rope.bacc(y[m], (s[m] > fit_thresh(s[idx], y[idx])).astype(int)))
            if not accs:
                continue
            boot = []
            for _ in range(400):
                bi = rng.integers(0, len(y), len(y))
                yb, sb = y[bi], s[bi]
                if len(np.unique(yb)) < 2:
                    continue
                inner = []
                for sd2 in range(10):
                    r2 = np.random.default_rng(77 * sd2 + k)
                    idx = r2.choice(len(yb), size=k, replace=False)
                    m = np.ones(len(yb), bool); m[idx] = False
                    if len(np.unique(yb[idx])) < 2 or len(np.unique(yb[m])) < 2:
                        continue
                    inner.append(rope.bacc(yb[m],
                                 (sb[m] > fit_thresh(sb[idx], yb[idx])).astype(int)))
                if inner:
                    boot.append(np.mean(inner))
            blo, bhi = (np.quantile(boot, [.025, .975]) if len(boot) > 50 else (np.nan, np.nan))
            out["shots"][k] = {"bacc_mean": float(np.mean(accs)),
                               "bacc_sd": float(np.std(accs)),
                               "ci95": [float(blo), float(bhi)], "n_draws": len(accs)}
            print(f"  {k:>3} {np.mean(accs):7.3f} {np.std(accs):7.3f}  "
                  f"[{blo:+.3f}, {bhi:+.3f}]")
        if out["shots"]:
            kmax = max(out["shots"])
            realised = out["shots"][kmax]["bacc_mean"] - d["bacc_native"]
            print(f"\n  predicted H {d['H']:+.3f}   realised at k={kmax} {realised:+.3f}")
            out["realised_gain"] = float(realised)

    json.dump(out, open(f"{R}/ckd_arm.json", "w"), indent=1)
    print(f"\nwrote {R}/ckd_arm.json")


if __name__ == "__main__":
    main()
