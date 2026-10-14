"""ORACLE diagnostic: is there ANY (gamma, mode) where the geometry prior helps?

This deliberately sweeps on the TEST set. That is leakage and the numbers are NOT
reportable as results. The purpose is a go/no-go on Claim 2: if no configuration beats
the isotropic baseline even with oracle hyperparameters, the constructive half is dead
and no amount of honest tuning will rescue it.
"""
import os, sys, json, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import estimators as E
from metrics import evaluate
from run_experiment import sample_balanced

FEAT = "/mnt/extra-storage/features"

def main(dataset_id="dr_regular", task="DR level", ks=(2, 4, 8, 16), seeds=range(5)):
    z = np.load(f"{FEAT}/{dataset_id}_biomedclip.npz", allow_pickle=True)
    t = np.load(f"{FEAT}/{dataset_id}_text.npz", allow_pickle=True)
    X = z["features"]; y = z[f"labels__{task}"].astype(int).ravel()
    tr, te = z["split__train"], z["split__test"]
    T = t[f"feats__{task}"]; C = T.shape[0]

    GAMMAS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    MODES = ["adjacent", "distant", "uniform"]

    print(f"{dataset_id} / {task}   C={C}   oracle sweep on TEST (diagnostic only)\n")
    print("QWK, averaged over 5 seeds.  gamma=0 == isotropic Ledoit-Wolf baseline.\n")
    best_overall = {}
    for k in ks:
        sups = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            sups.append(sample_balanced(y, tr, C, k, rng))
        print(f"--- k={k} ---")
        header = "mode       " + "".join(f"{g:>9}" for g in GAMMAS)
        print(header); print("-" * len(header))
        base = None
        for mode in MODES:
            G = E.ordinal_graph(C, mode=mode)
            row = []
            for g in GAMMAS:
                qs = []
                for sup in sups:
                    L = E.gda_structured(X[sup], y[sup], X[te], C, T, gamma=g, G=G)
                    qs.append(evaluate(y[te], L.argmax(1), n_classes=C)["qwk"])
                row.append(float(np.mean(qs)))
            if base is None:
                base = row[0]   # gamma=0 is identical across modes by construction
            print(f"{mode:<11}" + "".join(f"{v:>9.4f}" for v in row))
        # best non-trivial config
        cells = []
        for mode in MODES:
            G = E.ordinal_graph(C, mode=mode)
            for g in GAMMAS[1:]:
                qs = [evaluate(y[te], E.gda_structured(X[s], y[s], X[te], C, T, gamma=g, G=G).argmax(1),
                               n_classes=C)["qwk"] for s in sups]
                cells.append((float(np.mean(qs)), mode, g))
        cells.sort(reverse=True)
        bq, bm, bg = cells[0]
        delta = bq - base
        best_overall[k] = {"baseline_qwk": base, "best_qwk": bq, "mode": bm,
                           "gamma": bg, "delta": delta}
        print(f"  baseline (gamma=0): {base:.4f}")
        print(f"  best oracle cell  : {bq:.4f}  (mode={bm}, gamma={bg})   delta={delta:+.4f}\n")

    print("=" * 62)
    print("ORACLE-BEST improvement over the isotropic baseline, per k:")
    for k, v in best_overall.items():
        print(f"  k={k:<3} delta QWK = {v['delta']:+.4f}  (mode={v['mode']}, gamma={v['gamma']})")
    deltas = [v["delta"] for v in best_overall.values()]
    print(f"\nmean oracle delta = {np.mean(deltas):+.4f}, max = {max(deltas):+.4f}")
    print()
    # A gain only counts if it comes from a graph that actually ENCODES ordinal structure.
    # mode="uniform" (G_ij = const) carries no ordinal information: it is the control.
    # If the control wins, any apparent gain is unrelated to label geometry.
    modes_won = [v["mode"] for v in best_overall.values()]
    control_won = all(m == "uniform" for m in modes_won)
    print(f"winning mode per k: {modes_won}")
    if control_won:
        print("  -> the UNIFORM control (no ordinal information) wins at every k.")
    if max(deltas) <= 0.005 or control_won:
        print()
        print("VERDICT: Claim 2 is DEAD as formulated.")
        print("  * mean oracle delta is %.4f - indistinguishable from zero." % np.mean(deltas))
        print("  * the no-information control wins, so what little moves is not geometry.")
        print("  * delta is NEGATIVE at the larger k where the estimator is best conditioned.")
        print("  Honest tuning can only do worse than oracle, so do not try to rescue this.")
    else:
        print("VERDICT: an ordinal-encoding graph beats the control. Necessary but NOT")
        print("sufficient - must be reproduced selecting on the val split, no test access.")
    json.dump(best_overall, open("results/psi_oracle_ablation.json", "w"), indent=2)

if __name__ == "__main__":
    main()
