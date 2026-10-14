"""
Arm A - zero-shot BiomedCLIP on MedIMeta, nine binary clinical screening endpoints.

Uses only cached features, so it costs seconds and no GPU. For each endpoint we
ask: does the model rank patients well (AUROC) while deciding badly (native
BACC)? And does a zero-label rank rule recover the gap the diagnostic predicts?
"""
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rope

FEAT = "/mnt/extra-storage/features"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# endpoint := (dataset, label key, positive class indices, name)
# Binarisations are the clinically standard screening endpoints for each task.
ENDPOINTS = [
    ("bus",        "malignancy",       [1],          "BUS: malignant finding"),
    ("bus",        "case category",    [2],          "BUS: malignant vs rest"),
    ("bus",        "case category",    [1, 2],       "BUS: any lesion vs normal"),
    ("dr_regular", "Overall quality",  [0],          "DR: inadequate image quality"),
    ("dr_regular", "DR level",         [1, 2, 3, 4], "DR: any retinopathy (>=1)"),
    ("dr_regular", "DR level",         [2, 3, 4],    "DR: referable DR (>=2)"),
    ("dr_regular", "DR level",         [3, 4],       "DR: sight-threatening (>=3)"),
    ("dr_regular", "Clarity",          [0, 1],       "DR: poor clarity"),
    ("dr_regular", "Field definition", [0, 1],       "DR: poor field definition"),
]


def load(ds):
    img = np.load(f"{FEAT}/{ds}_biomedclip.npz", allow_pickle=True)
    txt = np.load(f"{FEAT}/{ds}_text.npz", allow_pickle=True)
    return img, txt


def zero_shot_scores(img, txt, label_key, pos_idx):
    """Binary score = logsumexp over positive class prompts minus negatives.
    This is exactly the model's own zero-shot decision, collapsed to one axis;
    native decision is score > 0, i.e. argmax over the two groups."""
    f = img["features"].astype(np.float64)
    f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-12
    t = txt[f"feats__{label_key}"].astype(np.float64)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
    logits = 100.0 * (f @ t.T)                      # CLIP logit scale
    C = logits.shape[1]
    pos = np.array(sorted(set(pos_idx)))
    neg = np.array([c for c in range(C) if c not in set(pos_idx)])
    def lse(a):
        m = a.max(1, keepdims=True)
        return (m + np.log(np.exp(a - m).sum(1, keepdims=True))).ravel()
    return lse(logits[:, pos]) - lse(logits[:, neg])


def run_endpoint(ds, label_key, pos_idx, name, seeds=50):
    img, txt = load(ds)
    y_all = img[f"labels__{label_key}"]
    if y_all.ndim > 1:                              # multilabel -> not used here
        return None
    s_all = zero_shot_scores(img, txt, label_key, pos_idx)
    te = img["split__test"]
    y = np.isin(y_all[te], pos_idx).astype(int)
    s = s_all[te]
    if y.sum() == 0 or y.sum() == len(y):
        return None

    h = rope.headroom(y, s)
    pi_true = float(y.mean())
    pi_mix = rope.estimate_pi_mixture(s)

    rows = [rope.evaluate(y, s, None, "native (s>0)"),
            rope.evaluate(y, s, pi_true, "ROPE-Q (pi known)"),
            rope.evaluate(y, s, pi_mix, "ROPE-M (pi from scores)")]

    # ROPE-S: k labels, prior = the mixture estimate, w by leave-one-out
    rng = np.random.default_rng(0)
    r_cohort = rope.ranks01(s)
    shots = {}
    for k in (4, 8, 16, 32):
        accs, ws, pis = [], [], []
        for sd in range(seeds):
            r = np.random.default_rng(1000 * sd + k)
            idx = r.choice(len(y), size=min(k, len(y)), replace=False)
            w, pi = rope.shrink_pi_loo(s, idx, y[idx], pi_mix)
            mask = np.ones(len(y), bool); mask[idx] = False   # score on held-out
            yh = rope.predict_quantile(s, pi, r=r_cohort)[mask]
            accs.append(rope.bacc(y[mask], yh)); ws.append(w); pis.append(pi)
        shots[k] = {"bacc_mean": float(np.mean(accs)), "bacc_std": float(np.std(accs)),
                    "w_mean": float(np.mean(ws)), "pi_mean": float(np.mean(pis))}

    # controls
    y_shuf = rng.permutation(y)
    ctrl_shuffled = rope.headroom(y_shuf, s)
    s_rand = rng.normal(size=len(y))
    ctrl_random = rope.headroom(y, s_rand)

    ci_auroc = rope.bootstrap_ci(rope.auroc, y, s, n=600)
    return {"endpoint": name, "dataset": ds, "task": label_key,
            "pos_classes": pos_idx, "n_test": int(len(y)),
            "prevalence": pi_true, "majority_acc": rope.majority_acc(y),
            "diagnostic": h, "auroc_ci95": ci_auroc,
            "pi_mixture": pi_mix, "rules": rows, "rope_s": shots,
            "control_shuffled_labels": ctrl_shuffled,
            "control_random_scores": ctrl_random}


if __name__ == "__main__":
    out = []
    for ds, lk, pos, nm in ENDPOINTS:
        try:
            r = run_endpoint(ds, lk, pos, nm)
        except Exception as e:
            print(f"SKIP {nm}: {type(e).__name__}: {e}", flush=True)
            continue
        if r is None:
            print(f"SKIP {nm}: degenerate", flush=True); continue
        out.append(r)
        d, rl = r["diagnostic"], {x["rule"]: x for x in r["rules"]}
        print(f"\n{nm}   n={r['n_test']}  pi={r['prevalence']:.3f}  "
              f"majority_acc={r['majority_acc']:.3f}", flush=True)
        print(f"  AUROC {d['auroc']:.3f} [{r['auroc_ci95'][0]:.3f},{r['auroc_ci95'][1]:.3f}]"
              f"   native BACC {d['bacc_native']:.3f}   reachable {d['bacc_binormal']:.3f}"
              f"   H {d['H']:+.3f}", flush=True)
        for k, v in rl.items():
            print(f"    {k:26} BACC {v['bacc']:.3f}  sens {v['sens']:.3f}  "
                  f"spec {v['spec']:.3f}  pred+ {v['pred_pos_rate']:.3f}", flush=True)
        print(f"    pi: true {r['prevalence']:.3f}  mixture {r['pi_mixture']:.3f}", flush=True)
        print("    ROPE-S " + "  ".join(
            f"k={k}:{v['bacc_mean']:.3f}(w={v['w_mean']:.2f})" for k, v in r["rope_s"].items()), flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/arm_a_medimeta.json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"\nwrote {OUT}/arm_a_medimeta.json  ({len(out)} endpoints)", flush=True)
