"""Phase A: reproduce TCLA on BUSI (= MedIMeta `bus`, verified identical: 133/437/210).

Protocol copied from refs/TCLA/utils/data.py FewShotImageFolder:
  * sample k images per class from ALL data -> support
  * test on EVERY remaining image (no fixed held-out split; test shrinks as k grows)
  * metric: balanced accuracy
  * 50 seeds (the paper's setting)

Baselines included so the comparison is ours end-to-end rather than copied from their table.
"""
import os, sys, json, time, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import tcla as T
from metrics import balanced_accuracy, accuracy

FEAT = "/mnt/extra-storage/features"
LAYERS = (5, 7, 9, 11)
ULTRASOUND_PROMPTS = ["a photo of ultrasound showing {}", "an ultrasound image showing {}."]


def _n(X, eps=1e-12):
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), eps)


def sample_support(y, C, k, rng):
    """k per class from ALL indices; the rest is the test set."""
    sup = []
    for c in range(C):
        pool = np.where(y == c)[0]
        if len(pool) < k:
            return None, None
        sup.append(rng.choice(pool, size=k, replace=False))
    sup = np.concatenate(sup)
    test = np.setdiff1d(np.arange(len(y)), sup)
    return sup, test


def run(dataset_id="bus", task="case category", shots=(1, 2, 4, 8, 16), n_seeds=50,
        lf_mode="aggressive"):
    z = np.load(f"{FEAT}/{dataset_id}_tcla_layers.npz", allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    proj = z["projected"].astype(np.float64)
    layer_feats_all = {l: z[f"layer__{l}"].astype(np.float64) for l in LAYERS}
    y = z[f"labels__{task}"].astype(int).ravel()
    classes = meta["tasks"][task]["classes"]
    names = [classes[k] for k in sorted(classes, key=int)]
    C = len(names)

    # text features per prompt per class (cached once)
    tp = f"{FEAT}/{dataset_id}_tcla_text.npz"
    if os.path.exists(tp):
        tz = np.load(tp, allow_pickle=True)
        text_per_prompt = [tz[f"cls__{c}"].astype(np.float64) for c in range(C)]
    else:
        os.environ.setdefault("HF_HOME", "/mnt/extra-storage/hf_cache")
        import torch, open_clip
        torch.set_num_threads(8)
        M = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        m, _, _ = open_clip.create_model_and_transforms(M)
        tok = open_clip.get_tokenizer(M); m.eval()
        text_per_prompt = []
        with torch.inference_mode():
            for nm in names:
                prompts = [t.format(nm) for t in ULTRASOUND_PROMPTS]
                text_per_prompt.append(m.encode_text(tok(prompts)).cpu().numpy().astype(np.float64))
        np.savez_compressed(tp, **{f"cls__{i}": f for i, f in enumerate(text_per_prompt)})
    # simple prompt-ensemble text prototypes, for the zero-shot baseline
    T_simple = _n(np.stack([_n(t).mean(0) for t in text_per_prompt]))

    print(f"TCLA reproduction | {dataset_id}/{task} | C={C} | N={len(y)} | "
          f"class counts {list(np.bincount(y))}")
    print(f"classes: {names}")
    print(f"lf_mode={lf_mode}  seeds={n_seeds}  protocol: k/class support, test = remainder\n")

    rows = []
    for k in shots:
        METHODS = ("zero_shot", "tip_adapter", "proto_only", "tcla",
                   "ours_anchor_ridge", "ours_no_zs", "ours_shuffled_labels")
        acc = {m: [] for m in METHODS}
        bac = {m: [] for m in acc}
        ntest = None
        t0 = time.time()
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            sup, te = sample_support(y, C, k, rng)
            if sup is None:
                continue
            ntest = len(te)
            ys = y[sup]
            lf_sup = {l: layer_feats_all[l][sup] for l in LAYERS}
            lf_te = {l: layer_feats_all[l][te] for l in LAYERS}
            protos = {l: T.build_prototypes(lf_sup[l], ys, C) for l in LAYERS}
            sel, w, _ = T.select_layers_and_weights(protos)
            # projected prototypes for PDA (built from the projected embedding)
            proto_proj = {l: T.build_prototypes(proj[sup], ys, C) for l in sel}
            T_pda = T.pda_text_prototypes(text_per_prompt, proto_proj, sel, w)
            rel = T.prototype_reliabilities(lf_sup, ys, protos, sel)
            kmode, strength = T.lf_mode_config(lf_mode, float(k))
            K_sup = T.proto_kernel(lf_sup, protos, sel, w, rel, kernel_mode=kmode)
            K_te = T.proto_kernel(lf_te, protos, sel, w, rel, kernel_mode=kmode)
            Zs_sup = _n(proj[sup]) @ T_pda.T
            Zs_te = _n(proj[te]) @ T_pda.T
            st = T.fit_rlc(Zs_sup, K_sup, ys, C)
            Z_tcla = T.apply_rlc(Zs_te, K_te, st, strength)

            # baselines on the identical split
            Z_zs = _n(proj[te]) @ T_simple.T
            A = np.exp(-5.5 * (1.0 - _n(proj[te]) @ _n(proj[sup]).T))
            Z_tip = 100.0 * Z_zs + 1.0 * (A @ np.eye(C)[ys])
            Z_proto = K_te   # prototype kernel alone, no residual correction

            # OURS: joint ridge over [K | Z_ZS | 1], lambda by analytic LOO
            st_o = T.fit_anchor_ridge(Zs_sup, K_sup, ys, C)
            Z_ours = T.apply_anchor_ridge(Zs_te, K_te, st_o)
            # ablation 1: drop the zero-shot block entirely
            st_nz = T.fit_anchor_ridge(Zs_sup, K_sup, ys, C, include_zs=False)
            Z_nz = T.apply_anchor_ridge(Zs_te, K_te, st_nz)
            # ablation 2: shuffled support labels. If this does not collapse, the fit is
            # not using label information and any gain is an artefact.
            ys_bad = rng.permutation(ys)
            st_sh = T.fit_anchor_ridge(Zs_sup, K_sup, ys_bad, C)
            Z_sh = T.apply_anchor_ridge(Zs_te, K_te, st_sh)

            for name, Z in (("zero_shot", Z_zs), ("tip_adapter", Z_tip),
                            ("proto_only", Z_proto), ("tcla", Z_tcla),
                            ("ours_anchor_ridge", Z_ours), ("ours_no_zs", Z_nz),
                            ("ours_shuffled_labels", Z_sh)):
                p = Z.argmax(1)
                acc[name].append(accuracy(y[te], p))
                bac[name].append(balanced_accuracy(y[te], p))
        dt = time.time() - t0
        print(f"--- {k}-shot   (test n={ntest}, {dt:.0f}s for {n_seeds} seeds) ---")
        print(f"{'method':<14}{'BACC':>9}{'+/-':>8}{'accuracy':>11}")
        for m in METHODS:
            b = np.array(bac[m]); a = np.array(acc[m])
            print(f"{m:<14}{100*b.mean():>9.2f}{100*b.std():>8.2f}{100*a.mean():>11.2f}")
            rows.append({"shots": k, "method": m, "bacc_mean": float(b.mean()),
                         "bacc_std": float(b.std()), "acc_mean": float(a.mean()),
                         "n_seeds": int(len(b)), "n_test": int(ntest),
                         "bacc_per_seed": [float(v) for v in b],
                         "acc_per_seed": [float(v) for v in a]})
        # paired comparisons on identical splits, with a t-test
        from scipy.stats import ttest_rel
        for a_, b_ in (("tcla", "proto_only"), ("ours_anchor_ridge", "tcla"),
                       ("ours_anchor_ridge", "proto_only"),
                       ("ours_anchor_ridge", "ours_shuffled_labels")):
            A = np.array(bac[a_]); B = np.array(bac[b_])
            d = A - B
            try:
                pv = ttest_rel(A, B).pvalue
            except Exception:
                pv = float("nan")
            print(f"  {a_} - {b_}: {100*d.mean():+6.2f} BACC  "
                  f"wins {int((d>0).sum())}/{len(d)}  p={pv:.2e}")
        print()

    os.makedirs("results", exist_ok=True)
    out = f"results/tcla_repro_{dataset_id}_{task.replace(' ','_')}_{lf_mode}.json"
    json.dump({"dataset": dataset_id, "task": task, "classes": names, "lf_mode": lf_mode,
               "n_seeds": n_seeds, "rows": rows}, open(out, "w"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    run(task=sys.argv[1] if len(sys.argv) > 1 else "case category",
        n_seeds=int(sys.argv[2]) if len(sys.argv) > 2 else 50,
        lf_mode=sys.argv[3] if len(sys.argv) > 3 else "aggressive")
