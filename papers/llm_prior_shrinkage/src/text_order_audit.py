"""Does the frozen medical VLM text tower represent clinical ordinal scales as ORDERED?

If it does not, then every text-side ordinal prior on a frozen encoder is doomed, and
that is a publishable negative result in its own right. Test across tasks and across
prompt templates, so the conclusion cannot be dismissed as one bad prompt.
"""
import os, json, itertools
os.environ.setdefault("HF_HOME", "/mnt/extra-storage/hf_cache")
import numpy as np, torch, open_clip
from scipy.stats import spearmanr, kendalltau

MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
DS = "/mnt/extra-storage/data/medimeta_extracted/dr_regular"

TEMPLATES = [
    "a {dom} photograph showing {lbl}",
    "{lbl}",
    "a photo of {lbl}",
    "retinal fundus image, {lbl}",
    "this fundus image shows {lbl}, a diabetic retinopathy severity grade",
    "diabetic retinopathy severity: {lbl}",
]

def norm(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)

def order_scores(T):
    """How well does the embedding geometry recover the known grade order 0..C-1?"""
    T = norm(T); C = len(T)
    # 1-D embedding via the leading principal component of the class embeddings
    Tc = T - T.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Tc, full_matrices=False)
    pc1 = Tc @ Vt[0]
    # orient so it correlates positively with true order
    if np.corrcoef(pc1, np.arange(C))[0, 1] < 0:
        pc1 = -pc1
    rho = spearmanr(pc1, np.arange(C)).statistic
    tau = kendalltau(pc1, np.arange(C)).statistic
    # projection onto the endpoint axis
    v = T[-1] - T[0]; v /= np.linalg.norm(v)
    proj = T @ v
    rho_ax = spearmanr(proj, np.arange(C)).statistic
    mono = bool(all(proj[i] < proj[i + 1] for i in range(C - 1)))
    Cs = T @ T.T
    adj = np.mean([Cs[i, i + 1] for i in range(C - 1)])
    dist = np.mean([Cs[i, j] for i in range(C) for j in range(C) if abs(i - j) >= 2])
    return {"spearman_pc1": round(float(rho), 4), "kendall_pc1": round(float(tau), 4),
            "spearman_endpoint_axis": round(float(rho_ax), 4), "monotonic_on_axis": mono,
            "mean_cos_adjacent": round(float(adj), 4), "mean_cos_distant": round(float(dist), 4),
            "adjacency_margin": round(float(adj - dist), 4),
            "pc1_var_explained": round(float(S[0]**2 / (S**2).sum()), 4)}

def main():
    import yaml
    info = yaml.safe_load(open(os.path.join(DS, "info.yaml")))
    dom = info["domain"].lower()
    torch.set_num_threads(8)
    model, _, _ = open_clip.create_model_and_transforms(MODEL)
    tok = open_clip.get_tokenizer(MODEL); model.eval()

    ordinal_tasks = [t for t in info["tasks"] if t["task_target"] == "ORDINAL_REGRESSION"]
    print(f"auditing {len(ordinal_tasks)} ORDINAL_REGRESSION tasks x {len(TEMPLATES)} templates\n")
    out = {}
    for t in ordinal_tasks:
        name = t["task_name"]
        keys = sorted(t["labels"], key=int)
        labels = [t["labels"][k] for k in keys]
        out[name] = {}
        print(f"--- {name}  (C={len(labels)}, declared ORDINAL_REGRESSION) ---")
        print(f"{'template':<52} {'rho(pc1)':>9} {'tau':>7} {'mono':>6} {'adj-dist':>9}")
        for tpl in TEMPLATES:
            prompts = [tpl.format(dom=dom, lbl=l) for l in labels]
            with torch.inference_mode():
                T = model.encode_text(tok(prompts)).cpu().numpy().astype(np.float64)
            s = order_scores(T)
            out[name][tpl] = s
            print(f"{tpl[:50]:<52} {s['spearman_pc1']:>9.3f} {s['kendall_pc1']:>7.3f} "
                  f"{str(s['monotonic_on_axis']):>6} {s['adjacency_margin']:>9.4f}")
        print()

    # summary across everything
    allrho = [v["spearman_pc1"] for d in out.values() for v in d.values()]
    allmono = [v["monotonic_on_axis"] for d in out.values() for v in d.values()]
    allmarg = [v["adjacency_margin"] for d in out.values() for v in d.values()]
    print("=" * 74)
    print(f"across {len(allrho)} (task, template) pairs:")
    print(f"  Spearman rho of PC1 vs true grade order: mean {np.mean(allrho):+.3f}  "
          f"median {np.median(allrho):+.3f}  min {min(allrho):+.3f}  max {max(allrho):+.3f}")
    print(f"  monotonic on the endpoint axis: {sum(allmono)}/{len(allmono)}")
    print(f"  adjacency margin (cos_adj - cos_distant): mean {np.mean(allmarg):+.4f}")
    print()
    print("A text tower that encoded these scales as ordered would give rho near +1 and")
    print("monotonicity in nearly every cell. Read the table above before concluding.")
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/text_order_audit.json", "w"), indent=2)

if __name__ == "__main__":
    main()
