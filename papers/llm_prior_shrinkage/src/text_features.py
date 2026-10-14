"""Build class text embeddings, selecting the prompt template by ORDER FIDELITY.

Why this is not test-set leakage: the selection criterion is Spearman correlation between
the template's embedding geometry and the class ordering declared in the dataset's
info.yaml metadata. It touches no images and no test labels. It is a property of the
prompt and the label names only, so it is available at "train" time in any deployment.

The axis-naming templates are the ones the Phase 0 audit found recover ordering best, and
the axis NAME is exactly what the local LLM supplies (verified: 'Gleason Grade',
'Clinical Severity', 'Severity').
"""
import os, json, sys
os.environ.setdefault("HF_HOME", "/mnt/extra-storage/hf_cache")
import numpy as np, torch, open_clip, yaml
from scipy.stats import spearmanr

MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
DATA = "/mnt/extra-storage/data/medimeta_extracted"
OUT = "/mnt/extra-storage/features"

# axis-naming templates first; generic ones kept as controls
BANK = [
    "{axis}: {lbl}",
    "{dom} image, {axis}: {lbl}",
    "this {dom} image shows {lbl} on the {axis} scale",
    "a {dom} photograph showing {lbl}",      # generic control (the original)
    "{lbl}",                                  # bare control
]


def order_fidelity(T, C):
    T = T / np.linalg.norm(T, axis=1, keepdims=True)
    Tc = T - T.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Tc, full_matrices=False)
    pc1 = Tc @ Vt[0]
    if np.corrcoef(pc1, np.arange(C))[0, 1] < 0:
        pc1 = -pc1
    rho = spearmanr(pc1, np.arange(C)).statistic
    v = T[-1] - T[0]; v /= np.linalg.norm(v)
    proj = T @ v
    mono = bool(all(proj[i] < proj[i + 1] for i in range(C - 1)))
    return (0.0 if np.isnan(rho) else float(rho)), mono


def build(dataset_id, axis_names=None):
    ds_dir = os.path.join(DATA, dataset_id)
    info = yaml.safe_load(open(os.path.join(ds_dir, "info.yaml")))
    dom = info["domain"].lower()
    torch.set_num_threads(8)
    model, _, _ = open_clip.create_model_and_transforms(MODEL)
    tok = open_clip.get_tokenizer(MODEL); model.eval()

    axis_names = axis_names or {}
    store, chosen = {}, {}
    for t in info["tasks"]:
        name = t["task_name"]
        lp = os.path.join(ds_dir, "task_labels", f"{name}.npy")
        if not os.path.exists(lp) or not t.get("labels"):
            continue
        keys = sorted(t["labels"], key=int)
        labels = [t["labels"][k] for k in keys]
        C = len(labels)
        is_ord = t["task_target"] == "ORDINAL_REGRESSION"
        # LLM-supplied axis name, falling back to the dataset/task name
        axis = axis_names.get(name) or f"{info['name'].lower()} {name.lower()}"

        best = None
        print(f"--- {name} (C={C}, ordinal={is_ord}) axis={axis!r}")
        for tpl in BANK:
            prompts = [tpl.format(dom=dom, lbl=l, axis=axis) for l in labels]
            with torch.inference_mode():
                T = model.encode_text(tok(prompts)).cpu().numpy().astype(np.float32)
            rho, mono = order_fidelity(T.astype(np.float64), C)
            print(f"    rho={rho:+.3f} mono={str(mono):<5} {tpl[:52]}")
            # select on order fidelity ONLY for ordinal tasks; otherwise keep the generic
            score = (rho, mono) if is_ord else (0, False)
            if best is None or score > best[0]:
                best = (score, tpl, T, rho, mono)
        if not is_ord:  # non-ordinal: use the generic template deterministically
            tpl = "a {dom} photograph showing {lbl}"
            prompts = [tpl.format(dom=dom, lbl=l, axis=axis) for l in labels]
            with torch.inference_mode():
                T = model.encode_text(tok(prompts)).cpu().numpy().astype(np.float32)
            best = ((0, False), tpl, T, float("nan"), False)
        _, tpl, T, rho, mono = best
        print(f"    -> chose {tpl!r}  (rho={rho:+.3f}, monotonic={mono})\n")
        store[name] = T
        chosen[name] = {"template": tpl, "spearman": rho, "monotonic": mono,
                        "axis_name": axis, "ordinal": is_ord, "keys": [int(k) for k in keys]}

    path = os.path.join(OUT, f"{dataset_id}_text.npz")
    np.savez_compressed(path,
        **{f"feats__{t}": v for t, v in store.items()},
        **{f"keys__{t}": np.array(chosen[t]["keys"]).astype(str) for t in store},
        meta=json.dumps(chosen))
    print(f"wrote {path}")
    json.dump(chosen, open("results/template_selection.json", "w"), indent=2)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    # axis names as elicited from llama3.2:3b in Phase 0.8
    build(sys.argv[1] if len(sys.argv) > 1 else "dr_regular",
          axis_names={"DR level": "diabetic retinopathy severity",
                      "Artifact": "artifact severity",
                      "Clarity": "image clarity",
                      "Field definition": "field definition quality"})
