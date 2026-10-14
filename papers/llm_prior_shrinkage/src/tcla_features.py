"""Multi-layer feature cache for the TCLA reproduction.

TCLA needs intermediate CLS tokens from the BiomedCLIP vision trunk at blocks
(5,7,9,11), UNPROJECTED (768-d for ViT-B/16), plus the usual projected 512-d
embedding for the zero-shot logits. Taken from refs/TCLA/methods/features.py:
forward hooks on model.visual.trunk.blocks[idx], output[:, 0].
"""
import os, sys, json, time
os.environ.setdefault("HF_HOME", "/mnt/extra-storage/hf_cache")
import numpy as np, h5py, torch, open_clip, yaml

MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
DATA = "/mnt/extra-storage/data/medimeta_extracted"
OUT = "/mnt/extra-storage/features"
LAYERS = (5, 7, 9, 11)


def main(dataset_id, batch=16, threads=8):
    torch.set_num_threads(threads)
    ds = os.path.join(DATA, dataset_id)
    out_path = os.path.join(OUT, f"{dataset_id}_tcla_layers.npz")
    if os.path.exists(out_path):
        print("exists:", out_path); return

    model, _, preprocess = open_clip.create_model_and_transforms(MODEL)
    model.eval()
    norm = [t for t in preprocess.transforms if t.__class__.__name__ == "Normalize"][0]
    mean = torch.tensor(norm.mean).view(1, 3, 1, 1)
    std = torch.tensor(norm.std).view(1, 3, 1, 1)

    trunk = model.visual.trunk
    nblocks = len(trunk.blocks)
    print(f"vision trunk has {nblocks} blocks; hooking {LAYERS}", flush=True)

    grabbed = {}
    def hook(idx):
        def fn(m, i, o):
            grabbed[idx] = o[:, 0].detach()   # CLS token, unprojected
        return fn
    handles = [trunk.blocks[i].register_forward_hook(hook(i)) for i in LAYERS]

    with h5py.File(os.path.join(ds, "images.hdf5"), "r") as f:
        imgs = f["images"]; n = imgs.shape[0]
        gray = (imgs.ndim == 3) or (imgs.ndim == 4 and imgs.shape[-1] == 1)
        layer_feats = None; proj = np.zeros((n, 512), dtype=np.float32)
        t0 = time.time()
        with torch.inference_mode():
            for s in range(0, n, batch):
                e = min(s + batch, n)
                a = np.asarray(imgs[s:e])
                if a.ndim == 3:
                    a = np.repeat(a[..., None], 3, axis=-1)
                elif a.ndim == 4 and a.shape[-1] == 1:
                    a = np.repeat(a, 3, axis=-1)
                x = torch.from_numpy(a).float().permute(0, 3, 1, 2) / 255.0
                x = (x - mean) / std
                p = model.encode_image(x)
                proj[s:e] = p.cpu().numpy()
                if layer_feats is None:
                    d = grabbed[LAYERS[0]].shape[-1]
                    layer_feats = {i: np.zeros((n, d), dtype=np.float32) for i in LAYERS}
                    print(f"intermediate CLS dim = {d}", flush=True)
                for i in LAYERS:
                    layer_feats[i][s:e] = grabbed[i].cpu().numpy()
                if s % (batch * 10) == 0:
                    el = time.time() - t0; r = e / max(el, 1e-9)
                    print(f"  {e}/{n} {r:.2f} img/s eta {(n-e)/max(r,1e-9)/60:.1f} min", flush=True)
    for h in handles:
        h.remove()
    dt = time.time() - t0
    print(f"done {n} in {dt/60:.1f} min ({n/dt:.2f} img/s)", flush=True)

    info = yaml.safe_load(open(os.path.join(ds, "info.yaml")))
    tasks = {}
    for t in info["tasks"]:
        lp = os.path.join(ds, "task_labels", f"{t['task_name']}.npy")
        if os.path.exists(lp):
            tasks[t["task_name"]] = {"target": t["task_target"],
                                     "classes": {str(k): v for k, v in (t.get("labels") or {}).items()}}
    np.savez_compressed(out_path, projected=proj,
        **{f"layer__{i}": layer_feats[i] for i in LAYERS},
        **{f"labels__{k}": np.load(os.path.join(ds, "task_labels", f"{k}.npy"))
           for k in tasks},
        meta=json.dumps({"dataset_id": dataset_id, "domain": info.get("domain"),
                         "layers": list(LAYERS), "n": int(n),
                         "grayscale_replicated": bool(gray), "tasks": tasks}))
    print("wrote", out_path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bus")
