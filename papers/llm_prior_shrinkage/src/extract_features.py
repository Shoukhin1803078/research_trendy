"""Extract frozen BiomedCLIP features once; cache to .npz. Everything else runs on these."""
import os, sys, time, json
os.environ.setdefault("HF_HOME", "/mnt/extra-storage/hf_cache")
import numpy as np, h5py, torch, open_clip

MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
DATA = "/mnt/extra-storage/data/medimeta_extracted"
OUT = "/mnt/extra-storage/features"


def load_splits(ds_dir):
    """split name -> array of integer image indices (filenames are images/NNNNNN.tiff)."""
    sp = {}
    for name in ("train", "val", "test"):
        p = os.path.join(ds_dir, "splits", f"{name}.txt")
        idx = []
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    idx.append(int(os.path.splitext(os.path.basename(line))[0]))
        sp[name] = np.array(idx, dtype=np.int64)
    return sp


def main(dataset_id, batch=16, threads=8):
    torch.set_num_threads(threads)
    ds_dir = os.path.join(DATA, dataset_id)
    os.makedirs(OUT, exist_ok=True)
    out_path = os.path.join(OUT, f"{dataset_id}_biomedclip.npz")
    if os.path.exists(out_path):
        print(f"exists, skipping: {out_path}"); return

    model, _, preprocess = open_clip.create_model_and_transforms(MODEL)
    model.eval()
    # BiomedCLIP's own normalization, taken from the preprocess pipeline (do not hardcode)
    norm = [t for t in preprocess.transforms if t.__class__.__name__ == "Normalize"][0]
    mean = torch.tensor(norm.mean).view(1, 3, 1, 1)
    std = torch.tensor(norm.std).view(1, 3, 1, 1)
    print(f"normalization mean={norm.mean} std={norm.std}", flush=True)

    with h5py.File(os.path.join(ds_dir, "images.hdf5"), "r") as f:
        imgs = f["images"]
        n = imgs.shape[0]
        gray = (imgs.ndim == 3) or (imgs.ndim == 4 and imgs.shape[-1] == 1)
        print(f"input {imgs.shape} grayscale={gray}", flush=True)
        feats = np.zeros((n, 512), dtype=np.float32)
        t0 = time.time()
        with torch.inference_mode():
            for s in range(0, n, batch):
                e = min(s + batch, n)
                a = np.asarray(imgs[s:e])
                # MedIMeta stores RGB as (N,H,W,3) and GRAYSCALE as (N,H,W).
                # Breast ultrasound and mammography are grayscale; replicate to 3
                # channels because BiomedCLIP's stem expects RGB.
                if a.ndim == 3:
                    a = np.repeat(a[..., None], 3, axis=-1)
                elif a.ndim == 4 and a.shape[-1] == 1:
                    a = np.repeat(a, 3, axis=-1)
                x = torch.from_numpy(a).float().permute(0, 3, 1, 2) / 255.0
                x = (x - mean) / std
                feats[s:e] = model.encode_image(x).cpu().numpy()
                if s % (batch * 10) == 0:
                    el = time.time() - t0
                    rate = e / el if el > 0 else 0
                    print(f"  {e}/{n}  {rate:.2f} img/s  eta {(n-e)/max(rate,1e-9)/60:.1f} min",
                          flush=True)
    dt = time.time() - t0
    print(f"done {n} images in {dt/60:.1f} min ({n/dt:.2f} img/s)", flush=True)

    # all declared tasks for this dataset, with their task_target
    import yaml
    info = yaml.safe_load(open(os.path.join(ds_dir, "info.yaml")))
    tasks = {}
    for t in info["tasks"]:
        name = t["task_name"]
        lp = os.path.join(ds_dir, "task_labels", f"{name}.npy")
        if os.path.exists(lp):
            tasks[name] = {"target": t["task_target"], "labels": np.load(lp),
                           "classes": t.get("labels", {})}
    sp = load_splits(ds_dir)
    np.savez_compressed(
        out_path, features=feats,
        **{f"labels__{k}": v["labels"] for k, v in tasks.items()},
        **{f"split__{k}": v for k, v in sp.items()},
        meta=json.dumps({"dataset_id": dataset_id, "domain": info.get("domain"),
                         "n": int(n), "feat_dim": 512, "img_per_s": round(n/dt, 2), "grayscale_replicated": bool(gray),
                         "tasks": {k: {"target": v["target"],
                                       "classes": {str(a): b for a, b in v["classes"].items()}}
                                   for k, v in tasks.items()}}))
    print(f"wrote {out_path}")
    for k, v in tasks.items():
        print(f"  task '{k}': {v['target']}, classes={len(v['classes'])}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dr_regular")
