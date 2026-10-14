"""Phase 0.3 - CPU feature-extraction throughput for BiomedCLIP ViT-B/16."""
import os, time, json
os.environ.setdefault("HF_HOME", "/mnt/extra-storage/hf_cache")
import torch, open_clip

torch.set_num_threads(8)
MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

t0 = time.time()
model, _, preprocess = open_clip.create_model_and_transforms(MODEL)
tokenizer = open_clip.get_tokenizer(MODEL)
model.eval()
load_s = time.time() - t0
print(f"model loaded in {load_s:.1f}s", flush=True)

n_params = sum(p.numel() for p in model.parameters())
print(f"params: {n_params/1e6:.1f}M", flush=True)

res = {"load_s": round(load_s, 1), "params_M": round(n_params / 1e6, 1),
       "threads": torch.get_num_threads(), "image": {}, "text": {}}

# ---- image encoder throughput ----
for bs in (1, 8, 32):
    x = torch.randn(bs, 3, 224, 224)
    with torch.no_grad():                      # warmup
        model.encode_image(x)
    reps = max(1, 32 // bs)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(reps):
            f = model.encode_image(x)
    dt = time.time() - t0
    ips = bs * reps / dt
    res["image"][f"bs{bs}"] = round(ips, 2)
    print(f"  image bs={bs:<3} {ips:6.2f} img/s   (feat dim {f.shape[-1]})", flush=True)

# ---- text encoder throughput ----
prompts = ["this is a photo of a benign breast tumor"] * 32
with torch.no_grad():
    tk = tokenizer(prompts)
    model.encode_text(tk[:4])
    t0 = time.time()
    tf = model.encode_text(tk)
dt = time.time() - t0
res["text"]["per_s"] = round(32 / dt, 2)
res["feat_dim"] = int(tf.shape[-1])
print(f"  text          {32/dt:6.2f} prompt/s  (feat dim {tf.shape[-1]})", flush=True)

# ---- projected cost for the full experiment grid ----
best = max(res["image"].values())
for n in (10_000, 50_000, 200_000):
    print(f"  => {n:>7,} images at {best:.1f} img/s = {n/best/3600:5.2f} h", flush=True)
res["est_hours_50k"] = round(50_000 / best / 3600, 2)

json.dump(res, open("phase0/bench_cpu.json", "w"), indent=2)
print("\nGATE: need >= 5 img/s ->", "PASS" if best >= 5 else "FAIL", f"({best:.2f} img/s)")
