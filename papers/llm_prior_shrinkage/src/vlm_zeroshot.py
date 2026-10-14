"""Zero-shot breast cancer classification with a LOCAL generative VLM (via Ollama).

This is a different kind of model from BiomedCLIP. BiomedCLIP is a contrastive
vision-language model: it scores image-text similarity. qwen2.5vl / moondream are
GENERATIVE VLMs: they look at the image and answer in words.

Published expectation to test against: Qwen2.5-VL-3B scores ~35.6% on medical diagnosis
benchmarks, so we should expect weak results. The point is to measure it on OUR task
rather than assume, and to compare against the majority-class baseline, which is the
number that actually matters.
"""
import base64, io, json, os, sys, time
import numpy as np, h5py, yaml
from PIL import Image

OLLAMA = "http://127.0.0.1:11434/api/generate"
D = "/mnt/extra-storage/data/medimeta_extracted"


def img_b64(arr):
    im = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO(); im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def ask(model, prompt, b64, schema, timeout=600):
    import subprocess
    pl = {"model": model, "prompt": prompt, "images": [b64], "stream": False,
          "format": schema, "options": {"temperature": 0, "num_predict": 120}}
    t0 = time.time()
    p = subprocess.run(["curl", "-s", OLLAMA, "-d", json.dumps(pl)],
                       capture_output=True, text=True, timeout=timeout)
    dt = time.time() - t0
    try:
        return json.loads(json.loads(p.stdout)["response"]), dt
    except Exception as e:
        return {"__error__": str(e)[:120]}, dt


PROMPTS = {
 "bus": ("This is a breast ultrasound image. Does it show a MALIGNANT (cancerous) lesion, "
         "or is it BENIGN / normal (not cancer)?\n"
         "Answer with JSON: {\"label\": \"malignant\" or \"benign\"}"),
 "mammo_mass": ("This is a mammogram showing a breast mass. Is the mass MALIGNANT "
                "(cancerous) or BENIGN (not cancer)?\n"
                "Answer with JSON: {\"label\": \"malignant\" or \"benign\"}"),
 "mammo_calc": ("This is a mammogram showing breast calcifications. Are they MALIGNANT "
                "(cancerous) or BENIGN (not cancer)?\n"
                "Answer with JSON: {\"label\": \"malignant\" or \"benign\"}"),
}
SCHEMA = {"type": "object",
          "properties": {"label": {"type": "string", "enum": ["malignant", "benign"]}},
          "required": ["label"]}


def run(dataset_id, task, model="qwen2.5vl:3b", n_test=None):
    sp = {s: np.array([int(os.path.splitext(os.path.basename(l.strip()))[0])
                       for l in open(f"{D}/{dataset_id}/splits/{s}.txt") if l.strip()])
          for s in ("test",)}
    y = np.load(f"{D}/{dataset_id}/task_labels/{task}.npy").astype(int).ravel()
    te = sp["test"]
    if n_test:
        rng = np.random.default_rng(0)
        te = rng.choice(te, size=min(n_test, len(te)), replace=False)
    yq = y[te]
    # binary: 1 == malignant in all three of these tasks
    maj = np.bincount(yq).argmax(); majacc = (yq == maj).mean()

    with h5py.File(f"{D}/{dataset_id}/images.hdf5", "r") as f:
        imgs = f["images"]
        preds, errs, secs = [], 0, 0.0
        for i, idx in enumerate(te):
            o, dt = ask(model, PROMPTS[dataset_id], img_b64(np.asarray(imgs[idx])), SCHEMA)
            secs += dt
            if "__error__" in o or "label" not in o:
                preds.append(-1); errs += 1
            else:
                preds.append(1 if o["label"] == "malignant" else 0)
            if (i + 1) % 20 == 0:
                done = np.array(preds); m = done >= 0
                acc = (done[m] == yq[:len(done)][m]).mean() if m.any() else float("nan")
                print(f"  {i+1}/{len(te)}  running acc={acc:.3f}  "
                      f"{secs/(i+1):.1f}s/img  errors={errs}", flush=True)

    preds = np.array(preds); valid = preds >= 0
    acc = (preds[valid] == yq[valid]).mean() if valid.any() else float("nan")
    # how often does it just say one class?
    frac_malig = (preds[valid] == 1).mean() if valid.any() else float("nan")
    out = {"dataset": dataset_id, "task": task, "model": model,
           "n_test": int(len(te)), "n_valid": int(valid.sum()), "n_parse_errors": int(errs),
           "accuracy": float(acc), "majority_baseline": float(majacc),
           "delta_vs_majority": float(acc - majacc),
           "fraction_predicted_malignant": float(frac_malig),
           "true_fraction_malignant": float((yq == 1).mean()),
           "secs_per_image": round(secs / max(len(te), 1), 2)}
    print(f"\n{model} on {dataset_id}/{task}")
    print(f"  accuracy            {acc:.3f}")
    print(f"  majority baseline   {majacc:.3f}")
    print(f"  delta               {acc-majacc:+.3f}")
    print(f"  predicted malignant {frac_malig:.3f}  (true rate {(yq==1).mean():.3f})")
    print(f"  parse errors        {errs}/{len(te)}   {out['secs_per_image']}s per image")
    os.makedirs("results", exist_ok=True)
    path = f"results/vlm_{model.replace(':','_')}_{dataset_id}_{task.replace(' ','_')}.json"
    json.dump(out, open(path, "w"), indent=2); print(f"  wrote {path}")
    return out


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "bus"
    task = sys.argv[2] if len(sys.argv) > 2 else "malignancy"
    model = sys.argv[3] if len(sys.argv) > 3 else "qwen2.5vl:3b"
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 60
    run(ds, task, model, n_test=n)
