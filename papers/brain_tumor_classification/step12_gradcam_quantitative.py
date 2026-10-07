"""
Step 12: Quantitative Grad-CAM evaluation against Figshare tumor masks --
implements improvements.md item 4.

step5_gradcam.py only inspects Grad-CAM heatmaps visually, and only for
EfficientNetB0, on a dataset (Kaggle 4-class) that has no ground-truth
tumor masks -- so "the network attends to the tumor" is currently a
qualitative impression, not a measurement. Figshare (loaded in
step11_figshare_patient_cv.py) has a pixel-level tumor mask for every
slice, which makes two standard localization metrics possible:

  - Pointing-game accuracy: does the single hottest pixel of the Grad-CAM
    heatmap fall inside the ground-truth tumor mask? Reported as the
    fraction of test images where it does.
  - IoU: binarize the heatmap at a fixed percentile threshold, compute
    intersection-over-union against the tumor mask.

Both metrics are computed for the config-A (baseline MobileNetV2) and
config-E (proposed MSAI) models saved by step11
(/kaggle/working/figshare_{config}_fold{k}.keras), for every test fold.
If MSAI's attention genuinely helps (rather than just improving softmax
accuracy for unrelated reasons), its pointing-game accuracy and IoU
should be higher than the baseline's -- that is the evidence this script
is designed to produce.

Run after step11 has completed and saved its fold models. Needs the same
X, y, masks, pids arrays step11 builds -- this script reloads the raw
.mat files itself (via step11's load_dataset) rather than assuming
they're still in memory from a different session.
"""

import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupKFold

import sys
# `__file__` is undefined in a Jupyter/Kaggle cell, so don't rely on it here.
# On Kaggle: %%writefile both step10_msai_module.py and
# step11_figshare_patient_cv.py into /kaggle/working in earlier cells FIRST.
for _candidate_dir in ("/kaggle/working", os.getcwd(), os.path.dirname(os.path.abspath(sys.argv[0] or "."))):
    if os.path.isfile(os.path.join(_candidate_dir, "step11_figshare_patient_cv.py")):
        sys.path.insert(0, _candidate_dir)
        break
else:
    raise FileNotFoundError(
        "step11_figshare_patient_cv.py not found in /kaggle/working or the "
        "current directory. On Kaggle, run `%%writefile /kaggle/working/"
        "step11_figshare_patient_cv.py` (and step10_msai_module.py, which "
        "step11 itself imports) with their full contents pasted below, in "
        "earlier cells, before running this script."
    )
from step11_figshare_patient_cv import load_dataset, DATA_DIR, IMG_SIZE, N_FOLDS, SEED, CLASS_NAMES

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_DIR = "/kaggle/working"
RESULTS_PATH = "/kaggle/working/gradcam_quantitative.jsonl"
CONFIGS_TO_EVAL = ["A", "E"]
IOU_THRESHOLD_PERCENTILE = 85  # heatmap pixels above this percentile count as "attended"


# ---------------------------------------------------------------------------
# Grad-CAM (same derivation as step5, generalized to find the backbone
# submodel by scanning model.layers instead of hardcoding its name)
# ---------------------------------------------------------------------------
def find_backbone_layer(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    raise ValueError("No nested backbone sub-model found in this model.")


def make_gradcam_heatmap(img_array, model, backbone, pred_index=None):
    backbone_idx = model.layers.index(backbone)
    pre_layers = model.layers[1:backbone_idx]
    post_layers = model.layers[backbone_idx + 1:]

    x = img_array
    for layer in pre_layers:
        x = layer(x)

    with tf.GradientTape() as tape:
        conv_output = backbone(x, training=False)
        tape.watch(conv_output)
        h = conv_output
        for layer in post_layers:
            # MSAI models branch internally (multi-scale concat + attention);
            # those are captured inside the single functional graph already
            # baked into `model`, so replaying post_layers sequentially only
            # works for the plain baseline (config A). For config E, fall
            # back to slicing the full functional model instead.
            try:
                h = layer(h, training=False)
            except Exception:
                raise RuntimeError(
                    "Sequential post-layer replay failed -- this model has "
                    "a branching graph (MSAI config). Use "
                    "make_gradcam_heatmap_functional() instead."
                )
        predictions = h
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index)


def make_gradcam_heatmap_functional(img_array, model, backbone, pred_index=None):
    """General-purpose Grad-CAM for branching graphs (MSAI configs):
    build a sub-model from `inputs` to the backbone's output tensor AND
    the full model's output tensor in one Keras functional call, instead
    of manually replaying layers -- this works regardless of how many
    branches sit between the backbone and the final softmax."""
    backbone_output = backbone.output if hasattr(backbone, "output") else None
    grad_model = tf.keras.models.Model(
        model.inputs, [backbone.get_output_at(0) if backbone_output is None else backbone_output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array, training=False)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index)


# ---------------------------------------------------------------------------
# Localization metrics
# ---------------------------------------------------------------------------
def pointing_game_hit(heatmap_resized, mask):
    hottest = np.unravel_index(np.argmax(heatmap_resized), heatmap_resized.shape)
    return bool(mask[hottest] > 0)


def iou_score(heatmap_resized, mask, percentile=IOU_THRESHOLD_PERCENTILE):
    thresh = np.percentile(heatmap_resized, percentile)
    pred_region = heatmap_resized >= thresh
    gt_region = mask > 0
    intersection = np.logical_and(pred_region, gt_region).sum()
    union = np.logical_or(pred_region, gt_region).sum()
    return float(intersection / union) if union > 0 else 0.0


def append_result(row):
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    print("Loading Figshare dataset (image, label, PID, tumor mask)...")
    X, y, pids, masks, paths = load_dataset(DATA_DIR)

    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_splits = list(gkf.split(X, y, groups=pids))

    for config_name in CONFIGS_TO_EVAL:
        print(f"\n{'#' * 70}\nCONFIG {config_name}\n{'#' * 70}")
        pg_hits, ious = [], []

        for fold_idx, (_, test_idx) in enumerate(fold_splits):
            model_path = os.path.join(MODEL_DIR, f"figshare_{config_name}_fold{fold_idx + 1}.keras")
            if not os.path.exists(model_path):
                print(f"  [SKIP] {model_path} not found -- run step11 first.")
                continue
            model = tf.keras.models.load_model(model_path, compile=False)
            backbone = find_backbone_layer(model)

            fold_pg, fold_iou = [], []
            for idx in test_idx:
                img_array = np.expand_dims(X[idx], axis=0).astype(np.float32)
                try:
                    heatmap, pred_idx = make_gradcam_heatmap(img_array, model, backbone)
                except RuntimeError:
                    heatmap, pred_idx = make_gradcam_heatmap_functional(img_array, model, backbone)

                heatmap_resized = tf.image.resize(
                    heatmap[..., np.newaxis], (IMG_SIZE, IMG_SIZE)
                ).numpy().squeeze()

                hit = pointing_game_hit(heatmap_resized, masks[idx])
                iou = iou_score(heatmap_resized, masks[idx])
                fold_pg.append(hit)
                fold_iou.append(iou)

            fold_pg_acc = float(np.mean(fold_pg))
            fold_iou_mean = float(np.mean(fold_iou))
            pg_hits.extend(fold_pg)
            ious.extend(fold_iou)
            print(f"  Fold {fold_idx + 1}: pointing-game={fold_pg_acc:.4f}  "
                  f"mean IoU={fold_iou_mean:.4f}  (n={len(test_idx)})")

            append_result({
                "config": config_name, "fold": fold_idx + 1,
                "n_test": len(test_idx),
                "pointing_game_acc": round(fold_pg_acc, 4),
                "mean_iou": round(fold_iou_mean, 4),
            })

            del model
            tf.keras.backend.clear_session()

        if pg_hits:
            print(f"\n{config_name} OVERALL: pointing-game accuracy = "
                  f"{np.mean(pg_hits) * 100:.2f}%  mean IoU = {np.mean(ious):.4f}")

    print(f"\nDone. Per-fold rows in {RESULTS_PATH}. Compare config A vs E: "
          f"a higher pointing-game accuracy and IoU for E is the evidence "
          f"that MSAI's attention moves focus closer to the tumor, not just "
          f"that it improves softmax accuracy for unrelated reasons.")


if __name__ == "__main__":
    main()
