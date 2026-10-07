"""
Step 11: Patient-level validation on the Figshare brain tumor dataset --
implements improvements.md item 3.

Cheng's Figshare dataset (3,064 T1 contrast-enhanced MRI slices from 233
patients, 3 classes: glioma, meningioma, pituitary; no "no-tumor" class)
ships as .mat files, each with cjdata.image, cjdata.label, cjdata.PID
(patient ID), and cjdata.tumorMask. Unlike the Kaggle 4-class dataset this
paper's main results use, Figshare has real patient identifiers, so a
GroupKFold split grouped by PID gives a genuinely patient-level train/test
split -- directly answering the leakage concern raised in
brain_tumor_preprint_verified.tex's limitations section, instead of
leaving it as an untestable open concern.

IMPORTANT (do not misuse this dataset): Figshare is one of the three
public sources merged into the Kaggle 4-class dataset this paper's main
results use (the other two being SARTAJ and Br35H). Do NOT use Figshare
as an "external test set" for models trained on the Kaggle dataset --
the images would overlap with that dataset's training data, which would
make any "external validation" claim false. This script trains and
evaluates entirely within Figshare, as a separate, patient-level-grouped
experiment, reported alongside the main image-level results, not as a
replacement for them.

Runs the same MSAI-MobileNetV2 model (config E from step10) plus the
unmodified MobileNetV2 baseline (config A), for a direct before/after
comparison of patient-level accuracy -- this also doubles as the input to
step12's quantitative Grad-CAM evaluation, since Figshare's tumor masks
are what make that measurement possible.

Setup: pip install h5py scipy; download the Figshare archive (4 folders of
.mat files, brainTumorDataPublic_*) and point DATA_DIR at the folder
containing all *.mat files (flatten the 4 subfolders first, filenames are
unique across them).
"""

import os
import glob
import json
import time
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import sys
# `__file__` is undefined in a Jupyter/Kaggle cell, so don't rely on it here.
# On Kaggle: %%writefile /kaggle/working/step10_msai_module.py (paste its
# contents) in an earlier cell FIRST, so this import can find it.
for _candidate_dir in ("/kaggle/working", os.getcwd(), os.path.dirname(os.path.abspath(sys.argv[0] or "."))):
    if os.path.isfile(os.path.join(_candidate_dir, "step10_msai_module.py")):
        sys.path.insert(0, _candidate_dir)
        break
else:
    raise FileNotFoundError(
        "step10_msai_module.py not found in /kaggle/working or the current "
        "directory. On Kaggle, run `%%writefile /kaggle/working/"
        "step10_msai_module.py` with its full contents pasted below, in an "
        "earlier cell, before running this script."
    )
from step10_msai_module import (
    build_model, make_augmenter, tta_predict, BACKBONE_REGISTRY,
    IMG_SIZE as MODEL_IMG_SIZE,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = "/kaggle/input/figshare-brain-tumor/all_mats"
RESULTS_PATH = "/kaggle/working/figshare_patient_cv.jsonl"
IMG_SIZE = MODEL_IMG_SIZE
BATCH_SIZE = 16
HEAD_EPOCHS = 8
FINE_TUNE_EPOCHS = 35
FINE_TUNE_AT = 0.8
TTA_ROUNDS = 5
N_FOLDS = 5
SEED = 42
CLASS_NAMES = ["meningioma", "glioma", "pituitary"]  # Figshare label order: 1,2,3
NUM_CLASSES = len(CLASS_NAMES)
CROP_MARGIN = 16  # pixels of margin around the mask-derived bounding box
CONFIGS_TO_RUN = ["A", "E"]  # baseline vs. proposed MSAI module


# ---------------------------------------------------------------------------
# 1. Load Figshare .mat files
# ---------------------------------------------------------------------------
def load_mat_record(path):
    """Figshare's .mat files are HDF5-backed (v7.3), so h5py reads them
    directly without the scipy.io.loadmat MATLAB-struct gymnastics."""
    with h5py.File(path, "r") as f:
        cjdata = f["cjdata"]
        image = np.array(cjdata["image"]).astype(np.float32)
        label = int(np.array(cjdata["label"]).flatten()[0])  # 1, 2, or 3
        pid_raw = cjdata["PID"]
        pid = "".join(chr(c) for c in np.array(pid_raw).flatten()) if pid_raw.shape else str(np.array(pid_raw))
        mask = np.array(cjdata["tumorMask"]).astype(np.uint8)
    return image, label, pid, mask


def mask_bbox(mask, margin=CROP_MARGIN):
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        h, w = mask.shape
        return 0, h, 0, w
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin, mask.shape[0])
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin, mask.shape[1])
    return y0, y1, x0, x1


def load_dataset(data_dir):
    paths = sorted(glob.glob(os.path.join(data_dir, "*.mat")))
    if not paths:
        raise FileNotFoundError(f"No .mat files found under {data_dir}")
    N = len(paths)
    X = np.zeros((N, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    y = np.zeros(N, dtype=np.int64)
    pids = []
    masks_resized = np.zeros((N, IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    for i, p in enumerate(paths):
        image, label, pid, mask = load_mat_record(p)
        y0, y1, x0, x1 = mask_bbox(mask)
        cropped_img = image[y0:y1, x0:x1]
        cropped_mask = mask[y0:y1, x0:x1]

        img_norm = (cropped_img - cropped_img.min()) / (cropped_img.max() - cropped_img.min() + 1e-8)
        img_u8 = (img_norm * 255).astype(np.uint8)
        img_resized = tf.image.resize(img_u8[..., None], [IMG_SIZE, IMG_SIZE]).numpy().astype(np.uint8)
        X[i] = np.repeat(img_resized, 3, axis=-1)
        y[i] = label - 1  # 1,2,3 -> 0,1,2
        pids.append(pid)

        mask_resized = tf.image.resize(
            cropped_mask[..., None].astype(np.float32), [IMG_SIZE, IMG_SIZE], method="nearest"
        ).numpy().astype(np.uint8).squeeze(-1)
        masks_resized[i] = mask_resized

        if (i + 1) % 500 == 0 or (i + 1) == N:
            print(f"  loaded {i + 1}/{N}")

    return X, y, np.array(pids), masks_resized, paths


# ---------------------------------------------------------------------------
# 2. Training (reuses step10's build_model, head/fine-tune recipe)
# ---------------------------------------------------------------------------
def train_fold_model(backbone_name, config_name, X_train, y_train, X_val, y_val, class_weight, augmenter):
    train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
                .shuffle(1000, seed=SEED).batch(BATCH_SIZE)
                .map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE))
    val_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
              .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

    model, backbone, embedding_model = build_model(backbone_name, config_name)
    # NOTE: build_model's final Dense layer uses NUM_CLASSES from step10 (4).
    # Figshare has 3 classes, so rebuild the head here instead of importing
    # the 4-class model as-is.
    x = model.layers[-2].output  # last Dropout before the old 4-way Dense
    new_outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="figshare_head")(x)
    model = models.Model(model.input, new_outputs)

    model.compile(optimizer=optimizers.Adam(1e-3),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=HEAD_EPOCHS,
              class_weight=class_weight, verbose=2)

    backbone.trainable = True
    n_layers = len(backbone.layers)
    freeze_until = int(n_layers * (1 - FINE_TUNE_AT))
    for layer in backbone.layers[:freeze_until]:
        layer.trainable = False

    model.compile(optimizer=optimizers.Adam(1e-5),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=FINE_TUNE_EPOCHS,
              callbacks=cbs, class_weight=class_weight, verbose=2)
    return model


def append_result(row):
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# 3. Driver -- GroupKFold by patient ID
# ---------------------------------------------------------------------------
def main():
    print("Loading Figshare .mat files (image, label, PID, tumor mask)...")
    X, y, pids, masks, paths = load_dataset(DATA_DIR)
    n_patients = len(set(pids))
    print(f"Loaded {len(X)} slices from {n_patients} patients, "
          f"classes: {CLASS_NAMES}")

    gkf = GroupKFold(n_splits=N_FOLDS)
    augmenter = make_augmenter()

    for config_name in CONFIGS_TO_RUN:
        print(f"\n{'#' * 70}\nCONFIG {config_name}\n{'#' * 70}")
        fold_accs, fold_f1s = [], []
        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=pids)):
            print(f"\n{'=' * 70}\nFigshare/{config_name} -- FOLD {fold_idx + 1}/{N_FOLDS}\n{'=' * 70}")
            train_pids = set(pids[train_idx])
            test_pids = set(pids[test_idx])
            assert train_pids.isdisjoint(test_pids), "Patient overlap between train and test fold!"

            n_val = max(1, int(0.15 * len(train_idx)))
            rng = np.random.RandomState(SEED + fold_idx)
            perm = rng.permutation(len(train_idx))
            val_sub_idx = train_idx[perm[:n_val]]
            train_sub_idx = train_idx[perm[n_val:]]

            X_train, y_train = X[train_sub_idx], y[train_sub_idx]
            X_val, y_val = X[val_sub_idx], y[val_sub_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            class_weight = dict(enumerate(
                compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y_train)
            ))

            t0 = time.perf_counter()
            model = train_fold_model("mobilenetv2", config_name, X_train, y_train,
                                      X_val, y_val, class_weight, augmenter)
            train_time_s = time.perf_counter() - t0

            probs_tta = tta_predict(model, augmenter, X_test, rounds=TTA_ROUNDS, batch_size=BATCH_SIZE)
            y_pred = np.argmax(probs_tta, axis=1)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")
            fold_accs.append(acc)
            fold_f1s.append(f1)

            print(f"\nFold {fold_idx + 1} [{config_name}]: n_train={len(train_sub_idx)} "
                  f"n_val={len(val_sub_idx)} n_test={len(test_idx)} "
                  f"({len(test_pids)} held-out patients) "
                  f"acc={acc:.4f} macro_f1={f1:.4f} train_time={train_time_s:.1f}s")
            print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
            print(confusion_matrix(y_test, y_pred))

            append_result({
                "config": config_name, "fold": fold_idx + 1,
                "n_train": len(train_sub_idx), "n_val": len(val_sub_idx), "n_test": len(test_idx),
                "n_test_patients": len(test_pids), "train_time_s": round(train_time_s, 2),
                "acc": round(acc, 4), "macro_f1": round(f1, 4),
            })

            model.save(f"/kaggle/working/figshare_{config_name}_fold{fold_idx + 1}.keras")

            del model
            tf.keras.backend.clear_session()

        accs = np.array(fold_accs)
        print(f"\n{config_name} patient-level 5-fold accuracy: "
              f"{accs.mean() * 100:.2f}% +/- {accs.std() * 100:.2f}%")

    print(f"\nDone. Per-fold rows in {RESULTS_PATH}. Saved fold models "
          f"(/kaggle/working/figshare_*_fold*.keras) are reused by "
          f"step12_gradcam_quantitative.py for the pointing-game/IoU "
          f"evaluation against these same tumor masks.")


if __name__ == "__main__":
    main()
