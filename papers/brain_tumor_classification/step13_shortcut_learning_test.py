"""
Step 13: Shortcut-learning stress test -- implements improvements.md
item 5.

step8_eda.py's EDA showed that no-tumor images skew brighter and come
from a different source (Br35H) than the three tumor classes. A model
could be partly "cheating" by learning overall image brightness/contrast
as a proxy for the no-tumor class, rather than learning tumor-relevant
anatomy -- a classic shortcut-learning failure mode, and one that would
inflate the headline accuracy without indicating real diagnostic skill.

This script tests for that directly: it histogram-matches every image
(train AND test) to a single common reference intensity distribution
(skimage.exposure.match_histograms) BEFORE classification, which removes
the brightness/contrast cue that differs by source dataset while leaving
tumor-relevant structure intact. It then re-evaluates a trained model
(same recipe as step4/step6) on the histogram-matched test set.

Two things to look for:
  1. If accuracy holds roughly steady after histogram matching, the model
     is not primarily relying on the brightness shortcut.
  2. If accuracy (especially no-tumor recall specifically) drops sharply,
     the brightness/source cue was doing real work, and this should be
     reported as a limitation rather than buried.

Also reports the reverse control: per-class mean intensity before and
after matching, to confirm the matching step actually removed the
brightness gap step8_eda.py found (a sanity check mirroring the dedup
audit's own sanity check in step9).

Run on Kaggle (GPU enabled, same protocol as step4/step6).
"""

import os
import gc
import glob
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from skimage.exposure import match_histograms

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = "/kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset"
RESULTS_PATH = "/kaggle/working/shortcut_test.json"
IMG_SIZE = 224
BATCH_SIZE = 16
HEAD_EPOCHS = 8
FINE_TUNE_EPOCHS = 35
FINE_TUNE_AT = 0.8
SEED = 42
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASS_NAMES)


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------
def list_image_paths(data_dir):
    paths, labels = [], []
    for split_dir in ["Training", "Testing"]:
        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = os.path.join(data_dir, split_dir, class_name)
            files = sorted(glob.glob(os.path.join(class_dir, "*")))
            paths.extend(files)
            labels.extend([class_idx] * len(files))
    return paths, np.array(labels)


def load_and_preprocess(paths):
    n = len(paths)
    X = np.zeros((n, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    for i, p in enumerate(paths):
        img = tf.io.read_file(p)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        X[i] = img.numpy().astype(np.uint8)
        if (i + 1) % 1000 == 0 or (i + 1) == n:
            print(f"  processed {i + 1}/{n}")
    return X


def histogram_match_all(X, reference):
    """Match every image's intensity histogram to a single fixed reference
    image, per-channel. Using ONE fixed reference (rather than matching
    each image to the dataset mean, which would still correlate with
    source/class) removes per-class/per-source brightness differences as
    a usable cue."""
    out = np.zeros_like(X)
    for i in range(len(X)):
        out[i] = np.clip(match_histograms(X[i], reference, channel_axis=-1), 0, 255).astype(np.uint8)
        if (i + 1) % 1000 == 0 or (i + 1) == len(X):
            print(f"  histogram-matched {i + 1}/{len(X)}")
    return out


# ---------------------------------------------------------------------------
# 2. Model (same recipe as step4/step6, single EfficientNetB0 split --
#    this is a diagnostic, not the paper's headline CV number)
# ---------------------------------------------------------------------------
def build_model():
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    backbone.trainable = False
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return models.Model(inputs, outputs), backbone


def make_augmenter():
    return tf.keras.Sequential([
        layers.RandomRotation(0.05), layers.RandomTranslation(0.1, 0.1),
        layers.RandomZoom(0.1), layers.RandomFlip("horizontal"), layers.RandomContrast(0.1),
    ])


def train_model(X_train, y_train, X_val, y_val, class_weight):
    augmenter = make_augmenter()
    train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
                .shuffle(1000, seed=SEED).batch(BATCH_SIZE)
                .map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE))
    val_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
              .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

    model, backbone = build_model()
    model.compile(optimizer=optimizers.Adam(1e-3),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=HEAD_EPOCHS,
              class_weight=class_weight, verbose=2)

    backbone.trainable = True
    freeze_until = int(len(backbone.layers) * (1 - FINE_TUNE_AT))
    for layer in backbone.layers[:freeze_until]:
        layer.trainable = False
    model.compile(optimizer=optimizers.Adam(1e-5),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cbs = [callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
           callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4)]
    model.fit(train_ds, validation_data=val_ds, epochs=FINE_TUNE_EPOCHS,
              callbacks=cbs, class_weight=class_weight, verbose=2)
    return model


# ---------------------------------------------------------------------------
# 3. Driver
# ---------------------------------------------------------------------------
def main():
    print("Listing + loading images...")
    paths, y = list_image_paths(DATA_DIR)
    X = load_and_preprocess(paths)
    gc.collect()

    print("\nPer-class mean intensity BEFORE histogram matching:")
    mean_intensity_before = {}
    for class_idx, class_name in enumerate(CLASS_NAMES):
        m = float(X[y == class_idx].mean())
        mean_intensity_before[class_name] = round(m, 2)
        print(f"  {class_name:12s}: {m:.2f}")

    trainval_idx, test_idx = train_test_split(np.arange(len(X)), test_size=0.2,
                                               random_state=SEED, stratify=y)
    train_idx, val_idx = train_test_split(trainval_idx, test_size=0.15,
                                           random_state=SEED, stratify=y[trainval_idx])
    class_weight = dict(enumerate(
        compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y[train_idx])
    ))

    print("\nTraining model on ORIGINAL (unmatched) images...")
    model = train_model(X[train_idx], y[train_idx], X[val_idx], y[val_idx], class_weight)

    y_pred_orig = np.argmax(model.predict(X[test_idx], batch_size=BATCH_SIZE, verbose=0), axis=1)
    acc_orig = accuracy_score(y[test_idx], y_pred_orig)
    f1_orig = f1_score(y[test_idx], y_pred_orig, average="macro")
    report_orig = classification_report(y[test_idx], y_pred_orig, target_names=CLASS_NAMES, output_dict=True)
    print(f"\nOriginal test accuracy: {acc_orig:.4f}  macro-F1: {f1_orig:.4f}")

    print("\nHistogram-matching the test set to a single fixed reference image...")
    reference = X[train_idx][0]  # any single training image fixes the target distribution
    X_test_matched = histogram_match_all(X[test_idx], reference)

    print("\nPer-class mean intensity AFTER histogram matching (test set only):")
    mean_intensity_after = {}
    for class_idx, class_name in enumerate(CLASS_NAMES):
        mask = y[test_idx] == class_idx
        if mask.sum() == 0:
            continue
        m = float(X_test_matched[mask].mean())
        mean_intensity_after[class_name] = round(m, 2)
        print(f"  {class_name:12s}: {m:.2f}")

    y_pred_matched = np.argmax(model.predict(X_test_matched, batch_size=BATCH_SIZE, verbose=0), axis=1)
    acc_matched = accuracy_score(y[test_idx], y_pred_matched)
    f1_matched = f1_score(y[test_idx], y_pred_matched, average="macro")
    report_matched = classification_report(y[test_idx], y_pred_matched, target_names=CLASS_NAMES, output_dict=True)
    print(f"\nHistogram-matched test accuracy: {acc_matched:.4f}  macro-F1: {f1_matched:.4f}")

    drop = acc_orig - acc_matched
    notumor_recall_orig = report_orig["notumor"]["recall"]
    notumor_recall_matched = report_matched["notumor"]["recall"]
    print(f"\nAccuracy drop after removing the brightness cue: {drop * 100:.2f} points")
    print(f"no-tumor recall: {notumor_recall_orig:.4f} -> {notumor_recall_matched:.4f}")
    if drop > 0.03 or (notumor_recall_orig - notumor_recall_matched) > 0.05:
        print("[FINDING] Non-trivial drop -- brightness/source was likely "
              "contributing a real shortcut cue; report this as a limitation.")
    else:
        print("[FINDING] Accuracy held steady -- model does not appear to "
              "rely primarily on the brightness/source shortcut.")

    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "mean_intensity_before": mean_intensity_before,
            "mean_intensity_after_test_only": mean_intensity_after,
            "acc_original": round(acc_orig, 4), "macro_f1_original": round(f1_orig, 4),
            "acc_histogram_matched": round(acc_matched, 4), "macro_f1_histogram_matched": round(f1_matched, 4),
            "accuracy_drop": round(drop, 4),
            "notumor_recall_original": round(notumor_recall_orig, 4),
            "notumor_recall_histogram_matched": round(notumor_recall_matched, 4),
        }, f, indent=2)
    print(f"\nResults written to {RESULTS_PATH}.")


if __name__ == "__main__":
    main()
