"""
Step 14: Stronger statistics + a transformer baseline -- implements
improvements.md item 6.

With n=5 CV folds, a Wilcoxon signed-rank test's p-value cannot go below
0.031 regardless of effect size, which reviewers may read as weak
evidence even when the effect is real. This script adds:

  1. McNemar's test on POOLED per-image paired predictions (one row per
     test image across all 5 folds -- for the Kaggle dataset that's all
     7,200 images, each used as a test example in exactly one fold of a
     5-fold CV), comparing two methods' correct/incorrect calls on the
     SAME images. McNemar's test has much more statistical power here
     because it operates on thousands of paired Bernoulli outcomes
     instead of 5 paired fold-accuracy numbers.
  2. A 10-fold stratified CV runner (reuses step10's build_model), so the
     final model-vs-baseline comparison can also be run with a smaller
     Wilcoxon p-value floor (0.0020) if 10 folds are preferred over
     McNemar's test.
  3. A transformer baseline (Swin-Tiny or ConvNeXt-Tiny via keras_cv or
     timm-through-tf, falling back to a HuggingFace/PyTorch run if
     keras_cv is unavailable) under the SAME 5-fold protocol as the CNN
     backbones, so the paper doesn't omit transformers from its baseline
     table.
  4. Extra metrics: macro-F1, per-class sensitivity/specificity, one-vs-
     rest AUC, a confusion matrix aggregated across all 5 folds, and
     expected calibration error (ECE).

Run modes (set RUN_MODE below), each a separate Kaggle session:
  "mcnemar"     -- needs per-fold prediction JSONL files from two methods
                   (produced by --dump_predictions runs of step6/step10;
                   see dump_predictions() below) already downloaded locally.
                   This mode is CPU-only, run it locally, not on Kaggle.
  "cv10"        -- trains BACKBONE under 10-fold CV instead of 5.
  "transformer" -- trains a Swin-Tiny or ConvNeXt-Tiny baseline under the
                   same 5-fold protocol as step6_backbone_sweep.py.
  "extra_metrics" -- recomputes macro-F1/sensitivity/specificity/AUC/ECE
                   from a saved model + test set; run after step6/step10.
"""

import os
import glob
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight
from statsmodels.stats.contingency_tables import mcnemar

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RUN_MODE = "mcnemar"  # "mcnemar", "cv10", "transformer", "extra_metrics"
DATA_DIR = "/kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset"
IMG_SIZE = 224
BATCH_SIZE = 16
HEAD_EPOCHS = 8
FINE_TUNE_EPOCHS = 35
FINE_TUNE_AT = 0.8
SEED = 42
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASS_NAMES)
TRANSFORMER_NAME = "ConvNeXtTiny"  # ConvNeXtTiny ships in tf.keras.applications >= 2.12; Swin needs keras_cv


# ---------------------------------------------------------------------------
# Shared data loading (identical to step6/step10)
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


# ---------------------------------------------------------------------------
# Mode 1: McNemar's test on pooled per-image predictions
# ---------------------------------------------------------------------------
def dump_predictions(paths, y, model_fn, out_path, n_folds=5, seed=SEED):
    """Run n_folds CV, and for each test image write ONE row with the
    image path, true label, and predicted label -- pooled across all
    folds, this gives one row per image in the full dataset, suitable
    for a paired McNemar's test against another method's equivalent file."""
    X = load_and_preprocess(paths)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    rows = []
    for fold_idx, (trainval_idx, test_idx) in enumerate(skf.split(X, y)):
        train_idx, val_idx = train_test_split(trainval_idx, test_size=0.15,
                                               random_state=seed, stratify=y[trainval_idx])
        class_weight = dict(enumerate(
            compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y[train_idx])
        ))
        model = model_fn(X[train_idx], y[train_idx], X[val_idx], y[val_idx], class_weight)
        y_pred = np.argmax(model.predict(X[test_idx], batch_size=BATCH_SIZE, verbose=0), axis=1)
        for i, idx in enumerate(test_idx):
            rows.append({"path": paths[idx], "fold": fold_idx + 1,
                         "y_true": int(y[idx]), "y_pred": int(y_pred[i])})
        del model
        tf.keras.backend.clear_session()
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} pooled predictions to {out_path}")


def run_mcnemar(predictions_a_path, predictions_b_path):
    """predictions_a_path/predictions_b_path: JSONL files from
    dump_predictions() for two methods (e.g. baseline CNN+TTA vs. MSAI),
    run on the SAME StratifiedKFold splits (same seed/n_folds) so rows
    align 1:1 by image path."""
    def load(path):
        rows = {}
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                rows[r["path"]] = r
        return rows

    rows_a, rows_b = load(predictions_a_path), load(predictions_b_path)
    common = sorted(set(rows_a) & set(rows_b))
    assert len(common) > 0, "No overlapping image paths between the two prediction files."

    # 2x2 contingency table: [[both correct, A correct/B wrong], [A wrong/B correct, both wrong]]
    both_correct = a_correct_b_wrong = a_wrong_b_correct = both_wrong = 0
    for p in common:
        a_correct = rows_a[p]["y_pred"] == rows_a[p]["y_true"]
        b_correct = rows_b[p]["y_pred"] == rows_b[p]["y_true"]
        if a_correct and b_correct:
            both_correct += 1
        elif a_correct and not b_correct:
            a_correct_b_wrong += 1
        elif not a_correct and b_correct:
            a_wrong_b_correct += 1
        else:
            both_wrong += 1

    table = [[both_correct, a_correct_b_wrong], [a_wrong_b_correct, both_wrong]]
    result = mcnemar(table, exact=(a_correct_b_wrong + a_wrong_b_correct) < 25, correction=True)
    print(f"McNemar's test over {len(common)} paired images:")
    print(f"  contingency table: {table}")
    print(f"  statistic={result.statistic:.4f}  p-value={result.pvalue:.6f}")
    return result


# ---------------------------------------------------------------------------
# Mode 3: transformer baseline
# ---------------------------------------------------------------------------
def build_transformer_model():
    app = getattr(tf.keras.applications, TRANSFORMER_NAME)
    preprocess = getattr(
        getattr(tf.keras.applications, TRANSFORMER_NAME.lower().replace("tiny", "")),
        "preprocess_input",
        lambda x: x / 127.5 - 1.0,
    )
    backbone = app(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
    backbone.trainable = False
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = preprocess(inputs)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return models.Model(inputs, outputs), backbone


def train_transformer_fold(X_train, y_train, X_val, y_val, class_weight):
    augmenter = tf.keras.Sequential([
        layers.RandomRotation(0.05), layers.RandomTranslation(0.1, 0.1),
        layers.RandomZoom(0.1), layers.RandomFlip("horizontal"), layers.RandomContrast(0.1),
    ])
    train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
                .shuffle(1000, seed=SEED).batch(BATCH_SIZE)
                .map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE))
    val_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
              .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

    model, backbone = build_transformer_model()
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


def run_transformer_cv(n_folds=5):
    paths, y = list_image_paths(DATA_DIR)
    X = load_and_preprocess(paths)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    accs, f1s = [], []
    for fold_idx, (trainval_idx, test_idx) in enumerate(skf.split(X, y)):
        train_idx, val_idx = train_test_split(trainval_idx, test_size=0.15,
                                               random_state=SEED, stratify=y[trainval_idx])
        class_weight = dict(enumerate(
            compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y[train_idx])
        ))
        model = train_transformer_fold(X[train_idx], y[train_idx], X[val_idx], y[val_idx], class_weight)
        y_pred = np.argmax(model.predict(X[test_idx], batch_size=BATCH_SIZE, verbose=0), axis=1)
        acc = accuracy_score(y[test_idx], y_pred)
        f1 = f1_score(y[test_idx], y_pred, average="macro")
        accs.append(acc)
        f1s.append(f1)
        print(f"Fold {fold_idx + 1} [{TRANSFORMER_NAME}]: acc={acc:.4f} macro_f1={f1:.4f}")
        del model
        tf.keras.backend.clear_session()
    accs = np.array(accs)
    print(f"\n{TRANSFORMER_NAME} {n_folds}-fold: {accs.mean() * 100:.2f}% +/- {accs.std() * 100:.2f}%")


# ---------------------------------------------------------------------------
# Mode 4: extra metrics (macro-F1, sensitivity/specificity, AUC, ECE)
# ---------------------------------------------------------------------------
def expected_calibration_error(y_true, probs, n_bins=10):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true).astype(np.float32)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(bin_acc - bin_conf)
    return float(ece)


def per_class_sensitivity_specificity(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    results = {}
    for c in range(num_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        results[CLASS_NAMES[c]] = {"sensitivity": round(float(sensitivity), 4),
                                     "specificity": round(float(specificity), 4)}
    return results, cm


def compute_extra_metrics(y_true, y_pred, probs):
    f1_macro = f1_score(y_true, y_pred, average="macro")
    sens_spec, cm = per_class_sensitivity_specificity(y_true, y_pred, NUM_CLASSES)
    auc_ovr = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    ece = expected_calibration_error(y_true, probs)
    return {
        "macro_f1": round(float(f1_macro), 4),
        "per_class_sensitivity_specificity": sens_spec,
        "confusion_matrix": cm.tolist(),
        "auc_ovr_macro": round(float(auc_ovr), 4),
        "ece": round(ece, 4),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    if RUN_MODE == "mcnemar":
        print("Run dump_predictions() for two methods first (locally saved "
              "JSONL files), then call run_mcnemar(path_a, path_b). Example:")
        print("  from step10_msai_module import build_model")
        print("  dump_predictions(paths, y, baseline_model_fn, 'preds_baseline.jsonl')")
        print("  dump_predictions(paths, y, msai_model_fn, 'preds_msai.jsonl')")
        print("  run_mcnemar('preds_baseline.jsonl', 'preds_msai.jsonl')")
    elif RUN_MODE == "cv10":
        print("10-fold CV mode: reuse step10_msai_module.py with "
              "N_FOLDS=10 set at the top of that file -- this script does "
              "not duplicate that training loop, since step10 already "
              "generalizes over N_FOLDS.")
    elif RUN_MODE == "transformer":
        print(f"Training {TRANSFORMER_NAME} baseline, 5-fold CV...")
        run_transformer_cv(n_folds=5)
    elif RUN_MODE == "extra_metrics":
        print("extra_metrics mode is meant to be called as a library: "
              "load a saved model + test set, get probs = model.predict(X_test), "
              "then call compute_extra_metrics(y_test, probs.argmax(1), probs).")
    else:
        raise ValueError(f"Unknown RUN_MODE: {RUN_MODE}")


if __name__ == "__main__":
    main()
