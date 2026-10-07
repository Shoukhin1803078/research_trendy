"""
Step 10: Multi-Scale Attention Injection (MSAI) module -- implements
improvements.md item 1, the paper's proposed, backbone-agnostic
contribution.

Design note / honest constraint: the four backbones used throughout this
paper (EfficientNetB0, ResNet50, DenseNet121, MobileNetV2) are loaded as
single opaque Keras Application submodels with ImageNet weights, which is
what lets step6_backbone_sweep.py swap backbones by changing one registry
entry. Reaching INSIDE a pretrained Application to splice injection points
at several internal resolutions (as a from-scratch architecture could)
would mean rewriting each backbone's internals by hand and losing the
"drop in any of four stock Applications" property that makes the
backbone-agnostic claim possible. MSAI therefore injects multi-scale
context at the OUTPUT of the backbone, not at intermediate stages:

    - The full-resolution image goes through the backbone as normal,
      producing its final conv feature map.
    - The image, downsampled to 1/2 and 1/4 resolution then upsampled back
      to the backbone's feature-map spatial size, each goes through a
      small trainable conv stem (not the backbone) to produce two
      auxiliary feature maps at matching channel count.
    - All three feature maps are concatenated channel-wise -- this is the
      "multi-scale injection" (configs B, E, F below).
    - CBAM is applied to the concatenated tensor: channel attention
      (configs C, E) followed by spatial attention (configs D, E).
    - The existing classification head (GAP, BN, Dropout, Dense-128
      embedding, BN, Dropout, softmax) is unchanged, so the CNN-embedding
      ensemble and TTA code from step4/step6 still apply unmodified.

This is a legitimate multi-scale-context + attention module and is fairly
described as "backbone-agnostic" since it only touches each backbone's
output tensor, but reviewers should not be told it injects at multiple
INTERNAL backbone resolutions -- it does not. Mention this explicitly in
the methods section.

Ablation configs (run on MobileNetV2 -- fastest backbone, ~23 min/fold):
    A: baseline (no MSAI; identical to step6's mobilenetv2 result)
    B: + multi-scale injection only (no attention)
    C: + channel attention only (CBAM channel branch, single-scale input)
    D: + spatial attention only (CBAM spatial branch, single-scale input)
    E: + full MSAI (multi-scale injection + CBAM channel + spatial) [proposed]
    F: + full MSAI but with ECA instead of CBAM channel attention

Then rerun config E only on resnet50, densenet121, efficientnetb0.

Protocol (5-fold stratified CV, two-stage fine-tuning, TTA) and output
format mirror step6_backbone_sweep.py exactly, with an added "config"
field and parameter/FLOP counts so the paper's ablation table (accuracy,
macro-F1, params, FLOPs, latency, Wilcoxon p vs. config A) can be built
directly from the JSONL rows. Each row also reports macro-F1 (item 6).

Run ONE (backbone, config) pair per Kaggle session -- same reasoning as
step6 about the ~12h session limit. Set RUN_MODE below.
"""

import os
import gc
import glob
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ---------------------------------------------------------------------------
# Config -- EDIT THIS before each Kaggle session
# ---------------------------------------------------------------------------
RUN_MODE = "ablation"        # "ablation" (configs A-F on MobileNetV2) or "cross_backbone" (config E on other backbones)
ABLATION_CONFIG = "A"        # one of A, B, C, D, E, F -- only used when RUN_MODE == "ablation"
CROSS_BACKBONE = "resnet50"  # one of resnet50, densenet121, efficientnetb0 -- only used when RUN_MODE == "cross_backbone"

DATA_DIR = "/kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset"
RESULTS_PATH = "/kaggle/working/msai_ablation.jsonl"
IMG_SIZE = 224
BATCH_SIZE = 16
HEAD_EPOCHS = 8
FINE_TUNE_EPOCHS = 35
FINE_TUNE_AT = 0.8
TTA_ROUNDS = 5
N_FOLDS = 5
SEED = 42
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASS_NAMES)
MSAI_CHANNELS = 64  # channel width of each auxiliary conv stem and the CBAM bottleneck

BACKBONE_REGISTRY = {
    "mobilenetv2": dict(app=tf.keras.applications.MobileNetV2,
                         preprocess=tf.keras.applications.mobilenet_v2.preprocess_input),
    "resnet50": dict(app=tf.keras.applications.ResNet50,
                      preprocess=tf.keras.applications.resnet50.preprocess_input),
    "densenet121": dict(app=tf.keras.applications.DenseNet121,
                         preprocess=tf.keras.applications.densenet.preprocess_input),
    "efficientnetb0": dict(app=tf.keras.applications.EfficientNetB0,
                            preprocess=tf.keras.applications.efficientnet.preprocess_input),
}

CONFIG_FLAGS = {
    # (use_multiscale, use_channel_attn, use_spatial_attn, attn_type)
    "A": dict(multiscale=False, channel_attn=False, spatial_attn=False, attn_type="cbam"),
    "B": dict(multiscale=True, channel_attn=False, spatial_attn=False, attn_type="cbam"),
    "C": dict(multiscale=False, channel_attn=True, spatial_attn=False, attn_type="cbam"),
    "D": dict(multiscale=False, channel_attn=False, spatial_attn=True, attn_type="cbam"),
    "E": dict(multiscale=True, channel_attn=True, spatial_attn=True, attn_type="cbam"),
    "F": dict(multiscale=True, channel_attn=True, spatial_attn=True, attn_type="eca"),
}


# ---------------------------------------------------------------------------
# 1. Data loading (identical to step6)
# ---------------------------------------------------------------------------
def list_image_paths(data_dir):
    paths, labels = [], []
    for split_dir in ["Training", "Testing"]:
        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = os.path.join(data_dir, split_dir, class_name)
            files = sorted(glob.glob(os.path.join(class_dir, "*")))
            paths.extend(files)
            labels.extend([class_idx] * len(files))
    if not paths:
        raise FileNotFoundError(f"No images found under {data_dir}")
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
# 2. Attention blocks
# ---------------------------------------------------------------------------
def channel_attention_cbam(x, reduction=8, name="ch_attn"):
    channels = x.shape[-1]
    avg_pool = layers.GlobalAveragePooling2D()(x)
    max_pool = layers.GlobalMaxPooling2D()(x)
    shared_dense1 = layers.Dense(max(channels // reduction, 8), activation="relu", name=f"{name}_fc1")
    shared_dense2 = layers.Dense(channels, name=f"{name}_fc2")
    avg_out = shared_dense2(shared_dense1(avg_pool))
    max_out = shared_dense2(shared_dense1(max_pool))
    scale = layers.Activation("sigmoid")(layers.Add()([avg_out, max_out]))
    scale = layers.Reshape((1, 1, channels))(scale)
    return layers.Multiply()([x, scale])


def spatial_attention_cbam(x, kernel_size=7, name="sp_attn"):
    avg_pool = layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True))(x)
    max_pool = layers.Lambda(lambda t: tf.reduce_max(t, axis=-1, keepdims=True))(x)
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
    scale = layers.Conv2D(1, kernel_size, padding="same", activation="sigmoid", name=f"{name}_conv")(concat)
    return layers.Multiply()([x, scale])


def eca_attention(x, gamma=2, b=1, name="eca_attn"):
    """Efficient Channel Attention: a single 1D conv over channel-pooled
    descriptor, no dimensionality reduction (config F, in place of CBAM's
    channel branch)."""
    channels = x.shape[-1]
    k = int(abs((np.log2(channels) + b) / gamma))
    k = k if k % 2 else k + 1
    k = max(k, 3)
    pooled = layers.GlobalAveragePooling2D()(x)
    pooled = layers.Reshape((channels, 1))(pooled)
    conv = layers.Conv1D(1, kernel_size=k, padding="same", use_bias=False, name=f"{name}_conv1d")(pooled)
    scale = layers.Activation("sigmoid")(conv)
    scale = layers.Reshape((1, 1, channels))(scale)
    return layers.Multiply()([x, scale])


def apply_attention(x, cfg, name="msai"):
    if cfg["channel_attn"]:
        x = (eca_attention(x, name=f"{name}_ch") if cfg["attn_type"] == "eca"
             else channel_attention_cbam(x, name=f"{name}_ch"))
    if cfg["spatial_attn"]:
        x = spatial_attention_cbam(x, name=f"{name}_sp")
    return x


# ---------------------------------------------------------------------------
# 3. Auxiliary multi-scale conv stems
# ---------------------------------------------------------------------------
def conv_stem(inputs, out_channels, name):
    x = layers.Conv2D(out_channels // 2, 3, padding="same", activation="relu", name=f"{name}_c1")(inputs)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Conv2D(out_channels, 3, padding="same", activation="relu", name=f"{name}_c2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    return x


# ---------------------------------------------------------------------------
# 4. Model builder
# ---------------------------------------------------------------------------
def build_model(backbone_name, config_name):
    cfg = CONFIG_FLAGS[config_name]
    reg = BACKBONE_REGISTRY[backbone_name]
    backbone = reg["app"](include_top=False, weights="imagenet",
                           input_shape=(IMG_SIZE, IMG_SIZE, 3))
    backbone.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x_pre = reg["preprocess"](inputs)
    feat = backbone(x_pre, training=False)
    feat_h, feat_w = feat.shape[1], feat.shape[2]

    if cfg["multiscale"]:
        half = layers.Resizing(IMG_SIZE // 2, IMG_SIZE // 2)(inputs)
        half = layers.Resizing(feat_h, feat_w)(half)
        half_feat = conv_stem(reg["preprocess"](half), MSAI_CHANNELS, name="half_scale")

        quarter = layers.Resizing(IMG_SIZE // 4, IMG_SIZE // 4)(inputs)
        quarter = layers.Resizing(feat_h, feat_w)(quarter)
        quarter_feat = conv_stem(reg["preprocess"](quarter), MSAI_CHANNELS, name="quarter_scale")

        feat = layers.Concatenate(axis=-1, name="msai_concat")([feat, half_feat, quarter_feat])

    feat = apply_attention(feat, cfg, name="msai")

    x = layers.GlobalAveragePooling2D()(feat)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    embedding = layers.Dense(128, activation="relu", name="embedding")(x)
    x = layers.BatchNormalization()(embedding)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    embedding_model = models.Model(inputs, embedding)
    return model, backbone, embedding_model


def make_augmenter():
    return tf.keras.Sequential([
        layers.RandomRotation(0.05),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomZoom(0.1),
        layers.RandomFlip("horizontal"),
        layers.RandomContrast(0.1),
    ])


def tta_predict(model, augmenter, X, rounds=TTA_ROUNDS, batch_size=BATCH_SIZE):
    probs = np.zeros((X.shape[0], NUM_CLASSES), dtype=np.float32)
    probs += model.predict(X, batch_size=batch_size, verbose=0)
    for _ in range(rounds):
        X_aug = augmenter(X, training=True).numpy()
        probs += model.predict(X_aug, batch_size=batch_size, verbose=0)
    probs /= (rounds + 1)
    return probs


def count_params_and_flops(model):
    trainable = int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))
    total = model.count_params()
    try:
        from keras_flops import get_flops
        flops = get_flops(model, batch_size=1)
    except Exception:
        flops = None  # keras_flops not installed -- report params only
    return total, trainable, flops


def train_fold_model(backbone_name, config_name, X_train, y_train, X_val, y_val, class_weight, augmenter):
    train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
                .shuffle(1000, seed=SEED).batch(BATCH_SIZE)
                .map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE))
    val_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
              .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

    model, backbone, embedding_model = build_model(backbone_name, config_name)

    model.compile(optimizer=optimizers.Adam(1e-3),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    t0 = time.perf_counter()
    model.fit(train_ds, validation_data=val_ds, epochs=HEAD_EPOCHS,
              class_weight=class_weight, verbose=2)
    t_stage1 = time.perf_counter() - t0

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
    t0 = time.perf_counter()
    model.fit(train_ds, validation_data=val_ds, epochs=FINE_TUNE_EPOCHS,
              callbacks=cbs, class_weight=class_weight, verbose=2)
    t_stage2 = time.perf_counter() - t0

    return model, embedding_model, t_stage1, t_stage2


def append_result(row):
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# 5. Driver
# ---------------------------------------------------------------------------
def main():
    if RUN_MODE == "ablation":
        backbone_name, config_name = "mobilenetv2", ABLATION_CONFIG
    elif RUN_MODE == "cross_backbone":
        backbone_name, config_name = CROSS_BACKBONE, "E"
    else:
        raise ValueError("RUN_MODE must be 'ablation' or 'cross_backbone'")

    assert config_name in CONFIG_FLAGS
    assert backbone_name in BACKBONE_REGISTRY
    print(f"Backbone: {backbone_name}  Config: {config_name}  "
          f"({CONFIG_FLAGS[config_name]})")

    print("Listing image paths...")
    paths, y = list_image_paths(DATA_DIR)
    print(f"Found {len(paths)} images across {NUM_CLASSES} classes: {CLASS_NAMES}")

    print("\nLoading + preprocessing images...")
    X = load_and_preprocess(paths)
    gc.collect()

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_results = {"cnn": [], "cnn_tta": []}
    params_logged = False

    for fold_idx, (trainval_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'=' * 70}\n{backbone_name}/{config_name} -- FOLD {fold_idx + 1}/{N_FOLDS}\n{'=' * 70}")

        train_idx, val_idx = train_test_split(
            trainval_idx, test_size=0.15, random_state=SEED, stratify=y[trainval_idx]
        )
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        class_weight = dict(enumerate(
            compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y_train)
        ))
        augmenter = make_augmenter()

        model, embedding_model, t_stage1, t_stage2 = train_fold_model(
            backbone_name, config_name, X_train, y_train, X_val, y_val, class_weight, augmenter
        )
        train_time_s = t_stage1 + t_stage2

        if not params_logged:
            total_params, trainable_params, flops = count_params_and_flops(model)
            print(f"  Params: {total_params:,} total / {trainable_params:,} trainable  "
                  f"FLOPs: {flops if flops else 'N/A (install keras_flops)'}")
            params_logged = True

        t0 = time.perf_counter()
        probs_cnn = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
        t_cnn_infer = time.perf_counter() - t0
        y_pred_cnn = np.argmax(probs_cnn, axis=1)
        acc_cnn = accuracy_score(y_test, y_pred_cnn)
        f1_cnn = f1_score(y_test, y_pred_cnn, average="macro")
        fold_results["cnn"].append(acc_cnn)
        ms_per_image_cnn = 1000.0 * t_cnn_infer / len(X_test)

        t0 = time.perf_counter()
        probs_tta = tta_predict(model, augmenter, X_test)
        t_tta_infer = time.perf_counter() - t0
        y_pred_tta = np.argmax(probs_tta, axis=1)
        acc_tta = accuracy_score(y_test, y_pred_tta)
        f1_tta = f1_score(y_test, y_pred_tta, average="macro")
        fold_results["cnn_tta"].append(acc_tta)
        ms_per_image_tta = 1000.0 * t_tta_infer / len(X_test)

        print(f"\nFold {fold_idx + 1} [{backbone_name}/{config_name}]: "
              f"CNN={acc_cnn:.4f} (F1={f1_cnn:.4f}, {ms_per_image_cnn:.2f} ms/img)  "
              f"CNN+TTA={acc_tta:.4f} (F1={f1_tta:.4f}, {ms_per_image_tta:.2f} ms/img)  "
              f"train_time={train_time_s:.1f}s")
        print(classification_report(y_test, y_pred_tta, target_names=CLASS_NAMES))
        print(confusion_matrix(y_test, y_pred_tta))

        append_result({
            "backbone": backbone_name, "config": config_name, "fold": fold_idx + 1,
            "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
            "train_time_total_s": round(train_time_s, 2),
            "cnn_acc": round(acc_cnn, 4), "cnn_macro_f1": round(f1_cnn, 4),
            "cnn_ms_per_image": round(ms_per_image_cnn, 3),
            "tta_acc": round(acc_tta, 4), "tta_macro_f1": round(f1_tta, 4),
            "tta_ms_per_image": round(ms_per_image_tta, 3),
        })

        del model, embedding_model
        tf.keras.backend.clear_session()
        gc.collect()

    print(f"\n{'=' * 70}\nFINAL 5-FOLD SUMMARY -- {backbone_name}/{config_name}\n{'=' * 70}")
    for key, label in [("cnn", "CNN (softmax)"), ("cnn_tta", "CNN + TTA")]:
        accs = np.array(fold_results[key])
        print(f"{label:20s}: {accs.mean() * 100:.2f}% +/- {accs.std() * 100:.2f}%")
    print(f"\nPer-fold rows appended to {RESULTS_PATH}. After collecting all "
          f"configs A-F on mobilenetv2 and config E on the other three "
          f"backbones, compute the Wilcoxon signed-rank test of each row's "
          f"5 paired fold accuracies against config A's, per backbone, to "
          f"fill in the ablation table.")


if __name__ == "__main__":
    main()
