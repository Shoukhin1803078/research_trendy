"""
Step 19: Lesion-supervised, plane-invariant classification (LSPI) on BRISC.

THE CONTRIBUTION
----------------
A classifier trained on mixed-plane 2D MRI can lean on plane-specific
anatomy instead of the lesion, so it degrades when the acquisition plane
shifts. BRISC is the first corpus in this project that provides both things
needed to attack that directly: the acquisition plane of every slice, and a
physician-validated lesion mask for every tumour slice.

LSPI adds two terms to an otherwise unchanged backbone + head:

  (1) LESION SUPERVISION. A 1x1 conv on the backbone's final feature map
      produces a single-channel spatial attention map, trained to agree with
      the downsampled ground-truth mask (Dice + BCE). Critically the map
      also GATES the features that reach the classifier,
      feat <- feat * (1 + A), so the supervision is on the decision path
      rather than on a side head that could be ignored. Samples without a
      mask (every no_tumor slice) contribute zero weight to this term.

  (2) PLANE INVARIANCE. The pooled embedding passes through a gradient
      reversal layer into a small plane classifier. The head learns to
      predict the plane; the reversal makes the trunk unlearn it. Lambda
      ramps 0 -> target over training, as is standard for adversarial
      domain-invariance, because a strong reversal from step 0 destabilises
      the trunk before it has useful features.

Evaluated under step18's leave-one-plane-out protocol: train on two planes,
test on the third, with no image, label or statistic from the held-out plane
used during training. That is DOMAIN GENERALIZATION. It is the distinction
from orientation-aware UDA work, which adapts using target-domain data and
specialises one model per plane -- here a single model must transfer unseen.

THE 2x2 IS THE CONTRIBUTION, NOT THE COMPONENTS
-----------------------------------------------
Mask-supervised attention and adversarial domain invariance are both
established. What is not established is whether they compose for plane shift,
and which one carries the effect. So the experiment is a 2x2:

    M0P0  baseline (neither)          -- the number to beat
    M1P0  lesion supervision only
    M0P1  plane invariance only
    M1P1  both (proposed)

Report ID and OOD accuracy for all four, per held-out plane, per backbone,
and test the OOD differences with the same exact Wilcoxon procedure used
elsewhere in this project. If M1P1 does not beat both single-component
configurations, say so: the honest outcome is that the components do not
compose, and that is still a result.

HONEST LIMITS, STATE THESE IN THE PAPER
---------------------------------------
* Supervision happens at the backbone's feature resolution, which is 7x7 for
  a 224x224 input on all four backbones here. That is coarse -- the same
  resolution Grad-CAM operates at -- so this constrains attention to the
  right REGION, not to a precise boundary. --attn-upsample can refine it at
  the cost of extra parameters.
* Only tumour slices carry masks, so lesion supervision sees three of four
  classes. The no_tumor class is shaped only by the classification loss.
* With three planes, leave-one-plane-out gives n=3 paired observations per
  comparison. An exact signed-rank test on n=3 cannot reach p<0.25, so run
  the per-plane folds across seeds (--seeds) if you need power, and report
  per-fold numbers rather than leaning on a p-value.

RUN ORDER
---------
  1. step18_brisc_data.py           -> lopo_splits.csv   (CPU, no GPU)
  2. this file with --self-test     -> verifies the custom layers (~30 s)
  3. this file per (backbone, config) -> one Kaggle session each

NOTE ON TESTING: the TensorFlow parts of this file were NOT executed on the
machine where it was written (no TF available there). --self-test exists for
exactly that reason: it checks, on random tensors, that gradient reversal
actually reverses, that the masked Dice+BCE term is finite and decreases when
attention approaches the mask, and that every tensor shape lines up. Run it
first; it costs seconds and will catch a broken loss before a 2-hour session
does.
"""

import os
import sys
import gc
import csv
import json
import time
import argparse
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Notebook config -- EDIT THIS before each Kaggle session.
# ---------------------------------------------------------------------------
NB_SPLITS_CSV = "/kaggle/working/brisc_protocol/lopo_splits.csv"
NB_RESULTS_PATH = "/kaggle/working/lspi_results.jsonl"
NB_BACKBONE = "mobilenetv2"      # mobilenetv2 | resnet50 | densenet121 | efficientnetb0
NB_CONFIG = "M0P0"               # M0P0 | M1P0 | M0P1 | M1P1
NB_FOLD = "all"                  # "all", or one fold name from step18
NB_SELF_TEST = False
NB_SEED = 42

IMG_SIZE = 224
BATCH_SIZE = 16
HEAD_EPOCHS = 8
FINE_TUNE_EPOCHS = 35
FINE_TUNE_AT = 0.8
MASK_LOSS_WEIGHT = 1.0           # alpha
PLANE_LOSS_WEIGHT = 1.0          # weight on the plane head's own CE
GRL_LAMBDA_MAX = 1.0             # target lambda for the reversal
ATTN_UPSAMPLE = 1                # 1 = supervise at feature resolution (7x7)
ATTN_INIT = 0.1                  # initial attention value; keeps the gate
                                 # feat*(1+A) near identity at step 0

CLASS_NAMES = ["glioma", "meningioma", "pituitary", "no_tumor"]
PLANE_NAMES = ["axial", "coronal", "sagittal"]
NUM_CLASSES = len(CLASS_NAMES)

CONFIGS = {
    "M0P0": dict(mask_sup=False, plane_inv=False, desc="baseline (neither)"),
    "M1P0": dict(mask_sup=True,  plane_inv=False, desc="lesion supervision only"),
    "M0P1": dict(mask_sup=False, plane_inv=True,  desc="plane invariance only"),
    "M1P1": dict(mask_sup=True,  plane_inv=True,  desc="LSPI (proposed)"),
}


def _tf():
    """Import TF lazily so step18-style inspection and --help work without it."""
    import tensorflow as tf
    return tf


# ---------------------------------------------------------------------------
# Custom layers
# ---------------------------------------------------------------------------
def make_grl_layer():
    tf = _tf()
    from tensorflow.keras import layers

    class GradientReversal(layers.Layer):
        """Identity forward; multiplies the gradient by -lambda backward.

        lambda_ is a non-trainable tf.Variable so a callback can ramp it
        during training without recompiling the model.
        """

        def __init__(self, lambda_init=0.0, **kw):
            super().__init__(**kw)
            self.lambda_ = tf.Variable(
                float(lambda_init), trainable=False, dtype="float32",
                name="grl_lambda")

        def call(self, x):
            lam = self.lambda_

            @tf.custom_gradient
            def _rev(z):
                def grad(dy):
                    return -lam * dy
                return tf.identity(z), grad

            return _rev(x)

        def get_config(self):
            cfg = super().get_config()
            cfg["lambda_init"] = float(self.lambda_.numpy())
            return cfg

    return GradientReversal


def make_lambda_ramp_callback(grl_layer, total_epochs, lambda_max,
                              epoch_offset=0):
    """lambda = lambda_max * (2/(1+exp(-10p)) - 1), p = global_epoch/total.

    The standard DANN schedule: near zero early, saturating at lambda_max.
    Without it the reversal fights the trunk before the trunk has learned
    anything worth making invariant.

    epoch_offset makes the ramp CONTINUOUS across the two training stages.
    Without it the stage-2 callback restarts at p=0, so lambda climbs during
    stage 1, collapses back to ~0 when fine-tuning begins, and re-climbs --
    which un-does the invariance pressure at exactly the point the backbone
    is unfrozen and most able to learn plane-specific features.
    """
    tf = _tf()
    from tensorflow.keras import callbacks

    class LambdaRamp(callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            g = epoch + 1 + epoch_offset
            p = min(1.0, g / max(1, total_epochs))
            lam = lambda_max * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)
            grl_layer.lambda_.assign(float(lam))
            print(f"  [GRL] global epoch {g}/{total_epochs}: lambda = {lam:.4f}")

    return LambdaRamp()


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def masked_dice_bce(smooth=1.0):
    """Per-sample Dice + BCE between the attention map and the GT mask.

    Returns shape (batch,) so Keras can apply a per-sample weight -- that is
    how slices without a mask are excluded (weight 0) rather than being
    taught to predict an empty map, which would be a different and wrong
    supervision signal.
    """
    tf = _tf()

    def loss(y_true, y_pred):
        yt = tf.cast(tf.reshape(y_true, (tf.shape(y_true)[0], -1)), tf.float32)
        yp = tf.cast(tf.reshape(y_pred, (tf.shape(y_pred)[0], -1)), tf.float32)
        yp = tf.clip_by_value(yp, 1e-7, 1.0 - 1e-7)
        inter = tf.reduce_sum(yt * yp, axis=1)
        dice = 1.0 - (2.0 * inter + smooth) / (
            tf.reduce_sum(yt, axis=1) + tf.reduce_sum(yp, axis=1) + smooth)
        bce = -tf.reduce_mean(
            yt * tf.math.log(yp) + (1.0 - yt) * tf.math.log(1.0 - yp), axis=1)
        return dice + bce

    return loss


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def backbone_registry():
    tf = _tf()
    return {
        "mobilenetv2": dict(app=tf.keras.applications.MobileNetV2,
                            preprocess=tf.keras.applications.mobilenet_v2.preprocess_input),
        "resnet50": dict(app=tf.keras.applications.ResNet50,
                         preprocess=tf.keras.applications.resnet50.preprocess_input),
        "densenet121": dict(app=tf.keras.applications.DenseNet121,
                            preprocess=tf.keras.applications.densenet.preprocess_input),
        "efficientnetb0": dict(app=tf.keras.applications.EfficientNetB0,
                               preprocess=tf.keras.applications.efficientnet.preprocess_input),
    }


def build_model(backbone_name, config_name, img_size=IMG_SIZE,
                attn_upsample=ATTN_UPSAMPLE, grl_cls=None):
    """Backbone + {optional lesion-gated attention} + head + {optional GRL
    plane head}. Returns (model, backbone, grl_layer_or_None, feat_hw)."""
    tf = _tf()
    from tensorflow.keras import layers, models

    cfg = CONFIGS[config_name]
    reg = backbone_registry()[backbone_name]
    grl_cls = grl_cls or make_grl_layer()

    backbone = reg["app"](include_top=False, weights="imagenet",
                          input_shape=(img_size, img_size, 3))
    backbone.trainable = False

    inp = layers.Input(shape=(img_size, img_size, 3), name="image")
    feat = backbone(reg["preprocess"](inp), training=False)
    fh, fw = int(feat.shape[1]), int(feat.shape[2])

    outputs, feat_used = {}, feat
    if cfg["mask_sup"]:
        # Negative bias init so sigmoid(.) starts near ATTN_INIT, making the
        # gate feat*(1+A) close to identity at step 0. With a default-zero
        # bias the map starts at 0.5 everywhere, scaling every feature by 1.5
        # before the classifier has learned anything; in the integration test
        # that drove an immediate collapse to a single predicted class. The
        # classifier establishes itself first, then the mask loss sharpens A.
        attn = layers.Conv2D(
            1, 1, activation="sigmoid", name="attn_conv",
            bias_initializer=tf.keras.initializers.Constant(
                float(np.log(ATTN_INIT / (1.0 - ATTN_INIT)))))(feat)
        if attn_upsample > 1:
            attn = layers.UpSampling2D(attn_upsample, interpolation="bilinear",
                                       name="attn_up")(attn)
            gate = layers.AveragePooling2D(attn_upsample, name="attn_down")(attn)
        else:
            gate = attn
        # Gate the decision path: supervision must influence classification,
        # not sit on an ignorable side branch. (1 + A) preserves scale at A=0.
        feat_used = layers.Multiply(name="attn_gate")(
            [feat, layers.Lambda(lambda a: 1.0 + a, name="one_plus_a")(gate)])
        outputs["mask"] = layers.Activation("linear", name="mask")(attn)

    x = layers.GlobalAveragePooling2D(name="gap")(feat_used)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Dropout(0.4, name="do1")(x)
    emb = layers.Dense(128, activation="relu", name="embedding")(x)
    h = layers.BatchNormalization(name="bn2")(emb)
    h = layers.Dropout(0.3, name="do2")(h)
    outputs["cls"] = layers.Dense(NUM_CLASSES, activation="softmax", name="cls")(h)

    grl = None
    if cfg["plane_inv"]:
        grl = grl_cls(lambda_init=0.0, name="grl")
        p = grl(emb)
        p = layers.Dense(64, activation="relu", name="plane_fc")(p)
        outputs["plane"] = layers.Dense(len(PLANE_NAMES), activation="softmax",
                                        name="plane")(p)

    model = models.Model(inp, outputs, name=f"{backbone_name}_{config_name}")
    embed_model = models.Model(inp, emb)
    return model, backbone, grl, (fh, fw), embed_model


def compile_model(model, config_name, lr):
    tf = _tf()
    from tensorflow.keras import optimizers

    cfg = CONFIGS[config_name]
    losses = {"cls": "sparse_categorical_crossentropy"}
    weights = {"cls": 1.0}
    metrics = {"cls": ["accuracy"]}
    if cfg["mask_sup"]:
        losses["mask"] = masked_dice_bce()
        weights["mask"] = MASK_LOSS_WEIGHT
    if cfg["plane_inv"]:
        losses["plane"] = "sparse_categorical_crossentropy"
        weights["plane"] = PLANE_LOSS_WEIGHT
    model.compile(optimizer=optimizers.Adam(lr), loss=losses,
                  loss_weights=weights, metrics=metrics)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_split_rows(csv_path):
    folds = defaultdict(lambda: defaultdict(list))
    held = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            folds[row["fold"]][row["part"]].append(row)
            held[row["fold"]] = row["held_out_plane"]
    return {k: dict(v) for k, v in folds.items()}, held


def decode_rows(rows, feat_hw, config_name, img_size=IMG_SIZE,
                attn_upsample=ATTN_UPSAMPLE):
    """Materialise a split into arrays. BRISC splits are small (a few
    thousand slices), so loading eagerly is simpler and faster than a
    streaming pipeline, and matches what step6 did."""
    tf = _tf()
    cfg = CONFIGS[config_name]
    mh, mw = feat_hw[0] * attn_upsample, feat_hw[1] * attn_upsample

    n = len(rows)
    X = np.zeros((n, img_size, img_size, 3), dtype=np.uint8)
    y_cls = np.zeros(n, dtype=np.int32)
    y_pln = np.zeros(n, dtype=np.int32)
    y_msk = np.zeros((n, mh, mw, 1), dtype=np.float32)
    w_msk = np.zeros(n, dtype=np.float32)

    for i, r in enumerate(rows):
        img = tf.io.decode_image(tf.io.read_file(r["path"]), channels=3,
                                 expand_animations=False)
        X[i] = tf.image.resize(img, [img_size, img_size]).numpy().astype(np.uint8)
        y_cls[i] = CLASS_NAMES.index(r["label"])
        y_pln[i] = PLANE_NAMES.index(r["plane"])
        if cfg["mask_sup"] and r.get("mask"):
            m = tf.io.decode_image(tf.io.read_file(r["mask"]), channels=1,
                                   expand_animations=False)
            m = tf.image.resize(m, [mh, mw]).numpy().astype(np.float32)
            y_msk[i] = (m > 127.0).astype(np.float32)
            w_msk[i] = 1.0
        if (i + 1) % 500 == 0:
            print(f"    loaded {i + 1}/{n}")

    ys, ws = {"cls": y_cls}, {"cls": np.ones(n, dtype=np.float32)}
    if cfg["mask_sup"]:
        ys["mask"] = y_msk
        ws["mask"] = w_msk          # 0 for slices with no mask -> excluded
    if cfg["plane_inv"]:
        ys["plane"] = y_pln
        ws["plane"] = np.ones(n, dtype=np.float32)
    return X, ys, ws


def make_augmenter():
    tf = _tf()
    from tensorflow.keras import layers
    return tf.keras.Sequential([
        layers.RandomRotation(0.05),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomZoom(0.1),
        layers.RandomFlip("horizontal"),
        layers.RandomContrast(0.1),
    ], name="augmenter")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def self_test():
    """Verify the custom pieces on random tensors. Seconds, no dataset."""
    tf = _tf()
    print(f"TensorFlow {tf.__version__}\n")
    ok = True

    # 1. gradient reversal really reverses
    print("[1] gradient reversal sign")
    GRL = make_grl_layer()
    g = GRL(lambda_init=1.0)
    x = tf.Variable([[1.0, 2.0, 3.0]])
    with tf.GradientTape() as t:
        out = tf.reduce_sum(g(x) * 2.0)
    grad_rev = t.gradient(out, x).numpy()
    with tf.GradientTape() as t:
        out2 = tf.reduce_sum(x * 2.0)
    grad_plain = t.gradient(out2, x).numpy()
    print(f"    plain grad   {grad_plain}")
    print(f"    reversed     {grad_rev}")
    if not np.allclose(grad_rev, -grad_plain):
        print("    FAIL: not the negation of the plain gradient"); ok = False
    else:
        print("    pass")

    g.lambda_.assign(0.0)
    with tf.GradientTape() as t:
        out = tf.reduce_sum(g(x) * 2.0)
    if not np.allclose(t.gradient(out, x).numpy(), 0.0):
        print("    FAIL: lambda=0 should zero the gradient"); ok = False
    else:
        print("    pass (lambda=0 blocks the adversarial gradient)")

    # 2. masked Dice+BCE: finite, per-sample, decreasing toward the mask
    print("\n[2] masked Dice+BCE")
    loss_fn = masked_dice_bce()
    m = np.zeros((3, 7, 7, 1), np.float32); m[:, 2:5, 2:5] = 1.0
    perfect = loss_fn(tf.constant(m), tf.constant(m * 0.99 + 0.005)).numpy()
    halfway = loss_fn(tf.constant(m), tf.constant(np.full_like(m, 0.5))).numpy()
    wrong = loss_fn(tf.constant(m), tf.constant(1.0 - (m * 0.99 + 0.005))).numpy()
    print(f"    shape {perfect.shape} (must be (3,) for per-sample weighting)")
    print(f"    aligned {perfect.mean():.4f} < uniform {halfway.mean():.4f} "
          f"< inverted {wrong.mean():.4f}")
    if perfect.shape != (3,):
        print("    FAIL: not per-sample"); ok = False
    elif not (perfect.mean() < halfway.mean() < wrong.mean()):
        print("    FAIL: not ordered as expected"); ok = False
    elif not np.all(np.isfinite([perfect.mean(), halfway.mean(), wrong.mean()])):
        print("    FAIL: non-finite"); ok = False
    else:
        print("    pass")

    # 3. all four configs build, and shapes line up
    print("\n[3] model build, all four configs (mobilenetv2)")
    for name in ("M0P0", "M1P0", "M0P1", "M1P1"):
        model, bb, grl, fhw, _ = build_model("mobilenetv2", name)
        compile_model(model, name, 1e-3)
        outs = {k: tuple(v.shape[1:]) for k, v in
                zip(model.output_names, model.outputs)} \
            if hasattr(model, "output_names") else \
            {k: tuple(v.shape[1:]) for k, v in model.output.items()}
        print(f"    {name:5s} {CONFIGS[name]['desc']:28s} feat {fhw}  outputs {outs}")
        if ("mask" in outs) != CONFIGS[name]["mask_sup"]:
            print("    FAIL: mask output presence wrong"); ok = False
        if ("plane" in outs) != CONFIGS[name]["plane_inv"]:
            print("    FAIL: plane output presence wrong"); ok = False
        del model, bb
        tf.keras.backend.clear_session(); gc.collect()

    # 4. two training steps on random data for the full config
    print("\n[4] two train steps on random data (M1P1)")
    model, bb, grl, fhw, _ = build_model("mobilenetv2", "M1P1")
    compile_model(model, "M1P1", 1e-3)
    n = 8
    X = np.random.randint(0, 255, (n, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    ys = {"cls": np.random.randint(0, NUM_CLASSES, n).astype(np.int32),
          "mask": (np.random.rand(n, fhw[0], fhw[1], 1) > 0.7).astype(np.float32),
          "plane": np.random.randint(0, len(PLANE_NAMES), n).astype(np.int32)}
    ws = {"cls": np.ones(n, np.float32),
          "mask": np.array([1, 1, 1, 1, 0, 0, 0, 0], np.float32),  # half unmasked
          "plane": np.ones(n, np.float32)}
    grl.lambda_.assign(0.5)
    h = model.fit(X, ys, sample_weight=ws, epochs=2, batch_size=4, verbose=0)
    losses = h.history["loss"]
    print(f"    losses {['%.4f' % v for v in losses]}")
    if not all(np.isfinite(losses)):
        print("    FAIL: non-finite loss -- the masked term or the GRL is broken")
        ok = False
    else:
        print("    pass (finite loss with zero-weighted mask samples present)")

    print(f"\n{'=' * 60}")
    print("SELF-TEST PASSED" if ok else "SELF-TEST FAILED -- do not run a full session")
    print(f"{'=' * 60}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# Train / evaluate one fold
# ---------------------------------------------------------------------------
def run_fold(backbone, config_name, fold_name, parts, seed, results_path):
    tf = _tf()
    from tensorflow.keras import callbacks, optimizers

    print(f"\n{'=' * 78}")
    print(f"{backbone} / {config_name} ({CONFIGS[config_name]['desc']}) / {fold_name}")
    print(f"{'=' * 78}")
    tf.keras.utils.set_random_seed(seed)

    model, bb, grl, fhw, _ = build_model(backbone, config_name)
    print(f"feature map {fhw}, mask supervised at "
          f"{fhw[0] * ATTN_UPSAMPLE}x{fhw[1] * ATTN_UPSAMPLE}")

    print("loading splits...")
    data = {}
    for part in ("train", "val", "id_test", "ood_test"):
        print(f"  {part} ({len(parts[part])})")
        data[part] = decode_rows(parts[part], fhw, config_name)

    aug = make_augmenter()
    Xtr, ytr, wtr = data["train"]
    n_masked = int(wtr.get("mask", np.zeros(1)).sum()) if "mask" in wtr else 0
    if CONFIGS[config_name]["mask_sup"]:
        print(f"  train slices with a mask: {n_masked}/{len(Xtr)}")
        if n_masked == 0:
            raise SystemExit(
                "No training masks found but this config needs them. Check the "
                "'mask' column in lopo_splits.csv -- step18 pairs masks by "
                "filename stem.")

    Xtr_aug = aug(Xtr.astype("float32"), training=True).numpy().astype("uint8")

    # Stage 1: frozen backbone
    compile_model(model, config_name, 1e-3)
    total_epochs = HEAD_EPOCHS + FINE_TUNE_EPOCHS
    cbs = []
    if grl is not None:
        cbs.append(make_lambda_ramp_callback(grl, total_epochs, GRL_LAMBDA_MAX,
                                             epoch_offset=0))
    t0 = time.perf_counter()
    model.fit(Xtr_aug, ytr, sample_weight=wtr,
              validation_data=(data["val"][0], data["val"][1], data["val"][2]),
              epochs=HEAD_EPOCHS, batch_size=BATCH_SIZE, callbacks=cbs, verbose=2)
    t_s1 = time.perf_counter() - t0

    # Stage 2: unfreeze the top FINE_TUNE_AT of the backbone
    bb.trainable = True
    for layer in bb.layers[:int(len(bb.layers) * (1 - FINE_TUNE_AT))]:
        layer.trainable = False
    compile_model(model, config_name, 1e-5)
    cbs2 = [callbacks.EarlyStopping(monitor="val_cls_accuracy", mode="max",
                                    patience=10, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4)]
    if grl is not None:
        # continue the SAME ramp; offset by the stage-1 epochs already run
        cbs2.append(make_lambda_ramp_callback(
            grl, total_epochs, GRL_LAMBDA_MAX, epoch_offset=HEAD_EPOCHS))
    t0 = time.perf_counter()
    model.fit(Xtr_aug, ytr, sample_weight=wtr,
              validation_data=(data["val"][0], data["val"][1], data["val"][2]),
              epochs=FINE_TUNE_EPOCHS, batch_size=BATCH_SIZE,
              callbacks=cbs2, verbose=2)
    t_s2 = time.perf_counter() - t0

    def evaluate(part):
        X, ys, _ws = data[part]
        pred = model.predict(X, batch_size=BATCH_SIZE, verbose=0)
        probs = pred["cls"] if isinstance(pred, dict) else (
            pred[list(model.output_names).index("cls")]
            if isinstance(pred, list) else pred)
        yp = np.argmax(probs, axis=1)
        yt = ys["cls"]
        from sklearn.metrics import accuracy_score, f1_score
        return (float(accuracy_score(yt, yp)),
                float(f1_score(yt, yp, average="macro")), yp.tolist(), yt.tolist())

    id_acc, id_f1, _, _ = evaluate("id_test")
    ood_acc, ood_f1, ood_pred, ood_true = evaluate("ood_test")

    print(f"\n{backbone}/{config_name}/{fold_name}")
    print(f"  ID  accuracy {id_acc * 100:6.2f}%  macro-F1 {id_f1:.4f}")
    print(f"  OOD accuracy {ood_acc * 100:6.2f}%  macro-F1 {ood_f1:.4f}")
    print(f"  GAP          {(id_acc - ood_acc) * 100:+6.2f} pp   "
          f"<- the quantity this method targets")

    # Collapse guard. A model that predicts one class for everything can post
    # a plausible-looking accuracy and a plausible-looking gap while having
    # learned nothing, and the macro-F1 is the giveaway. Flag it loudly
    # rather than letting it into the 2x2 table as a real number.
    n_pred_classes = len(set(ood_pred))
    if n_pred_classes == 1:
        print(f"  [COLLAPSE] every OOD prediction is class "
              f"{CLASS_NAMES[ood_pred[0]]} -- this run learned nothing. "
              f"Do not report it. Lower the learning rate, raise ATTN_INIT, "
              f"or reduce MASK_LOSS_WEIGHT.")
    elif ood_f1 < 0.4 and ood_acc > 0.5:
        print(f"  [WARN] macro-F1 {ood_f1:.3f} is far below accuracy "
              f"{ood_acc:.3f} -- predictions are concentrated on few classes.")

    row = {"backbone": backbone, "config": config_name, "fold": fold_name,
           "held_out_plane": parts["held_out_plane"], "seed": seed,
           "n_train": len(parts["train"]), "n_id_test": len(parts["id_test"]),
           "n_ood_test": len(parts["ood_test"]), "n_train_masked": n_masked,
           "id_acc": round(id_acc, 4), "id_macro_f1": round(id_f1, 4),
           "ood_acc": round(ood_acc, 4), "ood_macro_f1": round(ood_f1, 4),
           "gap_pp": round((id_acc - ood_acc) * 100, 3),
           "n_ood_pred_classes": n_pred_classes,
           "collapsed": bool(n_pred_classes == 1),
           "train_time_s": round(t_s1 + t_s2, 1)}
    with open(results_path, "a") as f:
        f.write(json.dumps(row) + "\n")

    pred_path = results_path.replace(".jsonl", "_ood_predictions.csv")
    new = not os.path.exists(pred_path)
    with open(pred_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["backbone", "config", "fold", "path", "y_true", "y_pred"])
        for r, yt, yp in zip(parts["ood_test"], ood_true, ood_pred):
            w.writerow([backbone, config_name, fold_name, r["path"], yt, yp])

    del model, bb
    tf.keras.backend.clear_session(); gc.collect()
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits-csv")
    ap.add_argument("--results-path")
    ap.add_argument("--backbone", choices=list(CONFIGS) and
                    ["mobilenetv2", "resnet50", "densenet121", "efficientnetb0"])
    ap.add_argument("--config", choices=list(CONFIGS))
    ap.add_argument("--fold", default=None)
    ap.add_argument("--seed", type=int, default=NB_SEED)
    ap.add_argument("--self-test", action="store_true")

    if argv is not None:
        args = ap.parse_args(argv)
    elif "ipykernel" in sys.modules or len(sys.argv) <= 1:
        args = argparse.Namespace(
            splits_csv=NB_SPLITS_CSV, results_path=NB_RESULTS_PATH,
            backbone=NB_BACKBONE, config=NB_CONFIG, fold=NB_FOLD,
            seed=NB_SEED, self_test=NB_SELF_TEST)
        print("No CLI arguments (notebook or bare run) -- using the NB_* "
              "config block at the top of this file:")
        print(f"  backbone : {args.backbone}   config: {args.config}"
              f"   fold: {args.fold}   self-test: {args.self_test}\n")
    else:
        args = ap.parse_args()

    if args.self_test:
        raise SystemExit(self_test())

    for need, name in ((args.splits_csv, "--splits-csv / NB_SPLITS_CSV"),
                       (args.results_path, "--results-path / NB_RESULTS_PATH"),
                       (args.backbone, "--backbone / NB_BACKBONE"),
                       (args.config, "--config / NB_CONFIG")):
        if not need:
            ap.error(f"missing {name}")
    if not os.path.exists(args.splits_csv):
        ap.error(f"no splits file at {args.splits_csv} -- run step18 first")

    folds, _held = load_split_rows(args.splits_csv)
    for name in folds:
        folds[name]["held_out_plane"] = folds[name]["ood_test"][0]["plane"]

    wanted = (list(folds) if args.fold in (None, "all", "")
              else [f for f in folds if f == args.fold])
    if not wanted:
        ap.error(f"fold {args.fold!r} not in {list(folds)}")

    rows = [run_fold(args.backbone, args.config, name, folds[name],
                     args.seed, args.results_path) for name in wanted]

    print(f"\n{'=' * 78}\nSUMMARY -- {args.backbone}/{args.config}\n{'=' * 78}")
    print(f"{'fold':<20}{'ID':>9}{'OOD':>9}{'gap':>9}")
    for r in rows:
        print(f"{r['fold']:<20}{r['id_acc'] * 100:>8.2f}%"
              f"{r['ood_acc'] * 100:>8.2f}%{r['gap_pp']:>8.2f}")
    if rows:
        print(f"{'mean':<20}{np.mean([r['id_acc'] for r in rows]) * 100:>8.2f}%"
              f"{np.mean([r['ood_acc'] for r in rows]) * 100:>8.2f}%"
              f"{np.mean([r['gap_pp'] for r in rows]):>8.2f}")
    print(f"\nRows appended to {args.results_path}. Run all four configs "
          f"(M0P0/M1P0/M0P1/M1P1) to fill the 2x2, then compare OOD accuracy "
          f"per fold.")


if __name__ == "__main__":
    main()
