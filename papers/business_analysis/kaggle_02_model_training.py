"""
Step 2 of 2 -- Empirical validation of the AI-assisted red-flag prioritisation
framework (Eq. 1 in the manuscript: s(r) = w1*a(x_r) + w2*b(x_r) + w3*c(x_r) + w4*f(x_r)).

Run this AFTER kaggle_01_eda.py, on the same Kaggle Notebook (dataset:
mlg-ulb/creditcardfraud -> /kaggle/input/creditcardfraud/creditcard.csv).

Dataset: ULB "Credit Card Fraud Detection" (Dal Pozzolo et al., 2015, IEEE SSCI,
doi:10.1109/SSCI.2015.33). 284,807 European-cardholder transactions from
September 2013, 492 (0.172%) fraudulent, features V1-V28 (PCA-anonymised),
Amount, Time.

What this script does:
  1. Operationalises each term of Eq. 1 as a concrete, reproducible signal
     computed from the raw transaction data ("Feature engineering").
       a(x_r): unsupervised anomaly score (Isolation Forest)
       b(x_r): behavioural-deviation score (distance to nearest cluster
                centroid from unsupervised peer-group clustering)
       c(x_r): binary control-rule violation (large-amount OR off-hours rule)
       f(x_r): frequency/severity (normalised amount + local transaction
                density in a +/-60s window)
  2. Trains a meta-learner (logistic regression) on [a,b,c,f] to CALIBRATE
     the weights w1..w4 empirically instead of assuming them a priori --
     "EC-ARP" (Empirically-Calibrated AI-assisted Red-flag Prioritisation).
     This turns the manuscript's conceptual Eq. 1 into a fitted model.
  3. Extends EC-ARP with an out-of-fold supervised XGBoost probability as a
     fifth stacking input -- "EC-ARP+" -- the proposed hybrid architecture.
  4. Benchmarks EC-ARP / EC-ARP+ against standalone baselines mirroring
     individual techniques reported in the reviewed literature: Logistic
     Regression, Random Forest, XGBoost, Isolation Forest alone, and a PCA
     reconstruction-error anomaly detector (classical stand-in for an
     autoencoder, since V1-V28 are already PCA components).
  5. Reports PR-AUC, ROC-AUC, F1/Precision/Recall at 0.5, and
     Precision/Recall at the top 1% of ranked transactions (operationalising
     threshold tau in Eq. 1 as "flag the top 1% of records for investigation",
     the metric that matters most for audit triage capacity).

No numbers in this script are invented; every reported figure comes from
actually running this code. Re-running with a different random_state or
resampling will shift the figures slightly -- see the printed LIMITATIONS
note at the end.

Run:  python kaggle_02_model_training.py
"""
# ============================================================
# STEP 2 OF 2
# EMPIRICAL VALIDATION OF AI-ASSISTED RED-FLAG PRIORITISATION
#
# Eq. (1):
# s(r) = w1*a(x_r) + w2*b(x_r) + w3*c(x_r) + w4*f(x_r)
#
# EC-ARP  : Empirically-Calibrated AI-assisted Red-flag Prioritisation
# EC-ARP+ : EC-ARP + supervised XGBoost stacking signal
# ============================================================

from pathlib import Path
import json
import time
import shutil

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split
)

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler
)

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    IsolationForest
)

from sklearn.cluster import MiniBatchKMeans

from sklearn.decomposition import PCA

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
)

import xgboost as xgb


# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_STATE = 42

DATA_PATH = Path(
    "/kaggle/input/datasets/alamintokdershoukhin/creditcard/creditcard.csv"
)

RESULTS_DIR = Path(
    "/kaggle/working/model_outputs"
)

# Remove previous results to avoid stale files
if RESULTS_DIR.exists():
    shutil.rmtree(RESULTS_DIR)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

V_COLS = [
    f"V{i}"
    for i in range(1, 29)
]


# ============================================================
# 2. DATA LOADING
# ============================================================

def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    df["Class"] = (
        df["Class"]
        .astype(int)
    )

    if "Time" not in df.columns:
        raise ValueError(
            "The dataset must contain a 'Time' column."
        )

    return df


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

def engineer_signals(
    df,
    fit_on_idx
):
    """
    Compute the four components of Eq. (1):

        a(x_r) = anomaly score
        b(x_r) = behavioural-deviation score
        c(x_r) = control-rule violation
        f(x_r) = frequency/severity score

    Unsupervised components are fitted only on the training
    portion and then applied to the complete dataset.
    """

    out = pd.DataFrame(
        index=df.index
    )

    V = df[V_COLS].values

    # --------------------------------------------------------
    # Standardise V1-V28
    # --------------------------------------------------------

    scaler_V = StandardScaler().fit(
        V[fit_on_idx]
    )

    Vs = scaler_V.transform(V)


    # ========================================================
    # a(x_r): ISOLATION FOREST ANOMALY SCORE
    # ========================================================

    iso = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    iso.fit(
        Vs[fit_on_idx]
    )

    # Isolation Forest score_samples:
    # lower = more anomalous
    #
    # Negate so that:
    # higher = more anomalous

    raw_a = -iso.score_samples(Vs)

    a_scaler = MinMaxScaler().fit(
        raw_a[fit_on_idx]
        .reshape(-1, 1)
    )

    out["a_anomaly"] = (
        a_scaler
        .transform(
            raw_a.reshape(-1, 1)
        )
        .ravel()
    )


    # ========================================================
    # b(x_r): BEHAVIOURAL-DEVIATION SCORE
    # ========================================================

    kmeans = MiniBatchKMeans(
        n_clusters=12,
        random_state=RANDOM_STATE,
        n_init=10,
        batch_size=2048
    )

    kmeans.fit(
        Vs[fit_on_idx]
    )

    centroids = (
        kmeans.cluster_centers_
    )

    labels = kmeans.predict(
        Vs
    )

    dist = np.linalg.norm(
        Vs - centroids[labels],
        axis=1
    )

    b_scaler = MinMaxScaler().fit(
        dist[fit_on_idx]
        .reshape(-1, 1)
    )

    out["b_behavioural"] = (
        b_scaler
        .transform(
            dist.reshape(-1, 1)
        )
        .ravel()
    )


    # ========================================================
    # c(x_r): CONTROL-RULE VIOLATION
    # ========================================================

    # 99th percentile amount threshold
    amount_hi = np.percentile(
        df["Amount"].values[fit_on_idx],
        99
    )

    # IMPORTANT:
    # Time is elapsed seconds, not actual wall-clock time.
    # This creates a relative time bucket.
    hour_bucket = (
        df["Time"].values // 3600
    ).astype(int)

    # Use first 5 relative hours as the off-hours proxy
    off_hours = (
        hour_bucket % 24 < 5
    )

    large_amount = (
        df["Amount"].values
        > amount_hi
    )

    out["c_rule"] = (
        off_hours | large_amount
    ).astype(float)


    # ========================================================
    # f(x_r): FREQUENCY / SEVERITY
    # ========================================================

    # --------------------------------------------------------
    # Transaction severity
    # --------------------------------------------------------

    amt_scaler = MinMaxScaler().fit(
        df["Amount"]
        .values[fit_on_idx]
        .reshape(-1, 1)
    )

    severity = (
        amt_scaler
        .transform(
            df["Amount"]
            .values
            .reshape(-1, 1)
        )
        .ravel()
    )


    # --------------------------------------------------------
    # Local transaction density
    # +/- 60 seconds
    # --------------------------------------------------------

    time_values = (
        df["Time"]
        .values
    )

    time_sorted_idx = np.argsort(
        time_values
    )

    t_sorted = (
        time_values[
            time_sorted_idx
        ]
    )

    window = 60.0

    left = np.searchsorted(
        t_sorted,
        t_sorted - window,
        side="left"
    )

    right = np.searchsorted(
        t_sorted,
        t_sorted + window,
        side="right"
    )

    density_sorted = (
        right - left
    ).astype(float)

    density = np.empty_like(
        density_sorted
    )

    density[
        time_sorted_idx
    ] = density_sorted


    dens_scaler = MinMaxScaler().fit(
        density[fit_on_idx]
        .reshape(-1, 1)
    )

    density_n = (
        dens_scaler
        .transform(
            density.reshape(-1, 1)
        )
        .ravel()
    )

    out["f_freq_severity"] = (
        0.5 * severity
        + 0.5 * density_n
    )


    # ========================================================
    # PCA RECONSTRUCTION ERROR
    # ========================================================
    #
    # Used only as a standalone baseline.
    #
    # It is NOT included in EC-ARP / EC-ARP+.

    pca = PCA(
        n_components=10,
        random_state=RANDOM_STATE
    )

    pca.fit(
        Vs[fit_on_idx]
    )

    reconstructed = (
        pca.inverse_transform(
            pca.transform(Vs)
        )
    )

    recon_error = np.mean(
        (Vs - reconstructed) ** 2,
        axis=1
    )

    out["recon_error"] = (
        recon_error
    )

    return out


# ============================================================
# 4. TOP-K METRICS
# ============================================================

def topk_precision_recall(
    y_true,
    scores,
    frac=0.01
):

    y_true = np.asarray(
        y_true
    )

    scores = np.asarray(
        scores
    )

    n = len(y_true)

    k = max(
        1,
        int(
            np.ceil(
                n * frac
            )
        )
    )

    order = np.argsort(
        -scores
    )

    top_indices = (
        order[:k]
    )

    tp = (
        y_true[
            top_indices
        ].sum()
    )

    precision = (
        tp / k
    )

    total_positive = (
        y_true.sum()
    )

    recall = (
        tp / total_positive
        if total_positive > 0
        else 0.0
    )

    return (
        precision,
        recall,
        k,
        int(tp)
    )


# ============================================================
# 5. MODEL EVALUATION
# ============================================================

def evaluate_probability_model(
    name,
    y_true,
    scores
):

    y_true = np.asarray(
        y_true
    )

    scores = np.asarray(
        scores
    )

    predictions = (
        scores >= 0.5
    ).astype(int)

    precision_1pct, recall_1pct, k, tp = (
        topk_precision_recall(
            y_true,
            scores,
            frac=0.01
        )
    )

    return {
        "model": name,

        "roc_auc": roc_auc_score(
            y_true,
            scores
        ),

        "pr_auc": average_precision_score(
            y_true,
            scores
        ),

        "f1_at_0.5": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "precision_at_0.5": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "recall_at_0.5": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "precision_at_top1pct": precision_1pct,

        "recall_at_top1pct": recall_1pct,

        "top1pct_k": k,

        "top1pct_true_positives": tp
    }


# ============================================================
# 6. ANOMALY MODEL EVALUATION
# ============================================================

def evaluate_anomaly_model(
    name,
    y_true,
    scores
):

    y_true = np.asarray(
        y_true
    )

    scores = np.asarray(
        scores
    )

    precision_1pct, recall_1pct, k, tp = (
        topk_precision_recall(
            y_true,
            scores,
            frac=0.01
        )
    )

    return {
        "model": name,

        "roc_auc": roc_auc_score(
            y_true,
            scores
        ),

        "pr_auc": average_precision_score(
            y_true,
            scores
        ),

        # Not reported as "at 0.5" because
        # anomaly scores are not probabilities.
        "f1_at_0.5": np.nan,

        "precision_at_0.5": np.nan,

        "recall_at_0.5": np.nan,

        "precision_at_top1pct": precision_1pct,

        "recall_at_top1pct": recall_1pct,

        "top1pct_k": k,

        "top1pct_true_positives": tp
    }


# ============================================================
# 7. MAIN
# ============================================================

def main():

    start_time = time.time()

    print("=" * 75)
    print("STEP 2 OF 2 — EMPIRICAL MODEL VALIDATION")
    print("=" * 75)

    print("\nLoading dataset...")
    print(f"Dataset: {DATA_PATH}")

    df = load_data()

    df = df.reset_index(
        drop=True
    )

    n = len(df)

    n_fraud = int(
        df["Class"].sum()
    )

    fraud_rate = (
        100 * n_fraud / n
    )

    print(
        f"\nDataset shape: {df.shape}"
    )

    print(
        f"Transactions: {n:,}"
    )

    print(
        f"Fraud transactions: {n_fraud:,}"
    )

    print(
        f"Fraud rate: {fraud_rate:.4f}%"
    )


    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    print("\n" + "=" * 75)
    print("TRAIN / TEST SPLIT")
    print("=" * 75)

    indices = np.arange(
        n
    )

    (
        train_idx,
        test_idx,
        y_train,
        y_test
    ) = train_test_split(
        indices,
        df["Class"].values,
        test_size=0.30,
        stratify=df["Class"].values,
        random_state=RANDOM_STATE
    )

    print(
        f"\nTraining samples: {len(train_idx):,}"
    )

    print(
        f"Test samples: {len(test_idx):,}"
    )

    print(
        f"Training fraud cases: {int(y_train.sum()):,}"
    )

    print(
        f"Test fraud cases: {int(y_test.sum()):,}"
    )


    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    print("\n" + "=" * 75)
    print("FEATURE ENGINEERING — EQ. (1)")
    print("=" * 75)

    print(
        "\nEngineering a(x), b(x), c(x), f(x)..."
    )

    signals = engineer_signals(
        df,
        train_idx
    )

    print(
        "\nGenerated signals:"
    )

    print(
        signals.columns.tolist()
    )

    print(
        "\nSignal summary:"
    )

    print(
        signals.describe()
        .round(4)
        .to_string()
    )


    # ========================================================
    # RAW FEATURES
    # ========================================================

    X_raw = (
        df[
            V_COLS + ["Amount"]
        ]
        .values
    )

    X_raw_train = (
        X_raw[train_idx]
    )

    X_raw_test = (
        X_raw[test_idx]
    )

    scaler_raw = (
        StandardScaler()
        .fit(X_raw_train)
    )

    X_raw_train_s = (
        scaler_raw
        .transform(X_raw_train)
    )

    X_raw_test_s = (
        scaler_raw
        .transform(X_raw_test)
    )


    # ========================================================
    # RESULTS
    # ========================================================

    results = []


    # ========================================================
    # BASELINE 1 — LOGISTIC REGRESSION
    # ========================================================

    print("\n" + "=" * 75)
    print("BASELINE 1 — LOGISTIC REGRESSION")
    print("=" * 75)

    print(
        "\nTraining Logistic Regression..."
    )

    lr = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    lr.fit(
        X_raw_train_s,
        y_train
    )

    lr_test_scores = (
        lr.predict_proba(
            X_raw_test_s
        )[:, 1]
    )

    results.append(
        evaluate_probability_model(
            "Logistic Regression (raw features)",
            y_test,
            lr_test_scores
        )
    )

    print("Completed.")


    # ========================================================
    # BASELINE 2 — RANDOM FOREST
    # ========================================================

    print("\n" + "=" * 75)
    print("BASELINE 2 — RANDOM FOREST")
    print("=" * 75)

    print(
        "\nTraining Random Forest..."
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    rf.fit(
        X_raw_train,
        y_train
    )

    rf_test_scores = (
        rf.predict_proba(
            X_raw_test
        )[:, 1]
    )

    results.append(
        evaluate_probability_model(
            "Random Forest (raw features)",
            y_test,
            rf_test_scores
        )
    )

    print("Completed.")


    # ========================================================
    # BASELINE 3 — XGBOOST
    # ========================================================

    print("\n" + "=" * 75)
    print("BASELINE 3 — XGBOOST")
    print("=" * 75)

    print(
        "\nTraining XGBoost..."
    )

    scale_pos_weight = (
        (len(y_train) - y_train.sum())
        / y_train.sum()
    )

    xgb_clf = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    xgb_clf.fit(
        X_raw_train,
        y_train
    )

    xgb_test_scores = (
        xgb_clf.predict_proba(
            X_raw_test
        )[:, 1]
    )

    results.append(
        evaluate_probability_model(
            "XGBoost (raw features)",
            y_test,
            xgb_test_scores
        )
    )

    print("Completed.")


    # ========================================================
    # BASELINE 4 — ISOLATION FOREST
    # ========================================================

    print("\n" + "=" * 75)
    print("BASELINE 4 — ISOLATION FOREST")
    print("=" * 75)

    print(
        "\nEvaluating Isolation Forest..."
    )

    iso_scores = (
        signals[
            "a_anomaly"
        ]
        .values[
            test_idx
        ]
    )

    results.append(
        evaluate_anomaly_model(
            "Isolation Forest alone (a only)",
            y_test,
            iso_scores
        )
    )

    print("Completed.")


    # ========================================================
    # BASELINE 5 — PCA RECONSTRUCTION
    # ========================================================

    print("\n" + "=" * 75)
    print("BASELINE 5 — PCA RECONSTRUCTION ERROR")
    print("=" * 75)

    print(
        "\nEvaluating PCA reconstruction error..."
    )

    pca_scores = (
        signals[
            "recon_error"
        ]
        .values[
            test_idx
        ]
    )

    results.append(
        evaluate_anomaly_model(
            "PCA reconstruction error (autoencoder proxy)",
            y_test,
            pca_scores
        )
    )

    print("Completed.")


    # ========================================================
    # EC-ARP
    # ========================================================

    print("\n" + "=" * 75)
    print("PROPOSED MODEL — EC-ARP")
    print("=" * 75)

    print(
        "\nTraining empirically calibrated Eq. (1) model..."
    )

    meta_cols = [
        "a_anomaly",
        "b_behavioural",
        "c_rule",
        "f_freq_severity"
    ]

    Xm_train = (
        signals
        .loc[
            train_idx,
            meta_cols
        ]
        .values
    )

    Xm_test = (
        signals
        .loc[
            test_idx,
            meta_cols
        ]
        .values
    )

    meta_ecarp = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    meta_ecarp.fit(
        Xm_train,
        y_train
    )

    ecarp_scores = (
        meta_ecarp
        .predict_proba(
            Xm_test
        )[:, 1]
    )

    results.append(
        evaluate_probability_model(
            "EC-ARP (proposed: calibrated Eq.1)",
            y_test,
            ecarp_scores
        )
    )


    # ========================================================
    # EC-ARP WEIGHTS
    # ========================================================

    raw_weights = (
        meta_ecarp
        .coef_
        .ravel()
    )

    positive_weights = np.clip(
        raw_weights,
        0,
        None
    )

    if positive_weights.sum() > 0:

        normalized_weights = (
            positive_weights
            / positive_weights.sum()
        )

    else:

        normalized_weights = (
            np.ones(
                len(positive_weights)
            )
            / len(positive_weights)
        )

    calibrated_weights = dict(
        zip(
            [
                "w1_a",
                "w2_b",
                "w3_c",
                "w4_f"
            ],
            normalized_weights.tolist()
        )
    )

    print(
        "\nEC-ARP calibrated weights:"
    )

    for key, value in calibrated_weights.items():

        print(
            f"  {key}: {value:.6f}"
        )


    # ========================================================
    # EC-ARP+
    # ========================================================

    print("\n" + "=" * 75)
    print("PROPOSED MODEL — EC-ARP+")
    print("=" * 75)

    print(
        "\nGenerating out-of-fold XGBoost predictions..."
    )

    oof_predictions = np.zeros(
        len(train_idx)
    )

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    for fold_number, (
        fold_train,
        fold_valid
    ) in enumerate(
        skf.split(
            X_raw_train,
            y_train
        ),
        start=1
    ):

        print(
            f"  Training OOF fold {fold_number}/5..."
        )

        fold_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        fold_model.fit(
            X_raw_train[
                fold_train
            ],
            y_train[
                fold_train
            ]
        )

        oof_predictions[
            fold_valid
        ] = (
            fold_model
            .predict_proba(
                X_raw_train[
                    fold_valid
                ]
            )[:, 1]
        )


    print(
        "\nTraining EC-ARP+ meta-learner..."
    )

    Xm_train_plus = np.column_stack(
        [
            Xm_train,
            oof_predictions
        ]
    )

    Xm_test_plus = np.column_stack(
        [
            Xm_test,
            xgb_test_scores
        ]
    )

    meta_plus = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    meta_plus.fit(
        Xm_train_plus,
        y_train
    )

    ecarp_plus_scores = (
        meta_plus
        .predict_proba(
            Xm_test_plus
        )[:, 1]
    )

    results.append(
        evaluate_probability_model(
            "EC-ARP+ (proposed: stacked hybrid)",
            y_test,
            ecarp_plus_scores
        )
    )


    # ========================================================
    # EC-ARP+ WEIGHTS
    # ========================================================

    raw_weights_plus = (
        meta_plus
        .coef_
        .ravel()
    )

    positive_weights_plus = np.clip(
        raw_weights_plus,
        0,
        None
    )

    if positive_weights_plus.sum() > 0:

        normalized_weights_plus = (
            positive_weights_plus
            / positive_weights_plus.sum()
        )

    else:

        normalized_weights_plus = (
            np.ones(
                len(positive_weights_plus)
            )
            / len(positive_weights_plus)
        )

    calibrated_weights_plus = dict(
        zip(
            [
                "w1_a",
                "w2_b",
                "w3_c",
                "w4_f",
                "w5_supervised"
            ],
            normalized_weights_plus.tolist()
        )
    )

    print(
        "\nEC-ARP+ calibrated weights:"
    )

    for key, value in calibrated_weights_plus.items():

        print(
            f"  {key}: {value:.6f}"
        )


    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            "pr_auc",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # SAVE MODEL RESULTS
    # ========================================================

    results_df.to_csv(
        RESULTS_DIR
        / "model_comparison.csv",
        index=False
    )

    with open(
        RESULTS_DIR
        / "calibrated_weights.json",
        "w"
    ) as file:

        json.dump(
            {
                "EC-ARP":
                    calibrated_weights,

                "EC-ARP+":
                    calibrated_weights_plus
            },
            file,
            indent=2
        )


    metadata = {

        "n_transactions":
            int(n),

        "n_fraud":
            int(n_fraud),

        "fraud_rate_pct":
            fraud_rate,

        "train_size":
            int(len(train_idx)),

        "test_size":
            int(len(test_idx)),

        "random_state":
            RANDOM_STATE,

        "runtime_seconds":
            time.time() - start_time,

        "dataset":
            "ULB Credit Card Fraud Detection "
            "(Dal Pozzolo et al., 2015, "
            "doi:10.1109/SSCI.2015.33)"
    }

    with open(
        RESULTS_DIR
        / "run_metadata.json",
        "w"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2
        )


    # ========================================================
    # PRINT MODEL COMPARISON
    # ========================================================

    print("\n" + "=" * 75)
    print("MODEL COMPARISON — HELD-OUT TEST SET")
    print("=" * 75)

    display_df = (
        results_df[
            [
                "model",
                "roc_auc",
                "pr_auc",
                "f1_at_0.5",
                "precision_at_0.5",
                "recall_at_0.5",
                "precision_at_top1pct",
                "recall_at_top1pct"
            ]
        ]
        .copy()
    )

    numeric_columns = (
        display_df
        .select_dtypes(
            include=np.number
        )
        .columns
    )

    display_df[
        numeric_columns
    ] = display_df[
        numeric_columns
    ].round(4)

    print(
        display_df.to_string(
            index=False
        )
    )


    # ========================================================
    # TOP 1% INVESTIGATION CAPACITY
    # ========================================================

    print("\n" + "=" * 75)
    print("TOP 1% INVESTIGATION PRIORITISATION")
    print("=" * 75)

    top1_df = (
        results_df[
            [
                "model",
                "precision_at_top1pct",
                "recall_at_top1pct",
                "top1pct_k",
                "top1pct_true_positives"
            ]
        ]
        .copy()
    )

    top1_df[
        [
            "precision_at_top1pct",
            "recall_at_top1pct"
        ]
    ] = (
        top1_df[
            [
                "precision_at_top1pct",
                "recall_at_top1pct"
            ]
        ]
        .round(4)
    )

    print(
        top1_df.to_string(
            index=False
        )
    )


    # ========================================================
    # FIGURE 1 — PRECISION-RECALL CURVES
    # ========================================================

    print("\nGenerating Figure 1: Precision-Recall Curves...")

    curve_models = [

        (
            "LogReg",
            lr_test_scores
        ),

        (
            "Random Forest",
            rf_test_scores
        ),

        (
            "XGBoost",
            xgb_test_scores
        ),

        (
            "EC-ARP",
            ecarp_scores
        ),

        (
            "EC-ARP+",
            ecarp_plus_scores
        )
    ]

    plt.figure(
        figsize=(7, 5.5)
    )

    for name, scores in curve_models:

        precision, recall, _ = (
            precision_recall_curve(
                y_test,
                scores
            )
        )

        ap = (
            average_precision_score(
                y_test,
                scores
            )
        )

        plt.plot(
            recall,
            precision,
            label=f"{name} (AP={ap:.3f})"
        )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        "Precision-Recall Curves — Held-Out Test Set"
    )

    plt.legend(
        fontsize=8
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / "pr_curves.png",
        dpi=300,
        bbox_inches="tight"
    )

    # DISPLAY DIRECTLY IN KAGGLE
    plt.show()

    plt.close()


    # ========================================================
    # FIGURE 2 — ROC CURVES
    # ========================================================

    print("\nGenerating Figure 2: ROC Curves...")

    plt.figure(
        figsize=(7, 5.5)
    )

    for name, scores in curve_models:

        fpr, tpr, _ = (
            roc_curve(
                y_test,
                scores
            )
        )

        auc = (
            roc_auc_score(
                y_test,
                scores
            )
        )

        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC={auc:.3f})"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        "k--",
        linewidth=0.8
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curves — Held-Out Test Set"
    )

    plt.legend(
        fontsize=8
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / "roc_curves.png",
        dpi=300,
        bbox_inches="tight"
    )

    # DISPLAY DIRECTLY IN KAGGLE
    plt.show()

    plt.close()


    # ========================================================
    # FIGURE 3 — CALIBRATED WEIGHTS
    # ========================================================

    print(
        "\nGenerating Figure 3: "
        "EC-ARP+ Calibrated Weights..."
    )

    labels = list(
        calibrated_weights_plus.keys()
    )

    values = list(
        calibrated_weights_plus.values()
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        labels,
        values
    )

    plt.ylabel(
        "Normalised calibrated weight"
    )

    plt.xlabel(
        "Eq. (1) component"
    )

    plt.title(
        "EC-ARP+ Calibrated Weights"
    )

    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / "calibrated_weights.png",
        dpi=300,
        bbox_inches="tight"
    )

    # DISPLAY DIRECTLY IN KAGGLE
    plt.show()

    plt.close()


    # ========================================================
    # FIGURE 4 — PR-AUC COMPARISON
    # ========================================================

    print(
        "\nGenerating Figure 4: "
        "PR-AUC Model Comparison..."
    )

    plot_df = results_df.copy()

    plt.figure(
        figsize=(10, 6)
    )

    sns.barplot(
        data=plot_df,
        x="pr_auc",
        y="model"
    )

    plt.xlabel(
        "PR-AUC / Average Precision"
    )

    plt.ylabel(
        "Model"
    )

    plt.title(
        "Model Comparison Based on PR-AUC"
    )

    plt.grid(
        axis="x",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / "pr_auc_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    # DISPLAY DIRECTLY IN KAGGLE
    plt.show()

    plt.close()


    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    runtime = (
        time.time()
        - start_time
    )

    print("\n" + "=" * 75)
    print("STEP 2 COMPLETED SUCCESSFULLY")
    print("=" * 75)

    print(
        f"\nTotal runtime: {runtime:.1f} seconds"
    )

    print(
        f"\nResults saved to:"
    )

    print(
        RESULTS_DIR
    )

    print(
        "\nGenerated files:"
    )

    for file_path in sorted(
        RESULTS_DIR.iterdir()
    ):

        print(
            f"  - {file_path.name}"
        )


    # ========================================================
    # LIMITATIONS
    # ========================================================

    print("\n" + "=" * 75)
    print("LIMITATIONS")
    print("=" * 75)

    print(
        """
1. The ULB dataset does not contain customer/entity identifiers.
   Therefore, the behavioural-deviation and frequency signals
   are dataset-level proxies rather than true per-account
   longitudinal behavioural profiles.

2. V1-V28 are PCA-anonymised features. Consequently, individual
   feature-level business interpretation is limited.

3. The c(x_r) control rule uses a large-amount threshold and a
   relative-time off-hours proxy. These are operational rules
   designed for reproducibility and should not be interpreted
   as universal fraud rules.

4. Results are based on one stratified 70/30 train-test split
   with RANDOM_STATE=42. They should therefore be interpreted
   as empirical results for this experimental configuration,
   rather than confidence intervals over repeated samples.

5. The top-1% metric represents an operational investigation
   capacity scenario: only the highest-ranked 1% of transactions
   are prioritised for review.

6. PR-AUC is particularly important for this highly imbalanced
   fraud-detection problem because ROC-AUC can remain high even
   when the precision of positive predictions is limited.
"""
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()