"""
Step 1 of 2 -- Dataset analysis / EDA for the "Credit Card Fraud Detection" dataset.

Run this on Kaggle as a notebook cell script attached to the dataset:
    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
(Add it via "+ Add Input" in a new Kaggle Notebook; the file will appear at
/kaggle/input/creditcardfraud/creditcard.csv)

Citation for the dataset (cite this in the paper, not Kaggle itself):
  Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015).
  "Calibrating Probability with Undersampling for Unbalanced Classification."
  2015 IEEE Symposium Series on Computational Intelligence, 159-166.
  doi:10.1109/SSCI.2015.33

What this script does (no modelling yet -- pure data understanding, feeds the
manuscript's "Dataset" subsection):
  1. Loads the raw CSV and reports shape, dtypes, missing values.
  2. Reports the class-imbalance ratio (fraud vs. non-fraud).
  3. Descriptive statistics for Amount and Time, split by class.
  4. Correlation of each PCA feature (V1-V28) with the fraud label.
  5. Saves figures: class balance bar chart, Amount distribution (log scale)
     by class, transactions-per-hour histogram by class, and a correlation
     bar chart -- all to ./eda_outputs/.
  6. Prints a plain-text summary (paste this into the paper's dataset
     description / share as the "log").

Run:  python kaggle_01_eda.py
"""

from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- locate the data: Kaggle input path first, local fallback second ---
CANDIDATE_PATHS = [
    Path("/kaggle/input/datasets/alamintokdershoukhin/creditcard/creditcard.csv"),
    Path("/kaggle/input/creditcardfraud/creditcard.csv"),
    Path(__file__).parent / "creditcard.csv",
    Path("creditcard.csv"),
]
DATA_PATH = next((p for p in CANDIDATE_PATHS if p.exists()), None)
if DATA_PATH is None:
    raise FileNotFoundError(
        "creditcard.csv not found. On Kaggle: add the "
        "'alamintokdershoukhin/creditcard' (or 'mlg-ulb/creditcardfraud') dataset "
        "as a notebook input. Locally: place creditcard.csv next to this script."
    )

OUT_DIR = Path("eda_outputs") if Path("/kaggle/working").exists() else Path(__file__).parent / "eda_outputs"
OUT_DIR.mkdir(exist_ok=True)

V_COLS = [f"V{i}" for i in range(1, 29)]


def main():
    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Missing values total: {int(df.isna().sum().sum())}")
    print(f"Duplicate rows: {int(df.duplicated().sum())}")

    n = len(df)
    n_fraud = int(df["Class"].sum())
    n_legit = n - n_fraud
    print("\n--- Class balance ---")
    print(f"Legitimate: {n_legit} ({100*n_legit/n:.4f}%)")
    print(f"Fraudulent: {n_fraud} ({100*n_fraud/n:.4f}%)")
    print(f"Imbalance ratio (legit:fraud): {n_legit/n_fraud:.1f}:1")

    print("\n--- Amount, by class ---")
    print(df.groupby("Class")["Amount"].describe())

    if "Time" in df.columns:
        df["hour"] = (df["Time"] % 86400) // 3600
        print("\n--- Transactions by hour-of-day (fraud rate) ---")
        hourly = df.groupby("hour")["Class"].agg(["count", "sum"])
        hourly["fraud_rate_pct"] = 100 * hourly["sum"] / hourly["count"]
        print(hourly)
    else:
        print("\n[WARN] No 'Time' column found; skipping hour-of-day analysis.")

    print("\n--- Correlation of V1-V28 with Class (top 10 by |corr|) ---")
    corr = df[V_COLS + ["Class"]].corr()["Class"].drop("Class")
    top_corr = corr.reindex(corr.abs().sort_values(ascending=False).index).head(10)
    print(top_corr)

    # ---------------- Figures ----------------
    plt.figure(figsize=(4, 4))
    plt.bar(["Legitimate", "Fraud"], [n_legit, n_fraud], color=["#4C72B0", "#C44E52"])
    plt.yscale("log")
    plt.ylabel("Count (log scale)")
    plt.title("Class balance")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "class_balance.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(df.loc[df.Class == 0, "Amount"], bins=60, alpha=0.6, label="Legitimate", density=True)
    plt.hist(df.loc[df.Class == 1, "Amount"], bins=60, alpha=0.6, label="Fraud", density=True)
    plt.xscale("symlog")
    plt.xlabel("Amount (symlog scale)")
    plt.ylabel("Density")
    plt.title("Transaction amount distribution by class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "amount_distribution.png", dpi=200)
    plt.close()

    if "hour" in df.columns:
        plt.figure(figsize=(7, 4))
        rate = df.groupby("hour")["Class"].mean() * 100
        plt.bar(rate.index, rate.values, color="#DD8452")
        plt.xlabel("Hour of day (derived from Time)")
        plt.ylabel("Fraud rate (%)")
        plt.title("Fraud rate by hour of day")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "fraud_rate_by_hour.png", dpi=200)
        plt.close()

    plt.figure(figsize=(6, 5))
    top_corr.sort_values().plot(kind="barh", color="#55A868")
    plt.xlabel("Pearson correlation with Class")
    plt.title("Top 10 |correlation| features vs. fraud label")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "feature_correlation.png", dpi=200)
    plt.close()

    summary = {
        "n_transactions": n,
        "n_fraud": n_fraud,
        "fraud_rate_pct": 100 * n_fraud / n,
        "n_features": len(V_COLS) + 1,  # + Amount
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    pd.Series(summary).to_csv(OUT_DIR / "dataset_summary.csv")
    print(f"\nFigures + summary written to {OUT_DIR}/")
    print("Copy dataset_summary.csv + the printed stats above into the paper's dataset subsection.")


if __name__ == "__main__":
    main()
