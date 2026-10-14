"""Flat vs order-aware metrics.

Claim 1 says the official benchmark's metrics (Accuracy, AUROC) cannot see label order,
so on an ordinal task they score a catastrophic confusion the same as a benign one.
This module provides both families so the same predictions can be scored each way and
the rankings compared.
"""
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score


# ---------- order-blind (what the official benchmark uses) ----------

def accuracy(y, yhat, **_):
    return float(accuracy_score(y, yhat))


def auroc_ovr(y, proba, **_):
    """One-vs-rest macro AUROC, as in torchcross.cd.metrics.AUROC."""
    classes = np.unique(y)
    if len(classes) < 2:
        return float("nan")
    try:
        if proba.shape[1] == 2:
            return float(roc_auc_score(y, proba[:, 1]))
        return float(roc_auc_score(y, proba, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


# ---------- order-aware ----------

def balanced_accuracy(y, yhat, **_):
    """Balanced accuracy = mean per-class recall. TCLA's reported metric (BACC).

    Unlike plain accuracy this is not inflated by predicting the majority class, so it is
    the right headline number on the skewed medical datasets we are using (majority-class
    baselines here run 0.50-0.72 in plain accuracy but 1/C in BACC).
    """
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y, yhat))


def qwk(y, yhat, n_classes=None, **_):
    """Quadratic weighted kappa. Penalises errors by SQUARED distance along the scale."""
    labels = list(range(n_classes)) if n_classes else None
    return float(cohen_kappa_score(y, yhat, weights="quadratic", labels=labels))


def lwk(y, yhat, n_classes=None, **_):
    """Linear weighted kappa."""
    labels = list(range(n_classes)) if n_classes else None
    return float(cohen_kappa_score(y, yhat, weights="linear", labels=labels))


def mae(y, yhat, **_):
    """Mean absolute error in grades. Clinically: how many grades off, on average."""
    return float(np.abs(np.asarray(y) - np.asarray(yhat)).mean())


def adjacent_accuracy(y, yhat, tol=1, **_):
    """Fraction predicted within `tol` grades. The clinically tolerable-error rate."""
    return float((np.abs(np.asarray(y) - np.asarray(yhat)) <= tol).mean())


def max_grade_error(y, yhat, **_):
    """Worst single error. Surfaces catastrophic confusions that means hide."""
    return float(np.abs(np.asarray(y) - np.asarray(yhat)).max())


FLAT = {"accuracy": accuracy, "bacc": balanced_accuracy}
FLAT_PROBA = {"auroc": auroc_ovr}
ORDINAL = {"qwk": qwk, "lwk": lwk, "mae": mae,
           "adjacent_acc": adjacent_accuracy, "max_grade_err": max_grade_error}


def evaluate(y, yhat, proba=None, n_classes=None):
    out = {}
    for name, fn in FLAT.items():
        out[name] = fn(y, yhat)
    if proba is not None:
        for name, fn in FLAT_PROBA.items():
            out[name] = fn(y, proba)
    for name, fn in ORDINAL.items():
        out[name] = fn(y, yhat, n_classes=n_classes)
    return out


if __name__ == "__main__":
    # The demonstration that motivates the whole paper.
    # 5-grade ICDR scale. Two classifiers, IDENTICAL accuracy, opposite clinical safety.
    C = 5
    y = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])

    # A: every error is off-by-one (clinically mild)
    a = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 3])
    # B: same NUMBER of errors, but they are maximal-distance (clinically catastrophic:
    #    proliferative DR called no-DR, and vice versa)
    b = np.array([0, 4, 1, 4, 2, 4, 3, 0, 4, 0])

    ra = evaluate(y, a, n_classes=C)
    rb = evaluate(y, b, n_classes=C)

    print("Two classifiers on a 5-grade ordinal scale (ICDR-like)\n")
    print(f"{'metric':<16} {'A (off-by-one)':>16} {'B (catastrophic)':>18} {'sees order?':>13}")
    print("-" * 68)
    order_blind = {"accuracy", "auroc"}
    for m in ("accuracy", "qwk", "lwk", "mae", "adjacent_acc", "max_grade_err"):
        tag = "NO" if m in order_blind else "yes"
        print(f"{m:<16} {ra[m]:>16.3f} {rb[m]:>18.3f} {tag:>13}")

    assert ra["accuracy"] == rb["accuracy"], "accuracy must be identical for the point to land"
    print(f"\nAccuracy is IDENTICAL ({ra['accuracy']:.1f}) for both.")
    print(f"QWK separates them decisively: {ra['qwk']:.3f} vs {rb['qwk']:.3f}")
    print(f"MAE:              {ra['mae']:.2f} grades vs {rb['mae']:.2f} grades")
    print(f"Worst error:      {ra['max_grade_err']:.0f} grade  vs {rb['max_grade_err']:.0f} grades")
    print("\nA benchmark that reports only accuracy/AUROC cannot distinguish these two")
    print("systems. The official MedIMeta FSL benchmark reports only accuracy and AUROC,")
    print("and selects hyperparameters on AUROC, on tasks its own metadata marks ordinal.")
