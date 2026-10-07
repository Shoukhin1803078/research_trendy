"""
Step 16: Leakage-stratified accuracy.

The dedup audit (step9 / step15) says what fraction of the dataset is
duplicated. It does not say what that duplication BOUGHT the model. A
contaminated-vs-cleaned comparison of two separate training runs gets at
that, but it confounds the leakage with everything else that changes when
you retrain on 16% less data (different folds, different class balance,
fewer gradient steps).

This script measures the effect directly, inside a single contaminated run,
with no retraining at all. For every test image in every fold it asks:

    did a duplicate or near-duplicate of this image sit in the part of the
    dataset the model was fitted on for THIS fold?

and then reports accuracy separately for the images where the answer is yes
and the images where it is no. Both groups come from the same model, the
same fold and the same forward pass, so the gap between them is attributable
to the leakage rather than to retraining.

A large gap -- leaked images scoring near-perfectly while clean images score
several points lower -- is direct causal evidence for the mechanism, and the
clean-subset accuracy is the honest estimate of what the pipeline does on
genuinely unseen data. A gap near zero is also a publishable result: it says
the contamination is real but did not inflate the headline number, which is
worth knowing and is not what most readers would assume.

Inputs
------
--predictions  per_image_predictions.csv written by the patched
               step6_backbone_sweep.py (one row per test image per fold).
               It must come from a CONTAMINATED run, i.e. one with
               CLEANED_FILE_LIST = None; a cleaned run has no leaked images
               left to stratify by and the script will say so.
--audit-dir    directory holding exact_duplicates.json and
               near_duplicate_confirmed.csv (step9's dedup_audit/, or any
               step15 --out-dir).

Path keying
-----------
The audit and the training run may record different absolute prefixes (a
Kaggle input mount vs. a local copy). Paths are therefore keyed on their
last --key-depth components (default 3, e.g. Training/glioma/Tr-gl_1.jpg).
The script prints how many prediction paths it matched into the audit's path
space; if that number is low, the keying is wrong and every result below is
meaningless, so it refuses to continue past --min-match.

Output: a printed table, plus stratified_accuracy.csv and .json.

Pure stdlib + numpy. No GPU, no retraining, seconds to run.
"""

import os
import sys
import csv
import json
import math
import argparse
from collections import defaultdict

import numpy as np


# ---------------------------------------------------------------------------
# Notebook config -- EDIT THIS when running inside a Kaggle/Jupyter cell.
# Pasting this file into a notebook cell means argparse sees the KERNEL's
# argv and exits with SystemExit: 2, so these values are used whenever the
# script runs in a notebook or with no arguments. A real command line, or
# main([...]) with explicit arguments, overrides them.
# ---------------------------------------------------------------------------
NB_PREDICTIONS = "/kaggle/working/per_image_predictions.csv"
NB_AUDIT_DIR = "/kaggle/working/dedup_audit"
NB_OUT_DIR = "/kaggle/working"
NB_KEY_DEPTH = 3
NB_MIN_MATCH = 0.95


def _running_in_notebook():
    return "ipykernel" in sys.modules


def _notebook_args():
    return argparse.Namespace(
        predictions=NB_PREDICTIONS,
        audit_dir=NB_AUDIT_DIR,
        out_dir=NB_OUT_DIR,
        key_depth=NB_KEY_DEPTH,
        min_match=NB_MIN_MATCH,
    )


METHODS = [("y_pred_cnn", "CNN (softmax)"),
           ("y_pred_tta", "CNN + TTA"),
           ("y_pred_ens", "Embedding ensemble")]


# ---------------------------------------------------------------------------
# Path keying
# ---------------------------------------------------------------------------
def key_of(path, depth):
    parts = path.replace("\\", "/").rstrip("/").split("/")
    return "/".join(parts[-depth:]) if depth > 0 else parts[-1]


# ---------------------------------------------------------------------------
# Partner graph from the audit outputs
# ---------------------------------------------------------------------------
def load_partners(audit_dir, depth):
    """Return {key: {"exact": {keys}, "near": {keys}}} from the audit files."""
    partners = defaultdict(lambda: {"exact": set(), "near": set()})

    exact_path = os.path.join(audit_dir, "exact_duplicates.json")
    n_exact_pairs = 0
    if os.path.exists(exact_path):
        with open(exact_path) as f:
            groups = json.load(f)
        for group in groups:
            keys = [key_of(p, depth) for p in group]
            for i in range(len(keys)):
                for j in range(len(keys)):
                    if i != j:
                        partners[keys[i]]["exact"].add(keys[j])
                        n_exact_pairs += 1
        n_exact_pairs //= 2
    else:
        print(f"  [WARN] no exact_duplicates.json in {audit_dir}")

    near_path = os.path.join(audit_dir, "near_duplicate_confirmed.csv")
    n_near_pairs = 0
    if os.path.exists(near_path):
        with open(near_path) as f:
            for row in csv.DictReader(f):
                k1, k2 = key_of(row["path1"], depth), key_of(row["path2"], depth)
                partners[k1]["near"].add(k2)
                partners[k2]["near"].add(k1)
                n_near_pairs += 1
    else:
        print(f"  [WARN] no near_duplicate_confirmed.csv in {audit_dir}")

    print(f"  partner graph: {n_exact_pairs} exact pairs, "
          f"{n_near_pairs} confirmed near-duplicate pairs, "
          f"{len(partners)} images with >=1 partner")
    return partners


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------
def load_predictions(path, depth):
    """Return {backbone: {fold: [row,...]}} with a 'key' added per row."""
    by_backbone = defaultdict(lambda: defaultdict(list))
    n = 0
    with open(path) as f:
        reader = csv.DictReader(f)
        missing = {"backbone", "fold", "path", "y_true"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path} is missing column(s) {sorted(missing)}. Expected the "
                f"per-image CSV from the patched step6_backbone_sweep.py; the "
                f"aggregate backbone_sweep.jsonl will not work here."
            )
        for row in reader:
            row["key"] = key_of(row["path"], depth)
            by_backbone[row["backbone"]][int(row["fold"])].append(row)
            n += 1
    print(f"  {n} prediction rows over {len(by_backbone)} backbone(s)")
    return by_backbone


# ---------------------------------------------------------------------------
# Statistics: two-proportion z-test (no scipy dependency)
# ---------------------------------------------------------------------------
def two_proportion_z(c1, n1, c2, n2):
    """Two-sided p-value for H0: the two accuracies are equal."""
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    p1, p2 = c1 / n1, c2 / n2
    pool = (c1 + c2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, (1.0 if p1 == p2 else 0.0)
    z = (p1 - p2) / se
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return z, p


# ---------------------------------------------------------------------------
# Core stratification
# ---------------------------------------------------------------------------
def stratify(rows_by_fold, partners):
    """Label every test row with the leakage status it had in its own fold.

    Every image is tested exactly once across the folds, so the union of all
    folds' test keys is the full dataset, and for a given fold everything
    outside that fold's test set is what the model was fitted on (train plus
    the validation split used for early stopping -- both are 'seen').
    """
    all_keys = set()
    for rows in rows_by_fold.values():
        all_keys.update(r["key"] for r in rows)

    tagged = []
    for fold, rows in sorted(rows_by_fold.items()):
        test_keys = {r["key"] for r in rows}
        seen_keys = all_keys - test_keys
        for r in rows:
            p = partners.get(r["key"])
            if p is None:
                status = "clean"
            elif p["exact"] & seen_keys:
                status = "exact_in_train"
            elif p["near"] & seen_keys:
                status = "near_in_train"
            else:
                status = "clean"
            out = dict(r)
            out["leak_status"] = status
            tagged.append(out)
    return tagged, len(all_keys)


def accuracy(rows, pred_col):
    if not rows:
        return 0, 0, float("nan")
    correct = sum(1 for r in rows if r[pred_col] == r["y_true"])
    return correct, len(rows), correct / len(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions",
                    help="per_image_predictions.csv from a CONTAMINATED run; "
                         "omit to use the NB_PREDICTIONS config block")
    ap.add_argument("--audit-dir",
                    help="dir with exact_duplicates.json + "
                         "near_duplicate_confirmed.csv; omit to use "
                         "NB_AUDIT_DIR")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--key-depth", type=int, default=3,
                    help="trailing path components used to match prediction "
                         "paths against audit paths (default 3)")
    ap.add_argument("--min-match", type=float, default=0.95,
                    help="abort if fewer than this fraction of prediction "
                         "paths key into the audit's path space (default 0.95)")

    if argv is not None:
        args = ap.parse_args(argv)
    elif _running_in_notebook() or len(sys.argv) <= 1:
        args = _notebook_args()
        print("No CLI arguments (notebook or bare run) -- using the NB_* "
              "config block at the top of this file:")
        print(f"  predictions : {args.predictions}")
        print(f"  audit-dir   : {args.audit_dir}\n")
    else:
        args = ap.parse_args()

    if not args.predictions:
        ap.error("no predictions CSV: pass --predictions or set NB_PREDICTIONS")
    if not args.audit_dir:
        ap.error("no audit dir: pass --audit-dir or set NB_AUDIT_DIR")
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading audit...")
    partners = load_partners(args.audit_dir, args.key_depth)
    print("Loading predictions...")
    preds = load_predictions(args.predictions, args.key_depth)

    # --- keying sanity check: do the two path spaces actually overlap? ----
    audit_keys = set(partners)
    pred_keys = {r["key"] for bb in preds.values()
                 for rows in bb.values() for r in rows}
    if audit_keys:
        overlap = len(audit_keys & pred_keys) / len(audit_keys)
        print(f"\n  keying check: {overlap * 100:.1f}% of the audit's "
              f"{len(audit_keys)} partnered images are present in the "
              f"prediction set")
        if overlap < args.min_match:
            raise SystemExit(
                f"\nABORT: only {overlap * 100:.1f}% of audited images were "
                f"found in the predictions (threshold {args.min_match * 100:.0f}%).\n"
                f"Either --key-depth {args.key_depth} is wrong for these path "
                f"layouts, or the predictions came from a CLEANED run (in "
                f"which case the duplicates are already gone and there is "
                f"nothing to stratify -- rerun step6 with "
                f"CLEANED_FILE_LIST = None).\n"
                f"  example audit key:      "
                f"{sorted(audit_keys)[0] if audit_keys else '-'}\n"
                f"  example prediction key: "
                f"{sorted(pred_keys)[0] if pred_keys else '-'}"
            )

    out_rows, summary = [], {}
    pooled = defaultdict(list)

    for backbone, rows_by_fold in sorted(preds.items()):
        tagged, n_images = stratify(rows_by_fold, partners)
        for r in tagged:
            pooled[r["leak_status"]].append(r)

        counts = defaultdict(int)
        for r in tagged:
            counts[r["leak_status"]] += 1
        n_leaked = counts["exact_in_train"] + counts["near_in_train"]

        print(f"\n{'=' * 74}")
        print(f"{backbone}  --  {n_images} images, "
              f"{n_leaked} ({100 * n_leaked / max(n_images, 1):.1f}%) had a "
              f"duplicate partner in their own fold's training data")
        print(f"  exact partner in train: {counts['exact_in_train']}")
        print(f"  near  partner in train: {counts['near_in_train']}")
        print(f"  clean (no partner seen): {counts['clean']}")
        print(f"{'=' * 74}")
        print(f"{'method':22s} {'leaked':>16s} {'clean':>16s} {'gap':>8s} {'p':>9s}")

        leaked = [r for r in tagged if r["leak_status"] != "clean"]
        clean = [r for r in tagged if r["leak_status"] == "clean"]
        summary[backbone] = {"n_images": n_images,
                             "counts": dict(counts), "methods": {}}

        for col, label in METHODS:
            if not tagged or col not in tagged[0]:
                continue
            lc, ln, la = accuracy(leaked, col)
            cc, cn, ca = accuracy(clean, col)
            gap = (la - ca) if (ln and cn) else float("nan")
            _z, p = two_proportion_z(lc, ln, cc, cn)
            print(f"{label:22s} {la * 100:7.2f}% ({ln:5d}) "
                  f"{ca * 100:7.2f}% ({cn:5d}) "
                  f"{gap * 100:+7.2f}pp {p:9.2e}")
            summary[backbone]["methods"][col] = {
                "leaked_acc": round(la, 6), "leaked_n": ln,
                "clean_acc": round(ca, 6), "clean_n": cn,
                "gap_pp": None if math.isnan(gap) else round(gap * 100, 4),
                "p_value": None if math.isnan(p) else p,
            }
            out_rows.append([backbone, label, round(la, 6), ln,
                             round(ca, 6), cn,
                             "" if math.isnan(gap) else round(gap * 100, 4),
                             "" if math.isnan(p) else f"{p:.6e}"])

            # exact vs near, to show the dose-response
            for status in ("exact_in_train", "near_in_train"):
                sub = [r for r in tagged if r["leak_status"] == status]
                if sub:
                    _c, _n, a = accuracy(sub, col)
                    print(f"{'  of which ' + status:22s} {a * 100:7.2f}% ({_n:5d})")

    # --- pooled across backbones ----------------------------------------
    if len(preds) > 1:
        print(f"\n{'=' * 74}\nPOOLED ACROSS BACKBONES\n{'=' * 74}")
        print(f"{'method':22s} {'leaked':>16s} {'clean':>16s} {'gap':>8s} {'p':>9s}")
        leaked = [r for s, rs in pooled.items() if s != "clean" for r in rs]
        clean = pooled["clean"]
        summary["_pooled"] = {"methods": {}}
        for col, label in METHODS:
            if not leaked or col not in leaked[0]:
                continue
            lc, ln, la = accuracy(leaked, col)
            cc, cn, ca = accuracy(clean, col)
            gap = (la - ca) if (ln and cn) else float("nan")
            _z, p = two_proportion_z(lc, ln, cc, cn)
            print(f"{label:22s} {la * 100:7.2f}% ({ln:5d}) "
                  f"{ca * 100:7.2f}% ({cn:5d}) "
                  f"{gap * 100:+7.2f}pp {p:9.2e}")
            summary["_pooled"]["methods"][col] = {
                "leaked_acc": round(la, 6), "leaked_n": ln,
                "clean_acc": round(ca, 6), "clean_n": cn,
                "gap_pp": None if math.isnan(gap) else round(gap * 100, 4),
                "p_value": None if math.isnan(p) else p,
            }

    csv_path = os.path.join(args.out_dir, "stratified_accuracy.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["backbone", "method", "leaked_acc", "leaked_n",
                    "clean_acc", "clean_n", "gap_pp", "p_value"])
        w.writerows(out_rows)
    json_path = os.path.join(args.out_dir, "stratified_accuracy.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote {csv_path} and {json_path}")
    print("\nHow to read this: the clean column is the honest accuracy on "
          "images with no duplicate partner in training. A large positive gap "
          "means the headline number was carried by leakage. A gap near zero "
          "means the contamination was real but not load-bearing -- report "
          "that too.")


if __name__ == "__main__":
    main()
