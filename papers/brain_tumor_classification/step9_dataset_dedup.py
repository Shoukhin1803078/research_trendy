"""
Step 9: Dataset integrity audit for the 4-class Kaggle "Brain Tumor MRI
Dataset" (masoudnickparvar) -- implements improvements.md item 2.

The paper currently admits (see the "Near-duplicate audit" paragraph in
brain_tumor_preprint_verified.tex) that a first attempt at near-duplicate
detection -- a coarse 8x8 average-hash Hamming-distance scan -- failed a
basic sanity check: only 49% of flagged pairs shared a class label, close
to the 25% base rate you'd expect from four balanced classes if the hash
carried no signal. That's because an 8x8 thumbnail is dominated by the
shared dark-background/centered-oval-skull layout common to all four
classes, not by class-specific content.

This script replaces that failed attempt with the two-stage procedure the
paper's limitations section calls for:

  Stage 0 (exact duplicates): SHA-256 over the raw file bytes. Catches
      byte-identical files, including the ~194 duplicates reported in this
      dataset's 7,023-image release by prior audits.

  Stage 1 (candidate filter): a much less lossy perceptual hash than the
      original 8x8 average-hash -- a 16x16 perceptual hash (phash, via
      imagehash) at Hamming distance <= HASH_THRESHOLD. This is still a
      CHEAP, HIGH-RECALL filter, not a verdict -- it is expected to have a
      nontrivial false-positive rate, which is exactly why stage 2 exists.

  Stage 2 (verification): every stage-1 candidate pair is verified by a
      pixel-level comparison of the two images at the pipeline's actual
      input resolution (224x224, grayscale, normalized) using normalized
      cross-correlation (NCC). A pair only counts as a confirmed
      near-duplicate if NCC >= NCC_THRESHOLD.

As a sanity check mirroring the one that caught the original hash's
failure, this script reports what fraction of CONFIRMED near-duplicate
pairs share a class label -- this should be close to 100%, not 25-49%.
If it isn't, the thresholds need revisiting before trusting the audit.

This script also does a lightweight non-MRI/CT screen (item 2, "remove CT
images and other non-MRI images"): CT slices in this dataset's source
mixture tend to be pure grayscale with a narrow intensity range and a
rectangular (non-skull-stripped, non-oval) foreground mask, unlike T1/T2
MRI slices. This heuristic is NOT a substitute for manual review -- it
only flags candidates into a CSV for a human to confirm/reject.

Outputs (under OUT_DIR):
  - exact_duplicates.json       groups of byte-identical files
  - near_duplicate_candidates.csv   stage-1 pairs with hash distance
  - near_duplicate_confirmed.csv    stage-2 verified pairs with NCC score
  - non_mri_candidates.csv      heuristic CT/non-MRI flags for manual review
  - removal_manifest.csv        final list of files to drop, with reason
  - cleaned_file_list.txt       paths to KEEP (original list minus removals)

Run locally or on Kaggle -- this is CPU-only (no training), so it does not
need a GPU session. Requires: pip install imagehash pillow scipy numpy.
"""

import os
import csv
import json
import glob
import hashlib
from collections import defaultdict

import numpy as np
from PIL import Image
import imagehash

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = "/kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset"
OUT_DIR = "/kaggle/working/dedup_audit"
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = 224

HASH_SIZE = 16            # 16x16 phash, vs. the failed 8x8 average-hash
HASH_THRESHOLD = 8        # Hamming distance cutoff for stage-1 candidates
NCC_THRESHOLD = 0.92      # stage-2 verification cutoff

# Heuristic non-MRI screen thresholds -- tune after inspecting flagged images.
NONMRI_MIN_NONZERO_FRAC = 0.55  # MRI slices are mostly dark background + oval skull
NONMRI_MAX_UNIQUE_GRAY_LEVELS = 40  # suspiciously low bit-depth / posterized


# ---------------------------------------------------------------------------
# File listing
# ---------------------------------------------------------------------------
def list_image_paths(data_dir):
    paths, labels = [], []
    for split_dir in ["Training", "Testing"]:
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(data_dir, split_dir, class_name)
            files = sorted(glob.glob(os.path.join(class_dir, "*")))
            paths.extend(files)
            labels.extend([class_name] * len(files))
    if not paths:
        raise FileNotFoundError(f"No images found under {data_dir}")
    return paths, labels


# ---------------------------------------------------------------------------
# Stage 0: exact duplicates via SHA-256
# ---------------------------------------------------------------------------
def sha256_of_file(path, chunk_size=1 << 16):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def find_exact_duplicates(paths):
    by_hash = defaultdict(list)
    for p in paths:
        by_hash[sha256_of_file(p)].append(p)
    groups = [v for v in by_hash.values() if len(v) > 1]
    return groups


# ---------------------------------------------------------------------------
# Stage 1: perceptual-hash candidate filter
# ---------------------------------------------------------------------------
def compute_phashes(paths):
    hashes = {}
    for i, p in enumerate(paths):
        try:
            img = Image.open(p).convert("L")
            hashes[p] = imagehash.phash(img, hash_size=HASH_SIZE)
        except Exception as e:
            print(f"  [WARN] could not hash {p}: {e}")
        if (i + 1) % 1000 == 0:
            print(f"  hashed {i + 1}/{len(paths)}")
    return hashes


def find_candidate_pairs(hashes, threshold=HASH_THRESHOLD):
    """O(n^2) Hamming-distance scan. For ~7,200 images this is ~26M pair
    comparisons on tiny 16x16-bit hashes -- a few minutes in pure Python,
    faster with a BK-tree if this becomes a bottleneck on a larger dataset."""
    items = list(hashes.items())
    candidates = []
    for i in range(len(items)):
        p1, h1 = items[i]
        for j in range(i + 1, len(items)):
            p2, h2 = items[j]
            dist = h1 - h2
            if dist <= threshold:
                candidates.append((p1, p2, dist))
    return candidates


# ---------------------------------------------------------------------------
# Stage 2: pixel-level verification via normalized cross-correlation
# ---------------------------------------------------------------------------
def load_gray_resized(path, size=IMG_SIZE):
    img = Image.open(path).convert("L").resize((size, size))
    return np.asarray(img, dtype=np.float32)


def ncc(a, b):
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    return float(np.mean(a * b))


def verify_candidates(candidates):
    confirmed = []
    cache = {}
    for p1, p2, dist in candidates:
        if p1 not in cache:
            cache[p1] = load_gray_resized(p1)
        if p2 not in cache:
            cache[p2] = load_gray_resized(p2)
        score = ncc(cache[p1], cache[p2])
        if score >= NCC_THRESHOLD:
            confirmed.append((p1, p2, dist, score))
    return confirmed


# ---------------------------------------------------------------------------
# Non-MRI / CT heuristic screen
# ---------------------------------------------------------------------------
def flag_non_mri(path):
    img = np.asarray(Image.open(path).convert("L"))
    nonzero_frac = float(np.mean(img > 10))
    n_unique = len(np.unique(img))
    reasons = []
    if nonzero_frac > NONMRI_MIN_NONZERO_FRAC + 0.25 or nonzero_frac < 0.05:
        reasons.append(f"nonzero_frac={nonzero_frac:.2f}")
    if n_unique <= NONMRI_MAX_UNIQUE_GRAY_LEVELS:
        reasons.append(f"unique_gray_levels={n_unique}")
    return reasons


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    print("Listing image paths...")
    paths, labels = list_image_paths(DATA_DIR)
    label_of = dict(zip(paths, labels))
    print(f"Found {len(paths)} images.")

    print("\nStage 0: exact duplicates (SHA-256)...")
    exact_groups = find_exact_duplicates(paths)
    n_exact_removed = sum(len(g) - 1 for g in exact_groups)
    print(f"  {len(exact_groups)} duplicate groups, "
          f"{n_exact_removed} files removable as exact duplicates.")
    with open(os.path.join(OUT_DIR, "exact_duplicates.json"), "w") as f:
        json.dump(exact_groups, f, indent=2)

    exact_dupe_set = set()
    for g in exact_groups:
        exact_dupe_set.update(g[1:])  # keep first of each group, drop rest

    remaining_paths = [p for p in paths if p not in exact_dupe_set]

    print(f"\nStage 1: perceptual-hash candidates over {len(remaining_paths)} "
          f"remaining images (16x16 phash, Hamming <= {HASH_THRESHOLD})...")
    hashes = compute_phashes(remaining_paths)
    candidates = find_candidate_pairs(hashes, HASH_THRESHOLD)
    print(f"  {len(candidates)} candidate pairs.")
    with open(os.path.join(OUT_DIR, "near_duplicate_candidates.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path1", "path2", "hamming_dist"])
        w.writerows(candidates)

    print(f"\nStage 2: pixel-level verification (NCC >= {NCC_THRESHOLD})...")
    confirmed = verify_candidates(candidates)
    print(f"  {len(confirmed)} confirmed near-duplicate pairs "
          f"(of {len(candidates)} candidates).")
    with open(os.path.join(OUT_DIR, "near_duplicate_confirmed.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path1", "path2", "hamming_dist", "ncc_score", "same_class"])
        same_class_count = 0
        for p1, p2, dist, score in confirmed:
            same_class = label_of[p1] == label_of[p2]
            same_class_count += int(same_class)
            w.writerow([p1, p2, dist, round(score, 4), same_class])

    if confirmed:
        frac_same_class = same_class_count / len(confirmed)
        print(f"\n  Sanity check: {frac_same_class * 100:.1f}% of confirmed pairs "
              f"share a class label (should be close to 100%; the original "
              f"8x8-average-hash attempt scored only 49%, i.e. near the 25% "
              f"base rate, which is why it was discarded).")
        if frac_same_class < 0.9:
            print("  [WARN] Still below 90% -- lower NCC_THRESHOLD's tolerance "
                  "or tighten HASH_THRESHOLD before trusting this audit.")

    near_dupe_removed = set()
    for p1, p2, _, _ in confirmed:
        # Keep the first (by sorted path) of each confirmed pair, drop the other.
        _, drop = sorted([p1, p2])
        near_dupe_removed.add(drop)

    print(f"\nNon-MRI / CT heuristic screen over "
          f"{len(remaining_paths) - len(near_dupe_removed)} remaining images...")
    non_mri_rows = []
    for p in remaining_paths:
        if p in near_dupe_removed:
            continue
        reasons = flag_non_mri(p)
        if reasons:
            non_mri_rows.append((p, label_of[p], "; ".join(reasons)))
    print(f"  {len(non_mri_rows)} images flagged for MANUAL review "
          f"(heuristic only -- do not auto-remove).")
    with open(os.path.join(OUT_DIR, "non_mri_candidates.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "class", "flag_reason"])
        w.writerows(non_mri_rows)

    # Final removal manifest: exact dupes + confirmed near-dupes only.
    # Non-MRI flags require manual confirmation before being added here.
    removal_manifest = (
        [(p, "exact_duplicate") for p in sorted(exact_dupe_set)] +
        [(p, "near_duplicate") for p in sorted(near_dupe_removed)]
    )
    with open(os.path.join(OUT_DIR, "removal_manifest.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "reason"])
        w.writerows(removal_manifest)

    removed_set = exact_dupe_set | near_dupe_removed
    cleaned = [p for p in paths if p not in removed_set]
    with open(os.path.join(OUT_DIR, "cleaned_file_list.txt"), "w") as f:
        f.write("\n".join(cleaned))

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    print(f"  Original images:          {len(paths)}")
    print(f"  Exact duplicates removed: {len(exact_dupe_set)}")
    print(f"  Near-duplicates removed:  {len(near_dupe_removed)}")
    print(f"  Non-MRI flagged (manual): {len(non_mri_rows)}")
    print(f"  Cleaned dataset size:     {len(cleaned)}")
    print(f"\nAll outputs written to {OUT_DIR}. Download the directory, manually "
          f"review non_mri_candidates.csv, then re-run cross-validation "
          f"(step6_backbone_sweep.py / step10_msai_module.py) by pointing "
          f"list_image_paths at cleaned_file_list.txt instead of walking "
          f"DATA_DIR directly, for both the proposed model and its baseline.")


if __name__ == "__main__":
    main()
