"""
Step 8: Exploratory Data Analysis (EDA) for the 4-class Kaggle "Brain Tumor
MRI Dataset" (masoudnickparvar) used throughout this study.

Runs entirely on CPU -- no TensorFlow/GPU needed, only image reading (PIL),
hashing, and plotting (numpy, matplotlib, scikit-learn). Safe to run in a
Kaggle CPU-only session, as long as the dataset is attached as an input
(same path as step4_kaggle4class.py / step6_backbone_sweep.py).

Produces, under images/ and results/:
    eda_a_class_distribution.png
    eda_a_image_dimensions.png
    eda_a_sample_grid.png
    eda_a_intensity_boxplot.png
    eda_a_mean_images.png
    eda_a_near_duplicate_examples.png
    results/eda_a_near_duplicates.csv   -- every near-duplicate pair found
    results/eda_a_summary.json          -- headline numbers for the paper

The most important part of this script for the paper is the near-duplicate
audit (Section "Near-Duplicate / Leakage Audit" below). Instead of only
arguing that an image-level split *could* leak near-duplicate slices across
the train/test boundary, this script directly measures it: it hashes every
image, finds near-duplicate pairs by Hamming distance, then replicates the
EXACT StratifiedKFold(n_splits=5, shuffle=True, random_state=42) split used
in step4_kaggle4class.py / step6_backbone_sweep.py and checks, for every
near-duplicate pair, whether the two images fall in different folds (which
is precisely the condition under which one image's near-duplicate would
have been seen during training while the other is being scored as "unseen"
test data).

Usage: edit DATA_DIR below if your Kaggle input path differs, then run the
whole script. Download the images/eda_a_*.png files and
results/eda_a_summary.json / eda_a_near_duplicates.csv for the paper.
"""

import os
import glob
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = "/kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset"
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = 224
N_FOLDS = 5
SEED = 42
HASH_SIZE = 8            # 8x8 -> 64-bit average hash
NEAR_DUP_THRESHOLD = 5   # Hamming distance <= this many differing bits (out
                          # of 64) counts as a near-duplicate candidate; 0
                          # would mean pixel-identical hashes only.

os.makedirs("images", exist_ok=True)
os.makedirs("results", exist_ok=True)

COLOR_A = "#0072B2"
CLASS_COLORS = {"glioma": "#0072B2", "meningioma": "#E69F00",
                 "notumor": "#009E73", "pituitary": "#D55E00"}

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#888888",
})


# ---------------------------------------------------------------------------
# Dataset loading + hashing
# ---------------------------------------------------------------------------
def list_image_paths(data_dir):
    """Same order as step4_kaggle4class.py's list_image_paths -- this exact
    order is what determines which StratifiedKFold fold each image falls in,
    so it must match bit-for-bit for the leakage audit below to be valid."""
    paths, labels = [], []
    for split_dir in ["Training", "Testing"]:
        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = os.path.join(data_dir, split_dir, class_name)
            files = sorted(glob.glob(os.path.join(class_dir, "*")))
            paths.extend(files)
            labels.extend([class_idx] * len(files))
    return paths, np.array(labels)


def average_hash(pil_img, hash_size=HASH_SIZE):
    gray = pil_img.convert("L").resize((hash_size, hash_size), Image.BILINEAR)
    arr = np.asarray(gray, dtype=np.float32)
    bits = (arr > arr.mean()).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return np.uint64(val)


_POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def popcount_u64(arr_u64):
    b = arr_u64.reshape(-1, 1).view(np.uint8)
    return _POPCOUNT_TABLE[b].sum(axis=1)


def find_near_duplicate_pairs(hashes, threshold=NEAR_DUP_THRESHOLD, row_chunk=400):
    """Returns list of (i, j, hamming_distance) with i < j and distance <= threshold."""
    n = len(hashes)
    hashes = hashes.astype(np.uint64)
    pairs = []
    for start in range(0, n, row_chunk):
        end = min(start + row_chunk, n)
        chunk = hashes[start:end]
        xor = chunk[:, None] ^ hashes[None, :]  # (chunk_len, n) uint64
        dist = popcount_u64(xor.reshape(-1)).reshape(xor.shape)
        for local_i, global_i in enumerate(range(start, end)):
            row = dist[local_i]
            close = np.where(row <= threshold)[0]
            for j in close:
                if j > global_i:
                    pairs.append((global_i, int(j), int(row[j])))
        if (start // row_chunk) % 5 == 0:
            print(f"  near-duplicate scan: {end}/{n} images compared")
    return pairs


def main():
    print("=" * 70)
    print("Dataset EDA")
    print("=" * 70)
    paths, y = list_image_paths(DATA_DIR)
    n = len(paths)
    print(f"Found {n} images across {len(CLASS_NAMES)} classes")

    # --- class distribution ---
    counts = {name: int((y == i).sum()) for i, name in enumerate(CLASS_NAMES)}
    print("Class counts:", counts)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(counts.keys(), counts.values(), color=[CLASS_COLORS[c] for c in counts])
    for i, (k, v) in enumerate(counts.items()):
        ax.text(i, v + 20, str(v), ha="center", fontsize=9)
    ax.set_ylabel("Number of images")
    ax.grid(axis="y", color="#e5e5e5", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig("images/eda_a_class_distribution.png", dpi=200)
    plt.close(fig)

    # --- image dimensions + intensity stats (single pass over all images) ---
    widths, heights, mean_intensity, hashes = [], [], [], []
    imgs_by_class_sample = {c: [] for c in CLASS_NAMES}
    sum_img_by_class = {c: np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float64) for c in CLASS_NAMES}
    count_by_class = {c: 0 for c in CLASS_NAMES}

    for idx, p in enumerate(paths):
        with Image.open(p) as im:
            widths.append(im.width)
            heights.append(im.height)
            gray = im.convert("L")
            arr_native = np.asarray(gray, dtype=np.float32)
            mean_intensity.append(float(arr_native.mean()))
            hashes.append(average_hash(im))

            cname = CLASS_NAMES[y[idx]]
            if len(imgs_by_class_sample[cname]) < 5:
                imgs_by_class_sample[cname].append(np.asarray(im.convert("RGB").resize((160, 160))))
            resized = np.asarray(gray.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float64)
            sum_img_by_class[cname] += resized
            count_by_class[cname] += 1
        if (idx + 1) % 1000 == 0 or (idx + 1) == n:
            print(f"  scanned {idx + 1}/{n}")

    widths, heights = np.array(widths), np.array(heights)
    mean_intensity = np.array(mean_intensity)
    hashes = np.array(hashes, dtype=np.uint64)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.hist(widths, bins=30, alpha=0.7, label="width", color=COLOR_A)
    ax.hist(heights, bins=30, alpha=0.7, label="height", color="#D55E00")
    ax.set_xlabel("Pixels")
    ax.set_ylabel("Number of images")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#e5e5e5", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig("images/eda_a_image_dimensions.png", dpi=200)
    plt.close(fig)
    print(f"Width range: [{widths.min()}, {widths.max()}], "
          f"Height range: [{heights.min()}, {heights.max()}]")

    # --- sample grid, 5 per class ---
    fig, axes = plt.subplots(len(CLASS_NAMES), 5, figsize=(10, 8))
    for r, cname in enumerate(CLASS_NAMES):
        for c in range(5):
            ax = axes[r, c]
            if c < len(imgs_by_class_sample[cname]):
                ax.imshow(imgs_by_class_sample[cname][c])
            ax.axis("off")
        axes[r, 0].text(-0.15, 0.5, cname, transform=axes[r, 0].transAxes,
                         rotation=90, ha="center", va="center", fontsize=11)
    fig.tight_layout()
    fig.savefig("images/eda_a_sample_grid.png", dpi=200)
    plt.close(fig)

    # --- pixel intensity boxplot by class ---
    fig, ax = plt.subplots(figsize=(5.5, 4))
    data_per_class = [mean_intensity[y == i] for i in range(len(CLASS_NAMES))]
    bp = ax.boxplot(data_per_class, labels=CLASS_NAMES, patch_artist=True)
    for patch, cname in zip(bp["boxes"], CLASS_NAMES):
        patch.set_facecolor(CLASS_COLORS[cname])
        patch.set_alpha(0.7)
    ax.set_ylabel("Mean pixel intensity (0-255)")
    ax.grid(axis="y", color="#e5e5e5", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig("images/eda_a_intensity_boxplot.png", dpi=200)
    plt.close(fig)

    # --- mean ("average") image per class ---
    fig, axes = plt.subplots(1, len(CLASS_NAMES), figsize=(12, 3.2))
    for ax, cname in zip(axes, CLASS_NAMES):
        mean_img = sum_img_by_class[cname] / max(count_by_class[cname], 1)
        ax.imshow(mean_img, cmap="gray")
        ax.set_title(cname, fontsize=10)
        ax.axis("off")
    fig.suptitle("Mean image per class (pixel-wise average, aligned to slice center)")
    fig.tight_layout()
    fig.savefig("images/eda_a_mean_images.png", dpi=200)
    plt.close(fig)

    # -----------------------------------------------------------------
    # Near-duplicate / leakage audit
    # -----------------------------------------------------------------
    print("\nScanning for near-duplicate image pairs (this is the leakage audit)...")
    pairs = find_near_duplicate_pairs(hashes)
    print(f"Found {len(pairs)} near-duplicate pairs (Hamming distance <= {NEAR_DUP_THRESHOLD}/64 bits)")

    same_class = sum(1 for i, j, d in pairs if y[i] == y[j])
    cross_class = len(pairs) - same_class
    exact_hash_matches = sum(1 for i, j, d in pairs if d == 0)

    # Replicate the EXACT fold assignment used in step4_kaggle4class.py / step6
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_of = np.full(n, -1, dtype=int)
    dummy_X = np.zeros((n, 1))
    for fold_idx, (_, test_idx) in enumerate(skf.split(dummy_X, y)):
        fold_of[test_idx] = fold_idx

    leaking_pairs = [(i, j, d) for i, j, d in pairs if fold_of[i] != fold_of[j]]
    affected_images = set()
    for i, j, d in leaking_pairs:
        affected_images.add(i)
        affected_images.add(j)

    print(f"Same-class pairs: {same_class}  Cross-class pairs: {cross_class}  "
          f"Exact hash matches (distance=0): {exact_hash_matches}")
    print(f"Pairs split across DIFFERENT folds under the actual CV protocol: "
          f"{len(leaking_pairs)} / {len(pairs)}")
    print(f"Unique images with >=1 near-duplicate in a different fold: "
          f"{len(affected_images)} / {n} ({100*len(affected_images)/n:.2f}%)")

    # save CSV of all pairs found
    import csv
    with open("results/eda_a_near_duplicates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path_i", "path_j", "class_i", "class_j", "hamming_distance",
                    "fold_i", "fold_j", "different_fold"])
        for i, j, d in pairs:
            w.writerow([paths[i], paths[j], CLASS_NAMES[y[i]], CLASS_NAMES[y[j]],
                        d, int(fold_of[i]), int(fold_of[j]), fold_of[i] != fold_of[j]])

    # visualize a handful of example near-duplicate pairs (closest distance first)
    pairs_sorted = sorted(pairs, key=lambda t: t[2])
    n_examples = min(4, len(pairs_sorted))
    if n_examples > 0:
        fig, axes = plt.subplots(n_examples, 2, figsize=(5, 2.4 * n_examples))
        if n_examples == 1:
            axes = axes[None, :]
        for row, (i, j, d) in enumerate(pairs_sorted[:n_examples]):
            for col, idx in enumerate((i, j)):
                with Image.open(paths[idx]) as im:
                    axes[row, col].imshow(im.convert("L"), cmap="gray")
                axes[row, col].axis("off")
                axes[row, col].set_title(
                    f"{CLASS_NAMES[y[idx]]}, fold {fold_of[idx]}", fontsize=8)
            axes[row, 0].set_ylabel(f"d={d}", fontsize=9)
        fig.suptitle("Closest near-duplicate pairs found (left/right = the two images in the pair)")
        fig.tight_layout()
        fig.savefig("images/eda_a_near_duplicate_examples.png", dpi=200)
        plt.close(fig)

    summary = {
        "n_images": n,
        "class_counts": counts,
        "image_width_range": [int(widths.min()), int(widths.max())],
        "image_height_range": [int(heights.min()), int(heights.max())],
        "near_duplicate_threshold_bits": NEAR_DUP_THRESHOLD,
        "near_duplicate_pairs_total": len(pairs),
        "near_duplicate_pairs_same_class": same_class,
        "near_duplicate_pairs_cross_class": cross_class,
        "exact_hash_matches": exact_hash_matches,
        "pairs_split_across_folds": len(leaking_pairs),
        "images_with_cross_fold_near_duplicate": len(affected_images),
        "images_with_cross_fold_near_duplicate_pct": round(100 * len(affected_images) / n, 3),
    }
    with open("results/eda_a_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved results/eda_a_summary.json:", json.dumps(summary, indent=2))
    print("\nDone. Download images/eda_a_*.png, results/eda_a_summary.json, "
          "and results/eda_a_near_duplicates.csv for the paper.")
    return summary


if __name__ == "__main__":
    main()
