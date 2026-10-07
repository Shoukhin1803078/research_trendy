"""
Step 15: Generalized, multi-dataset duplicate/near-duplicate audit.

This is step9_dataset_dedup.py generalized along three axes:

  1. ARBITRARY DATASET ROOTS. step9 hardcodes the masoudnickparvar layout
     (DATA_DIR / {Training,Testing} / {glioma,meningioma,notumor,pituitary}).
     This script discovers the layout by walking the root, so it runs
     unchanged on Figshare/Cheng, Br35H, SARTAJ, BraTS slice exports, etc.

  2. CROSS-DATASET OVERLAP. Pass more than one --data-dir and the script
     reports duplicate pairs partitioned into WITHIN-dataset and ACROSS-
     dataset. The second number is the one that matters for external
     validation: the 7,200-image Kaggle set is itself a merge of
     figshare/SARTAJ/Br35H, so "validate on Figshare" is only meaningful
     if Figshare images are not already sitting inside the training set.
     improvements.md item 3 warns about this; this measures it.

  3. SCALE. step9's find_candidate_pairs is an O(n^2) pure-Python loop over
     imagehash objects (~26M comparisons at n=7,200 -- minutes). That does
     not survive a BraTS-scale export. Here the hashes are bit-packed into
     a uint8 matrix and compared in vectorized chunks via XOR + a popcount
     lookup table, which is ~2 orders of magnitude faster and bounded in
     memory by --chunk-size.

The two-stage logic and the sanity check are unchanged from step9, because
that is the part that was validated: a cheap high-recall perceptual-hash
candidate filter, then pixel-level NCC verification, then a report of what
fraction of CONFIRMED pairs share a class label. On the Kaggle set step9
scored 99.6% on that check (the discarded 8x8 average-hash scored 49%,
against a 25% base rate). Any new dataset should be held to the same bar
before its numbers are trusted -- the script warns if it falls below 90%.

Outputs per run (under --out-dir), mirroring step9's filenames so the
downstream scripts keep working:
  - exact_duplicates.json           groups of byte-identical files
  - near_duplicate_candidates.csv   stage-1 pairs + hash distance
  - near_duplicate_confirmed.csv    stage-2 verified pairs + NCC + scope
  - non_mri_candidates.csv          heuristic flags for MANUAL review
  - removal_manifest.csv            files to drop, with reason
  - cleaned_file_list.txt           paths to KEEP
  - summary.json                    machine-readable counts for the paper

CPU-only, no GPU session needed. Requires: imagehash pillow numpy.

Examples
--------
Single dataset (reproduces step9's audit on a discovered layout):

    python3 step15_multidataset_audit.py \
        --data-dir kaggle4c=/data/brain-tumor-mri-dataset \
        --out-dir audit/kaggle4c

Cross-dataset overlap between the Kaggle merge and its claimed sources:

    python3 step15_multidataset_audit.py \
        --data-dir kaggle4c=/data/brain-tumor-mri-dataset \
        --data-dir figshare=/data/figshare-cheng \
        --data-dir br35h=/data/br35h \
        --out-dir audit/cross
"""

import os
import sys
import csv
import json
import argparse
import hashlib
from collections import defaultdict

import numpy as np
from PIL import Image
import imagehash


# ---------------------------------------------------------------------------
# Notebook config -- EDIT THIS when running inside a Kaggle/Jupyter cell.
#
# This script has a command-line interface, but pasting it into a notebook
# cell means argparse sees the KERNEL's argv (which has no --data-dir) and
# exits with SystemExit: 2. So, like the "Config -- EDIT THIS" blocks in
# step6/step9/step10, these values are used whenever the script runs inside a
# notebook or is invoked with no arguments. A real command line overrides
# them, and so does calling main(["--data-dir", ...]) explicitly.
# ---------------------------------------------------------------------------
NB_DATA_DIRS = [
    "kaggle4c=/kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset",
    # add more roots to measure CROSS-dataset overlap, e.g.
    # "figshare=/kaggle/input/figshare-brain-tumor",
    # "br35h=/kaggle/input/br35h",
]
NB_OUT_DIR = "/kaggle/working/audit_kaggle4c"
NB_CLASSES = "glioma,meningioma,notumor,pituitary"  # None to auto-discover
NB_HASH_SIZES = "16"          # "16" reproduces step9; see --hash-sizes
NB_HASH_THRESHOLD = 8
NB_NCC_THRESHOLD = 0.92
NB_CHUNK_SIZE = 512
NB_RECALL_PROBE = True        # measure stage-1 recall -- do this before
                              # quoting a contamination rate in the paper
NB_PROBE_THUMB = 32
NB_PROBE_NCC = 0.95
NB_SKIP_NON_MRI = False


def _running_in_notebook():
    return "ipykernel" in sys.modules


def _notebook_args():
    return argparse.Namespace(
        data_dir=list(NB_DATA_DIRS),
        out_dir=NB_OUT_DIR,
        classes=NB_CLASSES,
        hash_sizes=NB_HASH_SIZES,
        hash_threshold=NB_HASH_THRESHOLD,
        ncc_threshold=NB_NCC_THRESHOLD,
        chunk_size=NB_CHUNK_SIZE,
        recall_probe=NB_RECALL_PROBE,
        probe_thumb=NB_PROBE_THUMB,
        probe_ncc=NB_PROBE_NCC,
        skip_non_mri=NB_SKIP_NON_MRI,
    )


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Directory names that denote a train/test split rather than a class. Used
# only to pick the class component out of a path; harmless if absent.
SPLIT_DIR_NAMES = {
    "training", "train", "testing", "test", "val", "valid", "validation",
    "images", "data",
}

IMG_SIZE = 224

# Heuristic non-MRI screen thresholds, carried over from step9.
NONMRI_MIN_NONZERO_FRAC = 0.55
NONMRI_MAX_UNIQUE_GRAY_LEVELS = 40

# popcount[b] = number of set bits in byte b
POPCOUNT = np.unpackbits(
    np.arange(256, dtype=np.uint8)[:, None], axis=1
).sum(axis=1).astype(np.uint16)


# ---------------------------------------------------------------------------
# Layout discovery
# ---------------------------------------------------------------------------
def infer_class(rel_parts, known_classes):
    """Pick the class label out of a path's directory components.

    With --classes given, the class is whichever component matches (case
    insensitive). Otherwise it is the deepest directory component that is
    not a split/container name -- which handles root/class/*.jpg,
    root/split/class/*.jpg and root/split/class/subdir/*.jpg alike.
    """
    dirs = [p for p in rel_parts[:-1]]
    if known_classes:
        lowered = {c.lower(): c for c in known_classes}
        for part in reversed(dirs):
            if part.lower() in lowered:
                return lowered[part.lower()]
        return None
    for part in reversed(dirs):
        if part.lower() not in SPLIT_DIR_NAMES:
            return part
    return "_root"


def list_images(root, known_classes):
    """Walk a dataset root and return [(abs_path, class_label)]."""
    found = []
    root = os.path.abspath(root)
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in sorted(filenames):
            if os.path.splitext(fn)[1].lower() not in IMAGE_EXTS:
                continue
            abs_path = os.path.join(dirpath, fn)
            rel_parts = os.path.relpath(abs_path, root).split(os.sep)
            label = infer_class(rel_parts, known_classes)
            if label is None:
                continue  # --classes given and this file matched none of them
            found.append((abs_path, label))
    return found


# ---------------------------------------------------------------------------
# Stage 0: exact duplicates (SHA-256 over raw bytes)
# ---------------------------------------------------------------------------
def sha256_of_file(path, chunk_size=1 << 16):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def find_exact_duplicate_groups(paths):
    by_hash = defaultdict(list)
    for p in paths:
        by_hash[sha256_of_file(p)].append(p)
    return [sorted(v) for v in by_hash.values() if len(v) > 1]


# ---------------------------------------------------------------------------
# Stage 1: vectorized perceptual-hash candidate filter
# ---------------------------------------------------------------------------
def compute_packed_phashes(paths, hash_sizes):
    """Hash every readable image at each requested phash grid size.

    Returns (kept_paths, {hash_size: packed (n, nbytes) uint8}).

    Hashing at more than one grid size and UNIONING the candidate pairs is
    the fix for a recall bug in step9's single 16x16 configuration. A 16x16
    phash keeps high-frequency DCT coefficients, so a 1-pixel shift or mild
    noise can push a genuine near-duplicate pair to Hamming 12-18 -- past
    the threshold of 8 -- even when full-resolution NCC is 0.93, i.e. above
    the stage-2 cutoff that would have confirmed it. The same pairs sit at
    Hamming 0-4 under an 8x8 phash. Since stage 2 is the actual verifier,
    stage 1 should err toward recall; false positives cost only NCC time.
    """
    kept, rows = [], {hs: [] for hs in hash_sizes}
    for i, p in enumerate(paths):
        try:
            with Image.open(p) as im:
                gray = im.convert("L")
                hashes = {hs: imagehash.phash(gray, hash_size=hs)
                          for hs in hash_sizes}
        except Exception as e:  # unreadable/corrupt file -- skip, don't abort
            print(f"  [WARN] could not hash {p}: {e}")
            continue
        kept.append(p)
        for hs in hash_sizes:
            rows[hs].append(np.packbits(hashes[hs].hash.flatten()))
        if (i + 1) % 2000 == 0:
            print(f"  hashed {i + 1}/{len(paths)}")
    if not kept:
        return [], {hs: np.zeros((0, 0), dtype=np.uint8) for hs in hash_sizes}
    return kept, {hs: np.vstack(rows[hs]).astype(np.uint8) for hs in hash_sizes}


def find_candidate_pairs(packed, threshold, chunk_size=512):
    """Vectorized upper-triangle Hamming scan.

    For each chunk of rows, XOR against every row at a higher index and
    popcount via lookup. Memory is bounded by chunk_size * n * nbytes.
    """
    n = packed.shape[0]
    pairs = []
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        block = packed[start:stop]                       # (b, nbytes)
        xor = np.bitwise_xor(block[:, None, :], packed[None, :, :])
        dist = POPCOUNT[xor].sum(axis=2)                 # (b, n)
        # keep only j > i to get each unordered pair once
        rows, cols = np.nonzero(dist <= threshold)
        for r, c in zip(rows, cols):
            i = start + int(r)
            j = int(c)
            if j > i:
                pairs.append((i, j, int(dist[r, c])))
        print(f"  scanned rows {stop}/{n} ({len(pairs)} candidates so far)")
    return pairs


def union_candidates(packed_by_size, threshold, chunk_size):
    """Union stage-1 pairs across phash grid sizes, keeping the min distance."""
    best = {}
    for hs, packed in sorted(packed_by_size.items()):
        print(f"  stage-1 scan at phash {hs}x{hs}...")
        found = find_candidate_pairs(packed, threshold, chunk_size)
        print(f"    {len(found)} candidate pairs at {hs}x{hs}")
        for i, j, d in found:
            key = (i, j)
            if key not in best or d < best[key][0]:
                best[key] = (d, hs)
    return [(i, j, d, hs) for (i, j), (d, hs) in sorted(best.items())]


# ---------------------------------------------------------------------------
# Recall probe: an independent, exhaustive coarse-NCC oracle
# ---------------------------------------------------------------------------
def coarse_ncc_pairs(paths, thumb, threshold, chunk_size=512):
    """Every pair whose coarse-thumbnail NCC >= threshold, computed exactly.

    Each image is reduced to a thumb x thumb grayscale thumbnail, flattened
    and standardized to zero mean / unit variance. For standardized vectors
    the NCC is exactly the normalized inner product, so the full pairwise
    NCC matrix is a single matmul -- cheap enough to run exhaustively at
    n in the thousands, with no hashing heuristic involved at all.

    This is NOT the audit. It is a yardstick: because it never uses the
    perceptual hash, it gives an independent estimate of how many genuine
    near-duplicate pairs the stage-1 hash filter failed to propose, i.e. a
    measured recall rather than an assumed one.

    CAVEAT for anything quoted in a paper: the recall it reports is relative
    to THIS oracle, not to ground truth. Pairs that both the oracle and the
    hash miss are invisible to it, so the figure is an upper bound on true
    stage-1 recall. Lowering --probe-ncc widens the oracle (and slows
    verification); on synthetic ground truth, a 32x32 thumbnail at 0.95
    recovered every planted pair, including 1-pixel-shifted ones that the
    16x16 hash dropped.
    """
    vecs, kept = [], []
    for i, p in enumerate(paths):
        try:
            with Image.open(p) as im:
                a = np.asarray(im.convert("L").resize((thumb, thumb)),
                               dtype=np.float32).ravel()
        except Exception:
            continue
        a -= a.mean()
        s = a.std()
        if s < 1e-6:
            continue  # blank image: NCC undefined
        vecs.append(a / s)
        kept.append(i)
        if (i + 1) % 2000 == 0:
            print(f"  thumbnailed {i + 1}/{len(paths)}")
    if len(vecs) < 2:
        return []
    Z = np.vstack(vecs)                      # (m, thumb*thumb), standardized
    m, d = Z.shape
    out = []
    for start in range(0, m, chunk_size):
        stop = min(start + chunk_size, m)
        sim = (Z[start:stop] @ Z.T) / float(d)
        rows, cols = np.nonzero(sim >= threshold)
        for r, c in zip(rows, cols):
            i, j = start + int(r), int(c)
            if j > i:
                out.append((kept[i], kept[j], float(sim[r, c])))
        print(f"  probed rows {stop}/{m} ({len(out)} high-NCC pairs so far)")
    return out


# ---------------------------------------------------------------------------
# Stage 2: pixel-level verification (normalized cross-correlation)
# ---------------------------------------------------------------------------
def load_gray_resized(path, size=IMG_SIZE):
    with Image.open(path) as im:
        return np.asarray(im.convert("L").resize((size, size)), dtype=np.float32)


def ncc(a, b):
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    return float(np.mean(a * b))


def verify_candidates(pairs, paths, ncc_threshold, cache_limit=20000):
    confirmed, cache = [], {}

    def get(idx):
        if idx not in cache:
            if len(cache) >= cache_limit:
                cache.clear()  # crude but bounded; pairs are index-sorted
            cache[idx] = load_gray_resized(paths[idx])
        return cache[idx]

    for k, (i, j, dist) in enumerate(pairs):
        try:
            score = ncc(get(i), get(j))
        except Exception as e:
            print(f"  [WARN] could not compare {paths[i]} / {paths[j]}: {e}")
            continue
        if score >= ncc_threshold:
            confirmed.append((i, j, dist, score))
        if (k + 1) % 5000 == 0:
            print(f"  verified {k + 1}/{len(pairs)}")
    return confirmed


# ---------------------------------------------------------------------------
# Non-MRI / CT heuristic screen (flags for MANUAL review only)
# ---------------------------------------------------------------------------
def flag_non_mri(path):
    try:
        with Image.open(path) as im:
            img = np.asarray(im.convert("L"))
    except Exception:
        return ["unreadable"]
    nonzero_frac = float(np.mean(img > 10))
    n_unique = int(len(np.unique(img)))
    reasons = []
    if nonzero_frac > NONMRI_MIN_NONZERO_FRAC + 0.25 or nonzero_frac < 0.05:
        reasons.append(f"nonzero_frac={nonzero_frac:.2f}")
    if n_unique <= NONMRI_MAX_UNIQUE_GRAY_LEVELS:
        reasons.append(f"unique_gray_levels={n_unique}")
    return reasons


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def parse_roots(raw):
    """--data-dir accepts 'name=path' or bare 'path'."""
    roots = []
    for item in raw:
        if "=" in item:
            name, path = item.split("=", 1)
        else:
            path, name = item, os.path.basename(os.path.normpath(item))
        roots.append((name, os.path.abspath(path)))
    return roots


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", action="append", metavar="NAME=PATH",
                    help="dataset root; repeatable. 'name=path' or bare path. "
                         "Omit to use the NB_DATA_DIRS config block.")
    ap.add_argument("--out-dir", help="omit to use the NB_OUT_DIR config block")
    ap.add_argument("--classes", default=None,
                    help="comma-separated class names. Omit to auto-discover "
                         "from directory structure.")
    ap.add_argument("--hash-sizes", default="16",
                    help="comma-separated phash grid sizes for stage 1; "
                         "candidates are UNIONED across them. Default '16' "
                         "matches step9. Adding a smaller grid raises recall "
                         "but can collapse precision on layout-dominated "
                         "images -- measure with --recall-probe first, and "
                         "watch the same-class sanity check.")
    ap.add_argument("--hash-threshold", type=int, default=8,
                    help="stage-1 Hamming cutoff (default 8, as step9)")
    ap.add_argument("--recall-probe", action="store_true",
                    help="additionally run an exhaustive coarse-NCC oracle to "
                         "MEASURE what fraction of genuine near-duplicate "
                         "pairs stage 1 proposed. Recommended before quoting "
                         "a contamination rate in a paper.")
    ap.add_argument("--probe-thumb", type=int, default=32,
                    help="thumbnail grid for the recall probe (default 32)")
    ap.add_argument("--probe-ncc", type=float, default=0.95,
                    help="coarse-NCC cutoff defining a 'genuine' pair for the "
                         "recall probe (default 0.95; stricter than "
                         "--ncc-threshold because it is computed on a coarser "
                         "thumbnail)")
    ap.add_argument("--ncc-threshold", type=float, default=0.92,
                    help="stage-2 NCC cutoff (default 0.92, as step9)")
    ap.add_argument("--chunk-size", type=int, default=512,
                    help="rows per vectorized Hamming block")
    ap.add_argument("--skip-non-mri", action="store_true",
                    help="skip the heuristic non-MRI screen")

    if argv is not None:
        args = ap.parse_args(argv)
    elif _running_in_notebook() or len(sys.argv) <= 1:
        args = _notebook_args()
        print("No CLI arguments (notebook or bare run) -- using the NB_* "
              "config block at the top of this file:")
        print(f"  roots   : {args.data_dir}")
        print(f"  out-dir : {args.out_dir}")
        print(f"  grids   : {args.hash_sizes}   recall-probe: {args.recall_probe}\n")
    else:
        args = ap.parse_args()

    if not args.data_dir:
        ap.error("no dataset root: pass --data-dir or set NB_DATA_DIRS")
    if not args.out_dir:
        ap.error("no output dir: pass --out-dir or set NB_OUT_DIR")

    known_classes = ([c.strip() for c in args.classes.split(",")]
                     if args.classes else None)
    roots = parse_roots(args.data_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- discover -------------------------------------------------------
    paths, labels, datasets = [], [], []
    per_dataset_counts = {}
    for name, root in roots:
        if not os.path.isdir(root):
            raise FileNotFoundError(f"--data-dir {name}: not a directory: {root}")
        found = list_images(root, known_classes)
        if not found:
            raise FileNotFoundError(f"--data-dir {name}: no images under {root}")
        print(f"[{name}] {len(found)} images under {root}")
        by_class = defaultdict(int)
        for p, lab in found:
            paths.append(p)
            labels.append(lab)
            datasets.append(name)
            by_class[lab] += 1
        per_dataset_counts[name] = dict(sorted(by_class.items()))
        print(f"[{name}] classes: {per_dataset_counts[name]}")

    n_total = len(paths)
    print(f"\nTotal: {n_total} images across {len(roots)} dataset(s).")

    # ---- stage 0 --------------------------------------------------------
    print("\nStage 0: exact duplicates (SHA-256)...")
    exact_groups = find_exact_duplicate_groups(paths)
    exact_drop = set()
    for g in exact_groups:
        exact_drop.update(g[1:])  # keep the first of each group
    print(f"  {len(exact_groups)} groups, {len(exact_drop)} files removable.")
    with open(os.path.join(args.out_dir, "exact_duplicates.json"), "w") as f:
        json.dump(exact_groups, f, indent=2)

    # ---- stage 1 --------------------------------------------------------
    hash_sizes = [int(s) for s in args.hash_sizes.split(",") if s.strip()]
    keep_idx = [i for i, p in enumerate(paths) if p not in exact_drop]
    stage1_paths = [paths[i] for i in keep_idx]
    print(f"\nStage 1: phash grids {hash_sizes} over {len(stage1_paths)} "
          f"remaining images (Hamming <= {args.hash_threshold}, "
          f"candidates unioned across grids)...")
    hashed_paths, packed_by_size = compute_packed_phashes(stage1_paths, hash_sizes)
    print(f"  hashed {len(hashed_paths)} images; scanning pairs...")
    cand_full = union_candidates(packed_by_size, args.hash_threshold,
                                 args.chunk_size)
    cand = [(i, j, d) for i, j, d, _hs in cand_full]
    print(f"  {len(cand)} candidate pairs after union across {hash_sizes}.")

    meta = {p: (lab, ds) for p, lab, ds in zip(paths, labels, datasets)}
    with open(os.path.join(args.out_dir, "near_duplicate_candidates.csv"),
              "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path1", "path2", "hamming_dist"])
        for i, j, d in cand:
            w.writerow([hashed_paths[i], hashed_paths[j], d])

    # ---- stage 2 --------------------------------------------------------
    print(f"\nStage 2: pixel-level verification (NCC >= {args.ncc_threshold})...")
    confirmed = verify_candidates(cand, hashed_paths, args.ncc_threshold)
    print(f"  {len(confirmed)} confirmed pairs (of {len(cand)} candidates).")

    same_class = 0
    within, across = 0, 0
    cross_matrix = defaultdict(int)
    with open(os.path.join(args.out_dir, "near_duplicate_confirmed.csv"),
              "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path1", "path2", "hamming_dist", "ncc_score",
                    "same_class", "dataset1", "dataset2", "scope"])
        for i, j, d, s in confirmed:
            p1, p2 = hashed_paths[i], hashed_paths[j]
            lab1, ds1 = meta[p1]
            lab2, ds2 = meta[p2]
            sc = (lab1 == lab2)
            same_class += int(sc)
            scope = "within" if ds1 == ds2 else "across"
            if scope == "within":
                within += 1
            else:
                across += 1
                cross_matrix["|".join(sorted((ds1, ds2)))] += 1
            w.writerow([p1, p2, d, round(s, 4), sc, ds1, ds2, scope])

    frac_same = (same_class / len(confirmed)) if confirmed else 0.0
    if confirmed:
        print(f"\n  Sanity check: {frac_same * 100:.1f}% of confirmed pairs share "
              f"a class label (step9 scored 99.6% on the Kaggle 4-class set; "
              f"the discarded 8x8 average-hash scored 49% against a 25% base "
              f"rate).")
        if frac_same < 0.90:
            print("  [WARN] Below 90% -- tighten --hash-threshold or raise "
                  "--ncc-threshold before trusting this audit.")
    if len(roots) > 1:
        print(f"\n  Pair scope: {within} within-dataset, {across} across-dataset.")
        for k, v in sorted(cross_matrix.items(), key=lambda kv: -kv[1]):
            print(f"    {k}: {v} cross-dataset pairs")
        if across:
            print("  NOTE: across-dataset pairs mean these roots overlap. Any "
                  "'external validation' of a model trained on one root using "
                  "another is leakage to that extent.")

    # ---- recall probe ---------------------------------------------------
    probe = None
    if args.recall_probe:
        print(f"\nRecall probe: exhaustive {args.probe_thumb}x{args.probe_thumb} "
              f"coarse-NCC oracle (>= {args.probe_ncc}), independent of the "
              f"perceptual hash...")
        oracle = coarse_ncc_pairs(hashed_paths, args.probe_thumb,
                                  args.probe_ncc, args.chunk_size)
        proposed = {(i, j) for i, j, _ in cand}
        missed = [(i, j, s) for i, j, s in oracle if (i, j) not in proposed]
        recall = (1.0 - len(missed) / len(oracle)) if oracle else 1.0
        print(f"  oracle found {len(oracle)} high-NCC pairs; stage 1 proposed "
              f"{len(oracle) - len(missed)} of them.")
        print(f"  STAGE-1 RECALL = {recall * 100:.1f}%")
        if missed:
            print(f"  [WARN] {len(missed)} genuine-looking pairs were never "
                  f"proposed to stage 2. The contamination rate below is a "
                  f"LOWER BOUND. See stage1_missed_pairs.csv; consider adding "
                  f"a smaller grid to --hash-sizes or raising "
                  f"--hash-threshold.")
        with open(os.path.join(args.out_dir, "stage1_missed_pairs.csv"),
                  "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path1", "path2", "coarse_ncc", "dataset1", "dataset2"])
            for i, j, s in missed:
                p1, p2 = hashed_paths[i], hashed_paths[j]
                w.writerow([p1, p2, round(s, 4), meta[p1][1], meta[p2][1]])
        probe = {
            "probe_thumb": args.probe_thumb,
            "probe_ncc_threshold": args.probe_ncc,
            "n_oracle_pairs": len(oracle),
            "n_missed_by_stage1": len(missed),
            "stage1_recall": round(recall, 4),
        }

    # ---- removals -------------------------------------------------------
    near_drop = set()
    for i, j, _, _ in confirmed:
        _, drop = sorted((hashed_paths[i], hashed_paths[j]))
        near_drop.add(drop)

    non_mri_rows = []
    if not args.skip_non_mri:
        screen = [p for p in stage1_paths if p not in near_drop]
        print(f"\nNon-MRI heuristic screen over {len(screen)} remaining images...")
        for p in screen:
            reasons = flag_non_mri(p)
            if reasons:
                non_mri_rows.append((p, meta[p][0], meta[p][1], "; ".join(reasons)))
        print(f"  {len(non_mri_rows)} flagged for MANUAL review "
              f"(heuristic only -- not auto-removed).")
    with open(os.path.join(args.out_dir, "non_mri_candidates.csv"),
              "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "class", "dataset", "flag_reason"])
        w.writerows(non_mri_rows)

    with open(os.path.join(args.out_dir, "removal_manifest.csv"),
              "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "reason"])
        w.writerows([(p, "exact_duplicate") for p in sorted(exact_drop)] +
                    [(p, "near_duplicate") for p in sorted(near_drop)])

    removed = exact_drop | near_drop
    cleaned = [p for p in paths if p not in removed]
    with open(os.path.join(args.out_dir, "cleaned_file_list.txt"), "w") as f:
        f.write("\n".join(cleaned) + "\n")

    summary = {
        "datasets": {n: {"root": r, "classes": per_dataset_counts[n]}
                     for n, r in roots},
        "params": {
            "hash_sizes": hash_sizes,
            "hash_threshold": args.hash_threshold,
            "ncc_threshold": args.ncc_threshold,
        },
        "recall_probe": probe,
        "n_images": n_total,
        "n_exact_duplicate_groups": len(exact_groups),
        "n_exact_removed": len(exact_drop),
        "n_candidate_pairs": len(cand),
        "n_confirmed_pairs": len(confirmed),
        "n_confirmed_within": within,
        "n_confirmed_across": across,
        "cross_dataset_pairs": dict(cross_matrix),
        "frac_confirmed_pairs_same_class": round(frac_same, 4),
        "n_near_duplicate_removed": len(near_drop),
        "n_non_mri_flagged": len(non_mri_rows),
        "n_cleaned": len(cleaned),
        "contamination_rate": round(len(removed) / n_total, 4) if n_total else 0.0,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    print(f"  Original images:          {n_total}")
    print(f"  Exact duplicates removed: {len(exact_drop)}")
    print(f"  Near-duplicates removed:  {len(near_drop)}")
    print(f"  Non-MRI flagged (manual): {len(non_mri_rows)}")
    print(f"  Cleaned dataset size:     {len(cleaned)}")
    print(f"  Contamination rate:       "
          f"{summary['contamination_rate'] * 100:.2f}%"
          + ("" if probe is None else
             f"   (stage-1 recall {probe['stage1_recall'] * 100:.1f}%"
             + ("" if probe["n_missed_by_stage1"] == 0 else " -- LOWER BOUND")
             + ")"))
    print(f"\nOutputs in {args.out_dir} (summary.json is the machine-readable "
          f"version for the cross-dataset table).")


if __name__ == "__main__":
    main()
