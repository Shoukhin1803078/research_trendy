"""
Step 18: BRISC data layer and the leave-one-plane-out (LOPO) protocol.

This is the foundation for the plane-generalization contribution. It does not
train anything; it parses the dataset, reports its structure, and builds the
splits that step19 trains on. Keeping it separate means the protocol can be
inspected and sanity-checked before any GPU time is spent.

BRISC (Scientific Data, 2026) provides 2D T1 slices with four classes --
glioma, meningioma, pituitary, no_tumor -- plus physician-validated pixel
masks for the tumour classes, and it encodes the acquisition plane in each
filename:

    brisc2025_<split>_<index>_<tumor_code>_<plane_code>_<sequence>.<ext>

The plane code is the asset nothing else in this project had. On the Kaggle
corpus, the mixture of axial/coronal/sagittal slices inside each class was an
uncontrollable confound. Here the plane is a known variable, so it can be
held out, which turns it from a confound into a domain-generalization axis.

WHY LEAVE-ONE-PLANE-OUT, NOT LEAVE-TWO-OUT
------------------------------------------
The method in step19 makes features invariant ACROSS the training planes, so
it needs at least two of them visible during training. LOPO (train on two
planes, test on the third) is therefore the protocol the method requires, and
it is the standard multi-source domain-generalization setup. A single-source
condition (train on one plane, test on the other two) is harder and supports
no invariance penalty; it is available here via --single-source as an extra
stress condition, but it is not the main protocol.

Note this is DOMAIN GENERALIZATION, not domain adaptation: no image, label or
statistic from the held-out plane is used at any point in training. That is
the distinction from orientation-aware UDA work, which adapts using target
data and specialises a model per plane. Here a single model must transfer.

WHAT THE SPLITS LOOK LIKE
-------------------------
For each held-out plane P, every image whose plane != P is a source image and
every image whose plane == P is a target image. The source pool is then split
into train / val / in-distribution-test by stratified sampling on the class
label, so that two numbers can be compared for the same model:

    ID  accuracy: held-out images from the SOURCE planes
    OOD accuracy: all images from the HELD-OUT plane

The gap between them is the quantity this contribution is about. Reporting ID
alongside OOD matters: an OOD number on its own cannot distinguish a model
that generalizes badly from a model that is simply weak.

NO LEAKAGE ACROSS THE SPLIT BOUNDARY
------------------------------------
BRISC ships its own train/test directories. We merge them and re-split, the
same way step6 did for the Kaggle corpus, because the LOPO partition has to
be by plane rather than by the publisher's split. Since BRISC is curated we
do not expect the duplicate contamination that step9/step15 found in the
Kaggle corpus, but `audit_overlap()` below checks source/target overlap by
SHA-256 anyway -- cheap, and it would be embarrassing to assume.

Pure stdlib + numpy + PIL. No TensorFlow, no GPU.
"""

import os
import re
import csv
import json
import glob
import hashlib
import argparse
from collections import defaultdict, Counter

import numpy as np

# ---------------------------------------------------------------------------
# Notebook config -- EDIT THIS when running inside a Kaggle/Jupyter cell.
# Same rationale as step15-17: argparse cannot read a notebook cell's intent
# from the kernel's argv, so these defaults take over there.
# ---------------------------------------------------------------------------
NB_DATA_DIR = "/kaggle/input/notebooks/dipayanpaul07/brisc2025"
NB_OUT_DIR = "/kaggle/working/brisc_protocol"
NB_SEED = 42
NB_VAL_FRAC = 0.15
NB_ID_TEST_FRAC = 0.15
NB_SINGLE_SOURCE = False
NB_AUDIT = True

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_EXTS = {".png"}

# Canonical class names. BRISC's tumour codes are short; we discover the codes
# present in the filenames and map them, rather than assuming an encoding.
CANONICAL_CLASSES = ["glioma", "meningioma", "pituitary", "no_tumor"]
CANONICAL_PLANES = ["axial", "coronal", "sagittal"]

# Prefix-based resolution of a discovered code to a canonical name. Checked
# longest-first so "me" cannot shadow "meningioma".
CLASS_ALIASES = {
    "glioma": "glioma", "gl": "glioma", "g": "glioma",
    "meningioma": "meningioma", "men": "meningioma", "me": "meningioma",
    "pituitary": "pituitary", "pit": "pituitary", "pi": "pituitary",
    "no_tumor": "no_tumor", "notumor": "no_tumor", "no": "no_tumor",
    "none": "no_tumor", "healthy": "no_tumor", "n": "no_tumor",
}
PLANE_ALIASES = {
    "axial": "axial", "ax": "axial", "a": "axial", "tra": "axial",
    "coronal": "coronal", "cor": "coronal", "co": "coronal", "c": "coronal",
    "sagittal": "sagittal", "sag": "sagittal", "sa": "sagittal", "s": "sagittal",
}

FNAME_RE = re.compile(
    r"^brisc\d*_(?P<split>[a-z]+)_(?P<index>\d+)_(?P<tumor>[a-z_]+)_"
    r"(?P<plane>[a-z]+)_(?P<seq>[a-z0-9]+)$", re.I)


def _fail(msg, data_dir=None):
    """Raise with a message a notebook will actually show.

    argparse's ap.error() and bare SystemExit both call sys.exit(), which
    IPython renders as a terse "To exit: use 'exit', 'quit', or Ctrl-D"
    UserWarning with the real reason scrolled off into stderr. In a notebook
    that reads as a mysterious crash, so failures here raise RuntimeError
    with the diagnosis attached instead.
    """
    lines = ["", "=" * 70, "STEP 18 COULD NOT RUN", "=" * 70, msg]
    if data_dir is not None:
        lines += ["", f"Looked for: {data_dir}",
                  f"  exists: {os.path.exists(data_dir)}",
                  f"  is dir: {os.path.isdir(data_dir)}", ""]
        for probe in ("/kaggle/input", "/kaggle/working"):
            if not os.path.isdir(probe):
                continue
            lines.append(f"Directories under {probe} (depth 3):")
            found = []
            for root, dirs, _files in os.walk(probe):
                depth = root[len(probe):].count(os.sep)
                if depth >= 3:
                    dirs[:] = []
                    continue
                found.append(root)
                if len(found) > 60:
                    break
            lines += [f"    {d}" for d in found[:60]]
            if len(found) > 60:
                lines.append("    ... (truncated)")
            lines.append("")
        lines.append("Set NB_DATA_DIR to whichever of those holds the BRISC "
                     "images, then re-run this cell.")
    lines.append("=" * 70)
    text = "\n".join(lines)
    print(text)
    raise RuntimeError(msg)


def _resolve(code, aliases, what, path):
    c = code.lower()
    if c in aliases:
        return aliases[c]
    for k in sorted(aliases, key=len, reverse=True):
        if c.startswith(k):
            return aliases[k]
    raise ValueError(
        f"Could not resolve {what} code {code!r} from {os.path.basename(path)}. "
        f"Add it to the alias table at the top of step18_brisc_data.py "
        f"(known: {sorted(set(aliases.values()))})."
    )


def parse_name(path):
    """Parse a BRISC filename into its fields, or return None if it does not
    match. Returning None rather than raising lets the caller report how many
    files were unparseable, which is the signal that the convention differs
    from what this script expects."""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = FNAME_RE.match(stem)
    if not m:
        return None
    return {
        "path": path,
        "stem": stem,
        "pub_split": m.group("split").lower(),
        "index": int(m.group("index")),
        "class_code": m.group("tumor"),
        "plane_code": m.group("plane"),
        "seq": m.group("seq"),
    }


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def find_masks(data_dir):
    """Map basename-stem -> mask path, over anything that looks like a mask
    tree. BRISC pairs an image .jpg with a mask .png of the same basename."""
    masks = {}
    for dirpath, _dirs, files in os.walk(data_dir):
        low = dirpath.lower()
        if not any(k in low for k in ("mask", "label", "seg", "annotation", "gt")):
            continue
        for fn in files:
            if os.path.splitext(fn)[1].lower() in MASK_EXTS:
                masks[os.path.splitext(fn)[0]] = os.path.join(dirpath, fn)
    return masks


def load_records(data_dir):
    """Walk the dataset and return (records, diagnostics)."""
    mask_index = find_masks(data_dir)
    mask_dirs = {os.path.dirname(p) for p in mask_index.values()}

    records, unparsed, n_images = [], [], 0
    for dirpath, _dirs, files in os.walk(data_dir):
        if dirpath in mask_dirs:
            continue  # masks are paired in, not treated as samples
        for fn in sorted(files):
            if os.path.splitext(fn)[1].lower() not in IMAGE_EXTS:
                continue
            path = os.path.join(dirpath, fn)
            n_images += 1
            rec = parse_name(path)
            if rec is None:
                unparsed.append(path)
                continue
            rec["label"] = _resolve(rec["class_code"], CLASS_ALIASES, "class", path)
            rec["plane"] = _resolve(rec["plane_code"], PLANE_ALIASES, "plane", path)
            rec["mask"] = mask_index.get(rec["stem"])
            records.append(rec)

    diag = {
        "n_image_files_seen": n_images,
        "n_parsed": len(records),
        "n_unparsed": len(unparsed),
        "unparsed_examples": [os.path.basename(p) for p in unparsed[:10]],
        "n_mask_files_found": len(mask_index),
        "n_records_with_mask": sum(1 for r in records if r["mask"]),
    }
    return records, diag


def report_structure(records, diag):
    """Print the class x plane contingency table.

    This table is itself a result worth having. If the plane distribution is
    uneven across classes, then plane partially predicts the label, and a
    classifier can score well by reading the plane instead of the lesion --
    the same shortcut mechanism the Kaggle corpus's brightness difference
    raised, but here measurable and controllable.
    """
    print(f"\nimage files seen : {diag['n_image_files_seen']}")
    print(f"filenames parsed : {diag['n_parsed']}"
          f"   unparsed: {diag['n_unparsed']}")
    if diag["n_unparsed"]:
        print(f"  [WARN] unparsed examples: {diag['unparsed_examples']}")
        print(f"  [WARN] if this count is large, the filename convention "
              f"differs from FNAME_RE -- fix that before trusting anything below.")
    print(f"mask files found : {diag['n_mask_files_found']}"
          f"   records with a mask: {diag['n_records_with_mask']}")

    planes = sorted({r["plane"] for r in records},
                    key=lambda p: CANONICAL_PLANES.index(p)
                    if p in CANONICAL_PLANES else 99)
    classes = sorted({r["label"] for r in records},
                     key=lambda c: CANONICAL_CLASSES.index(c)
                     if c in CANONICAL_CLASSES else 99)

    grid = Counter((r["label"], r["plane"]) for r in records)
    mask_grid = Counter((r["label"], r["plane"]) for r in records if r["mask"])

    w = max(12, max(len(c) for c in classes) + 2)
    print(f"\nCLASS x PLANE (count, and [with mask])")
    print("-" * (w + 20 * len(planes) + 10))
    print(f"{'class':<{w}}" + "".join(f"{p:>20}" for p in planes) + f"{'total':>10}")
    for c in classes:
        row = f"{c:<{w}}"
        for p in planes:
            row += f"{grid[(c, p)]:>12} [{mask_grid[(c, p)]:>4}]"
        row += f"{sum(grid[(c, p)] for p in planes):>10}"
        print(row)
    row = f"{'total':<{w}}"
    for p in planes:
        row += f"{sum(grid[(c, p)] for c in classes):>12} " \
               f"[{sum(mask_grid[(c, p)] for c in classes):>4}]"
    print(row + f"{len(records):>10}")

    # Is plane informative about class on its own?
    tot = len(records)
    chi2 = 0.0
    for c in classes:
        rc = sum(grid[(c, p)] for p in planes)
        for p in planes:
            cp = sum(grid[(cc, p)] for cc in classes)
            exp = rc * cp / tot if tot else 0
            if exp > 0:
                chi2 += (grid[(c, p)] - exp) ** 2 / exp
    dof = (len(classes) - 1) * (len(planes) - 1)
    cramer = (chi2 / (tot * min(len(classes) - 1, len(planes) - 1))) ** 0.5 if tot else 0
    print(f"\nclass-plane association: chi2 = {chi2:.1f} (dof {dof}), "
          f"Cramer's V = {cramer:.3f}")
    print("  V near 0 means plane carries little information about the class, "
          "so plane cannot act as a label shortcut.")
    print("  V far from 0 means it can, and that belongs in the paper as a "
          "property of the benchmark.")
    return planes, classes, float(cramer)


# ---------------------------------------------------------------------------
# Overlap audit (cheap insurance)
# ---------------------------------------------------------------------------
def audit_overlap(records, planes):
    """Byte-identical duplicates, overall and specifically across planes.

    A duplicate that spans two planes would be mislabelled in one of them and
    would leak a source image into the held-out target, which is precisely the
    failure the LOPO protocol exists to avoid.
    """
    print("\nOverlap audit (SHA-256)...")
    by_hash = defaultdict(list)
    for r in records:
        h = hashlib.sha256()
        with open(r["path"], "rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
        by_hash[h.hexdigest()].append(r)

    dupe_groups = [v for v in by_hash.values() if len(v) > 1]
    cross_plane, cross_class = 0, 0
    for g in dupe_groups:
        if len({r["plane"] for r in g}) > 1:
            cross_plane += 1
        if len({r["label"] for r in g}) > 1:
            cross_class += 1
    n_dupe_files = sum(len(g) - 1 for g in dupe_groups)
    print(f"  {len(dupe_groups)} exact-duplicate groups, "
          f"{n_dupe_files} redundant files")
    print(f"  groups spanning >1 plane: {cross_plane}"
          f"   spanning >1 class: {cross_class}")
    if cross_plane:
        print("  [WARN] cross-plane duplicates leak source images into the "
              "held-out plane. Drop them before running step19.")
    if cross_class:
        print("  [WARN] identical images with conflicting class labels.")
    return {"n_duplicate_groups": len(dupe_groups),
            "n_redundant_files": n_dupe_files,
            "n_cross_plane_groups": cross_plane,
            "n_cross_class_groups": cross_class}


# ---------------------------------------------------------------------------
# LOPO splits
# ---------------------------------------------------------------------------
def stratified_three_way(items, key, fracs, rng):
    """Split items into (a, b, c) with the given fractions for b and c,
    stratified on key(item). Deterministic given rng."""
    f_b, f_c = fracs
    by = defaultdict(list)
    for it in items:
        by[key(it)].append(it)
    a, b, c = [], [], []
    for k in sorted(by):
        grp = by[k][:]
        rng.shuffle(grp)
        n = len(grp)
        n_b = int(round(n * f_b))
        n_c = int(round(n * f_c))
        n_b = min(n_b, max(0, n - n_c))
        c.extend(grp[:n_c])
        b.extend(grp[n_c:n_c + n_b])
        a.extend(grp[n_c + n_b:])
    for lst in (a, b, c):
        rng.shuffle(lst)
    return a, b, c


def build_lopo(records, planes, seed, val_frac, id_test_frac,
               single_source=False):
    """One fold per plane. In LOPO mode the held-out plane is the target and
    the other planes are sources; in single-source mode that is inverted, so
    one plane trains and the rest are the target."""
    folds = {}
    for held in planes:
        rng = np.random.default_rng(
            abs(hash((seed, held))) % (2 ** 32))

        if single_source:
            src = [r for r in records if r["plane"] == held]
            tgt = [r for r in records if r["plane"] != held]
            name, src_desc = f"only_{held}", [held]
        else:
            src = [r for r in records if r["plane"] != held]
            tgt = [r for r in records if r["plane"] == held]
            name = f"holdout_{held}"
            src_desc = [p for p in planes if p != held]

        train, val, id_test = stratified_three_way(
            src, lambda r: (r["label"], r["plane"]),
            (val_frac, id_test_frac), rng)

        folds[name] = {
            "held_out_plane": held,
            "source_planes": src_desc,
            "train": train, "val": val, "id_test": id_test, "ood_test": tgt,
        }
    return folds


def report_folds(folds):
    print(f"\n{'=' * 78}\nLEAVE-ONE-PLANE-OUT FOLDS\n{'=' * 78}")
    print(f"{'fold':<20}{'train':>8}{'val':>8}{'ID test':>9}"
          f"{'OOD test':>10}{'OOD masks':>11}")
    for name, f in folds.items():
        n_m = sum(1 for r in f["ood_test"] if r["mask"])
        print(f"{name:<20}{len(f['train']):>8}{len(f['val']):>8}"
              f"{len(f['id_test']):>9}{len(f['ood_test']):>10}{n_m:>11}")
    print("\nID test  = held-out images from the SOURCE planes (same domain)")
    print("OOD test = every image from the HELD-OUT plane (shifted domain)")
    print("The ID-minus-OOD gap is the quantity step19 attacks. Report both; "
          "an OOD number alone cannot separate poor transfer from a weak model.")


def write_splits(folds, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for name, f in folds.items():
        for part in ("train", "val", "id_test", "ood_test"):
            for r in f[part]:
                rows.append([name, f["held_out_plane"], part, r["path"],
                             r["label"], r["plane"], r["mask"] or ""])
    csv_path = os.path.join(out_dir, "lopo_splits.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fold", "held_out_plane", "part", "path", "label",
                    "plane", "mask"])
        w.writerows(rows)
    print(f"\nWrote {csv_path} ({len(rows)} rows)")
    return csv_path


def load_splits(csv_path):
    """Read lopo_splits.csv back. step19 uses this so that training consumes
    exactly the partition that was inspected here, with no re-derivation."""
    folds = defaultdict(lambda: defaultdict(list))
    meta = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            folds[row["fold"]][row["part"]].append({
                "path": row["path"], "label": row["label"],
                "plane": row["plane"], "mask": row["mask"] or None,
            })
            meta[row["fold"]] = row["held_out_plane"]
    out = {}
    for name, parts in folds.items():
        out[name] = dict(parts)
        out[name]["held_out_plane"] = meta[name]
    return out


def main(argv=None):
    import sys
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", help="BRISC root; omit to use NB_DATA_DIR")
    ap.add_argument("--out-dir", help="omit to use NB_OUT_DIR")
    ap.add_argument("--seed", type=int, default=NB_SEED)
    ap.add_argument("--val-frac", type=float, default=NB_VAL_FRAC)
    ap.add_argument("--id-test-frac", type=float, default=NB_ID_TEST_FRAC)
    ap.add_argument("--single-source", action="store_true",
                    help="invert the protocol: train on ONE plane, test on "
                         "the others. Harder, and incompatible with the "
                         "invariance penalty in step19.")
    ap.add_argument("--no-audit", action="store_true")

    if argv is not None:
        args = ap.parse_args(argv)
    elif "ipykernel" in sys.modules or len(sys.argv) <= 1:
        args = argparse.Namespace(
            data_dir=NB_DATA_DIR, out_dir=NB_OUT_DIR, seed=NB_SEED,
            val_frac=NB_VAL_FRAC, id_test_frac=NB_ID_TEST_FRAC,
            single_source=NB_SINGLE_SOURCE, no_audit=not NB_AUDIT)
        print("No CLI arguments (notebook or bare run) -- using the NB_* "
              "config block at the top of this file:")
        print(f"  data-dir : {args.data_dir}")
        print(f"  out-dir  : {args.out_dir}\n")
    else:
        args = ap.parse_args()

    if not args.data_dir:
        _fail("No data directory. Pass --data-dir or set NB_DATA_DIR.")
    if not args.out_dir:
        _fail("No output directory. Pass --out-dir or set NB_OUT_DIR.")
    if not os.path.isdir(args.data_dir):
        _fail(f"NB_DATA_DIR is not a directory on this machine.", args.data_dir)

    print(f"Scanning {args.data_dir} ...")
    records, diag = load_records(args.data_dir)
    if not records:
        _fail(
            f"Found {diag['n_image_files_seen']} image files but parsed 0 "
            f"BRISC-style filenames.\n"
            f"Expected names like 'brisc2025_train_00001_gl_ax_t1.jpg'.\n"
            f"Unparsed examples: {diag['unparsed_examples']}\n"
            f"If those look different, paste one to me and FNAME_RE at the "
            f"top of this file needs a one-line change.", args.data_dir)
    planes, classes, cramer = report_structure(records, diag)

    audit = None
    if not args.no_audit:
        audit = audit_overlap(records, planes)

    folds = build_lopo(records, planes, args.seed, args.val_frac,
                       args.id_test_frac, args.single_source)
    report_folds(folds)
    csv_path = write_splits(folds, args.out_dir)

    summary = {
        "data_dir": args.data_dir,
        "protocol": "single_source" if args.single_source else "leave_one_plane_out",
        "seed": args.seed,
        "n_records": len(records),
        "classes": classes, "planes": planes,
        "class_plane_cramers_v": round(cramer, 4),
        "diagnostics": diag,
        "overlap_audit": audit,
        "folds": {k: {"held_out_plane": v["held_out_plane"],
                      "source_planes": v["source_planes"],
                      "n_train": len(v["train"]), "n_val": len(v["val"]),
                      "n_id_test": len(v["id_test"]),
                      "n_ood_test": len(v["ood_test"]),
                      "n_ood_with_mask": sum(1 for r in v["ood_test"] if r["mask"])}
                  for k, v in folds.items()},
        "splits_csv": csv_path,
    }
    with open(os.path.join(args.out_dir, "protocol_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {os.path.join(args.out_dir, 'protocol_summary.json')}")
    print("\nNext: step19_brisc_plane_invariant.py consumes lopo_splits.csv.")


if __name__ == "__main__":
    main()
