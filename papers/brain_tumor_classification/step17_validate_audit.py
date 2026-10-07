"""
Step 17: Synthetic ground-truth validation of the duplicate detector.

The audit in step9/step15 reports how many near-duplicate pairs it FOUND.
That is not the same as how many EXIST, and the difference is exactly what a
reviewer will ask about, because a contamination rate quoted from a detector
of unknown recall is a lower bound rather than a measurement.

This script builds a synthetic dataset whose contamination is known by
construction, runs the detector over it, and reports per-stage recall. It is
what backs the recall figures quoted in the paper's audit-validation
subsection.

The fixture deliberately has realistic BETWEEN-image variation (varied oval
axes, centre offset, rotation, interior blobs, brightness). A naive fixture
in which every slice shares one skull outline is useless here: every pair
looks alike, the same-class sanity check collapses to the 25% base rate, and
the run says nothing about recall. That failure mode is the same one that
sank the original 8x8 average-hash attempt described in the paper.

Planted contamination:
  - 3 byte-identical pairs           (exact duplicates, Training -> Testing)
  - 6 within-dataset near-duplicates (1-pixel shift + mild noise)
  - 4 cross-dataset near-duplicates  (dsA image reappearing in dsB)

Usage:
    python3 step17_validate_audit.py --out-dir /tmp/audit_fixture
    # then point step15 at the fixture, e.g.
    python3 step15_multidataset_audit.py \
        --data-dir dsA=/tmp/audit_fixture/dsA \
        --data-dir dsB=/tmp/audit_fixture/dsB \
        --out-dir /tmp/audit_fixture/result \
        --classes glioma,notumor --hash-sizes 16 --recall-probe

Compare the confirmed pairs against truth.json. Requires: pillow numpy
(imagehash only for the --measure pass).
"""

import os
import sys
import json
import argparse

import numpy as np
from PIL import Image



# ---------------------------------------------------------------------------
# Notebook config -- EDIT THIS when running inside a Kaggle/Jupyter cell.
# See the equivalent note in step15/step16: argparse cannot read a notebook
# cell's intent from the kernel's argv, so these defaults take over there.
# ---------------------------------------------------------------------------
NB_OUT_DIR = "/kaggle/working/audit_fixture"
NB_N_FILLER = 40
NB_SEED = 0
NB_MEASURE = True
NB_HASH_SIZES = "16"       # "16" = step9's configuration; "8,16" = the union
NB_HASH_THRESHOLD = 8
NB_NCC_THRESHOLD = 0.92
NB_PROBE_NCC = 0.95


def _running_in_notebook():
    return "ipykernel" in sys.modules


def _notebook_args():
    return argparse.Namespace(
        out_dir=NB_OUT_DIR, n_filler=NB_N_FILLER, seed=NB_SEED,
        measure=NB_MEASURE, hash_sizes=NB_HASH_SIZES,
        hash_threshold=NB_HASH_THRESHOLD, ncc_threshold=NB_NCC_THRESHOLD,
        probe_ncc=NB_PROBE_NCC,
    )


def slice_img(seed):
    """A brain-slice-like image: dark background, bright skull ring, an oval
    interior of varying geometry, and a few gaussian blobs as anatomy."""
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:128, 0:128]
    cy, cx = 64 + r.integers(-10, 11), 64 + r.integers(-10, 11)
    ay, ax = r.uniform(36, 56), r.uniform(30, 48)
    th = r.uniform(0, np.pi)
    xr = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)
    yr = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)
    q = (xr / ax) ** 2 + (yr / ay) ** 2
    img = np.zeros((128, 128), np.float32)
    inner = q <= 1.0
    img[inner] = r.uniform(50, 110)
    img[(q > 1.0) & (q <= 1.18)] = r.uniform(180, 250)
    for _ in range(r.integers(3, 9)):
        by, bx = r.integers(30, 98), r.integers(30, 98)
        s, amp = r.uniform(4, 16), r.uniform(-45, 80)
        img += amp * np.exp(-(((xx - bx) ** 2 + (yy - by) ** 2) / (2 * s * s)))
    img[~inner & (q > 1.18)] = 0
    img += r.normal(0, 4, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def save(a, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(a).save(path)


def build(root, n_filler=40, seed=0):
    rng = np.random.default_rng(seed)
    A, B = os.path.join(root, "dsA"), os.path.join(root, "dsB")
    truth = {"exact": [], "near_within": [], "near_across": []}

    sid = 1000
    for split in ["Training", "Testing"]:
        for cls in ["glioma", "notumor"]:
            for k in range(n_filler):
                sid += 1
                save(slice_img(sid), f"{A}/{split}/{cls}/{split[:2]}-{cls}-{k}.png")
    for cls in ["glioma", "notumor"]:
        for k in range(n_filler // 2):
            sid += 1
            save(slice_img(sid), f"{B}/{cls}/b-{cls}-{k}.png")

    for i in range(3):
        src = f"{A}/Training/glioma/Tr-glioma-{i}.png"
        dst = f"{A}/Testing/glioma/EXACT-copy-{i}.png"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(src, "rb") as fi, open(dst, "wb") as fo:
            fo.write(fi.read())
        truth["exact"].append((src, dst))

    for i in range(6):
        base = slice_img(5000 + i)
        p1 = f"{A}/Training/notumor/NEARa-{i}.png"
        save(base, p1)
        nd = np.roll(base, 1, axis=0).astype(np.int16) + rng.integers(-4, 5, base.shape)
        p2 = f"{A}/Testing/notumor/NEARb-{i}.png"
        save(np.clip(nd, 0, 255).astype(np.uint8), p2)
        truth["near_within"].append((p1, p2))

    for i in range(4):
        base = slice_img(7000 + i)
        p1 = f"{A}/Training/glioma/XROSSa-{i}.png"
        save(base, p1)
        nd = base.astype(np.int16) + rng.integers(-3, 4, base.shape)
        p2 = f"{B}/glioma/XROSSb-{i}.png"
        save(np.clip(nd, 0, 255).astype(np.uint8), p2)
        truth["near_across"].append((p1, p2))

    with open(os.path.join(root, "truth.json"), "w") as f:
        json.dump(truth, f, indent=2)
    return truth


def measure(truth, hash_sizes, hash_threshold, ncc_threshold, probe_ncc):
    """Report, per planted pair, which stage would accept or drop it."""
    import imagehash

    def gray(p, n):
        with Image.open(p) as im:
            return np.asarray(im.convert("L").resize((n, n)), dtype=np.float32)

    def ncc(a, b):
        a = (a - a.mean()) / (a.std() + 1e-8)
        b = (b - b.mean()) / (b.std() + 1e-8)
        return float(np.mean(a * b))

    print(f"\n{'pair':26s} " + " ".join(f"h{h:<4d}" for h in hash_sizes) +
          f" {'ncc32':>7s} {'ncc224':>7s}  stage1  stage2  oracle")
    tally = {k: [0, 0, 0, 0] for k in ("near_within", "near_across")}
    for kind in ("near_within", "near_across"):
        for p1, p2 in truth[kind]:
            with Image.open(p1) as i1, Image.open(p2) as i2:
                g1, g2 = i1.convert("L"), i2.convert("L")
                ds = {h: imagehash.phash(g1, hash_size=h) -
                         imagehash.phash(g2, hash_size=h) for h in hash_sizes}
            n32, n224 = ncc(gray(p1, 32), gray(p2, 32)), ncc(gray(p1, 224), gray(p2, 224))
            s1 = any(d <= hash_threshold for d in ds.values())
            s2, orc = n224 >= ncc_threshold, n32 >= probe_ncc
            t = tally[kind]
            t[0] += 1
            t[1] += int(s1)
            t[2] += int(s1 and s2)
            t[3] += int(orc)
            print(f"{kind[5:] + '/' + os.path.basename(p1):26s} " +
                  " ".join(f"{ds[h]:<5d}" for h in hash_sizes) +
                  f" {n32:7.4f} {n224:7.4f}  "
                  f"{'PASS' if s1 else 'DROP':6s}  {'PASS' if s2 else 'DROP':6s}  "
                  f"{'sees' if orc else 'BLIND'}")

    print(f"\n{'=' * 70}\nRECALL vs PLANTED GROUND TRUTH "
          f"(phash grids {hash_sizes}, Hamming <= {hash_threshold}, "
          f"NCC >= {ncc_threshold})\n{'=' * 70}")
    tot = [sum(tally[k][i] for k in tally) for i in range(4)]
    for kind in ("near_within", "near_across"):
        n, s1, s2, orc = tally[kind]
        print(f"  {kind:14s} planted {n:2d} | stage1 proposed {s1:2d} "
              f"({100 * s1 / n:5.1f}%) | confirmed {s2:2d} ({100 * s2 / n:5.1f}%) "
              f"| oracle saw {orc:2d}")
    n, s1, s2, orc = tot
    print(f"  {'TOTAL':14s} planted {n:2d} | stage1 proposed {s1:2d} "
          f"({100 * s1 / n:5.1f}%) | confirmed {s2:2d} ({100 * s2 / n:5.1f}%) "
          f"| oracle saw {orc:2d}")
    print("\nStage-1 recall below 100% means the detector never showed those "
          "pairs to the verifier, so any contamination rate it reports on real "
          "data is a LOWER BOUND.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", help="omit to use the NB_OUT_DIR config block")
    ap.add_argument("--n-filler", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--measure", action="store_true",
                    help="also report per-stage recall against ground truth")
    ap.add_argument("--hash-sizes", default="8,16")
    ap.add_argument("--hash-threshold", type=int, default=8)
    ap.add_argument("--ncc-threshold", type=float, default=0.92)
    ap.add_argument("--probe-ncc", type=float, default=0.95)

    if argv is not None:
        args = ap.parse_args(argv)
    elif _running_in_notebook() or len(sys.argv) <= 1:
        args = _notebook_args()
        print("No CLI arguments (notebook or bare run) -- using the NB_* "
              "config block at the top of this file:")
        print(f"  out-dir : {args.out_dir}   grids: {args.hash_sizes}\n")
    else:
        args = ap.parse_args()

    if not args.out_dir:
        ap.error("no output dir: pass --out-dir or set NB_OUT_DIR")

    truth = build(args.out_dir, args.n_filler, args.seed)
    n_img = sum(1 for _r, _d, fs in os.walk(args.out_dir) for f in fs
                if f.endswith(".png"))
    print(f"Built fixture in {args.out_dir}: {n_img} images")
    print(json.dumps({k: len(v) for k, v in truth.items()}, indent=2))

    if args.measure:
        measure(truth, [int(s) for s in args.hash_sizes.split(",")],
                args.hash_threshold, args.ncc_threshold, args.probe_ncc)


if __name__ == "__main__":
    main()
