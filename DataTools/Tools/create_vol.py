#!/usr/bin/env python3
"""
glue_tiffs.py  —  Glue multi-folder reconstructions into a single stack.

RULES
------
•  N = central frame count (default 1030).
•  First folder   :  [HEAD] + central N.
•  Middle folders :  central N only.
•  Last folder    :  central N + [TAIL].
•  One global index stream  <prefix>_<idx>.tiff .
•  Output may be symlinks (default) or real copies (--copy).
•  Optional 8-/16-bit binned.

USAGE
-----
python glue_tiffs.py \
       --source_root /data/run001 \
       --dest_dir    /data/run001/VOL \
       --central     1030 \
       --prefix      recon \
       --binning     1 2 4 8 16 \
       --format      16 \
       --copy
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tifffile as tf


# ------------------------------------------------------------------ helpers --
def list_tiff_folders(root: Path) -> List[Path]:
    """Immediate sub-directories of *root* that contain at least one TIFF."""
    return sorted(
        p for p in root.iterdir()
        if p.is_dir() and any(p.glob("*.tif*"))
    )


def sorted_tiffs(folder: Path) -> List[Path]:
    """Natural-sort list of TIFFs in *folder*."""
    return sorted(folder.glob("*.tif*"))


def slice_indices(n_img: int, central: int, first: bool, last: bool) -> range:
    """Return the indices to keep for this folder."""
    start = (n_img - central) // 2
    if first and last:           # only one folder in total
        return range(n_img)
    if first:
        return range(0, start + central)          # head + central
    if last:
        return range(start, n_img)                # central + tail
    return range(start, start + central)          # middle


def assemble(root: Path, dest: Path, central: int,
             prefix: str, copy_files: bool) -> int:
    """
    Build glued stack → *dest*.  Return total output frames.
    """
    folders = list_tiff_folders(root)
    if not folders:
        raise RuntimeError(f"No TIFF folders inside {root}")

    selections: List[Tuple[List[Path], range]] = []
    total_frames = 0

    # pass-1  build selections + count output length
    for i, folder in enumerate(folders):
        files = sorted_tiffs(folder)
        if len(files) < central:
            raise RuntimeError(f"{folder} has {len(files)} files (<{central})")
        rng = slice_indices(len(files), central,
                            first=(i == 0), last=(i == len(folders) - 1))
        selections.append((files, rng))
        total_frames += len(rng)

    pad = len(str(total_frames))
    dest.mkdir(parents=True, exist_ok=True)

    # pass-2  copy / link
    idx = 1
    for files, rng in selections:
        for j in rng:
            src = files[j]
            dst = dest / f"{prefix}_{idx:0{pad}d}.tiff"
            if copy_files:
                shutil.copy2(src, dst)
            else:
                os.symlink(src.resolve(), dst)
            idx += 1

    return total_frames


# -------------------------------------------------------------- binning step --
def bin_convert(binning: List[int], src_dir: Path,
                start: int, end: int, bit_fmt: int):
    """
    Generate binned stacks (8/16-bit) from slices [start:end] in *src_dir*.
    """
    image_files = sorted(src_dir.glob("*.tif*"))
    end = min(end, len(image_files))

    # quick global min / max (use 100 images around the middle if large)
    probe = image_files[len(image_files)//2 - 50 : len(image_files)//2 + 50]
    overall_max = max(tf.imread(f).max() for f in probe)
    overall_min = min(tf.imread(f).min() for f in probe)

    out_dirs = {
        b: src_dir.parent / f"{src_dir.name}_{b}_{bit_fmt}"
        for b in binning
    }
    for d in out_dirs.values():
        d.mkdir(exist_ok=True)

    for n, f in enumerate(image_files[start:end], start=start):
        img = tf.imread(f)
        for b in binning:
            small = img.reshape(img.shape[0] // b, b,
                                img.shape[1] // b, b).mean(axis=(1, 3))
            scale = 255 if bit_fmt == 8 else 65535
            out = ((small - overall_min) /
                   (overall_max - overall_min) * scale).astype(
                       np.uint8 if bit_fmt == 8 else np.uint16)
            tf.imwrite(out_dirs[b] / f"recon_{n:05d}.tiff", out)


# ------------------------------------------------------------------  CLI -----
def main():
    p = argparse.ArgumentParser(
        description="Glue TIFF reconstructions from multiple folders "
                    "into one stack, then optionally create binned copies.")
    p.add_argument("--source_root", required=True,
                   help="Directory that holds sub-folders with TIFFs.")
    p.add_argument("--dest_dir", required=True,
                   help="Destination directory for the glued volume.")
    p.add_argument("--central", type=int, default=1030,
                   help="Central frame count per folder (default 1030).")
    p.add_argument("--prefix", default="img",
                   help="Output file prefix (default 'img').")
    p.add_argument("--copy", action="store_true",
                   help="Physically copy files instead of symlinking.")
    # binning options
    p.add_argument("--binning", type=int, nargs="+", default=[1, 2, 4, 8, 16],
                   help="Binning factors (default 1 2 4 8 16).")
    p.add_argument("--format", type=int, choices=[8, 16], default=16,
                   help="Bit depth for binned stacks (default 16).")
    p.add_argument("--start_vol", type=int, default=0,
                   help="First slice index for binning.")
    p.add_argument("--end_vol", type=int, default=10**9,
                   help="Last slice index (exclusive) for binning.")
    args = p.parse_args()

    root = Path(args.source_root).expanduser().resolve()
    dest = Path(args.dest_dir).expanduser().resolve()

    print(f"Assembling → {dest}")
    total = assemble(root, dest, args.central,
                     args.prefix, args.copy)
    print(f"   {total} slices written.")

    if args.binning:
        print(f"Binning {args.binning} → {args.format}-bit "
              f"[{args.start_vol}:{args.end_vol}]")
        bin_convert(args.binning, dest,
                    args.start_vol, args.end_vol, args.format)
        print("   Binning done.")


if __name__ == "__main__":
    main()

