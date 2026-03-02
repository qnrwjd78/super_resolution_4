#!/usr/bin/env python3
"""Extract DIV2K zip files into train/val/test HR/LR folders.

Expected inputs (default ./raw/DIV2K-raw):
- DIV2K_train_HR.zip -> ./DIV2K/train/HR
- DIV2K_train_LR_bicubic_X4.zip -> ./DIV2K/train/LR
- DIV2K_valid_HR.zip -> ./DIV2K/val/HR
- DIV2K_valid_LR_bicubic_X4.zip -> ./DIV2K/val/LR
- DIV2K_test_LR_bicubic_X4.zip -> ./DIV2K/test/LR

This script flattens a single top-level folder inside the zip so files land
directly under the target directory.
"""

import argparse
import os
import shutil
import tempfile
import zipfile
from typing import Dict


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _flatten_root(extract_dir: str) -> str:
    entries = [e for e in os.listdir(extract_dir) if not e.startswith(".")]
    if len(entries) != 1:
        return extract_dir

    only = os.path.join(extract_dir, entries[0])
    if not os.path.isdir(only):
        return extract_dir

    # Some LR zips contain an extra X4 folder; flatten it too.
    inner = [e for e in os.listdir(only) if not e.startswith(".")]
    if len(inner) == 1:
        inner_dir = os.path.join(only, inner[0])
        if os.path.isdir(inner_dir) and inner[0] == "X4":
            return inner_dir
    return only


def _move_contents(src_dir: str, dest_dir: str, overwrite: bool) -> None:
    _ensure_dir(dest_dir)
    for name in os.listdir(src_dir):
        src = os.path.join(src_dir, name)
        dst = os.path.join(dest_dir, name)
        if os.path.exists(dst):
            if not overwrite:
                print(f"[skip] {dst} exists")
                continue
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            else:
                os.remove(dst)
        shutil.move(src, dst)


def _extract_zip(zip_path: str, dest_dir: str, overwrite: bool) -> None:
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Missing zip: {zip_path}")
    print(f"[unzip] {os.path.basename(zip_path)} -> {dest_dir}")
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp)
        root = _flatten_root(tmp)
        _move_contents(root, dest_dir, overwrite)


def main() -> int:
    base_dir = os.path.abspath(os.path.dirname(__file__))
    default_src = os.path.join(base_dir, "_raw", "DIV2K-raw")
    default_out = os.path.join(base_dir, "DIV2K")

    parser = argparse.ArgumentParser(description="Extract DIV2K zip files.")
    parser.add_argument(
        "--src",
        default=default_src,
        help="Source directory containing DIV2K zip files (default: ./_raw/DIV2K-raw)",
    )
    parser.add_argument(
        "--out",
        default=default_out,
        help="Output base directory (default: ./DIV2K)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files/directories",
    )
    args = parser.parse_args()

    src_dir = os.path.abspath(args.src)
    out_dir = os.path.abspath(args.out)

    mapping: Dict[str, str] = {
        "DIV2K_train_LR_bicubic_X4.zip": os.path.join(out_dir, "train", "LR"),
        "DIV2K_train_HR.zip": os.path.join(out_dir, "train", "HR"),
        "DIV2K_valid_LR_bicubic_X4.zip": os.path.join(out_dir, "val", "LR"),
        "DIV2K_valid_HR.zip": os.path.join(out_dir, "val", "HR"),
        "DIV2K_test_LR_bicubic_X4.zip": os.path.join(out_dir, "test", "LR"),
    }

    print(f"Source: {src_dir}")
    print(f"Output: {out_dir}")

    for name, dest in mapping.items():
        zip_path = os.path.join(src_dir, name)
        _extract_zip(zip_path, dest, args.overwrite)

    print("Extraction complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
