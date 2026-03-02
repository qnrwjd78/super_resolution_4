#!/usr/bin/env python3
"""Extract LSDIR tar.gz files into train/val HR/LR folders.

Default behavior:
- Input: ./_raw/LSDIR-raw (same folder as this script)
- Output: ./LSDIR

Expected layout after extraction:
- ./LSDIR/train/HR
- ./LSDIR/train/LR (X4)
- ./LSDIR/val/HR
- ./LSDIR/val/LR (X4)
"""

import argparse
import os
import shutil
import tarfile
import tempfile
from typing import Iterable, List, Optional


def _find_archives(src_dir: str) -> List[str]:
    files = []
    for name in sorted(os.listdir(src_dir)):
        if name.endswith(".tar.gz"):
            files.append(os.path.join(src_dir, name))
    return files


def _is_within_dir(base: str, target: str) -> bool:
    base = os.path.abspath(base)
    target = os.path.abspath(target)
    return os.path.commonpath([base]) == os.path.commonpath([base, target])


def _safe_extract(tar: tarfile.TarFile, path: str) -> None:
    for member in tar.getmembers():
        target = os.path.join(path, member.name)
        if not _is_within_dir(path, target):
            raise RuntimeError(f"Unsafe path in tar: {member.name}")
    tar.extractall(path=path)


def _find_dir(root: str, parts: Iterable[str]) -> Optional[str]:
    target_parts = list(parts)
    for cur, dirs, _ in os.walk(root):
        for d in dirs:
            if d != target_parts[-1]:
                continue
            cand = os.path.join(cur, d)
            rel_parts = os.path.relpath(cand, root).split(os.sep)
            if rel_parts[-len(target_parts) :] == target_parts:
                return cand
    return None


def _move_tree(src_dir: str, dest_dir: str, overwrite: bool) -> None:
    for cur, _, files in os.walk(src_dir):
        rel = os.path.relpath(cur, src_dir)
        out_dir = os.path.join(dest_dir, rel) if rel != "." else dest_dir
        os.makedirs(out_dir, exist_ok=True)
        for name in files:
            src = os.path.join(cur, name)
            dst = os.path.join(out_dir, name)
            if os.path.exists(dst):
                if not overwrite:
                    continue
                os.remove(dst)
            shutil.move(src, dst)


def _extract_archive(archive_path: str, out_dir: str, overwrite: bool) -> None:
    base = os.path.basename(archive_path)
    if not os.path.exists(archive_path):
        raise FileNotFoundError(f"Missing tar: {archive_path}")

    print(f"[unzip] {base}")
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(archive_path, "r:gz") as tar:
            _safe_extract(tar, tmp)

        if base.startswith("shard-"):
            dest = os.path.join(out_dir, "train", "HR")
            _move_tree(tmp, dest, overwrite)
            return

        if base == "train_x4.tar.gz":
            train_dir = _find_dir(tmp, ["train"]) or tmp
            dest = os.path.join(out_dir, "train", "LR")
            _move_tree(train_dir, dest, overwrite)
            return

        if base == "val1.tar.gz":
            hr_val = _find_dir(tmp, ["HR", "val"]) or _find_dir(tmp, ["val", "HR"])
            lr_val = _find_dir(tmp, ["X4", "val"]) or _find_dir(tmp, ["val", "X4"])
            if not hr_val or not lr_val:
                raise RuntimeError("Could not locate val HR/LR folders in val1.tar.gz")
            _move_tree(hr_val, os.path.join(out_dir, "val", "HR"), overwrite)
            _move_tree(lr_val, os.path.join(out_dir, "val", "LR"), overwrite)
            return

        print(f"[skip] {base} (not handled)")


def main() -> int:
    default_src = os.path.join(os.path.dirname(__file__), "_raw", "LSDIR-raw")
    default_out = os.path.join(os.path.dirname(__file__), "LSDIR")

    parser = argparse.ArgumentParser(description="Extract LSDIR .tar.gz archives.")
    parser.add_argument(
        "--src",
        default=default_src,
        help="Source directory containing .tar.gz files (default: ./_raw/LSDIR-raw)",
    )
    parser.add_argument(
        "--out",
        default=default_out,
        help="Output directory (default: ./LSDIR)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing extracted folders",
    )
    args = parser.parse_args()

    src_dir = os.path.abspath(args.src)
    out_dir = os.path.abspath(args.out)

    if not os.path.isdir(src_dir):
        print(f"Source directory not found: {src_dir}")
        return 1

    archives = _find_archives(src_dir)
    if not archives:
        print(f"No .tar.gz files found in {src_dir}")
        return 1

    os.makedirs(out_dir, exist_ok=True)
    print(f"Source: {src_dir}")
    print(f"Output: {out_dir}")
    print(f"Archives: {len(archives)}")

    for archive in archives:
        _extract_archive(archive, out_dir, args.overwrite)

    print("Extraction complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
