#!/usr/bin/env python3
"""Extract Flickr2K.tar into ./Flickr2K.

Default behavior:
- Input: ./_raw/Flickr2K.tar (same folder as this script)
- Output: ./Flickr2K
"""

import argparse
import os
import shutil
import tarfile
import tempfile


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


def _flatten_root(extract_dir: str) -> str:
    entries = [e for e in os.listdir(extract_dir) if not e.startswith(".")]
    if len(entries) == 1:
        only = os.path.join(extract_dir, entries[0])
        if os.path.isdir(only):
            return only
    return extract_dir


def _move_contents(src_dir: str, dest_dir: str, overwrite: bool) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    for name in os.listdir(src_dir):
        src = os.path.join(src_dir, name)
        dst = os.path.join(dest_dir, name)
        if os.path.exists(dst):
            if not overwrite:
                print(f"[skip] {dst} exists")
                continue
            if os.path.isdir(dst):
                for root, dirs, files in os.walk(dst, topdown=False):
                    for f in files:
                        os.remove(os.path.join(root, f))
                    for d in dirs:
                        os.rmdir(os.path.join(root, d))
                os.rmdir(dst)
            else:
                os.remove(dst)
        shutil.move(src, dst)


def _remove_tree(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def _normalize_layout(out_dir: str, overwrite: bool) -> None:
    hr_src = os.path.join(out_dir, "Flickr2K_HR")
    lr_src = os.path.join(out_dir, "Flickr2K_LR_bicubic", "X4")
    hr_dst = os.path.join(out_dir, "HR")
    lr_dst = os.path.join(out_dir, "LR")

    if os.path.isdir(hr_src):
        _move_contents(hr_src, hr_dst, overwrite)
        _remove_tree(hr_src)

    if os.path.isdir(lr_src):
        _move_contents(lr_src, lr_dst, overwrite)
        _remove_tree(os.path.join(out_dir, "Flickr2K_LR_bicubic"))

    _remove_tree(os.path.join(out_dir, "Flickr2K_LR_unknown"))


def _extract_tar(tar_path: str, out_dir: str, overwrite: bool) -> None:
    if not os.path.exists(tar_path):
        raise FileNotFoundError(f"Missing tar: {tar_path}")

    print(f"[unzip] {os.path.basename(tar_path)} -> {out_dir}")
    with tarfile.open(tar_path, "r:") as tar:
        with tempfile.TemporaryDirectory() as tmp:
            _safe_extract(tar, tmp)
            root = _flatten_root(tmp)
            _move_contents(root, out_dir, overwrite)
            _normalize_layout(out_dir, overwrite)


def main() -> int:
    base_dir = os.path.abspath(os.path.dirname(__file__))
    default_src = os.path.join(base_dir, "_raw", "Flickr2K.tar")
    default_out = os.path.join(base_dir, "Flickr2K")

    parser = argparse.ArgumentParser(description="Extract Flickr2K.tar")
    parser.add_argument(
        "--src",
        default=default_src,
        help="Path to Flickr2K.tar (default: ./_raw/Flickr2K.tar)",
    )
    parser.add_argument(
        "--out",
        default=default_out,
        help="Output directory (default: ./Flickr2K)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files/directories",
    )
    args = parser.parse_args()

    src = os.path.abspath(args.src)
    out = os.path.abspath(args.out)

    if not os.path.isfile(src):
        print(f"Source tar not found: {src}")
        return 1

    _extract_tar(src, out, args.overwrite)
    print("Extraction complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
