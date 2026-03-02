#!/usr/bin/env python3
"""Build JSON manifests for SR datasets.

Outputs train.json, val.json, and test.json with absolute paths.
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMG_EXTS


def _norm_stem(name: str) -> str:
    stem = os.path.splitext(name)[0]
    lower = stem.lower()
    for token in ["_x4", "-x4", "x4"]:
        if lower.endswith(token):
            lower = lower[: -len(token)]
            break
    lower = lower.replace("x4", "")
    return lower


def _index_hr(hr_dir: str) -> Tuple[Dict[str, str], Dict[Tuple[str, str], str]]:
    by_stem: Dict[str, str] = {}
    by_parent_stem: Dict[Tuple[str, str], str] = {}
    for root, _, files in os.walk(hr_dir):
        for name in files:
            if not _is_image(name):
                continue
            stem = _norm_stem(name)
            path = os.path.join(root, name)
            parent = os.path.basename(root)
            key = (parent, stem)
            if key not in by_parent_stem:
                by_parent_stem[key] = path
            if stem not in by_stem:
                by_stem[stem] = path
    return by_stem, by_parent_stem


def _match_hr(lr_path: str, hr_idx: Dict[str, str], hr_parent_idx: Dict[Tuple[str, str], str]) -> Optional[str]:
    stem = _norm_stem(os.path.basename(lr_path))
    parent = os.path.basename(os.path.dirname(lr_path))
    key = (parent, stem)
    if key in hr_parent_idx:
        return hr_parent_idx[key]
    return hr_idx.get(stem)


def _collect_pairs(lr_dir: str, hr_dir: str) -> Tuple[List[Dict[str, str]], List[str]]:
    hr_idx, hr_parent_idx = _index_hr(hr_dir)
    pairs: List[Dict[str, str]] = []
    missing: List[str] = []

    for root, _, files in os.walk(lr_dir):
        for name in files:
            if not _is_image(name):
                continue
            lr_path = os.path.join(root, name)
            hr_path = _match_hr(lr_path, hr_idx, hr_parent_idx)
            if not hr_path:
                missing.append(lr_path)
                continue
            pairs.append({"lr": lr_path, "hr": hr_path})
    return pairs, missing


def _collect_lr_only(lr_dir: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    for root, _, files in os.walk(lr_dir):
        for name in files:
            if not _is_image(name):
                continue
            items.append({"lr": os.path.join(root, name)})
    return items


def main() -> int:
    base_dir = os.path.abspath(os.path.dirname(__file__))

    parser = argparse.ArgumentParser(description="Create SR dataset JSON manifest.")
    parser.add_argument(
        "--out-dir",
        default=base_dir,
        help="Output directory for train/val/test json (default: ./)",
    )
    args = parser.parse_args()

    train_pairs: List[Dict[str, str]] = []
    val_pairs: List[Dict[str, str]] = []
    test_items: List[Dict[str, str]] = []
    missing_total: List[str] = []

    mapping = [
        ("train", os.path.join(base_dir, "DIV2K", "train", "LR"), os.path.join(base_dir, "DIV2K", "train", "HR")),
        ("train", os.path.join(base_dir, "Flickr2K", "LR"), os.path.join(base_dir, "Flickr2K", "HR")),
        ("train", os.path.join(base_dir, "LSDIR", "train", "LR"), os.path.join(base_dir, "LSDIR", "train", "HR")),
        ("train", os.path.join(base_dir, "LSDIR", "val", "LR"), os.path.join(base_dir, "LSDIR", "val", "HR")),
        ("val", os.path.join(base_dir, "DIV2K", "val", "LR"), os.path.join(base_dir, "DIV2K", "val", "HR")),
    ]

    for split, lr_dir, hr_dir in mapping:
        if not os.path.isdir(lr_dir):
            print(f"[warn] LR dir missing: {lr_dir}")
            continue
        if not os.path.isdir(hr_dir):
            print(f"[warn] HR dir missing: {hr_dir}")
            continue
        pairs, missing = _collect_pairs(lr_dir, hr_dir)
        missing_total.extend(missing)
        if split == "train":
            train_pairs.extend(pairs)
        else:
            val_pairs.extend(pairs)

    test_dir = os.path.join(base_dir, "DIV2K", "test", "LR")
    if os.path.isdir(test_dir):
        test_items = _collect_lr_only(test_dir)
    else:
        print(f"[warn] Test LR dir missing: {test_dir}")

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train.json")
    val_path = os.path.join(out_dir, "val.json")
    test_path = os.path.join(out_dir, "test.json")

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_pairs, f, indent=2, sort_keys=False)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_pairs, f, indent=2, sort_keys=False)
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_items, f, indent=2, sort_keys=False)
    missing_path = ""
    if missing_total:
        missing_path = os.path.join(out_dir, "missing_hr.json")
        with open(missing_path, "w", encoding="utf-8") as f:
            json.dump(missing_total, f, indent=2, sort_keys=False)

    print(f"train: {len(train_pairs)}")
    print(f"val: {len(val_pairs)}")
    print(f"test: {len(test_items)}")
    print(f"missing_hr: {len(missing_total)}")
    print(f"wrote: {train_path}")
    print(f"wrote: {val_path}")
    print(f"wrote: {test_path}")
    if missing_path:
        print(f"wrote: {missing_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
