#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class StagePaths:
    stage_dir: Path
    lr_dir: Path


def _safe_name(s: str) -> str:
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s[:80] if len(s) > 80 else s


def _symlink_test(stage_root: Path) -> None:
    """Fail fast if symlink is not permitted under stage_root."""
    stage_root.mkdir(parents=True, exist_ok=True)
    src = stage_root  # existing path
    dst = stage_root / f".symlink_test_{os.getpid()}"
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)
        dst.unlink()
    except Exception as ex:
        raise RuntimeError(
            f"Symlink test failed under stage_root={stage_root}\n"
            f"Reason: {ex}\n"
            f"Fix: choose a different --stage_root on a Linux filesystem where symlink is allowed (e.g. /tmp)."
        ) from ex


def _link_only(src: Path, dst: Path) -> None:
    """symlink-only. No copy fallback."""
    if not src.exists():
        raise FileNotFoundError(f"Source does not exist: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    dst.symlink_to(src)


def make_lr_stage_from_json(
    val_json: Path,
    stage_root: Path = Path("/tmp/sr_stage"),
    lr_key: str = "lr",
    name_prefix: str = "swinir",
    prefix_index: bool = True,
    clean_if_exists: bool = True,
) -> StagePaths:
    """
    Create a staging directory containing ONLY LR files referenced in val_json.
    - symlink-only (no copy)
    - ignores HR completely even if present in JSON
    """
    val_json = Path(val_json).resolve()
    stage_root = Path(stage_root).resolve()

    _symlink_test(stage_root)

    data = json.loads(val_json.read_text())
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"Invalid JSON format: expected a non-empty list in {val_json}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stage_dir = stage_root / f"{_safe_name(name_prefix)}_{ts}"
    if stage_dir.exists() and clean_if_exists:
        shutil.rmtree(stage_dir)

    lr_dir = stage_dir / "LR"
    lr_dir.mkdir(parents=True, exist_ok=True)

    for i, e in enumerate(data):
        if not isinstance(e, dict) or lr_key not in e:
            raise KeyError(f"Missing key '{lr_key}' at index {i} in {val_json}")
        lr_src = Path(e[lr_key]).resolve()
        lr_name = f"{i:06d}__{lr_src.name}" if prefix_index else lr_src.name
        _link_only(lr_src, lr_dir / lr_name)

    return StagePaths(stage_dir=stage_dir, lr_dir=lr_dir)


def main():
    ap = argparse.ArgumentParser(description="Create LR-only symlink-only staging folder from VAL_JSON.")
    ap.add_argument("--val_json", required=True, help="Path to val_fixed.json")
    ap.add_argument("--stage_root", default="/tmp/sr_stage", help="Root dir for staging (must allow symlink)")
    ap.add_argument("--name_prefix", default="swinir", help="Prefix for stage folder name")
    args = ap.parse_args()

    sp = make_lr_stage_from_json(
        val_json=Path(args.val_json),
        stage_root=Path(args.stage_root),
        name_prefix=args.name_prefix,
    )

    # printed values can be copied/pasted if needed
    print(f"STAGE_DIR={sp.stage_dir}")
    print(f"STAGE_LR={sp.lr_dir}")


if __name__ == "__main__":
    main()
