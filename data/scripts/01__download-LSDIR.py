#!/usr/bin/env python3
"""Download the LSDIR dataset from Hugging Face.

This repo is gated; you must accept the access conditions on Hugging Face
and provide a token. Supply it via:
- --token argument (string)
- environment variables (HF_TOKEN or HUGGINGFACE_TOKEN)
"""

import argparse
import os
import sys
from typing import Iterable, List

REPO_ID = "ofsoundof/LSDIR"
REPO_TYPE = "model"

# Files listed in the repository.
ALL_FILES = [
    "shard-00.tar.gz",
    "shard-01.tar.gz",
    "shard-02.tar.gz",
    "shard-03.tar.gz",
    "shard-04.tar.gz",
    "shard-05.tar.gz",
    "shard-06.tar.gz",
    "shard-07.tar.gz",
    "shard-08.tar.gz",
    "shard-09.tar.gz",
    "shard-10.tar.gz",
    "shard-11.tar.gz",
    "shard-12.tar.gz",
    "shard-13.tar.gz",
    "shard-14.tar.gz",
    "shard-15.tar.gz",
    "shard-16.tar.gz",
    "train_x4.tar.gz",
    "val1.tar.gz",
]

ㅏ
def _get_token(cli_token: str) -> str:
    if cli_token:
        return cli_token
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or ""


def _resolve_files(args: argparse.Namespace) -> List[str]:
    if args.files:
        missing = [f for f in args.files if f not in ALL_FILES]
        if missing:
            raise SystemExit(f"Unknown file(s): {', '.join(missing)}")
        return args.files
    return ALL_FILES


def _download_files(files: Iterable[str], dest_dir: str, token: str) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        )

    os.makedirs(dest_dir, exist_ok=True)

    for filename in files:
        print(f"[down] {filename}")
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=filename,
            token=token or None,
            local_dir=dest_dir,
            local_dir_use_symlinks=False,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download LSDIR files from Hugging Face (gated repo)."
    )
    parser.add_argument(
        "--dest",
        default=os.path.join(os.path.dirname(__file__), "LSDIR-raw"),
        help="Output directory (default: ./LSDIR-raw)",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="Subset of files to download (default: all)",
    )
    parser.add_argument(
        "--token",
        default="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",
        help="HF access token string (overrides env vars)",
    )
    args = parser.parse_args()

    token = _get_token(args.token)
    if not token:
        print(
            "HF token not found. Provide --token, or set HF_TOKEN (or HUGGINGFACE_TOKEN)."
        )
        return 1

    files = _resolve_files(args)
    print(f"Repo: {REPO_ID}")
    print(f"Dest: {os.path.abspath(args.dest)}")
    print(f"Files: {len(files)}")

    _download_files(files, args.dest, token)
    print("All downloads complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
