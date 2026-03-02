#!/usr/bin/env python3
import os
import sys
import time
import urllib.request
from urllib.error import URLError, HTTPError

URLS = [
    "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
    "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip",
    "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip",
    "http://data.vision.ee.ethz.ch/cvl/DIV2K/validation_release/DIV2K_test_LR_bicubic_X4.zip",
]

CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB
PROGRESS_STEP = 5  # percent


def _human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(n)
    for u in units:
        if size < 1024 or u == units[-1]:
            return f"{size:.1f} {u}"
        size /= 1024
    return f"{size:.1f} TB"


def _download(url: str, dest_dir: str) -> None:
    filename = os.path.basename(url)
    dest_path = os.path.join(dest_dir, filename)

    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"[skip] {filename} already exists")
        return

    print(f"[down] {filename}")
    try:
        with urllib.request.urlopen(url) as resp:
            total = resp.headers.get("Content-Length")
            total_size = int(total) if total and total.isdigit() else None

            downloaded = 0
            last_percent = -PROGRESS_STEP
            last_print_time = 0.0

            with open(dest_path, "wb") as f:
                while True:
                    chunk = resp.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size:
                        percent = int(downloaded * 100 / total_size)
                        now = time.time()
                        if percent >= last_percent + PROGRESS_STEP and now - last_print_time > 0.2:
                            last_percent = percent
                            last_print_time = now
                            print(
                                f"  {percent:3d}% ({_human_bytes(downloaded)} / {_human_bytes(total_size)})",
                                end="\r",
                                flush=True,
                            )
            if total_size:
                print(
                    f"  100% ({_human_bytes(downloaded)} / {_human_bytes(total_size)})",
                    flush=True,
                )
            else:
                print(f"  done ({_human_bytes(downloaded)})")

    except (HTTPError, URLError) as e:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        print(f"[error] {filename}: {e}")
        raise


def main() -> int:
    dest_dir = os.path.abspath(os.path.dirname(__file__))
    dest_dir = os.path.join(dest_dir, "DIV2K-raw")
    os.makedirs(dest_dir, exist_ok=True)
    
    print(f"Download directory: {dest_dir}")
    for url in URLS:
        _download(url, dest_dir)
    print("All downloads complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
