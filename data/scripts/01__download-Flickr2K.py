#!/usr/bin/env python3
"""Download the Flickr2K dataset tarball with retry/resume.

Source: https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar
"""

import os
import sys
import time
import urllib.request
from urllib.error import HTTPError, URLError

URL = "https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar"
CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB
PROGRESS_STEP = 5  # percent
MAX_RETRIES = 5
BACKOFF_BASE = 2.0  # seconds
TIMEOUT = 30.0  # seconds
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Safari/537.36"


def _human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(n)
    for u in units:
        if size < 1024 or u == units[-1]:
            return f"{size:.1f} {u}"
        size /= 1024
    return f"{size:.1f} TB"


def _open_with_range(url: str, start: int) -> urllib.request.addinfourl:
    req = urllib.request.Request(url)
    req.add_header("User-Agent", USER_AGENT)
    req.add_header("Accept", "*/*")
    if start > 0:
        req.add_header("Range", f"bytes={start}-")
    return urllib.request.urlopen(req, timeout=TIMEOUT)


def _download(url: str, dest_path: str) -> None:
    filename = os.path.basename(dest_path)

    for attempt in range(1, MAX_RETRIES + 1):
        existing = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
        if attempt == 1 and existing == 0:
            print(f"[down] {filename}")
        elif existing > 0:
            print(f"[resume] {filename} from {_human_bytes(existing)} (attempt {attempt})")
        else:
            print(f"[retry] {filename} (attempt {attempt})")

        try:
            with _open_with_range(url, existing) as resp:
                total = resp.headers.get("Content-Length")
                total_size = int(total) if total and total.isdigit() else None

                # If server ignores Range, it will send full file with status 200.
                # In that case, start from scratch.
                if existing > 0 and getattr(resp, "status", 200) == 200:
                    existing = 0

                downloaded = existing
                last_percent = -PROGRESS_STEP
                last_print_time = 0.0

                mode = "ab" if existing > 0 else "wb"
                with open(dest_path, mode) as f:
                    while True:
                        chunk = resp.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total_size:
                            overall = total_size if existing == 0 else existing + total_size
                            percent = int(downloaded * 100 / overall)
                            now = time.time()
                            if percent >= last_percent + PROGRESS_STEP and now - last_print_time > 0.2:
                                last_percent = percent
                                last_print_time = now
                                print(
                                    f"  {percent:3d}% ({_human_bytes(downloaded)} / {_human_bytes(overall)})",
                                    end="\r",
                                    flush=True,
                                )

                print("", end="\r")
                print(f"  done ({_human_bytes(downloaded)})", flush=True)
                return

        except (HTTPError, URLError, ConnectionResetError) as e:
            if attempt == MAX_RETRIES:
                print(f"[error] {e}")
                raise
            wait = BACKOFF_BASE * (2 ** (attempt - 1))
            print(f"[warn] {e} -> retry in {wait:.1f}s")
            time.sleep(wait)


def main() -> int:
    dest_dir = os.path.abspath(os.path.dirname(__file__))
    dest_path = os.path.join(dest_dir, "Flickr2K.tar")

    print(f"Download directory: {dest_dir}")
    _download(URL, dest_path)
    print("Download complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
