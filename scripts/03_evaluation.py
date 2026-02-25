import argparse
import json
import sys
from pathlib import Path

SR_DIR = Path(__file__).resolve().parents[1]
if str(SR_DIR) not in sys.path:
    sys.path.insert(0, str(SR_DIR))

import torch

from utils.utils_perception import evaluate_perception, read_json_input
from utils.utils_restoration import evaluate_restoration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, default=None, help="Path to JSON file. If omitted, read stdin.")
    parser.add_argument("--json", type=str, default=None, help="JSON string input (alternative to --input).")
    parser.add_argument("--out", "-o", type=str, default=None, help="Output JSON path. If omitted, print to stdout.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--fr_resize",
        type=str,
        default="to_ref",
        choices=["to_ref", "none"],
        help="How to handle FR size mismatch (lpips/dists).",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    items = read_json_input(args.input, args.json)

    # 1) Perception metrics (existing 03_evaluation.py logic, moved to utils_perception.py)
    results = evaluate_perception(items, device=device, fr_resize=args.fr_resize)

    # 2) Restoration metrics (PSNR/SSIM) (03_evaluation_psnr.py logic, moved to utils_restoration.py)
    rest = evaluate_restoration(items)
    per_res = rest.get("per_res", {}) if isinstance(rest, dict) else {}
    mean = rest.get("mean", {}) if isinstance(rest, dict) else {}

    if isinstance(mean, dict):
        results["mean"].update(mean)

    for it in results.get("per_image", []):
        res_path = it.get("res")
        if res_path in per_res and isinstance(it.get("scores"), dict):
            it["scores"].update(per_res[res_path])

    if isinstance(rest, dict) and rest.get("note"):
        results["note"] = f'{results.get("note","")}\n{rest["note"]}'.strip()

    out_text = json.dumps(results, indent=2, ensure_ascii=False)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out_text, encoding="utf-8")
        print(f"Saved: {out_path}")
    else:
        print(out_text)


if __name__ == "__main__":
    main()
