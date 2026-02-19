import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

import pyiqa


def load_image_tensor(path: str, device: torch.device) -> torch.Tensor:
    """
    Load image as float tensor in [0,1], shape (1,3,H,W), RGB.
    """
    img = Image.open(path).convert("RGB")
    # PIL -> tensor (C,H,W) in [0m,1]
    x = torch.from_numpy(__import__("numpy").array(img)).to(torch.float32) / 255.0  # (H,W,3)
    x = x.permute(2, 0, 1).unsqueeze(0).to(device)  # (1,3,H,W)
    return x


def resize_to_match(x: torch.Tensor, ref: torch.Tensor, mode: str = "bicubic") -> torch.Tensor:
    """
    Resize x to match ref spatial size if needed.
    """
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    return F.interpolate(x, size=ref.shape[-2:], mode=mode, align_corners=False)


def safe_item(t: torch.Tensor) -> float:
    # pyiqa usually returns shape (N,) tensor
    if isinstance(t, torch.Tensor):
        t = t.detach().float().cpu()
        if t.numel() == 1:
            return float(t.item())
        return float(t.mean().item())
    return float(t)


def build_metrics(device: torch.device) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns (fr_metrics, nr_metrics)
    """
    # FR (need ref)
    fr = {
        "lpips": pyiqa.create_metric("lpips", device=device),
        "dists": pyiqa.create_metric("dists", device=device),
    }
    # NR (no ref)
    nr = {
        "niqe": pyiqa.create_metric("niqe", device=device),
        "maniqa": pyiqa.create_metric("maniqa", device=device),
        "musiq": pyiqa.create_metric("musiq", device=device),
        "clipiqa": pyiqa.create_metric("clipiqa", device=device),
    }
    return fr, nr


def evaluate_items(
    items: List[Dict[str, Any]],
    device: torch.device,
    fr_resize: str = "to_ref",  # or "none"
) -> Dict[str, Any]:
    fr_metrics, nr_metrics = build_metrics(device)

    per_image: List[Dict[str, Any]] = []
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    # Use inference mode
    with torch.inference_mode():
        for idx, it in enumerate(items):
            res_path = it.get("res")
            hr_path = it.get("hr", None)

            out: Dict[str, Any] = {
                "index": idx,
                "res": res_path,
                "hr": hr_path,
                "scores": {},
                "error": None,
            }

            try:
                if not res_path or not os.path.exists(res_path):
                    raise FileNotFoundError(f"res not found: {res_path}")

                x = load_image_tensor(res_path, device)

                has_hr = bool(hr_path) and os.path.exists(hr_path) if hr_path else False

                # NR metrics (always on x)
                for name, metric in nr_metrics.items():
                    val = metric(x)
                    out["scores"][name] = safe_item(val)

                # FR metrics if hr exists
                if has_hr:
                    y = load_image_tensor(hr_path, device)

                    x_fr = x
                    if fr_resize == "to_ref":
                        x_fr = resize_to_match(x_fr, y, mode="bicubic")
                    elif fr_resize == "none":
                        if x_fr.shape[-2:] != y.shape[-2:]:
                            raise ValueError(
                                f"FR size mismatch: res={tuple(x_fr.shape[-2:])}, hr={tuple(y.shape[-2:])}. "
                                f"Use --fr_resize to_ref to auto-resize."
                            )

                    for name, metric in fr_metrics.items():
                        val = metric(x_fr, y)
                        out["scores"][name] = safe_item(val)

                # accumulate
                for k, v in out["scores"].items():
                    sums[k] = sums.get(k, 0.0) + float(v)
                    counts[k] = counts.get(k, 0) + 1

            except Exception as e:
                out["error"] = str(e)

            per_image.append(out)

    mean_scores = {k: (sums[k] / max(counts.get(k, 1), 1)) for k in sums.keys()}

    return {
        "num_items": len(items),
        "device": str(device),
        "mean": mean_scores,
        "per_image": per_image,
        "note": (
            "If hr exists, FR metrics (lpips,dists) are computed. "
            "NR metrics (niqe,maniqa,musiq,clipiqa) are always computed on res. "
            "If sizes differ for FR, res is resized to hr when fr_resize=to_ref."
        ),
    }


def read_json_input(json_path: Optional[str], json_str: Optional[str]) -> List[Dict[str, Any]]:
    if json_str:
        data = json.loads(json_str)
    elif json_path:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        # stdin
        data = json.loads(__import__("sys").stdin.read())

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of objects like [{'res':..., 'hr':...}, ...]")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, default=None, help="Path to JSON file. If omitted, read stdin.")
    parser.add_argument("--json", type=str, default=None, help="JSON string input (alternative to --input).")
    parser.add_argument("--out", "-o", type=str, default=None, help="Output JSON path. If omitted, print to stdout.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--fr_resize", type=str, default="to_ref", choices=["to_ref", "none"],
                        help="How to handle FR size mismatch (lpips/dists).")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    items = read_json_input(args.input, args.json)
    results = evaluate_items(items, device=device, fr_resize=args.fr_resize)

    out_text = json.dumps(results, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out_text)
        print(f"Saved: {args.out}")
    else:
        print(out_text)


if __name__ == "__main__":
    main()
