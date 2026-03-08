from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from local_iqa import create_q_metric
from utils.utils_image_align import align_hr_to_res_crop_pil, pil_to_tensor_ready_numpy


def load_image_pil(path: str) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def load_image_tensor(path: str, device: torch.device) -> torch.Tensor:
    img = load_image_pil(path)
    x = torch.from_numpy(np.array(img)).to(torch.float32) / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0).to(device)
    return x


def pil_image_to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(pil_to_tensor_ready_numpy(image)).to(torch.float32)
    x = x.permute(2, 0, 1).unsqueeze(0).to(device)
    return x


def safe_item(t: torch.Tensor) -> float:
    if isinstance(t, torch.Tensor):
        t = t.detach().float().cpu()
        if t.numel() == 1:
            return float(t.item())
        return float(t.mean().item())
    return float(t)


def build_metrics(device: torch.device) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    fr = {
        "lpips": create_q_metric("lpips", device).module,
        "dists": create_q_metric("dists", device).module,
    }
    nr = {
        "niqe": create_q_metric("niqe", device).module,
        "maniqa": create_q_metric("maniqa", device).module,
        "musiq": create_q_metric("musiq", device).module,
    }
    return fr, nr


def evaluate_perception_local(
    items: List[Dict[str, Any]],
    device: torch.device,
    fr_resize: str = "to_res_crop",
) -> Dict[str, Any]:
    fr_metrics, nr_metrics = build_metrics(device)

    per_image: List[Dict[str, Any]] = []
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}

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
                x_pil = load_image_pil(res_path)

                has_hr = bool(hr_path) and os.path.exists(hr_path) if hr_path else False

                for name, metric in nr_metrics.items():
                    out["scores"][name] = safe_item(metric(x))

                if has_hr:
                    y_pil = load_image_pil(hr_path)

                    if fr_resize in ("to_res_crop", "to_ref"):
                        y_pil = align_hr_to_res_crop_pil(y_pil, x_pil)
                        y = pil_image_to_tensor(y_pil, device)
                    elif fr_resize == "none":
                        y = pil_image_to_tensor(y_pil, device)
                        if x.shape[-2:] != y.shape[-2:]:
                            raise ValueError(
                                f"FR size mismatch: res={tuple(x.shape[-2:])}, hr={tuple(y.shape[-2:])}. "
                                f"Use --fr_resize to_res_crop to auto-align HR to RES."
                            )
                    else:
                        raise ValueError(f"Unsupported fr_resize mode: {fr_resize}")

                    for name, metric in fr_metrics.items():
                        out["scores"][name] = safe_item(metric(x, y))

                for key, value in out["scores"].items():
                    sums[key] = sums.get(key, 0.0) + float(value)
                    counts[key] = counts.get(key, 0) + 1

            except Exception as exc:  # noqa: BLE001
                out["error"] = str(exc)

            per_image.append(out)

    mean_scores = {key: (sums[key] / max(counts.get(key, 1), 1)) for key in sums.keys()}

    return {
        "num_items": len(items),
        "device": str(device),
        "mean": mean_scores,
        "per_image": per_image,
        "note": (
            "If hr exists, FR metrics (lpips,dists) are computed. "
            "NR metrics (niqe,maniqa,musiq) are always computed on res. "
            "CLIP-IQA is excluded because it is not implemented in local_iqa. "
            "For FR alignment, hr is transformed to res size with diffusers resize_mode='crop' center-crop semantics "
            "when fr_resize=to_res_crop (legacy alias: to_ref)."
        ),
    }


def read_json_input(json_path: Optional[str], json_str: Optional[str]) -> List[Dict[str, Any]]:
    if json_str:
        data = json.loads(json_str)
    elif json_path:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = json.loads(sys.stdin.read())

    if isinstance(data, dict):
        items = data.get("items")
        if not isinstance(items, list):
            raise ValueError(
                "Input JSON object must include a list field 'items'. "
                "Expected {'items':[{'res':..., 'hr':...}, ...], ...}"
            )
        data = items

    if not isinstance(data, list):
        raise ValueError(
            "Input JSON must be either a list of items "
            "or an object with an 'items' list."
        )
    return data
