from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import pyiqa
from utils.utils_image_align import align_hr_to_res_crop_pil, pil_to_tensor_ready_numpy


def load_image_pil(path: str) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def load_image_tensor(path: str, device: torch.device) -> torch.Tensor:
    """
    Load image as float tensor in [0,1], shape (1,3,H,W), RGB.
    """
    img = load_image_pil(path)
    # PIL -> tensor (C,H,W) in [0,1]
    x = torch.from_numpy(np.array(img)).to(torch.float32) / 255.0  # (H,W,3)
    x = x.permute(2, 0, 1).unsqueeze(0).to(device)  # (1,3,H,W)
    return x


def pil_image_to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(pil_to_tensor_ready_numpy(image)).to(torch.float32)
    x = x.permute(2, 0, 1).unsqueeze(0).to(device)
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


def evaluate_perception(
    items: List[Dict[str, Any]],
    device: torch.device,
    fr_resize: str = "to_res_crop",  # "to_res_crop"(or legacy "to_ref"), "none"
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
                x_pil = load_image_pil(res_path)

                has_hr = bool(hr_path) and os.path.exists(hr_path) if hr_path else False

                # NR metrics (always on x)
                for name, metric in nr_metrics.items():
                    val = metric(x)
                    out["scores"][name] = safe_item(val)

                # FR metrics if hr exists
                if has_hr:
                    y_pil = load_image_pil(hr_path)

                    x_fr = x
                    if fr_resize in ("to_res_crop", "to_ref"):
                        # Keep backward compatibility for existing `to_ref` configs.
                        # We align HR to RES using diffusers resize_mode="crop" semantics.
                        y_pil = align_hr_to_res_crop_pil(y_pil, x_pil)
                        y = pil_image_to_tensor(y_pil, device)
                    elif fr_resize == "none":
                        y = pil_image_to_tensor(y_pil, device)
                        if x_fr.shape[-2:] != y.shape[-2:]:
                            raise ValueError(
                                f"FR size mismatch: res={tuple(x_fr.shape[-2:])}, hr={tuple(y.shape[-2:])}. "
                                f"Use --fr_resize to_res_crop to auto-align HR to RES."
                            )
                    else:
                        raise ValueError(f"Unsupported fr_resize mode: {fr_resize}")

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
        # stdin
        data = json.loads(sys.stdin.read())

    # Backward compatible:
    # - legacy: [ {"res": "...", "hr": "..."}, ... ]
    # - extended: { "items": [ ... ], "timing": {...} }
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
