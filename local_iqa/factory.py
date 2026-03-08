from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass
class QualityMetricSpec:
    name: str
    module: nn.Module
    lower_better: bool
    requires_reference: bool


class L2Metric(nn.Module):
    def forward(self, pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return ((pred - ref) ** 2).reshape(pred.shape[0], -1).mean(dim=1)


def _freeze_metric(metric: nn.Module, device: torch.device) -> nn.Module:
    metric = metric.to(device)
    metric.eval()
    for param in metric.parameters():
        param.requires_grad_(False)
    return metric


def _weights_root() -> Path | None:
    value = os.environ.get("LOCAL_IQA_WEIGHTS_DIR")
    if not value:
        return None
    root = Path(value).expanduser()
    return root if root.exists() else None


def _optional_weight_path(filename: str) -> str | None:
    root = _weights_root()
    if root is None:
        return None
    candidate = root / filename
    if candidate.exists():
        return str(candidate)
    return None


def _canonical_metric_key(metric_name: str) -> str:
    key = metric_name.strip().lower()
    aliases = {
        "mse": "l2",
        "l_2": "l2",
        "lpips-alex": "lpips",
        "lpips_alex": "lpips",
        "lpips_vgg": "lpips-vgg",
        "niqe-matlab": "niqe_matlab",
        "clip-iqa": "clipiqa",
        "clip_iqa": "clipiqa",
    }
    return aliases.get(key, key)


def create_q_metric(metric_name: str, device: torch.device) -> QualityMetricSpec:
    # Keep behavior aligned with pyiqa 0.1.14.1 metric keys/defaults.
    key = _canonical_metric_key(metric_name)

    if key == "l2":
        module = _freeze_metric(L2Metric(), device)
        return QualityMetricSpec("l2", module, lower_better=True, requires_reference=True)

    if key == "lpips":
        from .lpips_arch import LPIPS

        module = LPIPS(
            net="alex",
            pretrained_model_path=_optional_weight_path("LPIPS_v0.1_alex-df73285e.pth"),
        )
        module = _freeze_metric(module, device)
        return QualityMetricSpec("lpips", module, lower_better=True, requires_reference=True)

    if key in {"lpips-vgg", "lpips_vgg"}:
        from .lpips_arch import LPIPS

        module = LPIPS(
            net="vgg",
            pretrained_model_path=_optional_weight_path("LPIPS_v0.1_vgg-a78928a0.pth"),
        )
        module = _freeze_metric(module, device)
        return QualityMetricSpec("lpips-vgg", module, lower_better=True, requires_reference=True)

    if key == "dists":
        from .dists_arch import DISTS

        module = DISTS(
            pretrained_model_path=_optional_weight_path("DISTS_weights-f5e65c96.pth"),
        )
        module = _freeze_metric(module, device)
        return QualityMetricSpec("dists", module, lower_better=True, requires_reference=True)

    if key in {"niqe", "niqe_matlab"}:
        from .niqe_arch import NIQE

        version = "matlab" if key == "niqe_matlab" else "original"
        weight_name = "niqe_matlab_params.mat" if version == "matlab" else "niqe_modelparameters.mat"
        module = NIQE(
            version=version,
            pretrained_model_path=_optional_weight_path(weight_name),
        )
        module = _freeze_metric(module, device)
        return QualityMetricSpec(key, module, lower_better=True, requires_reference=False)

    if key in {"musiq", "musiq-ava", "musiq-paq2piq", "musiq-spaq"}:
        from .musiq_arch import MUSIQ

        musiq_pretrained = {
            "musiq": "koniq10k",
            "musiq-ava": "ava",
            "musiq-paq2piq": "paq2piq",
            "musiq-spaq": "spaq",
        }
        musiq_weight_name = {
            "musiq": "musiq_koniq_ckpt-e95806b9.pth",
            "musiq-ava": "musiq_ava_ckpt-e8d3f067.pth",
            "musiq-paq2piq": "musiq_paq2piq_ckpt-364c0c84.pth",
            "musiq-spaq": "musiq_spaq_ckpt-358bb6af.pth",
        }
        module = MUSIQ(
            pretrained=musiq_pretrained[key],
            pretrained_model_path=_optional_weight_path(musiq_weight_name[key]),
        )
        module = _freeze_metric(module, device)
        return QualityMetricSpec(key, module, lower_better=False, requires_reference=False)

    if key in {"maniqa", "maniqa-pipal", "maniqa-kadid"}:
        from .maniqa_arch import MANIQA

        maniqa_dataset = {
            "maniqa": "koniq",
            "maniqa-pipal": "pipal",
            "maniqa-kadid": "kadid",
        }
        maniqa_weight_name = {
            "maniqa": "ckpt_koniq10k.pt",
            "maniqa-pipal": "MANIQA_PIPAL-ae6d356b.pth",
            "maniqa-kadid": "ckpt_kadid10k.pt",
        }
        maniqa_kwargs = {
            "train_dataset": maniqa_dataset[key],
            "pretrained_model_path": _optional_weight_path(maniqa_weight_name[key]),
        }
        # pyiqa 0.1.14.1 default_model_configs.py:
        # - maniqa: scale=0.8
        # - maniqa-kadid: scale=0.8
        # - maniqa-pipal: no explicit scale override
        if key in {"maniqa", "maniqa-kadid"}:
            maniqa_kwargs["scale"] = 0.8

        module = MANIQA(**maniqa_kwargs)
        module = _freeze_metric(module, device)
        return QualityMetricSpec(key, module, lower_better=False, requires_reference=False)

    if key == "clipiqa":
        raise ValueError("CLIP-IQA is intentionally excluded from local_iqa.")

    supported = (
        "L2, LPIPS, LPIPS-VGG, DISTS, NIQE, NIQE_MATLAB, "
        "MUSIQ, MUSIQ-AVA, MUSIQ-PAQ2PIQ, MUSIQ-SPAQ, "
        "ManIQA, ManIQA-PIPAL, ManIQA-KADID"
    )
    raise ValueError(f"Unsupported q_metric='{metric_name}'. Supported labels: {supported}")
