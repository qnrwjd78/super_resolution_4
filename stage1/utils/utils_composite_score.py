from __future__ import annotations

import math
from typing import Any, Mapping, MutableMapping, Optional


SCORE_KEY = "score"
REQUIRED_SCORE_KEYS = ("lpips", "dists", "niqe", "maniqa", "musiq", "clipiqa")


def _as_finite_float(value: Any) -> Optional[float]:
    try:
        x = float(value)
    except Exception:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def composite_score_from_metrics(metrics: Mapping[str, Any]) -> Optional[float]:
    parsed: dict[str, float] = {}
    for key in REQUIRED_SCORE_KEYS:
        value = _as_finite_float(metrics.get(key))
        if value is None:
            return None
        parsed[key] = value

    return (
        (1.0 - parsed["lpips"])
        + (1.0 - parsed["dists"])
        + ((10.0 - parsed["niqe"]) / 10.0)
        + parsed["maniqa"]
        + (parsed["musiq"] / 100.0)
        + parsed["clipiqa"]
    )


def attach_composite_score(metrics: MutableMapping[str, Any], key: str = SCORE_KEY) -> Optional[float]:
    existing = _as_finite_float(metrics.get(key))
    if existing is not None:
        return existing

    score = composite_score_from_metrics(metrics)
    if score is not None:
        metrics[key] = float(score)
    return score
