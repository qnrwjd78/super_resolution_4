from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image


def _to_rgb_pil_from_numpy(image: np.ndarray) -> Image.Image:
    """Convert float/uint numpy image [H,W,C] into RGB PIL image."""
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.ndim == 3 and image.shape[2] > 3:
        image = image[:, :, :3]

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected image shape [H,W,C] with C in [1,3+], got {image.shape}")

    if image.dtype != np.uint8:
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).round().astype(np.uint8)

    return Image.fromarray(image, mode="RGB")


def _to_numpy_float32_rgb(image: Image.Image) -> np.ndarray:
    arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    return arr


def resize_mode_crop_center_pil(image: Image.Image, width: int, height: int) -> Image.Image:
    """
    Diffusers `VaeImageProcessor._resize_and_crop` equivalent.
    Keeps aspect ratio, resizes, then pastes centered on (width,height) canvas.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid target size: width={width}, height={height}")

    image = image.convert("RGB")
    ratio = width / height
    src_ratio = image.width / image.height

    src_w = width if ratio > src_ratio else image.width * height // image.height
    src_h = height if ratio <= src_ratio else image.height * width // image.width

    resized = image.resize((src_w, src_h), resample=Image.Resampling.LANCZOS)
    out = Image.new("RGB", (width, height))
    out.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
    return out


def align_hr_to_res_crop_pil(hr_image: Image.Image, res_image: Image.Image) -> Image.Image:
    """Align HR to RES spatial size with diffusers resize_mode='crop' center-crop semantics."""
    return resize_mode_crop_center_pil(hr_image, width=res_image.width, height=res_image.height)


def align_hr_to_res_crop_numpy(hr_image: np.ndarray, res_image: np.ndarray) -> np.ndarray:
    """
    Align HR numpy image to RES size using diffusers resize_mode='crop' center-crop semantics.
    Returns float32 RGB image in [0,1] with shape [H,W,3].
    """
    if hr_image.ndim < 2 or res_image.ndim < 2:
        raise ValueError("hr_image and res_image must have at least 2 dimensions")

    target_h, target_w = int(res_image.shape[0]), int(res_image.shape[1])
    hr_pil = _to_rgb_pil_from_numpy(hr_image)
    aligned = resize_mode_crop_center_pil(hr_pil, width=target_w, height=target_h)
    return _to_numpy_float32_rgb(aligned)


def pil_to_tensor_ready_numpy(image: Image.Image) -> np.ndarray:
    """Helper for callers that need float32 RGB numpy [0,1] from PIL."""
    return _to_numpy_float32_rgb(image)


def image_hw(image: np.ndarray) -> Tuple[int, int]:
    return int(image.shape[0]), int(image.shape[1])
