from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import json
import numpy as np
import tarfile
import torch
import zipfile
from torch.nn.functional import conv2d

from utils.utils_image_align import align_hr_to_res_crop_numpy

# ========== Global Constants ==========
SCALE = 4

# ========== Tool functions ==========
def extract_if_compressed(directory):
    """Check and extract compressed files"""
    extracted_files = []
    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} does not exist, skipping extraction", flush=True)
        return extracted_files
    
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.endswith('.zip'):
            print(f"Found ZIP file: {filename}", flush=True)
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                extract_path = os.path.join(directory, 'extracted')
                os.makedirs(extract_path, exist_ok=True)
                zip_ref.extractall(extract_path)
                extracted_files.extend([os.path.join(extract_path, f) for f in zip_ref.namelist()])
        elif filename.endswith('.tar.gz') or filename.endswith('.tgz'):
            print(f"Found TAR.GZ file: {filename}", flush=True)
            with tarfile.open(filepath, 'r:gz') as tar_ref:
                extract_path = os.path.join(directory, 'extracted')
                os.makedirs(extract_path, exist_ok=True)
                tar_ref.extractall(extract_path)
                extracted_files.extend([os.path.join(extract_path, f) for f in tar_ref.getnames()])
        elif filename.endswith('.tar'):
            print(f"Found TAR file: {filename}", flush=True)
            with tarfile.open(filepath, 'r:') as tar_ref:
                extract_path = os.path.join(directory, 'extracted')
                os.makedirs(extract_path, exist_ok=True)
                tar_ref.extractall(extract_path)
                extracted_files.extend([os.path.join(extract_path, f) for f in tar_ref.getnames()])
    return extracted_files

def find_image_files(directory):
    """Find image files"""
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    image_files = []
    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} does not exist, skipping image search", flush=True)
        return image_files
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    return sorted(image_files)

def read_image(image_path):
    """Read image using torchvision and return as [H, W, C] float32 array in [0, 1] range"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file does not exist: {image_path}")
    
    try:
        from torchvision.io import read_image as torch_read_image
        # Read image [C, H, W]
        img_tensor = torch_read_image(image_path)
        img_np = img_tensor.numpy().astype(np.float32) / 255.0
        # Convert [C, H, W] -> [H, W, C]
        img_np = img_np.transpose(1, 2, 0)
        return img_np
    except Exception as e:
        print(f"Failed to read image with torchvision: {e}", flush=True)
        raise

def to_y_channel(img):
    """Convert RGB/BGR image to Y channel based on provided logic"""
    # img is [H, W, C] in range [0, 1]
    if img.ndim == 3 and img.shape[2] >= 3:
        # Using RGB weights (swapped BGR [24.966, 128.553, 65.481] to RGB)
        y_img = np.dot(img[..., :3], [65.481, 128.553, 24.966]) + 16.0
        return y_img[..., None]
    return img * 255.

def resize_image(img, target_size):
    """Resize grayscale image"""
    # img: [H, W, 1] numpy array
    # target_size: (width, height)
    target_h, target_w = target_size[1], target_size[0]
    
    # Remove channel dimension for interpolation [H, W]
    img_2d = img[:, :, 0]
    
    # Convert to PyTorch tensor [1, 1, H, W]
    img_tensor = torch.from_numpy(img_2d).unsqueeze(0).unsqueeze(0).float()
    
    # Use bilinear interpolation
    resized_tensor = torch.nn.functional.interpolate(
        img_tensor, 
        size=(target_h, target_w), 
        mode='bilinear', 
        align_corners=False
    )
    
    # Convert back to numpy
    resized_img = resized_tensor.squeeze(0).permute(1, 2, 0).numpy()
    
    return resized_img

def matlab_style_gauss2D(shape=(11, 11), sigma=1.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian', shape, sigma)
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def calculate_ssim(img1, img2, data_range=255.0):
    """Calculate SSIM for single channel images using torch conv2d (valid padding)"""
    if img1.ndim == 3:
        img1 = img1[:, :, 0]
        img2 = img2[:, :, 0]
    
    img1_t = torch.from_numpy(img1).unsqueeze(0).unsqueeze(0).float()
    img2_t = torch.from_numpy(img2).unsqueeze(0).unsqueeze(0).float()
    
    window = matlab_style_gauss2D(shape=(11, 11), sigma=1.5)
    kernel = torch.from_numpy(window).float().unsqueeze(0).unsqueeze(0)
    
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    # Use padding=0 for 'valid' convolution
    mu1 = conv2d(img1_t, kernel, padding=0)
    mu2 = conv2d(img2_t, kernel, padding=0)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = conv2d(img1_t**2, kernel, padding=0) - mu1_sq
    sigma2_sq = conv2d(img2_t**2, kernel, padding=0) - mu2_sq
    sigma12 = conv2d(img1_t * img2_t, kernel, padding=0) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()

def calculate_psnr(img1, img2, data_range=255.0):
    """Calculate PSNR for single channel images"""
    # img1, img2: [H, W, 1]
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(data_range / np.sqrt(mse))

def calculate_metrics(ref_img, pred_img):
    """Calculate metrics for grayscale images"""
    # Ensure images are float32
    ref_img = ref_img.astype(np.float32)
    pred_img = pred_img.astype(np.float32)
    
    # Determine data range based on image values
    data_range = 255.0 if ref_img.max() > 1 else 1.0
    
    # Calculate SSIM and PSNR
    ssim_score = calculate_ssim(ref_img, pred_img, data_range=data_range)
    psnr_score = calculate_psnr(ref_img, pred_img, data_range=data_range)
    
    return ssim_score, psnr_score

def evaluate_restoration(
    items: List[Dict[str, Any]],
    scale: int = SCALE,
) -> Dict[str, Any]:
    """
    Evaluate restoration metrics (PSNR/SSIM) using the SAME logic as 03_evaluation_psnr.py,
    but with our JSON input format: [{'res':..., 'hr':...}, ...].
    - Only items that have a valid 'hr' path are evaluated.
    - Internally stages symlinks so ref/res filenames match and sorted() pairing stays correct.
    """
    # Stage to ref/res directories (symlink-only) to preserve original sorted pairing logic.
    stage_ctx = tempfile.TemporaryDirectory(prefix="sr_eval_restoration_")
    stage_dir = Path(stage_ctx.name)
    ref_dir = stage_dir / "ref"
    res_dir = stage_dir / "res"
    ref_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    name_to_pair: Dict[str, Dict[str, str]] = {}
    for it in items:
        res_path = str(it.get("res", "")).strip()
        hr_path = str(it.get("hr", "")).strip()
        if not res_path or not hr_path:
            continue
        if not os.path.exists(res_path) or not os.path.exists(hr_path):
            continue

        name = os.path.basename(res_path)
        if not name:
            continue
        if name in name_to_pair and name_to_pair[name].get("res") != res_path:
            raise ValueError(f"Duplicate res basename detected for restoration eval: {name}")
        name_to_pair[name] = {"res": res_path, "hr": hr_path}

        # Create symlinks with the SAME filename in both ref/res.
        ref_link = ref_dir / name
        res_link = res_dir / name
        if not ref_link.exists():
            ref_link.symlink_to(Path(hr_path).resolve())
        if not res_link.exists():
            res_link.symlink_to(Path(res_path).resolve())

    reference_dir = str(ref_dir)
    prediction_dir = str(res_dir)

    # Extract files (kept for logic parity; no-op for normal images)
    extract_if_compressed(reference_dir)
    extract_if_compressed(prediction_dir)

    # Find images (original logic)
    ref_images = find_image_files(reference_dir)
    pred_images = find_image_files(prediction_dir)

    ref_images = sorted(ref_images)
    pred_images = sorted(pred_images)

    all_ssim = []
    all_psnr = []
    per_res: Dict[str, Dict[str, float]] = {}
    num_pairs = min(len(ref_images), len(pred_images))

    if num_pairs == 0:
        stage_ctx.cleanup()
        return {
            "mean": {},
            "per_res": {},
            "note": "No valid (res,hr) pairs found for restoration metrics.",
        }

    for i in range(num_pairs):
        try:
            ref_img = read_image(ref_images[i])
            pred_img = read_image(pred_images[i])

            # Size matching:
            # Align HR(ref) to RES(pred) using the same semantics as diffusers resize_mode="crop".
            if ref_img.shape[:2] != pred_img.shape[:2]:
                ref_img = align_hr_to_res_crop_numpy(ref_img, pred_img)

            # Convert to Y channel
            ref_y = to_y_channel(ref_img)
            pred_y = to_y_channel(pred_img)

            # Cropping logic
            h, w = ref_y.shape[:2]
            h_new, w_new = h - h % scale, w - w % scale
            ref_y = ref_y[:h_new, :w_new, :]
            pred_y = pred_y[:h_new, :w_new, :]

            boundary = scale
            if h_new > 2 * boundary and w_new > 2 * boundary:
                ref_y = ref_y[boundary:-boundary, boundary:-boundary, :]
                pred_y = pred_y[boundary:-boundary, boundary:-boundary, :]

            ssim_score, psnr_score = calculate_metrics(ref_y, pred_y)
            all_ssim.append(ssim_score)
            all_psnr.append(psnr_score)

            name = os.path.basename(pred_images[i])
            pair = name_to_pair.get(name)
            if pair:
                per_res[pair["res"]] = {"ssim": float(ssim_score), "psnr": float(psnr_score)}

        except Exception:
            continue

    if len(all_ssim) == 0:
        stage_ctx.cleanup()
        return {
            "mean": {},
            "per_res": per_res,
            "note": "Failed to calculate restoration metrics for any image pairs.",
        }

    avg_ssim = float(np.mean(all_ssim))
    avg_psnr = float(np.mean(all_psnr))

    stage_ctx.cleanup()
    return {
        "mean": {"ssim": avg_ssim, "psnr": avg_psnr},
        "per_res": per_res,
        "note": (
            "Restoration metrics are computed on Y channel with crop_border=scale. "
            "If shape mismatches, hr is aligned to res using diffusers resize_mode='crop' center-crop semantics."
        ),
    }
