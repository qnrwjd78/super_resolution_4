import argparse
import contextlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import Flux2KleinPipeline
from PIL import Image
from PIL.ImageOps import exif_transpose
from tqdm.auto import tqdm

from lora_data import load_manifest_entries


STAGE2_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS_PATH = STAGE2_DIR / "prompts.json"


# -----------------------------
# Diffusers Flux2KleinPipeline source-compatible helpers
# -----------------------------
def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b
    return float(mu)


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[torch.device] = None,
    timesteps: Optional[list[int]] = None,
    sigmas: Optional[list[float]] = None,
    **kwargs,
):
    import inspect

    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")

    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(f"{scheduler.__class__} does not support custom `timesteps`.")
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accepts_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_sigmas:
            raise ValueError(f"{scheduler.__class__} does not support custom `sigmas`.")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps

    return timesteps, num_inference_steps


# -----------------------------
# CLI / basic utils
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run FLUX.2 image-conditioned inference from a local JSON manifest."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to a local FLUX.2 checkpoint or a model id.",
    )
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to a JSON array of samples. Each sample must contain `hr` and either `res` or `lr`.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where generated images and the output JSON manifest will be written.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional explicit path for the output manifest. Defaults to <output_dir>/results.json.",
    )
    parser.add_argument(
        "--lora_weights_path",
        type=str,
        default=None,
        help="Optional directory containing saved LoRA weights to load before inference.",
    )
    parser.add_argument(
        "--prompts_json",
        type=str,
        default=str(DEFAULT_PROMPTS_PATH),
        help="Path to the prompt definition JSON file.",
    )
    parser.add_argument(
        "--prompt_name",
        type=str,
        default=None,
        help="Prompt entry name to select from prompts.json. If set, the selected prompt is used for all samples.",
    )
    parser.add_argument(
        "--default_prompt",
        type=str,
        default=None,
        help="Fallback prompt used when a sample in `--input_json` does not define `prompt` (ignored when `--prompt_name` is set).",
    )
    parser.add_argument("--revision", type=str, default=None, help="Optional model revision.")
    parser.add_argument("--variant", type=str, default=None, help="Optional model variant.")

    # plain / canvas_tile
    parser.add_argument(
        "--mode",
        type=str,
        default="plain",
        choices=["plain", "canvas_tile"],
        help="Inference mode.",
    )

    # 기존 crop 기능 유지
    parser.add_argument(
        "--crop_mode",
        type=str,
        default="full",
        choices=["full", "center_crop", "random_crop"],
        help="Input preprocessing mode.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Patch size used when `--crop_mode` is `center_crop` or `random_crop`.",
    )

    # canvas tile args
    parser.add_argument(
        "--canvas_height",
        type=int,
        default=None,
        help="Optional explicit output height in pixels for canvas_tile mode. Defaults to condition image height.",
    )
    parser.add_argument(
        "--canvas_width",
        type=int,
        default=None,
        help="Optional explicit output width in pixels for canvas_tile mode. Defaults to condition image width.",
    )
    parser.add_argument(
        "--tile_size_px",
        type=int,
        default=1024,
        help="Tile size in pixel space for canvas_tile mode.",
    )
    parser.add_argument(
        "--tile_overlap_px",
        type=int,
        default=256,
        help="Tile overlap in pixel space for canvas_tile mode.",
    )
    parser.add_argument(
        "--tile_batch_size",
        type=int,
        default=4,
        help="How many tiles to evaluate together inside one transformer forward.",
    )
    parser.add_argument(
        "--tile_sigma_ratio",
        type=float,
        default=0.15,
        help="Gaussian blending width ratio for overlapping tiles.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="Torch dtype used when loading the pipeline.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Execution device.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=4.0,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed. Each sample uses `seed + index`. Set to a negative value to disable fixed seeding.",
    )
    parser.add_argument(
        "--cpu_offload",
        action="store_true",
        help="Enable `enable_model_cpu_offload()` instead of keeping the whole pipeline on the target device.",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name: str):
    if dtype_name == "fp32":
        return torch.float32
    if dtype_name == "fp16":
        return torch.float16
    return torch.bfloat16


def build_generator(device: str, seed: Optional[int]):
    if seed is None:
        return None
    generator_device = "cuda" if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    return torch.Generator(device=generator_device).manual_seed(seed)


def load_rgb_image(path: Path):
    with Image.open(path) as image:
        image = exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image.copy()


def load_prompt_by_name(prompts_json_path: str, prompt_name: str):
    prompts_path = Path(prompts_json_path).expanduser().resolve()
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts JSON does not exist: {prompts_path}")

    with prompts_path.open("r", encoding="utf-8") as handle:
        entries = json.load(handle)

    if not isinstance(entries, list):
        raise ValueError("Prompts JSON must be a JSON array.")

    available_names = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip()
        prompt = str(entry.get("prompt", "")).strip()
        if name:
            available_names.append(name)
        if name == prompt_name:
            if not prompt:
                raise ValueError(f"Prompt entry `{prompt_name}` is missing `prompt` text.")
            return prompt

    choices = ", ".join(sorted(available_names)) if available_names else "<none>"
    raise ValueError(f"Prompt name `{prompt_name}` was not found. Available names: {choices}")


def ensure_min_crop_size(cond_image: Image.Image, hr_image: Image.Image, patch_size: int):
    cond_width, cond_height = cond_image.size
    min_side = min(cond_width, cond_height)
    if min_side >= patch_size:
        return cond_image, hr_image

    scale = patch_size / float(min_side)
    new_size = (
        max(patch_size, round(cond_width * scale)),
        max(patch_size, round(cond_height * scale)),
    )
    cond_image = cond_image.resize(new_size, Image.Resampling.BICUBIC)
    hr_image = hr_image.resize(new_size, Image.Resampling.BICUBIC)
    return cond_image, hr_image


def crop_pair(cond_image: Image.Image, hr_image: Image.Image, mode: str, patch_size: int, seed: Optional[int]):
    if hr_image.size != cond_image.size:
        hr_image = hr_image.resize(cond_image.size, Image.Resampling.BICUBIC)

    cond_image, hr_image = ensure_min_crop_size(cond_image, hr_image, patch_size)
    width, height = cond_image.size

    if mode == "center_crop":
        left = max((width - patch_size) // 2, 0)
        top = max((height - patch_size) // 2, 0)
    else:
        rng = random.Random(seed) if seed is not None else random
        left = rng.randint(0, width - patch_size)
        top = rng.randint(0, height - patch_size)

    crop_box = (left, top, left + patch_size, top + patch_size)
    return cond_image.crop(crop_box), hr_image.crop(crop_box)


def resolve_output_image_path(output_dir: Path, cond_path: Path, used_names: set[str]) -> Path:
    candidate_name = cond_path.name
    candidate_path = output_dir / candidate_name
    if candidate_name not in used_names and not candidate_path.exists():
        used_names.add(candidate_name)
        return candidate_path

    stem = cond_path.stem
    suffix = cond_path.suffix or ".png"
    idx = 1
    while True:
        candidate_name = f"{stem}_{idx}{suffix}"
        candidate_path = output_dir / candidate_name
        if candidate_name not in used_names and not candidate_path.exists():
            used_names.add(candidate_name)
            return candidate_path
        idx += 1


# -----------------------------
# Canvas tiling helpers
# -----------------------------
@dataclass
class TileCoord:
    y0: int
    y1: int
    x0: int
    x1: int


def make_starts(total: int, tile: int, overlap: int) -> List[int]:
    assert tile > overlap, "tile must be larger than overlap"
    if total <= tile:
        return [0]

    stride = tile - overlap
    starts = []
    cur = 0
    while cur + tile < total:
        starts.append(cur)
        cur += stride

    starts.append(total - tile)
    return sorted(set(starts))


def make_tile_coords(h: int, w: int, tile_h: int, tile_w: int, overlap_h: int, overlap_w: int) -> List[TileCoord]:
    ys = make_starts(h, tile_h, overlap_h)
    xs = make_starts(w, tile_w, overlap_w)
    coords = []
    for y0 in ys:
        for x0 in xs:
            coords.append(TileCoord(y0=y0, y1=min(y0 + tile_h, h), x0=x0, x1=min(x0 + tile_w, w)))
    return coords


def make_gaussian_weight(tile_h: int, tile_w: int, channels: int, device, sigma_ratio: float = 0.15):
    sigma_y = max(tile_h * sigma_ratio, 1.0)
    sigma_x = max(tile_w * sigma_ratio, 1.0)

    ys = torch.arange(tile_h, device=device, dtype=torch.float32)
    xs = torch.arange(tile_w, device=device, dtype=torch.float32)

    cy = (tile_h - 1) / 2.0
    cx = (tile_w - 1) / 2.0

    gy = torch.exp(-((ys - cy) ** 2) / (2.0 * sigma_y * sigma_y))
    gx = torch.exp(-((xs - cx) ** 2) / (2.0 * sigma_x * sigma_x))

    w2d = gy[:, None] * gx[None, :]
    w2d = w2d / w2d.max()
    return w2d[None, None].repeat(1, channels, 1, 1)  # [1, C, H, W]


def offset_ids_hw(ids: torch.Tensor, coord: TileCoord) -> torch.Tensor:
    out = ids.clone()
    out[..., 1] += coord.y0
    out[..., 2] += coord.x0
    return out


def build_absolute_image_ids_for_tile(pipe: Flux2KleinPipeline, cond_tile_map: torch.Tensor, coord: TileCoord) -> torch.Tensor:
    cond_tile_list = [cond_tile_map[i].unsqueeze(0) for i in range(cond_tile_map.shape[0])]
    cond_ids = pipe._prepare_image_ids(cond_tile_list).to(device=cond_tile_map.device)
    cond_ids = cond_ids.view(cond_tile_map.shape[0], -1, 4)
    return offset_ids_hw(cond_ids, coord)


def maybe_cache_context(transformer, name: str):
    if hasattr(transformer, "cache_context"):
        return transformer.cache_context(name)
    return contextlib.nullcontext()


# -----------------------------
# Canvas runner
# -----------------------------
class Flux2CanvasRunner:
    def __init__(
        self,
        pipe: Flux2KleinPipeline,
        tile_size_px: int = 1024,
        tile_overlap_px: int = 256,
        tile_batch_size: int = 4,
        sigma_ratio: float = 0.15,
    ):
        self.pipe = pipe
        self.tile_size_px = tile_size_px
        self.tile_overlap_px = tile_overlap_px
        self.tile_batch_size = tile_batch_size
        self.sigma_ratio = sigma_ratio

    def _round_down_multiple(self, x: int, multiple: int) -> int:
        return max(multiple, (int(x) // multiple) * multiple)

    def _prepare_condition_images(
        self,
        image: Image.Image | List[Image.Image] | None,
        height: int,
        width: int,
    ):
        pipe = self.pipe
        if image is None:
            return None

        if not isinstance(image, list):
            image = [image]

        condition_images = []
        for img in image:
            pipe.image_processor.check_image_input(img)
            img_tensor = pipe.image_processor.preprocess(
                img,
                height=height,
                width=width,
                resize_mode="crop",
            )
            condition_images.append(img_tensor)

        return condition_images

    def _resolve_canvas_size(
        self,
        input_image: Image.Image | List[Image.Image] | None,
        canvas_height: Optional[int],
        canvas_width: Optional[int],
    ):
        pipe = self.pipe
        multiple_of = pipe.vae_scale_factor * 2

        if input_image is not None:
            base_image = input_image[0] if isinstance(input_image, list) else input_image
            default_h = base_image.height
            default_w = base_image.width
        else:
            default_h = pipe.default_sample_size * pipe.vae_scale_factor
            default_w = pipe.default_sample_size * pipe.vae_scale_factor

        height = canvas_height if canvas_height is not None else default_h
        width = canvas_width if canvas_width is not None else default_w

        height = self._round_down_multiple(height, multiple_of)
        width = self._round_down_multiple(width, multiple_of)
        return height, width

    @torch.no_grad()
    def _predict_noise_tiled(
        self,
        latents_map: torch.Tensor,           # [B, C, H_lat, W_lat]
        t_scalar: torch.Tensor,              # scalar tensor
        prompt_embeds: torch.Tensor,         # [B, L, D]
        text_ids: torch.Tensor,              # [B, L, 4]
        image_latents_map: Optional[torch.Tensor],  # [B, C, H_lat, W_lat]
        guidance_scale: float,
        negative_prompt_embeds: Optional[torch.Tensor],
        negative_text_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        pipe = self.pipe
        _, C, H, W = latents_map.shape
        device = latents_map.device

        latent_multiple = pipe.vae_scale_factor * 2
        tile_h = max(1, self.tile_size_px // latent_multiple)
        tile_w = max(1, self.tile_size_px // latent_multiple)
        overlap_h = max(0, self.tile_overlap_px // latent_multiple)
        overlap_w = max(0, self.tile_overlap_px // latent_multiple)

        if overlap_h >= tile_h or overlap_w >= tile_w:
            raise ValueError(
                f"tile overlap is too large after latent conversion: "
                f"tile=({tile_h},{tile_w}), overlap=({overlap_h},{overlap_w})"
            )

        coords = make_tile_coords(H, W, tile_h, tile_w, overlap_h, overlap_w)
        weight_cache: dict[tuple[int, int], torch.Tensor] = {}

        noise_accum = torch.zeros_like(latents_map)
        weight_accum = torch.zeros_like(latents_map)

        do_cfg = (guidance_scale > 1.0) and (not pipe.config.is_distilled)

        for start in range(0, len(coords), self.tile_batch_size):
            batch_coords = coords[start : start + self.tile_batch_size]
            nt = len(batch_coords)

            packed_tiles = []
            tile_local_ids = []
            tile_abs_ids = []
            cond_packed_tiles = []
            cond_abs_ids = []

            for coord in batch_coords:
                tile_map = latents_map[:, :, coord.y0:coord.y1, coord.x0:coord.x1]
                packed_tiles.append(pipe._pack_latents(tile_map))
                local_ids = pipe._prepare_latent_ids(tile_map).to(device=tile_map.device)
                tile_local_ids.append(local_ids)
                tile_abs_ids.append(offset_ids_hw(local_ids, coord))

                if image_latents_map is not None:
                    cond_tile_map = image_latents_map[:, :, coord.y0:coord.y1, coord.x0:coord.x1]
                    cond_packed_tiles.append(pipe._pack_latents(cond_tile_map))
                    cond_abs_ids.append(build_absolute_image_ids_for_tile(pipe, cond_tile_map, coord))

            tile_latents_packed = torch.cat(packed_tiles, dim=0)      # [nt*B, tile_seq, C]
            tile_latent_local_ids = torch.cat(tile_local_ids, dim=0)  # [nt*B, tile_seq, 4]
            tile_latent_ids = torch.cat(tile_abs_ids, dim=0)          # [nt*B, tile_seq, 4]
            tile_timestep = t_scalar.expand(tile_latents_packed.shape[0]).to(tile_latents_packed.dtype)

            prompt_rep = prompt_embeds.repeat(nt, 1, 1)
            text_ids_rep = text_ids.repeat(nt, 1, 1)

            if image_latents_map is not None:
                cond_latents_packed = torch.cat(cond_packed_tiles, dim=0)
                cond_latent_ids = torch.cat(cond_abs_ids, dim=0)

                latent_model_input = torch.cat(
                    [tile_latents_packed, cond_latents_packed.to(tile_latents_packed.dtype)],
                    dim=1,
                ).to(pipe.transformer.dtype)

                img_ids = torch.cat([tile_latent_ids, cond_latent_ids], dim=1)
            else:
                latent_model_input = tile_latents_packed.to(pipe.transformer.dtype)
                img_ids = tile_latent_ids

            with maybe_cache_context(pipe.transformer, "cond"):
                noise_pred = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=tile_timestep / 1000,
                    guidance=None,
                    encoder_hidden_states=prompt_rep,
                    txt_ids=text_ids_rep,
                    img_ids=img_ids,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]

            noise_pred = noise_pred[:, : tile_latents_packed.size(1), :]

            if do_cfg:
                neg_prompt_rep = negative_prompt_embeds.repeat(nt, 1, 1)
                neg_text_ids_rep = negative_text_ids.repeat(nt, 1, 1)

                with maybe_cache_context(pipe.transformer, "uncond"):
                    neg_noise_pred = pipe.transformer(
                        hidden_states=latent_model_input,
                        timestep=tile_timestep / 1000,
                        guidance=None,
                        encoder_hidden_states=neg_prompt_rep,
                        txt_ids=neg_text_ids_rep,
                        img_ids=img_ids,
                        joint_attention_kwargs=None,
                        return_dict=False,
                    )[0]
                neg_noise_pred = neg_noise_pred[:, : tile_latents_packed.size(1), :]
                noise_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)

            noise_pred_chunks = noise_pred.chunk(nt, dim=0)
            tile_local_id_chunks = tile_latent_local_ids.chunk(nt, dim=0)

            for chunk, local_ids_chunk, coord in zip(noise_pred_chunks, tile_local_id_chunks, batch_coords):
                local_h = coord.y1 - coord.y0
                local_w = coord.x1 - coord.x0
                weight_key = (local_h, local_w)
                if weight_key not in weight_cache:
                    weight_cache[weight_key] = make_gaussian_weight(
                        local_h,
                        local_w,
                        C,
                        device,
                        self.sigma_ratio,
                    ).to(latents_map.dtype)

                weight = weight_cache[weight_key]
                noise_tile = pipe._unpack_latents_with_ids(chunk, local_ids_chunk)
                noise_accum[:, :, coord.y0:coord.y1, coord.x0:coord.x1] += noise_tile * weight
                weight_accum[:, :, coord.y0:coord.y1, coord.x0:coord.x1] += weight

        return noise_accum / weight_accum.clamp_min(1e-8)

    @torch.no_grad()
    def generate(
        self,
        image: Image.Image | List[Image.Image] | None,
        prompt: str | List[str],
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        generator: Optional[torch.Generator] = None,
        canvas_height: Optional[int] = None,
        canvas_width: Optional[int] = None,
        max_sequence_length: int = 512,
        text_encoder_out_layers: tuple[int, ...] = (9, 18, 27),
    ):
        pipe = self.pipe
        device = pipe._execution_device

        pipe._guidance_scale = guidance_scale
        pipe._attention_kwargs = None
        pipe._current_timestep = None
        pipe._interrupt = False

        height, width = self._resolve_canvas_size(image, canvas_height, canvas_width)

        pipe.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            prompt_embeds=None,
            callback_on_step_end_tensor_inputs=["latents"],
            guidance_scale=guidance_scale,
        )

        if isinstance(prompt, str):
            batch_size = 1
        else:
            batch_size = len(prompt)

        prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=prompt,
            prompt_embeds=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )

        do_cfg = (guidance_scale > 1.0) and (not pipe.config.is_distilled)
        negative_prompt_embeds = None
        negative_text_ids = None
        if do_cfg:
            negative_prompt = ""
            if isinstance(prompt, list):
                negative_prompt = [negative_prompt] * len(prompt)

            negative_prompt_embeds, negative_text_ids = pipe.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=None,
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=max_sequence_length,
                text_encoder_out_layers=text_encoder_out_layers,
            )

        condition_images = self._prepare_condition_images(image, height, width)

        num_channels_latents = pipe.transformer.config.in_channels // 4
        latents_packed, latent_ids_full = pipe.prepare_latents(
            batch_size=batch_size,
            num_latents_channels=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=None,
        )

        image_latents_map = None
        if condition_images is not None:
            image_latents, image_latent_ids = pipe.prepare_image_latents(
                images=condition_images,
                batch_size=batch_size,
                generator=generator,
                device=device,
                dtype=pipe.vae.dtype,
            )
            image_latents_map = pipe._unpack_latents_with_ids(image_latents, image_latent_ids)

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        if hasattr(pipe.scheduler.config, "use_flow_sigmas") and pipe.scheduler.config.use_flow_sigmas:
            sigmas = None

        image_seq_len = latents_packed.shape[1]
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
        timesteps, num_inference_steps = retrieve_timesteps(
            pipe.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )

        pipe.scheduler.set_begin_index(0)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * pipe.scheduler.order, 0)
        pipe._num_timesteps = len(timesteps)

        latents_map = pipe._unpack_latents_with_ids(latents_packed, latent_ids_full)
        if image_latents_map is not None and image_latents_map.shape[-2:] != latents_map.shape[-2:]:
            image_latents_map = F.interpolate(
                image_latents_map.float(),
                size=latents_map.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).to(latents_map.dtype)

        with pipe.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if pipe.interrupt:
                    continue

                pipe._current_timestep = t

                noise_pred_map = self._predict_noise_tiled(
                    latents_map=latents_map,
                    t_scalar=t,
                    prompt_embeds=prompt_embeds,
                    text_ids=text_ids,
                    image_latents_map=image_latents_map,
                    guidance_scale=guidance_scale,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_text_ids=negative_text_ids,
                )

                noise_pred_packed = pipe._pack_latents(noise_pred_map)
                latents_dtype = latents_packed.dtype
                latents_packed = pipe.scheduler.step(noise_pred_packed, t, latents_packed, return_dict=False)[0]

                if latents_packed.dtype != latents_dtype and torch.backends.mps.is_available():
                    latents_packed = latents_packed.to(latents_dtype)

                latents_map = pipe._unpack_latents_with_ids(latents_packed, latent_ids_full)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                    progress_bar.update()

        pipe._current_timestep = None

        final_latents = pipe._unpack_latents_with_ids(latents_packed, latent_ids_full)
        latents_bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(final_latents.device, final_latents.dtype)
        latents_bn_std = torch.sqrt(
            pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps
        ).to(final_latents.device, final_latents.dtype)

        final_latents = final_latents * latents_bn_std + latents_bn_mean
        final_latents = pipe._unpatchify_latents(final_latents)
        decoded = pipe.vae.decode(final_latents, return_dict=False)[0]
        images = pipe.image_processor.postprocess(decoded, output_type="pil")

        pipe.maybe_free_model_hooks()
        return images[0]


# -----------------------------
# Inference wrappers
# -----------------------------
@torch.no_grad()
def run_plain_inference(
    pipeline: Flux2KleinPipeline,
    condition_image: Image.Image,
    prompt: str,
    guidance_scale: float,
    num_inference_steps: int,
    generator: Optional[torch.Generator],
):
    out = pipeline(
        image=condition_image,
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    )
    return out.images[0]


@torch.no_grad()
def run_canvas_tile_inference(
    pipeline: Flux2KleinPipeline,
    condition_image: Image.Image,
    prompt: str,
    guidance_scale: float,
    num_inference_steps: int,
    generator: Optional[torch.Generator],
    canvas_height: Optional[int],
    canvas_width: Optional[int],
    tile_size_px: int,
    tile_overlap_px: int,
    tile_batch_size: int,
    tile_sigma_ratio: float,
):
    runner = Flux2CanvasRunner(
        pipe=pipeline,
        tile_size_px=tile_size_px,
        tile_overlap_px=tile_overlap_px,
        tile_batch_size=tile_batch_size,
        sigma_ratio=tile_sigma_ratio,
    )
    return runner.generate(
        image=condition_image,
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
        canvas_height=canvas_height,
        canvas_width=canvas_width,
    )


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()

    if args.crop_mode != "full" and args.resolution <= 0:
        raise ValueError("`--resolution` must be a positive integer when using crop modes.")

    if args.mode == "canvas_tile" and args.tile_size_px <= args.tile_overlap_px:
        raise ValueError("`--tile_size_px` must be larger than `--tile_overlap_px`.")

    fixed_prompt = None
    if args.prompt_name:
        fixed_prompt = load_prompt_by_name(args.prompts_json, args.prompt_name)

    fallback_prompt = fixed_prompt if fixed_prompt is not None else args.default_prompt

    torch_dtype = resolve_dtype(args.dtype)
    samples, _ = load_manifest_entries(args.input_json, fallback_prompt)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_json = Path(args.output_json).expanduser().resolve() if args.output_json else output_dir / "results.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)

    hr_output_dir = output_dir / "hr"
    if args.crop_mode != "full":
        hr_output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = Flux2KleinPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=torch_dtype,
    )

    if args.lora_weights_path:
        pipeline.load_lora_weights(args.lora_weights_path)

    if args.cpu_offload and args.device.startswith("cuda"):
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to(args.device)

    # Keep only the dataset-level progress bar for cleaner logs.
    pipeline.set_progress_bar_config(disable=True)

    results = []
    used_output_names: set[str] = set()
    sample_iter = tqdm(enumerate(samples), total=len(samples), desc="Inference", unit="sample", dynamic_ncols=True)
    for index, sample in sample_iter:
        cond_path = Path(sample["cond_path"])
        hr_path = Path(sample["hr_path"]).resolve()
        prompt = fixed_prompt if fixed_prompt is not None else sample["prompt"]
        sample_iter.set_postfix_str(cond_path.name)

        seed = None if args.seed < 0 else args.seed + index
        generator = build_generator(args.device, seed)

        condition_image = load_rgb_image(cond_path)
        hr_image = None

        if args.crop_mode != "full":
            hr_image = load_rgb_image(hr_path)
            condition_image, hr_image = crop_pair(
                condition_image,
                hr_image,
                args.crop_mode,
                args.resolution,
                seed,
            )

        if args.mode == "plain":
            generated = run_plain_inference(
                pipeline=pipeline,
                condition_image=condition_image,
                prompt=prompt,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
            )
        elif args.mode == "canvas_tile":
            generated = run_canvas_tile_inference(
                pipeline=pipeline,
                condition_image=condition_image,
                prompt=prompt,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                canvas_height=args.canvas_height,
                canvas_width=args.canvas_width,
                tile_size_px=args.tile_size_px,
                tile_overlap_px=args.tile_overlap_px,
                tile_batch_size=args.tile_batch_size,
                tile_sigma_ratio=args.tile_sigma_ratio,
            )
        else:
            raise ValueError(f"Unknown mode: {args.mode}")

        output_path = resolve_output_image_path(output_dir, cond_path, used_output_names)
        generated.save(output_path)

        saved_hr_path = hr_path
        if args.crop_mode != "full":
            saved_hr_path = hr_output_dir / f"{index:05d}_{hr_path.stem}_hr.png"
            hr_image.save(saved_hr_path)

        results.append(
            {
                "res": str(output_path.resolve()),
                "hr": str(Path(saved_hr_path).resolve()),
            }
        )

    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
