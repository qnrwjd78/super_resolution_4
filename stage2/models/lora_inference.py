import argparse
import json
import random
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline
from PIL import Image
from PIL.ImageOps import exif_transpose

from lora_data import load_manifest_entries


def parse_args():
    parser = argparse.ArgumentParser(description="Run FLUX.2 image-conditioned inference from a local JSON manifest.")
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
        "--default_prompt",
        type=str,
        default=None,
        help="Fallback prompt used when a sample in `--input_json` does not define `prompt`.",
    )
    parser.add_argument("--revision", type=str, default=None, help="Optional model revision.")
    parser.add_argument("--variant", type=str, default=None, help="Optional model variant.")
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "center_crop", "random_crop"],
        help="`full` uses the whole condition image. Crop modes use a fixed square patch from the condition image.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Patch size used when `--mode` is `center_crop` or `random_crop`.",
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


def build_generator(device: str, seed: int | None):
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


def crop_pair(cond_image: Image.Image, hr_image: Image.Image, mode: str, patch_size: int, seed: int | None):
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


def main():
    args = parse_args()
    if args.mode != "full" and args.resolution <= 0:
        raise ValueError("`--resolution` must be a positive integer when using crop modes.")

    torch_dtype = resolve_dtype(args.dtype)
    samples, _ = load_manifest_entries(args.input_json, args.default_prompt)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = Path(args.output_json).expanduser().resolve() if args.output_json else output_dir / "results.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    hr_output_dir = output_dir / "hr"
    if args.mode != "full":
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

    results = []
    for index, sample in enumerate(samples):
        cond_path = Path(sample["cond_path"])
        hr_path = Path(sample["hr_path"]).resolve()
        prompt = sample["prompt"]
        seed = None if args.seed < 0 else args.seed + index
        generator = build_generator(args.device, seed)

        condition_image = load_rgb_image(cond_path)
        hr_image = None
        if args.mode != "full":
            hr_image = load_rgb_image(hr_path)
            condition_image, hr_image = crop_pair(condition_image, hr_image, args.mode, args.resolution, seed)

        generated = pipeline(
            image=condition_image,
            prompt=prompt,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
        ).images[0]

        output_path = output_dir / f"{index:05d}_{cond_path.stem}.png"
        generated.save(output_path)
        saved_hr_path = hr_path
        if args.mode != "full":
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
