import argparse
import json
import os
import random
from pathlib import Path

from PIL import Image
from PIL.ImageOps import exif_transpose


STAGE2_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = STAGE2_DIR / "weights" / "flux2-klein-base-9b"
DEFAULT_PROMPTS_PATH = STAGE2_DIR / "prompts.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Run batch base FLUX.2 inference from a local JSON manifest.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Local Diffusers model directory. Defaults to stage2/weights/flux2-klein-base-9b.",
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
        help="Directory where generated images, cropped HR images, and results.json will be written.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional explicit path for the output manifest. Defaults to <output_dir>/results.json.",
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
        required=True,
        help="Prompt entry name to select from prompts.json. The selected prompt is used for all samples.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        required=True,
        help="Square patch size used for paired random crops.",
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
        default="cuda",
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
    parser.add_argument(
        "--gpu_devices",
        type=str,
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES value, such as `0` or `0,1`.",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name: str, torch_module):
    if dtype_name == "fp32":
        return torch_module.float32
    if dtype_name == "fp16":
        return torch_module.float16
    return torch_module.bfloat16


def resolve_manifest_image_path(base_dir: Path, path_value: str, field_name: str, sample_index: int) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.exists():
        raise ValueError(f"Sample {sample_index} has missing `{field_name}` image: {path}")
    return path


def load_manifest(data_json_path: str):
    manifest_path = Path(data_json_path).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Input manifest does not exist: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as handle:
        entries = json.load(handle)

    if not isinstance(entries, list):
        raise ValueError("Input manifest must be a JSON array of samples.")

    samples = []
    base_dir = manifest_path.parent

    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Sample {index} must be a JSON object.")

        hr_path_value = entry.get("hr")
        cond_path_value = entry.get("res") or entry.get("lr")

        if not hr_path_value:
            raise ValueError(f"Sample {index} is missing `hr`.")
        if not cond_path_value:
            raise ValueError(f"Sample {index} must include either `res` or `lr`.")

        samples.append(
            {
                "hr_path": resolve_manifest_image_path(base_dir, str(hr_path_value), "hr", index),
                "cond_path": resolve_manifest_image_path(base_dir, str(cond_path_value), "res/lr", index),
            }
        )

    if not samples:
        raise ValueError("Input manifest is empty.")

    return samples


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


def paired_random_crop(cond_image: Image.Image, hr_image: Image.Image, patch_size: int, seed: int | None):
    if hr_image.size != cond_image.size:
        hr_image = hr_image.resize(cond_image.size, Image.Resampling.BICUBIC)

    cond_image, hr_image = ensure_min_crop_size(cond_image, hr_image, patch_size)
    width, height = cond_image.size

    rng = random.Random(seed) if seed is not None else random
    left = rng.randint(0, width - patch_size)
    top = rng.randint(0, height - patch_size)

    crop_box = (left, top, left + patch_size, top + patch_size)
    return cond_image.crop(crop_box), hr_image.crop(crop_box)


def build_generator(device: str, seed: int | None, torch_module):
    if seed is None:
        return None
    generator_device = "cuda" if device.startswith("cuda") and torch_module.cuda.is_available() else "cpu"
    return torch_module.Generator(device=generator_device).manual_seed(seed)


def main():
    args = parse_args()

    if args.patch_size <= 0:
        raise ValueError("`--patch_size` must be a positive integer.")

    if args.gpu_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

    import torch
    from diffusers import Flux2KleinPipeline

    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    prompt = load_prompt_by_name(args.prompts_json, args.prompt_name)
    samples = load_manifest(args.input_json)
    torch_dtype = resolve_dtype(args.dtype, torch)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    hr_output_dir = output_dir / "hr"
    hr_output_dir.mkdir(parents=True, exist_ok=True)

    output_json = Path(args.output_json).expanduser().resolve() if args.output_json else output_dir / "results.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)

    pipe = Flux2KleinPipeline.from_pretrained(
        str(model_path),
        torch_dtype=torch_dtype,
    )

    if args.cpu_offload and args.device.startswith("cuda"):
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(args.device)

    results = []
    for index, sample in enumerate(samples):
        sample_seed = None if args.seed < 0 else args.seed + index
        generator = build_generator(args.device, sample_seed, torch)

        cond_image = load_rgb_image(sample["cond_path"])
        hr_image = load_rgb_image(sample["hr_path"])
        cond_crop, hr_crop = paired_random_crop(cond_image, hr_image, args.patch_size, sample_seed)

        generated = pipe(
            image=cond_crop,
            prompt=prompt,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
        ).images[0]

        output_path = output_dir / f"{index:05d}_{sample['cond_path'].stem}.png"
        hr_output_path = hr_output_dir / f"{index:05d}_{sample['hr_path'].stem}_hr.png"
        generated.save(output_path)
        hr_crop.save(hr_output_path)

        results.append(
            {
                "res": str(output_path.resolve()),
                "hr": str(hr_output_path.resolve()),
            }
        )

    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
