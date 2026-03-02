import json
import random
from pathlib import Path

import torch
from accelerate.logging import get_logger
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

logger = get_logger(__name__)


def _resolve_image_path(base_dir: Path, path_value: str, field_name: str, sample_index: int) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()

    if not path.exists():
        raise ValueError(f"Sample {sample_index} has missing `{field_name}` image: {path}")

    return path


def load_manifest_entries(data_json_path: str, default_prompt: str | None = None):
    manifest_path = Path(data_json_path).expanduser().resolve()
    if not manifest_path.exists():
        raise ValueError(f"Training manifest does not exist: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as handle:
        entries = json.load(handle)

    if not isinstance(entries, list):
        raise ValueError("Training manifest must be a JSON array of samples.")

    samples = []
    has_custom_prompts = False
    base_dir = manifest_path.parent

    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Sample {index} must be a JSON object.")

        hr_path_value = entry.get("hr")
        cond_path_value = entry.get("res") or entry.get("lr")
        prompt_value = entry.get("prompt", default_prompt)

        if not hr_path_value:
            raise ValueError(f"Sample {index} is missing `hr`.")
        if not cond_path_value:
            raise ValueError(f"Sample {index} must include either `res` or `lr`.")
        if prompt_value is None or not str(prompt_value).strip():
            raise ValueError(
                f"Sample {index} is missing `prompt`, and no fallback `--instance_prompt` was provided."
            )

        if "prompt" in entry and str(entry["prompt"]).strip():
            has_custom_prompts = True

        samples.append(
            {
                "hr_path": str(_resolve_image_path(base_dir, hr_path_value, "hr", index)),
                "cond_path": str(_resolve_image_path(base_dir, cond_path_value, "res/lr", index)),
                "prompt": str(prompt_value).strip(),
            }
        )

    if not samples:
        raise ValueError("Training manifest is empty.")

    return samples, has_custom_prompts


class DreamBoothDataset(Dataset):
    def __init__(
        self,
        args,
        train_data_json,
        instance_prompt,
        size=1024,
        repeats=1,
        center_crop=False,
    ):
        self.args = args
        self.patch_size = (size, size)
        self.center_crop = center_crop
        self.instance_prompt = instance_prompt
        self.samples = []

        base_samples, self.custom_instance_prompts = load_manifest_entries(train_data_json, instance_prompt)
        for sample in base_samples:
            for _ in range(repeats):
                self.samples.append(sample.copy())

        self.num_instance_images = len(self.samples)
        self._length = self.num_instance_images
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.5], [0.5])

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        sample = self.samples[index % self.num_instance_images]
        target_image = self._load_rgb_image(sample["hr_path"])
        cond_image = self._load_rgb_image(sample["cond_path"])
        target_image, cond_image = self.paired_transform(target_image, cond_image)

        return {
            "instance_images": target_image,
            "cond_images": cond_image,
            "instance_prompt": sample["prompt"],
        }

    @staticmethod
    def _load_rgb_image(path: str) -> Image.Image:
        with Image.open(path) as image:
            image = exif_transpose(image)
            if image.mode != "RGB":
                image = image.convert("RGB")
            return image.copy()

    def paired_transform(self, image: Image.Image, cond_image: Image.Image):
        if cond_image.size != image.size:
            cond_image = cond_image.resize(image.size, Image.Resampling.BICUBIC)

        image, cond_image = self._resize_pair_for_patch(image, cond_image)

        if self.center_crop:
            top, left, height, width = self._center_crop_params(image)
        else:
            top, left, height, width = transforms.RandomCrop.get_params(image, output_size=self.patch_size)

        image = TF.crop(image, top, left, height, width)
        cond_image = TF.crop(cond_image, top, left, height, width)

        if self.args.random_flip and random.random() < 0.5:
            image = TF.hflip(image)
            cond_image = TF.hflip(cond_image)

        image = self.normalize(self.to_tensor(image))
        cond_image = self.normalize(self.to_tensor(cond_image))
        return image, cond_image

    def _resize_pair_for_patch(self, image: Image.Image, cond_image: Image.Image):
        width, height = image.size
        patch = self.patch_size[0]
        min_side = min(width, height)

        if min_side >= patch:
            return image, cond_image

        scale = patch / float(min_side)
        new_width = max(patch, round(width * scale))
        new_height = max(patch, round(height * scale))
        new_size = (new_width, new_height)

        image = image.resize(new_size, Image.Resampling.BICUBIC)
        cond_image = cond_image.resize(new_size, Image.Resampling.BICUBIC)
        return image, cond_image

    def _center_crop_params(self, image: Image.Image):
        width, height = image.size
        crop_h, crop_w = self.patch_size
        top = max((height - crop_h) // 2, 0)
        left = max((width - crop_w) // 2, 0)
        return top, left, crop_h, crop_w


def collate_fn(examples):
    pixel_values = [example["instance_images"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    cond_pixel_values = [example["cond_images"] for example in examples]
    cond_pixel_values = torch.stack(cond_pixel_values)
    cond_pixel_values = cond_pixel_values.to(memory_format=torch.contiguous_format).float()

    return {
        "pixel_values": pixel_values,
        "cond_pixel_values": cond_pixel_values,
        "prompts": prompts,
    }
