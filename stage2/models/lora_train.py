import copy
import json
import logging
import math
import os
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from peft import LoraConfig, prepare_model_for_kbit_training, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from tqdm.auto import tqdm
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

# Temporary compatibility shim: some diffusers builds reference torch.xpu
# unconditionally, but torch 2.0.x may not expose torch.xpu.
if not hasattr(torch, "xpu"):
    class _TorchXPUStub:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def empty_cache():
            return None

        @staticmethod
        def device_count():
            return 0

        @staticmethod
        def manual_seed(_seed):
            return None

        @staticmethod
        def synchronize():
            return None

    torch.xpu = _TorchXPUStub()  # type: ignore[attr-defined]

import diffusers
from diffusers import (
    AutoencoderKLFlux2,
    BitsAndBytesConfig,
    FlowMatchEulerDiscreteScheduler,
    Flux2KleinPipeline,
    Flux2Transformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _to_cpu_contiguous,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
    get_fsdp_kwargs_from_accelerator,
    offload_models,
    wrap_with_fsdp,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
    load_image,
)
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import is_compiled_module

from lora_data import DreamBoothDataset, collate_fn
from lora_hub import save_model_card

if getattr(torch, "distributed", None) is not None:
    import torch.distributed as dist

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.37.0.dev0")

logger = get_logger(__name__)


def _load_aesop_autoencoder(args, device: torch.device):
    metric_labels = {label.strip().lower() for label in _parse_csv_str(getattr(args, "nr_iqa_metric", None))}
    if "aesop" not in metric_labels:
        return None
    if not args.aesop_autoencoder_path:
        raise ValueError("`nr_iqa_metric` includes `aesop`, but `--aesop_autoencoder_path` is missing.")

    aesop_root = (
        Path(__file__).resolve().parents[1] / "repos" / "AESOP-SR" / "AESOP"
    )
    if not aesop_root.exists():
        raise FileNotFoundError(f"AESOP repo not found: {aesop_root}")

    aesop_root_str = str(aesop_root)
    if aesop_root_str not in sys.path:
        sys.path.insert(0, aesop_root_str)

    from basicsr.archs.autoencoder_arch import AutoEncoder_RRDBNet

    ae_net = AutoEncoder_RRDBNet(
        enc_opt={"placeholder": 0},
        dec_opt={
            "type": "RRDBNet",
            "num_in_ch": 3,
            "num_out_ch": 3,
            "num_feat": 64,
            "num_block": 23,
            "num_grow_ch": 32,
        },
    )

    checkpoint = torch.load(args.aesop_autoencoder_path, map_location="cpu")
    state_dict = checkpoint[args.aesop_autoencoder_key]
    ae_net.load_state_dict(state_dict)
    ae_net.eval()
    ae_net.requires_grad_(False)
    ae_net.to(device=device, dtype=torch.float32)
    logger.info("Loaded AESOP autoencoder from %s", args.aesop_autoencoder_path)
    return ae_net


def module_filter_fn(mod: torch.nn.Module, fqn: str):
    # don't convert the output module
    if fqn == "proj_out":
        return False
    # don't convert linear modules with weight dimensions not divisible by 16
    if isinstance(mod, torch.nn.Linear):
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            return False
    return True


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def _parse_csv_str(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items = value
    else:
        items = str(value).split(",")
    return [str(item).strip() for item in items if str(item).strip()]


def _parse_csv_floats(value: Any) -> list[float]:
    values = _parse_csv_str(value)
    try:
        return [float(item) for item in values]
    except ValueError as exc:
        raise ValueError(f"Expected comma-separated float values, got: {value!r}") from exc


def _parse_csv_ints(value: Any) -> list[int]:
    values = _parse_csv_str(value)
    try:
        return [int(item) for item in values]
    except ValueError as exc:
        raise ValueError(f"Expected comma-separated integer values, got: {value!r}") from exc


def _parse_csv_nullable_str(value: Any) -> list[str | None]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items = value
    else:
        items = str(value).split(",")

    parsed: list[str | None] = []
    for item in items:
        item_str = str(item).strip()
        if item_str == "" or item_str.lower() in {"none", "null", "__none__"}:
            parsed.append(None)
        else:
            parsed.append(item_str)
    return parsed


def _expand_sem_stage2_values(values: list[Any], count: int, default: Any, field_name: str) -> list[Any]:
    if not values:
        return [default] * count
    if len(values) == 1 and count > 1:
        return values * count
    if len(values) != count:
        raise ValueError(
            f"`{field_name}` expects either 1 value or {count} values to match the semantic adapters, got {len(values)}."
        )
    return values


def _build_sem_stage2_specs(args: Any) -> list[dict[str, Any]]:
    sem_adapter_names = _parse_csv_str(getattr(args, "sem_adapter_names", None))
    if sem_adapter_names:
        sem_weight_paths = _expand_sem_stage2_values(
            _parse_csv_nullable_str(getattr(args, "sem_lora_weights_paths", None)),
            len(sem_adapter_names),
            getattr(args, "sem_lora_weights_path", None),
            "sem_lora_weights_paths",
        )
        sem_adapter_scales = _expand_sem_stage2_values(
            _parse_csv_floats(getattr(args, "sem_adapter_scales", None)),
            len(sem_adapter_names),
            float(args.sem_adapter_scale),
            "sem_adapter_scales",
        )
        sem_ranks = _expand_sem_stage2_values(
            _parse_csv_ints(getattr(args, "sem_ranks", None)),
            len(sem_adapter_names),
            int(args.sem_rank if getattr(args, "sem_rank", None) is not None else args.rank),
            "sem_ranks",
        )
        sem_lora_alphas = _expand_sem_stage2_values(
            _parse_csv_ints(getattr(args, "sem_lora_alphas", None)),
            len(sem_adapter_names),
            int(args.sem_lora_alpha if getattr(args, "sem_lora_alpha", None) is not None else args.lora_alpha),
            "sem_lora_alphas",
        )
    elif getattr(args, "sem2_adapter_name", None):
        sem_adapter_names = [args.sem_adapter_name, args.sem2_adapter_name]
        sem_weight_paths = [getattr(args, "sem_lora_weights_path", None), getattr(args, "sem2_lora_weights_path", None)]
        sem_adapter_scales = [float(args.sem_adapter_scale), float(getattr(args, "sem2_adapter_scale", 1.0))]
        sem_ranks = [
            int(args.sem_rank if getattr(args, "sem_rank", None) is not None else args.rank),
            int(args.sem2_rank if getattr(args, "sem2_rank", None) is not None else args.rank),
        ]
        sem_lora_alphas = [
            int(args.sem_lora_alpha if getattr(args, "sem_lora_alpha", None) is not None else args.lora_alpha),
            int(args.sem2_lora_alpha if getattr(args, "sem2_lora_alpha", None) is not None else args.lora_alpha),
        ]
    else:
        sem_adapter_names = [args.sem_adapter_name]
        sem_weight_paths = [getattr(args, "sem_lora_weights_path", None)]
        sem_adapter_scales = [float(args.sem_adapter_scale)]
        sem_ranks = [int(args.sem_rank if getattr(args, "sem_rank", None) is not None else args.rank)]
        sem_lora_alphas = [
            int(args.sem_lora_alpha if getattr(args, "sem_lora_alpha", None) is not None else args.lora_alpha)
        ]

    return [
        {
            "name": name,
            "weight_path": weight_path,
            "scale": float(scale),
            "rank": int(rank),
            "lora_alpha": int(alpha),
        }
        for name, weight_path, scale, rank, alpha in zip(
            sem_adapter_names, sem_weight_paths, sem_adapter_scales, sem_ranks, sem_lora_alphas
        )
    ]


def _expand_q_metric_values(
    values: list[float],
    count: int,
    default: float,
    field_name: str,
) -> list[float]:
    if count < 1:
        raise ValueError("Q-loss metric count must be >= 1.")
    if not values:
        return [default] * count
    if len(values) == 1 and count > 1:
        return values * count
    if len(values) != count:
        raise ValueError(
            f"`{field_name}` expects either 1 value or {count} values to match `nr_iqa_metric`, got {len(values)}."
        )
    return values


def _default_q_metric_scale(metric_name: str) -> float:
    metric_key = str(metric_name).strip().lower()
    if metric_key == "niqe":
        return 10.0
    if metric_key == "musiq":
        return 100.0
    return 1.0


def _reduce_q_score_per_sample(q_score: Any, batch_size: int, device: torch.device) -> torch.Tensor:
    if isinstance(q_score, tuple):
        q_score = q_score[0]
    if not torch.is_tensor(q_score):
        q_score = torch.as_tensor(q_score, device=device, dtype=torch.float32)
    else:
        q_score = q_score.to(device=device, dtype=torch.float32)

    if q_score.ndim == 0:
        return q_score.reshape(1).expand(batch_size)
    if q_score.ndim == 1:
        if q_score.shape[0] == 1 and batch_size > 1:
            return q_score.expand(batch_size)
        if q_score.shape[0] != batch_size:
            raise ValueError(f"Q-loss metric returned shape {tuple(q_score.shape)} for batch_size={batch_size}.")
        return q_score

    q_score = q_score.reshape(q_score.shape[0], -1).mean(dim=1)
    if q_score.shape[0] == 1 and batch_size > 1:
        return q_score.expand(batch_size)
    if q_score.shape[0] != batch_size:
        raise ValueError(f"Q-loss metric returned shape {tuple(q_score.shape)} for batch_size={batch_size}.")
    return q_score


def _is_linalg_svd_failure(exc: Exception) -> bool:
    torch_c = getattr(torch, "_C", None)
    lin_alg_error = getattr(torch_c, "_LinAlgError", None)
    if lin_alg_error is not None and isinstance(exc, lin_alg_error):
        return True

    message = str(exc).lower()
    return "linalg.svd" in message or ("svd" in message and "converge" in message)


def _warn_q_metric_skip(args: Any, log_name: str, reason: str) -> None:
    if not hasattr(args, "_q_metric_skip_counts"):
        args._q_metric_skip_counts = {}
    skip_counts = args._q_metric_skip_counts
    count = int(skip_counts.get(log_name, 0)) + 1
    skip_counts[log_name] = count

    if count <= 3 or count % 100 == 0:
        logger.warning(
            "[Q-loss] Skipping metric '%s' for current batch (%s). skip_count=%d",
            log_name,
            reason,
            count,
        )


def _build_q_metric_specs(args: Any, device: torch.device) -> list[dict[str, Any]]:
    metric_aliases = {
        "l2": "l2",
        "mse": "l2",
        "l_2": "l2",
        "niqe": "niqe",
        "maniqa": "maniqa",
        "musiq": "musiq",
        "lpips": "lpips",
        "dists": "dists",
        "aesop": "aesop",
    }
    clip_aliases = {"clip-iqa", "clip_iqa", "clipiqa"}

    metric_labels = _parse_csv_str(getattr(args, "nr_iqa_metric", None))
    if not metric_labels:
        raise ValueError("`--nr_iqa_metric` must contain at least one metric.")

    weights = _expand_q_metric_values(
        _parse_csv_floats(getattr(args, "q_metric_weights", None)),
        len(metric_labels),
        1.0,
        "q_metric_weights",
    )
    create_q_metric = None
    needs_local_iqa = any(metric_aliases.get(label.strip().lower()) not in {None, "aesop"} for label in metric_labels)
    if needs_local_iqa:
        try:
            from local_iqa import create_q_metric
        except Exception:
            project_root = Path(__file__).resolve().parents[2]
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            try:
                from local_iqa import create_q_metric
            except Exception as nested_exc:
                raise RuntimeError("NR-IQA loss requested but `local_iqa` is not available in the runtime.") from nested_exc

    name_counts: dict[str, int] = {}
    specs: list[dict[str, Any]] = []
    for idx, raw_label in enumerate(metric_labels):
        metric_key = raw_label.strip().lower()
        if metric_key in clip_aliases:
            raise ValueError(
                "nr_iqa_metric='clipiqa' is excluded in this trainer. "
                "Use one of: L2, NIQE, ManIQA, MUSIQ, LPIPS, DISTS."
            )
        metric_name = metric_aliases.get(metric_key)
        if metric_name is None:
            supported = "NIQE, ManIQA, MUSIQ, L2, LPIPS, DISTS"
            raise ValueError(
                f"Unsupported nr_iqa_metric='{raw_label}'. "
                f"Supported labels: {supported}"
            )

        if metric_name == "aesop":
            if not args.aesop_autoencoder_path:
                raise ValueError("`nr_iqa_metric=aesop` requires `--aesop_autoencoder_path`.")
            spec_name = "aesop"
            count = name_counts.get(spec_name, 0) + 1
            name_counts[spec_name] = count
            log_name = spec_name if count == 1 else f"{spec_name}_{count}"
            specs.append(
                {
                    "name": spec_name,
                    "log_name": log_name,
                    "module": None,
                    "weight": float(weights[idx]),
                    "scale": 1.0,
                    "lower_better": True,
                    "requires_reference": True,
                }
            )
            continue

        if create_q_metric is None:
            raise RuntimeError(
                f"Metric '{metric_name}' requires local_iqa, but the helper was not initialized."
            )
        q_metric_spec = create_q_metric(metric_name, device=device)
        count = name_counts.get(q_metric_spec.name, 0) + 1
        name_counts[q_metric_spec.name] = count
        log_name = q_metric_spec.name if count == 1 else f"{q_metric_spec.name}_{count}"
        specs.append(
            {
                "name": q_metric_spec.name,
                "log_name": log_name,
                "module": q_metric_spec.module,
                "weight": float(weights[idx]),
                "scale": _default_q_metric_scale(q_metric_spec.name),
                "lower_better": bool(q_metric_spec.lower_better),
                "requires_reference": bool(q_metric_spec.requires_reference),
            }
        )

    logger.info(
        "Q-loss metrics enabled: %s",
        ", ".join(
            f"{spec['name']}(weight={spec['weight']}, scale={spec['scale']}, lower_better={spec['lower_better']}, "
            f"requires_reference={spec['requires_reference']})"
            for spec in specs
        ),
    )
    return specs


def build_inference_timestep_pool(
    scheduler: FlowMatchEulerDiscreteScheduler,
    image_seq_len: int,
    num_inference_steps: int,
    device: torch.device,
    sigma_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if num_inference_steps < 1:
        raise ValueError("`num_inference_steps` must be >= 1.")

    if num_inference_steps == 1:
        sigmas = [1.0]
    else:
        step = (1.0 - 1.0 / num_inference_steps) / (num_inference_steps - 1)
        sigmas = [1.0 - i * step for i in range(num_inference_steps)]

    if hasattr(scheduler.config, "use_flow_sigmas") and scheduler.config.use_flow_sigmas:
        sigmas = None

    mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
    scheduler_for_sampling = copy.deepcopy(scheduler)
    scheduler_for_sampling.set_timesteps(
        num_inference_steps=num_inference_steps,
        device=device,
        sigmas=sigmas,
        mu=mu,
    )

    timesteps = scheduler_for_sampling.timesteps.to(device=device, dtype=torch.float32)
    sigma_pool = scheduler_for_sampling.sigmas[: timesteps.shape[0]].to(device=device, dtype=sigma_dtype)
    return timesteps, sigma_pool


def add_lora_adapter(model: torch.nn.Module, adapter_config: LoraConfig, adapter_name: str) -> None:
    try:
        model.add_adapter(adapter_config, adapter_name=adapter_name)
        return
    except TypeError:
        pass

    try:
        model.add_adapter(adapter_name, adapter_config)
        return
    except TypeError:
        if adapter_name != "default":
            raise
        model.add_adapter(adapter_config)


def set_active_lora_adapters(
    model: torch.nn.Module, adapter_names: list[str], adapter_weights: list[float] | None = None
) -> None:
    if len(adapter_names) == 1 and hasattr(model, "set_adapter"):
        model.set_adapter(adapter_names[0])
        return

    if hasattr(model, "set_adapters"):
        try:
            if adapter_weights is None:
                model.set_adapters(adapter_names)
            else:
                model.set_adapters(adapter_names, adapter_weights=adapter_weights)
            return
        except TypeError:
            if adapter_weights is None:
                model.set_adapters(adapter_names)
            else:
                model.set_adapters(adapter_names, adapter_weights)
            return

    if hasattr(model, "set_adapter"):
        try:
            model.set_adapter(adapter_names)
            return
        except Exception:
            pass

    raise RuntimeError(f"This model does not support setting multiple active adapters: {adapter_names}.")


def set_adapter_requires_grad(model: torch.nn.Module, adapter_name: str, requires_grad: bool) -> int:
    marker = f".{adapter_name}."
    matched = 0
    for name, param in model.named_parameters():
        if "lora_" not in name:
            continue
        if marker in name:
            param.requires_grad_(requires_grad)
            matched += 1

    if matched == 0:
        logger.warning("No LoRA parameters found for adapter '%s'.", adapter_name)
    return matched


def get_adapter_peft_state_dict(model: torch.nn.Module, adapter_name: str, state_dict: dict[str, Any] | None = None):
    peft_kwargs: dict[str, Any] = {}
    if state_dict is not None:
        peft_kwargs["state_dict"] = state_dict

    try:
        return get_peft_model_state_dict(model, adapter_name=adapter_name, **peft_kwargs)
    except TypeError:
        if adapter_name != "default":
            raise
        return get_peft_model_state_dict(model, **peft_kwargs)


def filter_state_dict_for_adapter(
    state_dict: dict[str, Any],
    adapter_name: str,
    known_adapter_names: list[str],
) -> dict[str, Any]:
    other_markers = tuple(f".{name}." for name in known_adapter_names if name != adapter_name)
    if not other_markers:
        return state_dict

    filtered_state_dict = {
        key: value
        for key, value in state_dict.items()
        if not any(marker in key for marker in other_markers)
    }
    removed_keys = sorted(set(state_dict) - set(filtered_state_dict))
    if removed_keys:
        preview = ", ".join(removed_keys[:4])
        if len(removed_keys) > 4:
            preview += ", ..."
        logger.warning(
            "Filtered %d unexpected tensors while saving adapter '%s': %s",
            len(removed_keys),
            adapter_name,
            preview,
        )
    return filtered_state_dict


def collate_lora_metadata_for_adapter(modules_to_save: dict[str, Any], adapter_name: str) -> dict[str, Any]:
    metadatas: dict[str, Any] = {}

    for module_name, module in modules_to_save.items():
        if not hasattr(module, "peft_config"):
            continue

        peft_config = module.peft_config
        if adapter_name not in peft_config:
            available = ", ".join(sorted(peft_config.keys()))
            raise KeyError(
                f"Adapter '{adapter_name}' not found in `{module_name}.peft_config`. Available adapters: [{available}]"
            )

        metadatas[f"{module_name}_lora_adapter_metadata"] = peft_config[adapter_name].to_dict()

    return metadatas


def activate_stage2_adapters(
    model: torch.nn.Module,
    pix_adapter_name: str,
    sem_adapter_specs: list[dict[str, Any]],
    pix_adapter_scale: float,
) -> None:
    set_active_lora_adapters(
        model,
        [pix_adapter_name, *[spec["name"] for spec in sem_adapter_specs]],
        [float(pix_adapter_scale), *[float(spec["scale"]) for spec in sem_adapter_specs]],
    )


def set_stage2_adapter_trainability(
    model: torch.nn.Module,
    pix_adapter_name: str,
    sem_adapter_specs: list[dict[str, Any]],
    trainable_adapter_names: list[str],
) -> tuple[int, dict[str, int]]:
    trainable_name_set = set(trainable_adapter_names)
    pix_param_count = set_adapter_requires_grad(model, pix_adapter_name, False)
    sem_param_counts: dict[str, int] = {}
    for spec in sem_adapter_specs:
        adapter_name = spec["name"]
        sem_param_counts[adapter_name] = set_adapter_requires_grad(
            model, adapter_name, adapter_name in trainable_name_set
        )
    return pix_param_count, sem_param_counts


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `hf auth login` to authenticate with the Hub."
        )
    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )
    if args.do_fp8_training:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    sem_stage2_mode = bool(getattr(args, "train_sem_only", False))
    pix_adapter_name = args.pix_adapter_name if sem_stage2_mode else "default"
    sem_adapter_specs = _build_sem_stage2_specs(args) if sem_stage2_mode else []
    sem_adapter_names = [spec["name"] for spec in sem_adapter_specs]
    trainable_adapter_names = (
        _parse_csv_str(getattr(args, "sem_trainable_adapter_names", None)) or sem_adapter_names
    ) if sem_stage2_mode else ["default"]
    trainable_adapter_name_set = set(trainable_adapter_names)
    frozen_sem_adapter_names = [
        adapter_name for adapter_name in sem_adapter_names if adapter_name not in trainable_adapter_name_set
    ]
    save_adapter_names = trainable_adapter_names if sem_stage2_mode else ["default"]
    known_adapter_names_for_save = [pix_adapter_name, *sem_adapter_names] if sem_stage2_mode else save_adapter_names
    pix_lora_source = None

    if sem_stage2_mode:
        if not sem_adapter_specs:
            raise ValueError("Stage2 sem-only mode requires at least one semantic adapter spec.")
        pix_lora_source = args.pix_lora_weights_path or args.lora_weights_path
        if not pix_lora_source:
            raise ValueError(
                "Stage2 sem-only mode requires an existing pixel LoRA path via `--pix_lora_weights_path` "
                "(or legacy `--lora_weights_path`)."
            )
        if not args.use_nr_iqa_loss or args.lambda_q == 0.0:
            raise ValueError(
                "`--train_sem_only` requires Q-loss to be active. Set `--use_nr_iqa_loss` and `--lambda_q > 0`."
            )
        if args.lambda_fm != 0.0:
            logger.info("Stage2 sem-only mode overrides lambda_fm from %.6f to 0.0.", args.lambda_fm)
            args.lambda_fm = 0.0

    if args.cache_latents:
        logger.warning("Ignoring --cache_latents because random patch sampling changes the input crop every step.")
        args.cache_latents = False

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

    # Load the tokenizers
    tokenizer = Qwen2TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        revision=args.revision,
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    vae = AutoencoderKLFlux2.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(accelerator.device)
    latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
        accelerator.device
    )

    quantization_config = None
    if args.bnb_quantization_config_path is not None:
        with open(args.bnb_quantization_config_path, "r") as f:
            config_kwargs = json.load(f)
            if "load_in_4bit" in config_kwargs and config_kwargs["load_in_4bit"]:
                config_kwargs["bnb_4bit_compute_dtype"] = weight_dtype
        quantization_config = BitsAndBytesConfig(**config_kwargs)

    transformer = Flux2Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
        quantization_config=quantization_config,
        torch_dtype=weight_dtype,
    )
    if args.bnb_quantization_config_path is not None:
        transformer = prepare_model_for_kbit_training(transformer, use_gradient_checkpointing=False)

    text_encoder = Qwen3ForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder.requires_grad_(False)

    # We only train the additional adapter LoRA layers
    transformer.requires_grad_(False)
    vae.requires_grad_(False)

    if args.enable_npu_flash_attention:
        if is_torch_npu_available():
            logger.info("npu flash attention enabled.")
            transformer.set_attention_backend("_native_npu")
        else:
            raise ValueError("npu flash attention requires torch_npu extensions and is supported only on npu device ")

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    to_kwargs = {"dtype": weight_dtype, "device": accelerator.device} if not args.offload else {"dtype": weight_dtype}
    # flux vae is stable in bf16 so load it in weight_dtype to reduce memory
    vae.to(**to_kwargs)
    # we never offload the transformer to CPU, so we can just use the accelerator device
    transformer_to_kwargs = (
        {"device": accelerator.device}
        if args.bnb_quantization_config_path is not None
        else {"device": accelerator.device, "dtype": weight_dtype}
    )

    is_fsdp = getattr(accelerator.state, "fsdp_plugin", None) is not None
    if not is_fsdp:
        transformer.to(**transformer_to_kwargs)

    if args.do_fp8_training:
        convert_to_float8_training(
            transformer, module_filter_fn=module_filter_fn, config=Float8LinearConfig(pad_inner_dim=True)
        )

    text_encoder.to(**to_kwargs)
    # Initialize a text encoding pipeline and keep it to CPU for now.
    text_encoding_pipeline = Flux2KleinPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=None,
        transformer=None,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=None,
        revision=args.revision,
    )
    aesop_autoencoder = _load_aesop_autoencoder(args, accelerator.device)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    if args.lora_layers is not None:
        target_modules = [layer.strip() for layer in args.lora_layers.split(",") if layer.strip()]
        if not target_modules:
            target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
    else:
        target_modules = ["to_k", "to_q", "to_v", "to_out.0"]

    pix_transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    sem_transformer_lora_configs = {
        spec["name"]: LoraConfig(
            r=spec["rank"],
            lora_alpha=spec["lora_alpha"],
            lora_dropout=args.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        for spec in sem_adapter_specs
    }
    if sem_stage2_mode:
        add_lora_adapter(transformer, pix_transformer_lora_config, adapter_name=pix_adapter_name)
        for spec in sem_adapter_specs:
            add_lora_adapter(transformer, sem_transformer_lora_configs[spec["name"]], adapter_name=spec["name"])
        logger.info(
            "Stage2 adapter config: pix(rank=%d, alpha=%d), sem=%s",
            args.rank,
            args.lora_alpha,
            ", ".join(
                f"{spec['name']}(rank={spec['rank']}, alpha={spec['lora_alpha']}, scale={spec['scale']})"
                for spec in sem_adapter_specs
            ),
        )
    else:
        add_lora_adapter(transformer, pix_transformer_lora_config, adapter_name="default")

    def load_lora_into_transformer(transformer_model, input_dir, adapter_name="default"):
        lora_state_dict = Flux2KleinPipeline.lora_state_dict(input_dir)
        transformer_state_dict = {
            f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_model, transformer_state_dict, adapter_name=adapter_name)
        if incompatible_keys is not None:
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    "Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f"{unexpected_keys}."
                )
        return incompatible_keys

    if sem_stage2_mode:
        logger.info("Loading frozen pix LoRA from %s", pix_lora_source)
        load_lora_into_transformer(transformer, pix_lora_source, adapter_name=pix_adapter_name)

        for spec in sem_adapter_specs:
            if spec["weight_path"]:
                logger.info("Loading initial sem LoRA weights for '%s' from %s", spec["name"], spec["weight_path"])
                load_lora_into_transformer(transformer, spec["weight_path"], adapter_name=spec["name"])

        activate_stage2_adapters(transformer, pix_adapter_name, sem_adapter_specs, args.pix_adapter_scale)
        pix_param_count, sem_param_counts = set_stage2_adapter_trainability(
            transformer, pix_adapter_name, sem_adapter_specs, trainable_adapter_names
        )
        logger.info(
            "Stage2 sem-only mode active: pix adapter '%s' frozen (%d params), trainable sem adapters: %s, frozen sem adapters: %s",
            pix_adapter_name,
            pix_param_count,
            ", ".join(f"{name} ({sem_param_counts[name]} params)" for name in trainable_adapter_names) or "<none>",
            ", ".join(frozen_sem_adapter_names) or "<none>",
        )
    elif args.lora_weights_path:
        logger.info(f"Loading initial LoRA weights from {args.lora_weights_path}")
        load_lora_into_transformer(transformer, args.lora_weights_path, adapter_name="default")

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        transformer_cls = type(unwrap_model(transformer))

        # 1) Validate and pick the transformer model
        modules_to_save: dict[str, Any] = {}
        transformer_model = None

        for model in models:
            if isinstance(unwrap_model(model), transformer_cls):
                transformer_model = model
                modules_to_save["transformer"] = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        if transformer_model is None:
            raise ValueError("No transformer model found in 'models'")

        # 2) Optionally gather FSDP state dict once
        state_dict = accelerator.get_state_dict(transformer_model) if is_fsdp else None

        # 3) Only main process materializes the LoRA state dict
        if accelerator.is_main_process:
            for adapter_name in save_adapter_names:
                adapter_output_dir = output_dir if len(save_adapter_names) == 1 else os.path.join(output_dir, adapter_name)
                transformer_lora_layers_to_save = get_adapter_peft_state_dict(
                    unwrap_model(transformer_model) if is_fsdp else transformer_model,
                    adapter_name=adapter_name,
                    state_dict=state_dict,
                )
                transformer_lora_layers_to_save = filter_state_dict_for_adapter(
                    transformer_lora_layers_to_save,
                    adapter_name=adapter_name,
                    known_adapter_names=known_adapter_names_for_save,
                )

                if is_fsdp:
                    transformer_lora_layers_to_save = _to_cpu_contiguous(transformer_lora_layers_to_save)

                Flux2KleinPipeline.save_lora_weights(
                    adapter_output_dir,
                    transformer_lora_layers=transformer_lora_layers_to_save,
                    **collate_lora_metadata_for_adapter(modules_to_save, adapter_name),
                )

        # make sure to pop weight so that corresponding model is not saved again
        if weights:
            weights.pop()

    def load_model_hook(models, input_dir):
        transformer_ = None

        if not is_fsdp:
            while len(models) > 0:
                model = models.pop()

                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    transformer_ = unwrap_model(model)
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")
        else:
            transformer_ = Flux2Transformer2DModel.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="transformer",
            )
            if sem_stage2_mode:
                add_lora_adapter(transformer_, pix_transformer_lora_config, adapter_name=pix_adapter_name)
                for spec in sem_adapter_specs:
                    add_lora_adapter(transformer_, sem_transformer_lora_configs[spec["name"]], adapter_name=spec["name"])
                load_lora_into_transformer(transformer_, pix_lora_source, adapter_name=pix_adapter_name)
                for spec in sem_adapter_specs:
                    if spec["weight_path"]:
                        load_lora_into_transformer(transformer_, spec["weight_path"], adapter_name=spec["name"])
                activate_stage2_adapters(transformer_, pix_adapter_name, sem_adapter_specs, args.pix_adapter_scale)
                set_stage2_adapter_trainability(transformer_, pix_adapter_name, sem_adapter_specs, trainable_adapter_names)
            else:
                add_lora_adapter(transformer_, pix_transformer_lora_config, adapter_name="default")

        for adapter_name in save_adapter_names:
            adapter_input_dir = input_dir if len(save_adapter_names) == 1 else os.path.join(input_dir, adapter_name)
            load_lora_into_transformer(transformer_, adapter_input_dir, adapter_name=adapter_name)
        if sem_stage2_mode:
            activate_stage2_adapters(transformer_, pix_adapter_name, sem_adapter_specs, args.pix_adapter_scale)
            set_stage2_adapter_trainability(transformer_, pix_adapter_name, sem_adapter_specs, trainable_adapter_names)

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [transformer_]
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    # Optimization parameters
    transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.learning_rate}
    params_to_optimize = [transformer_parameters_with_lr]

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    # Dataset and DataLoaders creation:
    train_dataset = DreamBoothDataset(
        args=args,
        train_data_json=args.train_data_json,
        instance_prompt=args.instance_prompt,
        size=args.resolution,
        repeats=args.repeats,
        center_crop=args.center_crop,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    def compute_text_embeddings(prompt, text_encoding_pipeline):
        with torch.no_grad():
            prompt_embeds, text_ids = text_encoding_pipeline.encode_prompt(
                prompt=prompt, max_sequence_length=args.max_sequence_length
            )
        return prompt_embeds, text_ids

    static_prompt_embeds = None
    static_text_ids = None
    if not train_dataset.custom_instance_prompts:
        # Reuse one prompt embedding for every batch when the dataset does not override prompts per sample.
        with offload_models(text_encoding_pipeline, device=accelerator.device, offload=args.offload):
            instance_prompt_hidden_states, instance_text_ids = compute_text_embeddings(
                args.instance_prompt, text_encoding_pipeline
            )

    # Init FSDP for text encoder
    if args.fsdp_text_encoder:
        fsdp_kwargs = get_fsdp_kwargs_from_accelerator(accelerator)
        text_encoder_fsdp = wrap_with_fsdp(
            model=text_encoding_pipeline.text_encoder,
            device=accelerator.device,
            offload=args.offload,
            limit_all_gathers=True,
            use_orig_params=True,
            fsdp_kwargs=fsdp_kwargs,
        )

        text_encoding_pipeline.text_encoder = text_encoder_fsdp
        dist.barrier()

    if not train_dataset.custom_instance_prompts:
        static_prompt_embeds = instance_prompt_hidden_states
        static_text_ids = instance_text_ids
        text_encoding_pipeline = text_encoding_pipeline.to("cpu")
        del text_encoding_pipeline, text_encoder, tokenizer
        free_memory()

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * accelerator.num_processes * num_update_steps_per_epoch
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )
    if sem_stage2_mode:
        unwrapped_transformer = unwrap_model(transformer)
        activate_stage2_adapters(unwrapped_transformer, pix_adapter_name, sem_adapter_specs, args.pix_adapter_scale)
        set_stage2_adapter_trainability(
            unwrapped_transformer, pix_adapter_name, sem_adapter_specs, trainable_adapter_names
        )

    transformer_guidance_embeds = bool(getattr(unwrap_model(transformer).config, "guidance_embeds", False))

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "dreambooth-flux2-image2img-lora"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    infer50_timestep_pool = None
    infer50_sigma_pool = None
    infer50_seq_len = None

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()

        for step, batch in enumerate(train_dataloader):
            prompts = batch["prompts"]

            with accelerator.accumulate(transformer):
                if train_dataset.custom_instance_prompts:
                    if args.fsdp_text_encoder:
                        prompt_embeds, text_ids = compute_text_embeddings(prompts, text_encoding_pipeline)
                    else:
                        with offload_models(text_encoding_pipeline, device=accelerator.device, offload=args.offload):
                            prompt_embeds, text_ids = compute_text_embeddings(prompts, text_encoding_pipeline)
                else:
                    num_repeat_elements = len(prompts)
                    prompt_embeds = static_prompt_embeds.repeat(num_repeat_elements, 1, 1)
                    text_ids = static_text_ids.repeat(num_repeat_elements, 1, 1)

                # Convert images to latent space
                with offload_models(vae, device=accelerator.device, offload=args.offload):
                    pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                    cond_pixel_values = batch["cond_pixel_values"].to(dtype=vae.dtype)
                    model_input = vae.encode(pixel_values).latent_dist.mode()
                    cond_model_input = vae.encode(cond_pixel_values).latent_dist.mode()

                model_input = Flux2KleinPipeline._patchify_latents(model_input)
                model_input = (model_input - latents_bn_mean) / latents_bn_std

                cond_model_input = Flux2KleinPipeline._patchify_latents(cond_model_input)
                cond_model_input = (cond_model_input - latents_bn_mean) / latents_bn_std

                model_input_ids = Flux2KleinPipeline._prepare_latent_ids(model_input).to(device=model_input.device)
                cond_model_input_list = [cond_model_input[i].unsqueeze(0) for i in range(cond_model_input.shape[0])]
                cond_model_input_ids = Flux2KleinPipeline._prepare_image_ids(cond_model_input_list).to(
                    device=cond_model_input.device
                )
                cond_model_input_ids = cond_model_input_ids.view(
                    cond_model_input.shape[0], -1, model_input_ids.shape[-1]
                )

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                if args.train_timestep_mode == "infer50_random":
                    image_seq_len = int(model_input.shape[-2] * model_input.shape[-1])
                    if (
                        infer50_timestep_pool is None
                        or infer50_sigma_pool is None
                        or infer50_seq_len != image_seq_len
                    ):
                        infer50_timestep_pool, infer50_sigma_pool = build_inference_timestep_pool(
                            scheduler=noise_scheduler,
                            image_seq_len=image_seq_len,
                            num_inference_steps=50,
                            device=model_input.device,
                            sigma_dtype=model_input.dtype,
                        )
                        infer50_seq_len = image_seq_len
                        if accelerator.is_local_main_process:
                            logger.info(
                                "Initialized infer50 timestep pool for image_seq_len=%s (%s steps).",
                                infer50_seq_len,
                                infer50_timestep_pool.shape[0],
                            )

                    indices = torch.randint(
                        low=0,
                        high=infer50_timestep_pool.shape[0],
                        size=(bsz,),
                        device=model_input.device,
                    )
                    timesteps = infer50_timestep_pool[indices]
                    sigma_values = infer50_sigma_pool[indices]
                    sigmas = sigma_values.view(bsz, *([1] * (model_input.ndim - 1)))
                else:
                    # Sample a random timestep for each image
                    # for weighting schemes where we sample timesteps non-uniformly
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme=args.weighting_scheme,
                        batch_size=bsz,
                        logit_mean=args.logit_mean,
                        logit_std=args.logit_std,
                        mode_scale=args.mode_scale,
                    )
                    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                    timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                    # Add noise according to flow matching.
                    # zt = (1 - texp) * x + texp * z1
                    sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)

                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                # [B, C, H, W] -> [B, H*W, C]
                # concatenate the model inputs with the cond inputs
                packed_noisy_model_input = Flux2KleinPipeline._pack_latents(noisy_model_input)
                packed_cond_model_input = Flux2KleinPipeline._pack_latents(cond_model_input)
                orig_input_shape = packed_noisy_model_input.shape
                orig_input_ids_shape = model_input_ids.shape

                # concatenate the model inputs with the cond inputs
                packed_noisy_model_input = torch.cat([packed_noisy_model_input, packed_cond_model_input], dim=1)
                model_input_ids = torch.cat([model_input_ids, cond_model_input_ids], dim=1)

                # handle guidance
                if transformer_guidance_embeds:
                    guidance = torch.full([1], args.guidance_scale, device=accelerator.device)
                    guidance = guidance.expand(model_input.shape[0])
                else:
                    guidance = None

                # Predict the noise residual
                model_pred = transformer(
                    hidden_states=packed_noisy_model_input,  # (B, image_seq_len, C)
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,  # B, text_seq_len, 4
                    img_ids=model_input_ids,  # B, image_seq_len, 4
                    return_dict=False,
                )[0]
                # pruning the condition information
                model_pred = model_pred[:, : orig_input_shape[1], :]
                model_input_ids = model_input_ids[:, : orig_input_ids_shape[1], :]

                model_pred = Flux2KleinPipeline._unpack_latents_with_ids(model_pred, model_input_ids)

                lambda_fm = args.lambda_fm
                if lambda_fm != 0.0:
                    # these weighting schemes use a uniform timestep sampling
                    # and instead post-weight the loss
                    weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                    # flow matching loss
                    target = noise - model_input

                    # Compute regular loss.
                    loss_fm = torch.mean(
                        (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                        1,
                    )
                    loss_fm = loss_fm.mean()
                else:
                    loss_fm = torch.zeros([], device=model_input.device, dtype=model_pred.dtype)

                # Optional image-space regularization on current-step x0 estimate.
                use_nr_iqa_loss = args.use_nr_iqa_loss
                lambda_q = args.lambda_q
                q_sigma_max = args.q_sigma_max
                q_loss_active = use_nr_iqa_loss and lambda_q != 0.0

                # Keep VAE on GPU through backward when image-space regularizers are active with offload.
                image_reg_active = q_loss_active
                qloss_vae_scope = (
                    offload_models(vae, device=accelerator.device, offload=args.offload)
                    if image_reg_active and args.offload
                    else nullcontext()
                )

                with qloss_vae_scope:
                    loss_q = torch.zeros([], device=model_input.device, dtype=loss_fm.dtype)
                    q_metric_log_values: dict[str, torch.Tensor] = {}
                    if image_reg_active:
                        if q_loss_active and not hasattr(args, "_q_metric_specs"):
                            args._q_metric_specs = _build_q_metric_specs(args, accelerator.device)

                        sigma_per_sample = sigmas.reshape(sigmas.shape[0], -1)[:, 0]
                        q_mask = (sigma_per_sample <= q_sigma_max).float()

                        if q_mask.sum() > 0:
                            # model predicts v_hat = eps - x, so x_hat = z_t - sigma * v_hat
                            x0_hat = noisy_model_input - sigmas * model_pred
                            x0_hat = x0_hat * latents_bn_std.to(dtype=x0_hat.dtype) + latents_bn_mean.to(dtype=x0_hat.dtype)
                            x0_hat = Flux2KleinPipeline._unpatchify_latents(x0_hat)

                            sr_hat = vae.decode(x0_hat.to(dtype=vae.dtype), return_dict=False)[0]
                            sr_hat = (sr_hat / 2 + 0.5).clamp(0, 1)
                            target_pixels = (pixel_values / 2 + 0.5).clamp(0, 1)

                            batch_size = sr_hat.shape[0]
                            q_weighted_terms: list[torch.Tensor] = []
                            for spec in args._q_metric_specs:
                                nan_scalar = torch.full([], float("nan"), device=sr_hat.device, dtype=torch.float32)
                                try:
                                    if spec["name"] == "aesop":
                                        with offload_models(
                                            aesop_autoencoder,
                                            device=accelerator.device,
                                            offload=args.offload,
                                        ) if args.offload else nullcontext():
                                            sr_aesop = aesop_autoencoder(sr_hat.float())
                                            with torch.no_grad():
                                                gt_aesop = aesop_autoencoder(target_pixels.float().detach())
                                        q_score = (sr_aesop - gt_aesop).abs().reshape(batch_size, -1).mean(dim=1)
                                    elif spec["requires_reference"]:
                                        q_score = spec["module"](sr_hat.float(), target_pixels.float())
                                    else:
                                        q_score = spec["module"](sr_hat.float())
                                except Exception as exc:
                                    if _is_linalg_svd_failure(exc):
                                        _warn_q_metric_skip(args, spec["log_name"], f"{type(exc).__name__}: {exc}")
                                        q_metric_log_values[f"q_{spec['log_name']}"] = nan_scalar
                                        q_metric_log_values[f"q_loss_{spec['log_name']}"] = nan_scalar
                                        continue
                                    raise

                                q_score = _reduce_q_score_per_sample(q_score, batch_size=batch_size, device=sr_hat.device)
                                finite_mask = torch.isfinite(q_score).float()
                                valid_mask = q_mask * finite_mask
                                valid_count = valid_mask.sum()
                                if valid_count.item() <= 0:
                                    _warn_q_metric_skip(args, spec["log_name"], "all scores are non-finite")
                                    q_metric_log_values[f"q_{spec['log_name']}"] = nan_scalar
                                    q_metric_log_values[f"q_loss_{spec['log_name']}"] = nan_scalar
                                    continue

                                q_score = torch.nan_to_num(q_score, nan=0.0, posinf=0.0, neginf=0.0)
                                q_score = (q_score * valid_mask).sum() / valid_count.clamp_min(1.0)
                                q_loss_metric = q_score if spec["lower_better"] else -q_score
                                q_scaled = q_loss_metric / spec["scale"]
                                q_weighted = spec["weight"] * q_scaled
                                q_weighted_terms.append(q_weighted.to(dtype=loss_fm.dtype))
                                q_metric_log_values[f"q_{spec['log_name']}"] = q_score.detach()
                                q_metric_log_values[f"q_loss_{spec['log_name']}"] = q_scaled.detach()

                            if q_weighted_terms:
                                loss_q = torch.stack(q_weighted_terms).sum()

                    loss = lambda_fm * loss_fm + lambda_q * loss_q
                    accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process and args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        # before we save the new checkpoint, we need to have at most `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    accelerator.wait_for_everyone()
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        logger.info(f"Saved state to {save_path}")

            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "loss_fm": loss_fm.detach().item(),
                "loss_q": loss_q.detach().item(),
            }
            for key, value in q_metric_log_values.items():
                logs[key] = value.item()
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Save the lora layers
    accelerator.wait_for_everyone()

    if is_fsdp:
        transformer = unwrap_model(transformer)
        state_dict = accelerator.get_state_dict(transformer)
    if accelerator.is_main_process:
        modules_to_save = {}
        if is_fsdp:
            if args.bnb_quantization_config_path is None:
                if args.upcast_before_saving:
                    state_dict = {
                        k: v.to(torch.float32) if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()
                    }
                else:
                    state_dict = {
                        k: v.to(weight_dtype) if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()
                    }

            transformer_lora_layers_by_adapter = {}
            for adapter_name in save_adapter_names:
                transformer_lora_layers = get_adapter_peft_state_dict(
                    transformer,
                    adapter_name=adapter_name,
                    state_dict=state_dict,
                )
                transformer_lora_layers = filter_state_dict_for_adapter(
                    transformer_lora_layers,
                    adapter_name=adapter_name,
                    known_adapter_names=known_adapter_names_for_save,
                )
                transformer_lora_layers_by_adapter[adapter_name] = {
                    k: v.detach().cpu().contiguous() if isinstance(v, torch.Tensor) else v
                    for k, v in transformer_lora_layers.items()
                }

        else:
            transformer = unwrap_model(transformer)
            if args.bnb_quantization_config_path is None:
                if args.upcast_before_saving:
                    transformer.to(torch.float32)
                else:
                    transformer = transformer.to(weight_dtype)
            transformer_lora_layers_by_adapter = {
                adapter_name: filter_state_dict_for_adapter(
                    get_adapter_peft_state_dict(transformer, adapter_name=adapter_name),
                    adapter_name=adapter_name,
                    known_adapter_names=known_adapter_names_for_save,
                )
                for adapter_name in save_adapter_names
            }

        modules_to_save["transformer"] = transformer

        for adapter_name, transformer_lora_layers in transformer_lora_layers_by_adapter.items():
            adapter_output_dir = args.output_dir if len(save_adapter_names) == 1 else os.path.join(args.output_dir, adapter_name)
            Flux2KleinPipeline.save_lora_weights(
                save_directory=adapter_output_dir,
                transformer_lora_layers=transformer_lora_layers,
                **collate_lora_metadata_for_adapter(modules_to_save, adapter_name),
            )

        save_model_card(
            (args.hub_model_id or Path(args.output_dir).name) if not args.push_to_hub else repo_id,
            images=[],
            base_model=args.pretrained_model_name_or_path,
            instance_prompt=args.instance_prompt,
            repo_folder=args.output_dir,
            fp8_training=args.do_fp8_training,
        )

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()
