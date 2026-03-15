import argparse
import os


def _parse_csv_items(value):
    if value is None:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _format_metric_weight(value):
    return format(float(value), "g")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--bnb_quantization_config_path",
        type=str,
        default=None,
        help="Quantization config in a JSON file that will be used to define the bitsandbytes quant config of the DiT.",
    )
    parser.add_argument(
        "--do_fp8_training",
        action="store_true",
        help="if we are doing FP8 training.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--lora_weights_path",
        type=str,
        default=None,
        help=(
            "Optional path to existing LoRA weights to load before training. If omitted, training starts from a new "
            "LoRA adapter."
        ),
    )
    parser.add_argument(
        "--train_sem_only",
        action="store_true",
        help=(
            "Enable stage2 semantic LoRA training mode. In this mode, the existing pixel LoRA is frozen and only the "
            "semantic LoRA adapter is optimized using Q-loss."
        ),
    )
    parser.add_argument(
        "--pix_lora_weights_path",
        type=str,
        default=None,
        help="Path to an existing stage1 pixel LoRA to load into the frozen `pix` adapter in stage2 mode.",
    )
    parser.add_argument(
        "--sem_lora_weights_path",
        type=str,
        default=None,
        help="Optional path to initialize the trainable `sem` adapter before stage2 training.",
    )
    parser.add_argument(
        "--pix_adapter_name",
        type=str,
        default="pix",
        help="Adapter name used for the frozen pixel LoRA in stage2 mode.",
    )
    parser.add_argument(
        "--sem_adapter_name",
        type=str,
        default="sem",
        help="Adapter name used for the trainable semantic LoRA in stage2 mode.",
    )
    parser.add_argument(
        "--sem2_lora_weights_path",
        type=str,
        default=None,
        help="Optional path to initialize an additional semantic adapter (`sem2`) before stage2 training.",
    )
    parser.add_argument(
        "--sem2_adapter_name",
        type=str,
        default=None,
        help="Optional second semantic adapter name for stage2 training.",
    )
    parser.add_argument(
        "--pix_adapter_scale",
        type=float,
        default=1.0,
        help="Runtime adapter scale used for the frozen `pix` adapter in stage2 mode.",
    )
    parser.add_argument(
        "--sem_adapter_scale",
        type=float,
        default=1.0,
        help="Runtime adapter scale used for the trainable `sem` adapter in stage2 mode.",
    )
    parser.add_argument(
        "--sem2_adapter_scale",
        type=float,
        default=1.0,
        help="Runtime adapter scale used for the optional `sem2` adapter in stage2 mode.",
    )
    parser.add_argument(
        "--sem_adapter_names",
        type=str,
        default=None,
        help="Optional comma-separated semantic adapter names for multi-sem stage2 training.",
    )
    parser.add_argument(
        "--sem_lora_weights_paths",
        type=str,
        default=None,
        help=(
            "Optional comma-separated semantic adapter initialization paths aligned with `--sem_adapter_names`. "
            "Use empty items or 'none' to skip initialization for a given adapter."
        ),
    )
    parser.add_argument(
        "--sem_adapter_scales",
        type=str,
        default=None,
        help="Optional comma-separated runtime scales aligned with `--sem_adapter_names`.",
    )
    parser.add_argument(
        "--sem_trainable_adapter_names",
        type=str,
        default=None,
        help=(
            "Optional comma-separated subset of semantic adapter names to optimize. "
            "Defaults to all semantic adapters."
        ),
    )
    parser.add_argument(
        "--sem_rank",
        type=int,
        default=None,
        help="Optional rank override for semantic adapters when stage2 multi-sem mode is enabled.",
    )
    parser.add_argument(
        "--sem2_rank",
        type=int,
        default=None,
        help="Optional rank override for the legacy `sem2` adapter alias.",
    )
    parser.add_argument(
        "--sem_ranks",
        type=str,
        default=None,
        help="Optional comma-separated semantic adapter ranks aligned with `--sem_adapter_names`.",
    )
    parser.add_argument(
        "--sem_lora_alpha",
        type=int,
        default=None,
        help="Optional alpha override for semantic adapters when stage2 multi-sem mode is enabled.",
    )
    parser.add_argument(
        "--sem2_lora_alpha",
        type=int,
        default=None,
        help="Optional alpha override for the legacy `sem2` adapter alias.",
    )
    parser.add_argument(
        "--sem_lora_alphas",
        type=str,
        default=None,
        help="Optional comma-separated semantic adapter alpha values aligned with `--sem_adapter_names`.",
    )
    parser.add_argument(
        "--train_data_json",
        type=str,
        required=True,
        help=(
            "Path to a local JSON array of training samples. Each sample must contain `hr` and either `res` or `lr`, "
            "and may optionally include a per-sample `prompt`."
        ),
    )

    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")

    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=False,
        help="Fallback prompt used when a sample in `--train_data_json` does not define `prompt`.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=4,
        help="LoRA alpha to be used for additional scaling.",
    )
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="Dropout probability for LoRA layers")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="flux-dreambooth-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Fixed square patch size used for both HR and condition inputs during training.",
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help="Use a center crop for the fixed patch instead of random crop coordinates.",
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="the FLUX.1 dev variant is a guidance distilled model",
    )

    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--train_timestep_mode",
        type=str,
        default="random",
        choices=["random", "infer50_random"],
        help="Timestep sampling mode for training (`random` or inference-aligned 50-step random pool).",
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--use_nr_iqa_loss",
        action="store_true",
        help="Enable Q-loss regularization on current-step x0 estimates using local_iqa metrics.",
    )
    parser.add_argument(
        "--lambda_q",
        type=float,
        default=0.0,
        help="Weight for NR-IQA regularization term. 0.0 disables the term.",
    )
    parser.add_argument(
        "--lambda_fm",
        type=float,
        default=1.0,
        help="Weight for flow-matching loss term. 0.0 disables the term.",
    )
    parser.add_argument(
        "--q_sigma_max",
        type=float,
        default=0.4,
        help="Apply NR-IQA regularization only to samples with sigma <= q_sigma_max.",
    )
    parser.add_argument(
        "--aesop_autoencoder_path",
        type=str,
        default=None,
        help="Path to the pretrained AESOP autoencoder checkpoint when `nr_iqa_metric` includes `aesop`.",
    )
    parser.add_argument(
        "--aesop_autoencoder_key",
        type=str,
        default="params_ema",
        help="State-dict key inside the AESOP autoencoder checkpoint.",
    )
    parser.add_argument(
        "--nr_iqa_metric",
        type=str,
        default="musiq",
        help=(
            "Q-loss metric label or comma-separated metric labels. Supported labels: "
            "NIQE, ManIQA, MUSIQ, L2, LPIPS, DISTS, AESOP. (CLIP-IQA excluded)"
        ),
    )
    parser.add_argument(
        "--q_metric_weights",
        type=str,
        default=None,
        help=(
            "Optional comma-separated per-metric weights aligned with `--nr_iqa_metric`. "
            "Defaults to 1.0 for each metric."
        ),
    )
    parser.add_argument(
        "--iqa_metric1",
        type=str,
        default=None,
        help="Primary Q-loss metric label. Use this instead of `--nr_iqa_metric` for the new config format.",
    )
    parser.add_argument(
        "--iqa_metric1_weight",
        type=float,
        default=None,
        help="Optional weight for `--iqa_metric1`. Defaults to 1.0.",
    )
    parser.add_argument(
        "--iqa_metric2",
        type=str,
        default=None,
        help="Optional secondary Q-loss metric label. Omit it to use single-metric Q-loss.",
    )
    parser.add_argument(
        "--iqa_metric2_weight",
        type=float,
        default=None,
        help="Optional weight for `--iqa_metric2`. Defaults to 1.0.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder"
    )

    parser.add_argument(
        "--lora_layers",
        type=str,
        default=None,
        help=(
            'The transformer modules to apply LoRA training on. Please specify the layers in a comma separated. E.g. - "to_k,to_q,to_v,to_out.0" will result in lora training of attention layers only'
        ),
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--cache_latents",
        action="store_true",
        default=False,
        help="Cache the VAE latents",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--upcast_before_saving",
        action="store_true",
        default=False,
        help=(
            "Whether to upcast the trained transformer layers to float32 before saving (at the end of training). "
            "Defaults to precision dtype used for training to save memory"
        ),
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        help="Whether to offload the VAE and the text encoder to CPU when they are not used.",
    )

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--enable_npu_flash_attention", action="store_true", help="Enabla Flash Attention for NPU")
    parser.add_argument("--fsdp_text_encoder", action="store_true", help="Use FSDP for text encoder")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.resolution <= 0:
        raise ValueError("`--resolution` must be a positive integer.")

    new_metric_fields = (
        args.iqa_metric1,
        args.iqa_metric2,
        args.iqa_metric1_weight,
        args.iqa_metric2_weight,
    )
    uses_new_iqa_fields = any(value is not None for value in new_metric_fields)
    uses_legacy_iqa_fields = args.nr_iqa_metric != parser.get_default("nr_iqa_metric") or args.q_metric_weights is not None

    if uses_new_iqa_fields and uses_legacy_iqa_fields:
        raise ValueError(
            "Use either `--iqa_metric1/2` or legacy `--nr_iqa_metric/--q_metric_weights`, not both."
        )

    if uses_new_iqa_fields:
        if not args.iqa_metric1 or not str(args.iqa_metric1).strip():
            raise ValueError("`--iqa_metric1` is required when using the new IQA metric fields.")
        if args.iqa_metric2_weight is not None and (not args.iqa_metric2 or not str(args.iqa_metric2).strip()):
            raise ValueError("`--iqa_metric2_weight` requires `--iqa_metric2`.")

        q_metrics = [str(args.iqa_metric1).strip()]
        q_metric_weights = [_format_metric_weight(args.iqa_metric1_weight or 1.0)]
        if args.iqa_metric2 and str(args.iqa_metric2).strip():
            q_metrics.append(str(args.iqa_metric2).strip())
            q_metric_weights.append(_format_metric_weight(args.iqa_metric2_weight or 1.0))

        args.nr_iqa_metric = ",".join(q_metrics)
        args.q_metric_weights = ",".join(q_metric_weights)

    q_metrics = _parse_csv_items(args.nr_iqa_metric)
    if not q_metrics:
        raise ValueError("`--nr_iqa_metric` must contain at least one metric.")

    for field_name in ("q_metric_weights",):
        field_value = getattr(args, field_name)
        field_items = _parse_csv_items(field_value)
        if field_items and len(field_items) not in {1, len(q_metrics)}:
            raise ValueError(
                f"`--{field_name}` expects either 1 value or {len(q_metrics)} values to match `--nr_iqa_metric`."
            )

    if args.train_sem_only:
        if args.pix_lora_weights_path is None and args.lora_weights_path is None:
            raise ValueError(
                "Stage2 mode requires an existing pixel LoRA. Set `--pix_lora_weights_path` (or legacy `--lora_weights_path`)."
            )

        sem_adapter_names = _parse_csv_items(args.sem_adapter_names)
        if not sem_adapter_names:
            sem_adapter_names = [args.sem_adapter_name]
            if args.sem2_adapter_name and str(args.sem2_adapter_name).strip():
                sem_adapter_names.append(str(args.sem2_adapter_name).strip())

        if any(not name for name in sem_adapter_names):
            raise ValueError("Semantic adapter names must be non-empty strings.")

        if len(set(sem_adapter_names)) != len(sem_adapter_names):
            raise ValueError(f"Semantic adapter names must be unique. Got: {sem_adapter_names}")

        if args.pix_adapter_name in set(sem_adapter_names):
            raise ValueError("`--pix_adapter_name` must be different from every semantic adapter name.")

        trainable_adapter_names = _parse_csv_items(args.sem_trainable_adapter_names) or sem_adapter_names
        if len(set(trainable_adapter_names)) != len(trainable_adapter_names):
            raise ValueError(f"Trainable semantic adapter names must be unique. Got: {trainable_adapter_names}")

        missing_trainable_names = [name for name in trainable_adapter_names if name not in set(sem_adapter_names)]
        if missing_trainable_names:
            raise ValueError(
                "`--sem_trainable_adapter_names` must be a subset of the semantic adapter names. "
                f"Unknown names: {missing_trainable_names}"
            )

        if not trainable_adapter_names:
            raise ValueError("Stage2 multi-sem mode requires at least one trainable semantic adapter.")

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
