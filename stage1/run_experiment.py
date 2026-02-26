#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

VALID_ENVS = ("sr", "dat", "eval")


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"[ERROR] JSON not found: {path}")
    except json.JSONDecodeError as e:
        raise SystemExit(f"[ERROR] Invalid JSON: {path}\n{e}") from e


def _expect_dict(obj: Any, *, what: str) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        raise SystemExit(f"[ERROR] {what} must be a JSON object (dict). Got: {type(obj).__name__}")
    return obj


def _expect_str(obj: Any, *, what: str) -> str:
    if not isinstance(obj, str) or obj.strip() == "":
        raise SystemExit(f"[ERROR] {what} must be a non-empty string.")
    return obj


def _expect_optional_str(obj: Any, *, what: str) -> str:
    if obj is None:
        return ""
    if not isinstance(obj, str):
        raise SystemExit(f"[ERROR] {what} must be a string when provided.")
    return obj.strip()


def _normalize_env_name(raw: str, *, what: str) -> str:
    name = raw.strip()
    if not name:
        return ""
    if name not in VALID_ENVS:
        raise SystemExit(f"[ERROR] {what} must be one of {list(VALID_ENVS)}. Got: {name!r}")
    return name


def _python_prefix_for_env(env_name: str, *, fallback_python: str) -> List[str]:
    if not env_name:
        return [fallback_python]

    # Prefer short wrappers when available (sr/dat/eval python ...).
    env_launcher = shutil.which(env_name)
    if env_launcher:
        return [env_launcher, "python"]

    # Fallback: conda run -n <env> python ...
    conda = shutil.which("conda")
    if conda:
        return [conda, "run", "-n", env_name, "--no-capture-output", "python"]

    raise SystemExit(
        f"[ERROR] Cannot launch env={env_name!r}. "
        f"Neither `{env_name}` launcher nor `conda` command was found in PATH."
    )


def _env_label(env_name: str) -> str:
    return env_name if env_name else "current"


def _script_supports_flag(model_script: Path, flag: str) -> bool:
    text = model_script.read_text(encoding="utf-8", errors="ignore")
    pattern = rf"add_argument\(\s*['\"]{re.escape(flag)}['\"]"
    return re.search(pattern, text) is not None


def _resolve_script_path(path_str: str, sr_dir: Path) -> Path:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = sr_dir / p
    return p.resolve()


def _resolve_workspace_path(path_str: str, workspace_dir: Path) -> Path:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = workspace_dir / p
    return p.resolve()


def _detect_weight_flag(model_script: Path) -> str:
    """
    Our 02_run_*.py wrappers use one of:
      - --weight
      - --model_path
    """
    if _script_supports_flag(model_script, "--weight"):
        return "--weight"
    if _script_supports_flag(model_script, "--model_path"):
        return "--model_path"
    raise SystemExit(
        "[ERROR] Cannot infer weight argument for model script.\n"
        f"  script: {model_script}\n"
        "Expected argparse to define either --weight or --model_path."
    )


def _load_registry(reg_path: Path, sr_dir: Path) -> Tuple[Dict[str, Path], Dict[str, Tuple[Path, Optional[Path]]]]:
    reg = _expect_dict(_load_json(reg_path), what="model_registry")
    model_map = _expect_dict(reg.get("model"), what="model_registry.model")
    weight_map = _expect_dict(reg.get("weight"), what="model_registry.weight")

    models: Dict[str, Path] = {}
    for k, v in model_map.items():
        key = _expect_str(k, what="model key")
        path_str = _expect_str(v, what=f"model_registry.model[{key!r}] path")
        p = _resolve_script_path(path_str, sr_dir)
        if not p.is_file():
            raise SystemExit(f"[ERROR] Model script not found: key={key!r} path={p}")
        models[key] = p

    weights: Dict[str, Tuple[Path, Optional[Path]]] = {}
    for k, v in weight_map.items():
        key = _expect_str(k, what="weight key")
        weight_path_str = ""
        opt_path_str = ""
        if isinstance(v, str):
            weight_path_str = _expect_str(v, what=f"model_registry.weight[{key!r}] path")
        elif isinstance(v, dict):
            weight_path_str = _expect_str(v.get("weight_path"), what=f"model_registry.weight[{key!r}].weight_path")
            opt_path_str = _expect_optional_str(v.get("opt_path"), what=f"model_registry.weight[{key!r}].opt_path")
        else:
            raise SystemExit(
                f"[ERROR] model_registry.weight[{key!r}] must be string path or object "
                "{weight_path,opt_path}."
            )

        weight_path = _resolve_script_path(weight_path_str, sr_dir)
        if not weight_path.is_file():
            raise SystemExit(f"[ERROR] Weight not found: key={key!r} path={weight_path}")

        opt_path: Optional[Path] = None
        if opt_path_str:
            opt_path = _resolve_script_path(opt_path_str, sr_dir)
            if not opt_path.is_file():
                raise SystemExit(f"[ERROR] Opt not found: key={key!r} opt_path={opt_path}")

        weights[key] = (weight_path, opt_path)

    return models, weights


def main() -> None:
    sr_dir = Path(__file__).resolve().parent
    stage1_dir = sr_dir.resolve()
    scripts_dir = (stage1_dir / "scripts").resolve()
    eval_dir = (stage1_dir.parent / "eval").resolve()
    workspace_dir = (sr_dir / "../../..").resolve()

    ap = argparse.ArgumentParser(description="Run multiple SR tests from experiment JSON + model registry JSON.")
    ap.add_argument("--exp", required=True, help="Path to experiment JSON (data_input/output_path/setting).")
    ap.add_argument(
        "--models",
        default=str(sr_dir / "model_registry.json"),
        help="Path to model registry JSON (default: model_registry.json).",
    )
    ap.add_argument("--dry_run", action="store_true", help="Print commands only; do not execute.")
    ap.add_argument("--gpu", default="", help="CUDA_VISIBLE_DEVICES to pass through to each 02_run_*.py wrapper.")
    ap.add_argument(
        "--default_env",
        default="",
        help="Default execution env for model runs: sr|dat|eval. Empty = current python.",
    )
    ap.add_argument("--eval_env", default="", help="Execution env for evaluation stage: sr|dat|eval.")
    ap.add_argument("--viz_env", default="", help="Execution env for visualization stage: sr|dat|eval.")
    ap.add_argument("--stage_root", default="", help="Override --stage_root for each 02_run_*.py wrapper (optional).")
    ap.add_argument("--keep_stage", action="store_true", help="Pass --keep_stage to each 02_run_*.py wrapper.")

    ap.add_argument("--skip_eval", action="store_true", help="Skip running eval/03_evaluation.py")
    ap.add_argument("--skip_viz", action="store_true", help="Skip running eval/04_visualization.py")
    ap.add_argument("--eval_device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--fr_resize", default="to_ref", choices=["to_ref", "none"])
    ap.add_argument("--viz_idx", type=int, nargs="+", default=None, help="Indices to render row images in visualization.")
    args = ap.parse_args()

    exp_path = Path(args.exp).expanduser().resolve()
    exp = _expect_dict(_load_json(exp_path), what="experiment")

    data_input = _expect_str(exp.get("data_input"), what="experiment.data_input")
    output_path = _expect_str(exp.get("output_path"), what="experiment.output_path")
    exp_default_env = _normalize_env_name(
        _expect_optional_str(exp.get("default_env"), what="experiment.default_env"),
        what="experiment.default_env",
    )
    exp_eval_env = _normalize_env_name(
        _expect_optional_str(exp.get("eval_env"), what="experiment.eval_env"),
        what="experiment.eval_env",
    )
    exp_viz_env = _normalize_env_name(
        _expect_optional_str(exp.get("viz_env"), what="experiment.viz_env"),
        what="experiment.viz_env",
    )

    cli_default_env = _normalize_env_name(_expect_optional_str(args.default_env, what="--default_env"), what="--default_env")
    cli_eval_env = _normalize_env_name(_expect_optional_str(args.eval_env, what="--eval_env"), what="--eval_env")
    cli_viz_env = _normalize_env_name(_expect_optional_str(args.viz_env, what="--viz_env"), what="--viz_env")

    default_env = cli_default_env or exp_default_env
    eval_env = cli_eval_env or exp_eval_env or default_env
    viz_env = cli_viz_env or exp_viz_env or eval_env

    setting = exp.get("setting")
    if not isinstance(setting, list) or len(setting) == 0:
        raise SystemExit("[ERROR] experiment.setting must be a non-empty list.")

    data_json = _resolve_workspace_path(data_input, workspace_dir)
    if not data_json.is_file():
        raise SystemExit(f"[ERROR] data_input not found: {data_json}")

    out_root = _resolve_workspace_path(output_path, workspace_dir)
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    reg_path = Path(args.models).expanduser().resolve()
    models, weights = _load_registry(reg_path, sr_dir)

    eval_py = (eval_dir / "03_evaluation.py").resolve()
    viz_py = (eval_dir / "04_visualization.py").resolve()
    if not eval_py.is_file():
        raise SystemExit(f"[ERROR] missing: {eval_py}")
    if not viz_py.is_file():
        raise SystemExit(f"[ERROR] missing: {viz_py}")

    seen_tests = set()

    for idx, s in enumerate(setting):
        if not isinstance(s, dict):
            raise SystemExit(f"[ERROR] experiment.setting[{idx}] must be an object.")

        test_name = _expect_str(s.get("test_name"), what=f"experiment.setting[{idx}].test_name")
        model_key = _expect_str(s.get("model"), what=f"experiment.setting[{idx}].model")
        weight_key = _expect_str(s.get("weight"), what=f"experiment.setting[{idx}].weight")
        setting_env = _normalize_env_name(
            _expect_optional_str(s.get("env"), what=f"experiment.setting[{idx}].env"),
            what=f"experiment.setting[{idx}].env",
        )
        setting_opt_raw = _expect_optional_str(s.get("opt"), what=f"experiment.setting[{idx}].opt")
        run_env = setting_env or default_env
        run_py = _python_prefix_for_env(run_env, fallback_python=sys.executable)

        if test_name in seen_tests:
            raise SystemExit(f"[ERROR] Duplicate test_name: {test_name!r}")
        seen_tests.add(test_name)

        if model_key not in models:
            raise SystemExit(f"[ERROR] Unknown model key: {model_key!r}. Available: {sorted(models.keys())}")
        if weight_key not in weights:
            raise SystemExit(f"[ERROR] Unknown weight key: {weight_key!r}. Available: {sorted(weights.keys())}")

        model_script = models[model_key]
        weight_path, registry_opt_path = weights[weight_key]
        setting_opt_path: Optional[Path] = None
        if setting_opt_raw:
            setting_opt_path = _resolve_script_path(setting_opt_raw, sr_dir)
            if not setting_opt_path.is_file():
                raise SystemExit(
                    f"[ERROR] opt not found: setting={test_name!r} key={weight_key!r} opt={setting_opt_path}"
                )
        opt_path = setting_opt_path if setting_opt_path is not None else registry_opt_path
        weight_flag = _detect_weight_flag(model_script)
        supports_opt = _script_supports_flag(model_script, "--opt")

        test_dir = (out_root / test_name).resolve()
        if not args.dry_run:
            test_dir.mkdir(parents=True, exist_ok=True)

        # Always store images + result.json under the per-test directory.
        cmd: List[str] = [
            *run_py,
            str(model_script),
            "--json",
            str(data_json),
            weight_flag,
            str(weight_path),
            "--out_dir",
            str(test_dir),
            "--meta_dir",
            str(test_dir),
        ]
        if supports_opt:
            if opt_path is None:
                raise SystemExit(
                    f"[ERROR] model script requires --opt but no opt configured: model={model_key!r}, weight={weight_key!r}"
                )
            cmd += ["--opt", str(opt_path)]
        elif setting_opt_raw:
            print(
                f"[WARN] setting.opt is ignored because script has no --opt: "
                f"model={model_key!r} script={model_script.name}"
            )
        if args.gpu.strip():
            cmd += ["--gpu", args.gpu.strip()]
        if args.stage_root.strip():
            cmd += ["--stage_root", args.stage_root.strip()]
        if args.keep_stage:
            cmd += ["--keep_stage"]

        print()
        print(f"[RUN] {test_name}  model={model_key}  weight={weight_key}  env={_env_label(run_env)}")
        print("      " + " ".join(cmd))

        if not args.dry_run:
            subprocess.run(cmd, check=True)

        result_json = test_dir / "result.json"
        if not args.dry_run and not result_json.is_file():
            raise SystemExit(f"[ERROR] Missing result.json after run: {result_json}")

        if not args.skip_eval:
            eval_out = (out_root / f"{test_name}.eval.json").resolve()
            eval_py_cmd = _python_prefix_for_env(eval_env, fallback_python=sys.executable)
            eval_cmd = [
                *eval_py_cmd,
                str(eval_py),
                "--input",
                str(result_json),
                "--out",
                str(eval_out),
                "--device",
                args.eval_device,
                "--fr_resize",
                args.fr_resize,
            ]
            print(f"[EVAL] {test_name}  env={_env_label(eval_env)}")
            print("      " + " ".join(eval_cmd))
            if not args.dry_run:
                subprocess.run(eval_cmd, check=True)

    if args.skip_viz:
        return
    if args.skip_eval:
        print("[WARN] skip_eval=true: visualization will use any existing *.eval.json under output_path.")

    viz_out_dir = (out_root / "viz").resolve()
    viz_py_cmd = _python_prefix_for_env(viz_env, fallback_python=sys.executable)
    viz_cmd = [*viz_py_cmd, str(viz_py), str(out_root), str(viz_out_dir)]
    if args.viz_idx:
        viz_cmd += ["--idx", *[str(int(x)) for x in args.viz_idx]]
    print()
    print(f"[VIZ] env={_env_label(viz_env)} " + " ".join(viz_cmd))
    if not args.dry_run:
        subprocess.run(viz_cmd, check=True)


if __name__ == "__main__":
    main()
