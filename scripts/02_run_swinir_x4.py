#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

SR_DIR = Path(__file__).resolve().parents[1]
if str(SR_DIR) not in sys.path:
    sys.path.insert(0, str(SR_DIR))

from utils.utils_symlink import make_lr_stage_from_json


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except ValueError:
        raise SystemExit(f"[ERROR] Invalid ${name}={v!r}: expected integer")


def _cli_has(flag: str) -> bool:
    return flag in sys.argv[1:]


def _as_int(v: Any, *, what: str) -> int:
    try:
        return int(v)
    except Exception as ex:
        raise SystemExit(f"[ERROR] {what} must be an integer. Got: {v!r}") from ex


def _load_opt_overrides(opt_path_raw: str) -> Dict[str, Any]:
    opt_path_raw = str(opt_path_raw).strip()
    if not opt_path_raw:
        return {}
    opt_path = Path(opt_path_raw).expanduser().resolve()
    if not opt_path.is_file():
        raise SystemExit(f"[ERROR] opt not found: {opt_path}")
    try:
        import yaml
    except Exception as ex:
        raise SystemExit("[ERROR] PyYAML is required to use --opt in 02_run_swinir_x4.py") from ex
    data = yaml.safe_load(opt_path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SystemExit(f"[ERROR] opt must be a YAML mapping: {opt_path}")
    return data


def main():
    script_dir = Path(__file__).resolve().parent
    sr_dir = (script_dir / "..").resolve()
    workspace_dir = (sr_dir / "../..").resolve()

    default_val_json = os.environ.get("VAL_JSON", str(workspace_dir / "data" / "val_fixed.json"))
    default_repo_dir = str((sr_dir / "repos" / "SwinIR").resolve())

    ap = argparse.ArgumentParser(
        description="Run SwinIR using symlink staging from VAL_JSON. "
        "For classical_sr/lightweight_sr, HR is required (paired). For real_sr, LR-only is fine."
    )
    ap.add_argument("--json", default=default_val_json, help="Path to val_fixed.json ('lr' required, 'hr' optional)")
    ap.add_argument("--repo_dir", default=default_repo_dir, help="Path to SwinIR repo")
    ap.add_argument("--opt", default=os.environ.get("OPT", ""), help="YAML with SwinIR args (optional).")
    ap.add_argument("--stage_root", default=os.environ.get("STAGE_ROOT", "/tmp/sr_stage"), help="Root dir for staging (must allow symlink)")
    ap.add_argument("--keep_stage", action="store_true", help="Do not delete staging folder after run")

    # SwinIR args
    ap.add_argument("--task", default=os.environ.get("TASK", "classical_sr"))
    ap.add_argument("--scale", type=int, default=_env_int("SCALE", 4))
    ap.add_argument("--training_patch_size", type=int, default=_env_int("TRAINING_PATCH_SIZE", 48))
    ap.add_argument(
        "--model_path",
        default=os.environ.get("MODEL_PATH", "model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth"),
    )
    ap.add_argument("--tile", type=int, default=_env_int("TILE", 0))
    ap.add_argument("--tile_overlap", type=int, default=_env_int("TILE_OVERLAP", 32))
    ap.add_argument(
        "--out_dir",
        default=os.environ.get("OUT_DIR", ""),
        help="Output directory for SR images (empty = SwinIR default results/...)",
    )
    ap.add_argument(
        "--meta_dir",
        default=os.environ.get("META_DIR", os.environ.get("PATH_DIR", "")),
        help=(
            "If set, write <meta_dir>/result.json "
            "with {'items':[{'res':..., 'hr':...}, ...], 'timing':{...}}."
        ),
    )
    
    # GPU selection
    ap.add_argument("--gpu", default="", help="CUDA_VISIBLE_DEVICES, e.g. '2' or '0,1'. Empty = do not set.")
    args = ap.parse_args()

    opt_cfg = _load_opt_overrides(args.opt)
    if not _cli_has("--task") and "task" in opt_cfg and str(opt_cfg.get("task", "")).strip():
        args.task = str(opt_cfg["task"]).strip()
    if not _cli_has("--scale") and "scale" in opt_cfg:
        args.scale = _as_int(opt_cfg["scale"], what="opt.scale")
    if not _cli_has("--training_patch_size") and "training_patch_size" in opt_cfg:
        args.training_patch_size = _as_int(opt_cfg["training_patch_size"], what="opt.training_patch_size")
    if not _cli_has("--model_path") and "model_path" in opt_cfg and str(opt_cfg.get("model_path", "")).strip():
        args.model_path = str(opt_cfg["model_path"]).strip()
    if not _cli_has("--tile") and "tile" in opt_cfg:
        args.tile = _as_int(opt_cfg["tile"], what="opt.tile")
    if not _cli_has("--tile_overlap") and "tile_overlap" in opt_cfg:
        args.tile_overlap = _as_int(opt_cfg["tile_overlap"], what="opt.tile_overlap")

    repo_dir = Path(args.repo_dir).resolve()

    val_json = Path(args.json).resolve()
    data = json.loads(val_json.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Invalid val_json format: expected list in {val_json}")

    # Detect basename collisions (would overwrite outputs).
    seen_lr = {}
    for i, e in enumerate(data):
        if not isinstance(e, dict) or "lr" not in e:
            raise KeyError(f"Missing key 'lr' at index {i} in {val_json}")
        name = Path(e["lr"]).name
        if name in seen_lr and str(e.get("lr")) != str(seen_lr[name]):
            raise ValueError(
                "Duplicate LR basename detected in val_json. "
                f"basename={name!r} lr1={seen_lr[name]!r} lr2={e.get('lr')!r}. "
                "This would overwrite outputs; rename files or use unique basenames."
            )
        seen_lr[name] = e.get("lr")

    needs_gt = args.task in ["classical_sr", "lightweight_sr"]

    # 1) Create LR-only symlink staging
    sp = make_lr_stage_from_json(
        val_json=Path(args.json),
        stage_root=Path(args.stage_root),
        name_prefix="swinir",
        prefix_index=False,  # keep original basenames for stable ordering and output filenames
    )

    hr_dir = None
    if needs_gt:
        hr_dir = sp.stage_dir / "HR"
        hr_dir.mkdir(parents=True, exist_ok=True)

        seen_hr = {}
        seen_hr_stem = {}
        for i, e in enumerate(data):
            hr = e.get("hr") if isinstance(e, dict) else None
            if hr is None or str(hr).strip() == "":
                raise SystemExit(
                    f"[ERROR] SwinIR task={args.task!r} needs HR, but entry {i} has no 'hr'. "
                    "Provide paired HR paths in the input JSON, or use --task real_sr for LR-only inference."
                )
            hr_src = Path(str(hr)).expanduser().resolve()
            hr_name = hr_src.name
            if hr_name in seen_hr and str(seen_hr[hr_name]) != str(hr_src):
                raise ValueError(
                    "Duplicate HR basename detected in val_json. "
                    f"basename={hr_name!r} hr1={seen_hr[hr_name]!r} hr2={hr_src!r}. "
                    "This would break SwinIR paired loading; rename files or use unique basenames."
                )
            seen_hr[hr_name] = hr_src
            hr_stem = hr_src.stem
            if hr_stem in seen_hr_stem and str(seen_hr_stem[hr_stem]) != str(hr_src):
                raise ValueError(
                    "Duplicate HR stem detected in val_json (would overwrite SwinIR outputs). "
                    f"stem={hr_stem!r} hr1={seen_hr_stem[hr_stem]!r} hr2={hr_src!r}. "
                    "Rename files to have unique stems."
                )
            seen_hr_stem[hr_stem] = hr_src

            dst = hr_dir / hr_name
            if not (dst.exists() or dst.is_symlink()):
                dst.symlink_to(hr_src)
    else:
        seen_lr_stem = {}
        for i, e in enumerate(data):
            lr_src = Path(str(e["lr"])).expanduser().resolve()
            lr_stem = lr_src.stem
            if lr_stem in seen_lr_stem and str(seen_lr_stem[lr_stem]) != str(lr_src):
                raise ValueError(
                    "Duplicate LR stem detected in val_json (would overwrite SwinIR outputs). "
                    f"stem={lr_stem!r} lr1={seen_lr_stem[lr_stem]!r} lr2={lr_src!r}. "
                    "Rename files to have unique stems."
                )
            seen_lr_stem[lr_stem] = lr_src

    # 2) Build command
    cmd = [
        sys.executable, "main_test_swinir.py",
        "--task", args.task,
        "--scale", str(args.scale),
        "--training_patch_size", str(args.training_patch_size),
        "--model_path", args.model_path,
        "--folder_lq", str(sp.lr_dir),
        "--tile_overlap", str(args.tile_overlap),
    ]
    if needs_gt and hr_dir is not None:
        cmd += ["--folder_gt", str(hr_dir)]

    out_dir_raw = args.out_dir.strip()
    meta_dir_raw = args.meta_dir.strip()

    out_dir = str(Path(out_dir_raw).expanduser().resolve()) if out_dir_raw else ""
    meta_dir = str(Path(meta_dir_raw).expanduser().resolve()) if meta_dir_raw else ""
    if not meta_dir and out_dir:
        meta_dir = out_dir
    timing_json_path = (Path(meta_dir).resolve() / "inference_timing.json") if meta_dir else None

    if args.tile and args.tile > 0:
        cmd += ["--tile", str(args.tile)]

    run_env = os.environ.copy()
    run_env["PYTHONPATH"] = f"{repo_dir}{os.pathsep}{run_env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    if args.gpu.strip():
        run_env["CUDA_VISIBLE_DEVICES"] = args.gpu.strip()
    if timing_json_path is not None:
        run_env["SR_TIMING_OUT"] = str(timing_json_path)
        try:
            timing_json_path.unlink()
        except FileNotFoundError:
            pass

    print("[INFO] repo_dir :", repo_dir)
    if str(args.opt).strip():
        print("[INFO] opt      :", Path(args.opt).expanduser().resolve())
    print("[INFO] stage_lr :", sp.lr_dir)
    if needs_gt and hr_dir is not None:
        print("[INFO] stage_hr :", hr_dir)
    if out_dir:
        print("[INFO] out_dir  :", out_dir)
    if meta_dir:
        print("[INFO] meta_dir :", meta_dir)
    if args.gpu.strip():
        print("[INFO] CUDA_VISIBLE_DEVICES =", run_env["CUDA_VISIBLE_DEVICES"])
    print("[INFO] cmd      :", " ".join(cmd))

    try:
        subprocess.run(cmd, cwd=str(repo_dir), check=True, env=run_env)

        if args.task in ["classical_sr", "lightweight_sr", "real_sr"]:
            raw_dir = (repo_dir / "results" / f"swinir_{args.task}_x{args.scale}").resolve()
        elif args.task in ["gray_dn", "color_dn"]:
            raw_dir = (repo_dir / "results" / f"swinir_{args.task}_noise15").resolve()
        elif args.task in ["jpeg_car", "color_jpeg_car"]:
            raw_dir = (repo_dir / "results" / f"swinir_{args.task}_jpeg40").resolve()
        else:
            raw_dir = (repo_dir / "results").resolve()
        if not raw_dir.exists():
            raise FileNotFoundError(f"Expected output folder not found: {raw_dir}")

        final_dir = Path(out_dir).resolve() if out_dir else raw_dir
        final_dir.mkdir(parents=True, exist_ok=True)

        for e in data:
            lr_name = Path(e["lr"]).name
            raw_stem = Path(e["hr"]).stem if needs_gt else Path(e["lr"]).stem
            raw_file = raw_dir / f"{raw_stem}_SwinIR.png"
            if not raw_file.exists():
                raise FileNotFoundError(f"Expected output image not found: {raw_file}")
            dst = final_dir / lr_name
            if final_dir == raw_dir:
                os.replace(raw_file, dst)
            else:
                shutil.copy2(raw_file, dst)

        if meta_dir:
            out_root = final_dir
            items = []
            for e in data:
                lr_name = Path(e["lr"]).name
                it = {"res": str((out_root / lr_name).resolve())}
                hr = e.get("hr", None) if isinstance(e, dict) else None
                if hr is not None and str(hr).strip() != "":
                    it["hr"] = hr
                items.append(it)

            items.sort(key=lambda x: Path(x["res"]).name)

            meta_dir_p = Path(meta_dir).resolve()
            meta_dir_p.mkdir(parents=True, exist_ok=True)
            out_json = meta_dir_p / "result.json"
            payload = {"items": items}
            if timing_json_path is not None and timing_json_path.is_file():
                try:
                    payload["timing"] = json.loads(timing_json_path.read_text(encoding="utf-8"))
                except Exception as ex:
                    print(f"[WARN] Failed to read timing JSON: {timing_json_path} ({ex})")
            out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"[INFO] wrote: {out_json}")
    finally:
        if not args.keep_stage:
            shutil.rmtree(sp.stage_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
