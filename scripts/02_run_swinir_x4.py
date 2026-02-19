#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from utils import make_lr_stage_from_json


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except ValueError:
        raise SystemExit(f"[ERROR] Invalid ${name}={v!r}: expected integer")


def main():
    script_dir = Path(__file__).resolve().parent
    sr_dir = (script_dir / "..").resolve()
    workspace_dir = (sr_dir / "../..").resolve()

    default_val_json = os.environ.get("VAL_JSON", str(workspace_dir / "data" / "val_fixed.json"))
    default_repo_dir = str((sr_dir / "repos" / "SwinIR").resolve())

    ap = argparse.ArgumentParser(description="Run SwinIR inference using LR-only symlink staging from VAL_JSON (NO HR).")
    ap.add_argument("--json", default=default_val_json, help="Path to val_fixed.json (only 'lr' is used)")
    ap.add_argument("--repo_dir", default=default_repo_dir, help="Path to SwinIR repo")
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
        help="If set, write <meta_dir>/result.json with [{'res':..., 'hr':...}, ...].",
    )
    
    # GPU selection
    ap.add_argument("--gpu", default="", help="CUDA_VISIBLE_DEVICES, e.g. '2' or '0,1'. Empty = do not set.")
    args = ap.parse_args()

    repo_dir = Path(args.repo_dir).resolve()

    # 1) Create LR-only symlink staging
    sp = make_lr_stage_from_json(
        val_json=Path(args.json),
        stage_root=Path(args.stage_root),
        name_prefix="swinir",
        prefix_index=False,  # keep original basenames for stable ordering and output filenames
    )

    # 2) Build command (NO HR / NO folder_gt / NO save_img_only)
    cmd = [
        sys.executable, "main_test_swinir.py",
        "--task", args.task,
        "--scale", str(args.scale),
        "--training_patch_size", str(args.training_patch_size),
        "--model_path", args.model_path,
        "--folder_lq", str(sp.lr_dir),
        "--tile_overlap", str(args.tile_overlap),
    ]

    out_dir_raw = args.out_dir.strip()
    meta_dir_raw = args.meta_dir.strip()

    out_dir = str(Path(out_dir_raw).expanduser().resolve()) if out_dir_raw else ""
    meta_dir = str(Path(meta_dir_raw).expanduser().resolve()) if meta_dir_raw else ""

    if out_dir:
        cmd += ["--save_dir", out_dir]
    if args.tile and args.tile > 0:
        cmd += ["--tile", str(args.tile)]

    run_env = os.environ.copy()
    if args.gpu.strip():
        run_env["CUDA_VISIBLE_DEVICES"] = args.gpu.strip()

    print("[INFO] repo_dir :", repo_dir)
    print("[INFO] stage_lr :", sp.lr_dir)
    if out_dir:
        print("[INFO] out_dir  :", out_dir)
    if meta_dir:
        print("[INFO] meta_dir :", meta_dir)
    if args.gpu.strip():
        print("[INFO] CUDA_VISIBLE_DEVICES =", run_env["CUDA_VISIBLE_DEVICES"])
    print("[INFO] cmd      :", " ".join(cmd))

    try:
        subprocess.run(cmd, cwd=str(repo_dir), check=True, env=run_env)

        if meta_dir:
            val_json = Path(args.json).resolve()
            data = json.loads(val_json.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                raise ValueError(f"Invalid val_json format: expected list in {val_json}")

            # Detect basename collisions (would overwrite outputs).
            seen = {}
            for i, e in enumerate(data):
                if not isinstance(e, dict) or "lr" not in e:
                    raise KeyError(f"Missing key 'lr' at index {i} in {val_json}")
                name = Path(e["lr"]).name
                if name in seen and str(e.get("lr")) != str(seen[name]):
                    raise ValueError(
                        "Duplicate LR basename detected in val_json. "
                        f"basename={name!r} lr1={seen[name]!r} lr2={e.get('lr')!r}. "
                        "This would overwrite outputs; rename files or use unique basenames."
                    )
                seen[name] = e.get("lr")

            if out_dir:
                out_root = Path(out_dir).resolve()
            else:
                if args.task in ["classical_sr", "lightweight_sr", "real_sr"]:
                    out_root = (repo_dir / "results" / f"swinir_{args.task}_x{args.scale}").resolve()
                elif args.task in ["gray_dn", "color_dn"]:
                    out_root = (repo_dir / "results" / f"swinir_{args.task}_noise15").resolve()
                elif args.task in ["jpeg_car", "color_jpeg_car"]:
                    out_root = (repo_dir / "results" / f"swinir_{args.task}_jpeg40").resolve()
                else:
                    out_root = (repo_dir / "results").resolve()
            items = []
            for e in data:
                lr_name = Path(e["lr"]).name
                it = {"res": str((out_root / lr_name).resolve())}
                hr = e.get("hr", None) if isinstance(e, dict) else None
                if hr is not None and str(hr).strip() != "":
                    it["hr"] = hr
                items.append(it)

            # Match processing order (sorted by filename).
            items.sort(key=lambda x: Path(x["res"]).name)

            meta_dir_p = Path(meta_dir).resolve()
            meta_dir_p.mkdir(parents=True, exist_ok=True)
            out_json = meta_dir_p / "result.json"
            out_json.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"[INFO] wrote: {out_json}")
    finally:
        if not args.keep_stage:
            shutil.rmtree(sp.stage_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
