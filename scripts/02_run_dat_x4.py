#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from utils import make_lr_stage_from_json


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    sr_dir = (script_dir / "..").resolve()
    workspace_dir = (sr_dir / "../..").resolve()

    default_val_json = os.environ.get("VAL_JSON", str(workspace_dir / "data" / "val_fixed.json"))
    default_repo_dir = str((sr_dir / "repos" / "DAT").resolve())
    default_opt = str((script_dir / "options" / "dat_x4.yml").resolve())

    ap = argparse.ArgumentParser(description="Run DAT x4 on VAL_JSON LR-only (symlink stage), optionally write result.json.")
    ap.add_argument("--json", default=default_val_json, help="Path to val_fixed.json (only 'lr' is used)")
    ap.add_argument("--repo_dir", default=default_repo_dir, help="Path to DAT repo")
    ap.add_argument("--stage_root", default=os.environ.get("STAGE_ROOT", "/tmp/sr_stage"), help="Root dir for staging (must allow symlink)")
    ap.add_argument("--keep_stage", action="store_true", help="Do not delete staging folder after run")

    ap.add_argument("--opt", default=os.environ.get("OPT", default_opt), help="Path to DAT option YAML (template)")
    ap.add_argument("--name", default=os.environ.get("NAME", "DAT_x4"))
    ap.add_argument("--weight", default=os.environ.get("WEIGHT", ""), help="Path to DAT_x4.pth (empty = repo default)")
    ap.add_argument("--use_chop", action="store_true", default=False, help="Enable chop testing")

    ap.add_argument("--out_dir", default=os.environ.get("OUT_DIR", ""), help="Save final SR images here (empty = repo results dir)")
    ap.add_argument(
        "--meta_dir",
        default=os.environ.get("META_DIR", os.environ.get("PATH_DIR", "")),
        help="If set, write <meta_dir>/result.json with [{'res':..., 'hr':...}, ...].",
    )

    ap.add_argument("--gpu", default="", help="CUDA_VISIBLE_DEVICES, e.g. '2' or '0,1'. Empty = do not set.")
    args = ap.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    opt_path = Path(str(args.opt)).expanduser().resolve()
    if not opt_path.is_file():
        raise SystemExit(f"[ERROR] opt not found: {opt_path}")

    weight_raw = str(args.weight).strip()
    weight = (
        Path(weight_raw).expanduser()
        if weight_raw
        else (repo_dir / "experiments" / "pretrained_models" / "DAT" / "DAT_x4.pth")
    ).resolve()
    if not weight.is_file():
        raise SystemExit(f"[ERROR] weight not found: {weight}")

    out_dir_raw = str(args.out_dir).strip()
    meta_dir_raw = str(args.meta_dir).strip()
    out_dir = str(Path(out_dir_raw).expanduser().resolve()) if out_dir_raw else ""
    meta_dir = str(Path(meta_dir_raw).expanduser().resolve()) if meta_dir_raw else ""
    if not meta_dir and out_dir:
        meta_dir = out_dir

    sp = make_lr_stage_from_json(
        val_json=Path(args.json),
        stage_root=Path(args.stage_root),
        name_prefix="dat",
        prefix_index=False,  # keep original basenames for stable ordering and output filenames
    )

    force_yml = [
        f"name={args.name}",
        f"datasets:test_1:dataroot_lq={sp.lr_dir}",
        f"path:pretrain_network_g={weight}",
        "val:suffix=x4",
    ]
    if args.use_chop:
        force_yml.append("val:use_chop=true")

    cmd = [
        sys.executable, "basicsr/test.py",
        "-opt", str(opt_path),
        "--force_yml",
        *force_yml,
    ]

    run_env = os.environ.copy()
    run_env["PYTHONPATH"] = f"{repo_dir}{os.pathsep}{run_env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    if args.gpu.strip():
        run_env["CUDA_VISIBLE_DEVICES"] = args.gpu.strip()

    print("[INFO] repo_dir :", repo_dir)
    print("[INFO] stage_lr :", sp.lr_dir)
    print("[INFO] opt      :", opt_path)
    print("[INFO] weight   :", weight)
    if out_dir:
        print("[INFO] out_dir  :", out_dir)
    if meta_dir:
        print("[INFO] meta_dir :", meta_dir)
    if args.gpu.strip():
        print("[INFO] CUDA_VISIBLE_DEVICES =", run_env["CUDA_VISIBLE_DEVICES"])
    print("[INFO] cmd      :", " ".join(cmd))

    try:
        subprocess.run(cmd, cwd=str(repo_dir), check=True, env=run_env)

        raw_dir = (repo_dir / "results" / args.name / "visualization" / "DIV2K_val").resolve()
        if not raw_dir.exists():
            raise FileNotFoundError(f"Expected output folder not found: {raw_dir}")

        final_dir = Path(out_dir).resolve() if out_dir else raw_dir
        final_dir.mkdir(parents=True, exist_ok=True)

        for stage_lr in sorted(sp.lr_dir.glob("*")):
            lr_name = stage_lr.name
            lr_stem = stage_lr.stem
            raw_file = raw_dir / f"{lr_stem}_x4.png"
            if not raw_file.exists():
                raise FileNotFoundError(f"Expected output image not found: {raw_file}")
            dst = final_dir / lr_name
            if final_dir == raw_dir:
                os.replace(raw_file, dst)
            else:
                shutil.copy2(raw_file, dst)

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

            out_root = Path(out_dir).resolve() if out_dir else raw_dir
            items = []
            for e in data:
                lr_name = Path(e["lr"]).name
                it = {"res": str((out_root / lr_name).resolve())}
                hr = e.get("hr", None)
                if hr is not None and str(hr).strip() != "":
                    it["hr"] = hr
                items.append(it)

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
