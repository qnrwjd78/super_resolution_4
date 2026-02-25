import argparse
import sys
from pathlib import Path

SR_DIR = Path(__file__).resolve().parents[1]
if str(SR_DIR) not in sys.path:
    sys.path.insert(0, str(SR_DIR))

from utils.utils_visualization import (
    DEFAULT_METRICS,
    available_mean_metrics,
    load_eval_runs,
    make_idx_row_image,
    make_mean_metrics_figure,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare mean metrics across multiple eval JSON files.")
    ap.add_argument("input", type=str, help="Input directory containing *.eval.json (or similar) files.")
    ap.add_argument("output", type=str, help="Output directory to write mean_metrics.png into.")
    ap.add_argument(
        "--idx",
        type=int,
        nargs="+",
        default=None,
        help="If provided, also write idx row images: [HR][res(run1)][res(run2)]...",
    )
    args = ap.parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = load_eval_runs(input_dir)

    metrics = available_mean_metrics(runs, metrics=DEFAULT_METRICS)
    if not metrics:
        raise SystemExit("No known metrics found in mean. Expected keys like niqe/psnr/lpips...")

    print(f"Loaded runs : {len(runs)}")
    print(f"Metrics     : {', '.join(metrics)}")

    out_png = make_mean_metrics_figure(runs, output_dir / "mean_metrics.png", metrics=metrics)
    print(f"Saved       : {out_png}")

    if args.idx:
        row_dir = output_dir / "idx_rows"
        row_dir.mkdir(parents=True, exist_ok=True)
        for idx in args.idx:
            out_row = make_idx_row_image(runs, idx, row_dir / f"idx_{int(idx):04d}.png")
            print(f"Saved       : {out_row}")


if __name__ == "__main__":
    main()
