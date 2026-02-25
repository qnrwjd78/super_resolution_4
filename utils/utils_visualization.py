from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


@dataclass
class EvalRun:
    name: str
    path: Path
    mean: Dict[str, Any]
    per_image: List[Dict[str, Any]]


_DEFAULT_METRICS = ["niqe", "maniqa", "musiq", "clipiqa", "lpips", "dists", "psnr", "ssim"]
DEFAULT_METRICS = list(_DEFAULT_METRICS)
TEXT_SCALE = 2.0


def _infer_run_name(p: Path) -> str:
    name = p.name
    if name.endswith(".json"):
        name = name[:-5]
    if name.endswith(".eval"):
        name = name[:-5]
    return name


def load_eval_runs(input_dir: Path) -> List[EvalRun]:
    """
    Load all evaluation JSON files under input_dir (non-recursive).
    A file is considered an eval output if it has a top-level "mean" dict.
    """
    input_dir = Path(input_dir).resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input_dir is not a directory: {input_dir}")

    runs: List[EvalRun] = []
    for p in sorted(input_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        mean = data.get("mean")
        if not isinstance(mean, dict):
            continue
        per_image = data.get("per_image")
        if not isinstance(per_image, list):
            per_image = []
        # Filter non-dict entries defensively.
        per_image = [it for it in per_image if isinstance(it, dict)]
        runs.append(EvalRun(name=_infer_run_name(p), path=p, mean=mean, per_image=per_image))

    if not runs:
        raise ValueError(f"No evaluation JSON files found in: {input_dir}")
    return runs


def _as_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
    except Exception:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def available_mean_metrics(runs: Sequence[EvalRun], metrics: Sequence[str] = _DEFAULT_METRICS) -> List[str]:
    present = set()
    for run in runs:
        for m in metrics:
            if _as_float(run.mean.get(m)) is not None:
                present.add(m)
    return [m for m in metrics if m in present]


def _suffix_labels(names: Sequence[str]) -> List[str]:
    """
    Prefer the last underscore-suffix for each name (e.g. eval_dat -> dat).
    If that collides, expand to last 2/3/... suffix parts to make labels unique.
    """
    parts = [str(n).split("_") for n in names]
    use_k = [1 for _ in parts]

    def label(i: int) -> str:
        p = parts[i]
        k = use_k[i]
        if k >= len(p):
            return "_".join(p)
        return "_".join(p[-k:])

    labels = [label(i) for i in range(len(parts))]
    for _ in range(max((len(p) for p in parts), default=1)):
        seen: Dict[str, int] = {}
        dups: List[int] = []
        for i, l in enumerate(labels):
            if l in seen:
                dups.append(i)
                dups.append(seen[l])
            else:
                seen[l] = i
        dups = sorted(set(dups))
        if not dups:
            break
        changed = False
        for i in dups:
            if use_k[i] < len(parts[i]):
                use_k[i] += 1
                changed = True
        labels = [label(i) for i in range(len(parts))]
        if not changed:
            break

    # Still duplicated? Append small disambiguator.
    seen: Dict[str, int] = {}
    for i, l in enumerate(labels):
        if l in seen:
            labels[i] = f"{l}#{i+1}"
        else:
            seen[l] = i
    return labels


def _try_load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size=size)
        except Exception:
            pass
    # Pillow>=10 supports load_default(size=...), which gives a scalable font.
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _text_bbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    try:
        b = draw.textbbox((0, 0), text, font=font)
        return int(b[2] - b[0]), int(b[3] - b[1])
    except Exception:
        return draw.textsize(text, font=font)  # type: ignore[attr-defined]


def _palette(n: int) -> List[Tuple[int, int, int]]:
    base = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
        (188, 189, 34),
        (23, 190, 207),
    ]
    if n <= len(base):
        return base[:n]
    out = list(base)
    k = 0
    while len(out) < n:
        r, g, b = base[k % len(base)]
        k += 1
        out.append((min(255, r + 25), min(255, g + 25), min(255, b + 25)))
    return out[:n]


def _nice_num(x: float, round_: bool) -> float:
    if x <= 0:
        return 1.0
    exp = math.floor(math.log10(x))
    f = x / (10**exp)
    if round_:
        if f < 1.5:
            nf = 1.0
        elif f < 3.0:
            nf = 2.0
        elif f < 7.0:
            nf = 5.0
        else:
            nf = 10.0
    else:
        if f <= 1.0:
            nf = 1.0
        elif f <= 2.0:
            nf = 2.0
        elif f <= 5.0:
            nf = 5.0
        else:
            nf = 10.0
    return nf * (10**exp)


def _nice_ticks(vmin: float, vmax: float, nticks: int = 6) -> List[float]:
    if nticks < 2:
        return [vmin, vmax]
    if vmin == vmax:
        pad = abs(vmin) * 0.1 + 1e-6
        vmin -= pad
        vmax += pad
    rng = _nice_num(vmax - vmin, round_=False)
    step = _nice_num(rng / (nticks - 1), round_=True)
    tmin = math.floor(vmin / step) * step
    tmax = math.ceil(vmax / step) * step
    ticks: List[float] = []
    t = tmin
    for _ in range(1000):
        if t > tmax + 1e-12:
            break
        ticks.append(t)
        t += step
    return ticks or [vmin, vmax]


def _fmt_tick(v: float) -> str:
    av = abs(v)
    if av >= 100:
        return f"{v:.0f}"
    if av >= 10:
        return f"{v:.2f}".rstrip("0").rstrip(".")
    if av >= 1:
        return f"{v:.3f}".rstrip("0").rstrip(".")
    return f"{v:.4f}".rstrip("0").rstrip(".")


def make_mean_metric_bar(
    runs: Sequence[EvalRun],
    metric: str,
    out_path: Path,
    *,
    size: Tuple[int, int] = (1400, 900),
    scale: int = 2,
) -> Path:
    """
    Bar chart comparing per-run mean values for a single metric.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metric = str(metric).strip()
    values = [_as_float(r.mean.get(metric)) for r in runs]
    valid = [v for v in values if v is not None]

    fig_w, fig_h = int(size[0]), int(size[1])
    w, h = fig_w * max(1, scale), fig_h * max(1, scale)
    img = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img, "RGBA")

    font_title = _try_load_font(30 * max(1, scale))
    font_axis = _try_load_font(18 * max(1, scale))
    font_tick = _try_load_font(16 * max(1, scale))
    font_legend = _try_load_font(16 * max(1, scale))

    title = f"mean {metric} (by run)"
    draw.text((28 * scale, 18 * scale), title, fill=(0, 0, 0, 255), font=font_title)

    left = 140 * scale
    right = 40 * scale
    top = 90 * scale
    bottom = 210 * scale  # for x labels + legend

    x0, y0 = left, top
    x1, y1 = w - right, h - bottom

    # axes
    axis_col = (20, 20, 20, 255)
    draw.line([(x0, y1), (x1, y1)], fill=axis_col, width=3)
    draw.line([(x0, y0), (x0, y1)], fill=axis_col, width=3)

    if not valid:
        msg = f"No mean.{metric} values found."
        tw, th = _text_bbox(draw, msg, font_title)
        draw.text(((x0 + x1 - tw) // 2, (y0 + y1 - th) // 2), msg, fill=(120, 120, 120, 255), font=font_title)
        img_final = img
        if scale > 1:
            resample = getattr(Image, "Resampling", Image).LANCZOS
            img_final = img.resize((fig_w, fig_h), resample=resample)
        img_final.convert("RGB").save(out_path)
        return out_path

    vmin = min(valid)
    vmax = max(valid)
    if vmin == vmax:
        pad = abs(vmin) * 0.1 + 1e-6
        vmin -= pad
        vmax += pad
    else:
        pad = (vmax - vmin) * 0.12
        vmin -= pad
        vmax += pad

    yticks = _nice_ticks(vmin, vmax, nticks=6)
    y_min = yticks[0]
    y_max = yticks[-1]

    # grid + y tick labels
    grid_col = (235, 235, 235, 255)
    for t in yticks:
        yp = int(y1 - (t - y_min) / (y_max - y_min) * (y1 - y0))
        draw.line([(x0, yp), (x1, yp)], fill=grid_col, width=1)
        draw.line([(x0 - 6 * scale, yp), (x0, yp)], fill=axis_col, width=1)
        lab = _fmt_tick(t)
        tw, th = _text_bbox(draw, lab, font_tick)
        draw.text((x0 - 12 * scale - tw, yp - th // 2), lab, fill=(0, 0, 0, 255), font=font_tick)

    # axis label
    yw, yh = _text_bbox(draw, metric, font_axis)
    draw.text((28 * scale, y0 + (y1 - y0 - yh) // 2), metric, fill=(0, 0, 0, 255), font=font_axis)

    n = len(runs)
    colors = _palette(n)
    label_ids = [str(i + 1) for i in range(n)]

    # bars
    cw = x1 - x0
    bar_w = max(10 * scale, int(cw / (n * 1.6))) if n else 10 * scale
    gap = int((cw - n * bar_w) / (n + 1)) if n and cw > n * bar_w else 6 * scale
    if gap < 4 * scale:
        gap = 4 * scale

    for i, v in enumerate(values):
        cx = x0 + gap + i * (bar_w + gap)
        if v is None:
            draw.rectangle([cx, y0, cx + bar_w, y1], outline=(210, 210, 210, 255), width=2)
        else:
            frac = (v - y_min) / (y_max - y_min)
            frac = min(1.0, max(0.0, frac))
            topy = int(y1 - frac * (y1 - y0))
            r, g, b = colors[i]
            draw.rectangle([cx, topy, cx + bar_w, y1], fill=(r, g, b, 200), outline=(r, g, b, 255), width=2)

        lid = label_ids[i]
        tw, th = _text_bbox(draw, lid, font_tick)
        draw.text((cx + (bar_w - tw) // 2, y1 + 10 * scale), lid, fill=(0, 0, 0, 255), font=font_tick)

    # legend
    legend_y = y1 + 52 * scale
    draw.line([(x0, legend_y - 16 * scale), (x1, legend_y - 16 * scale)], fill=(210, 210, 210, 255), width=1)
    draw.text((x0, legend_y), "Runs:", fill=(0, 0, 0, 255), font=font_axis)

    lx = x0
    ly = legend_y + 36 * scale
    for i, run in enumerate(runs):
        r, g, b = colors[i]
        box = 18 * scale
        draw.rectangle([lx, ly + 3 * scale, lx + box, ly + 3 * scale + box], fill=(r, g, b, 255), outline=(0, 0, 0, 255), width=1)
        txt = f"{label_ids[i]} = {run.name}"
        draw.text((lx + box + 10 * scale, ly), txt, fill=(0, 0, 0, 255), font=font_legend)
        tw, th = _text_bbox(draw, txt, font_legend)
        lx += box + 10 * scale + tw + 22 * scale
        if lx > x1 - 340 * scale:
            lx = x0
            ly += 26 * scale

    # Downsample for antialiasing
    img_final = img
    if scale > 1:
        resample = getattr(Image, "Resampling", Image).LANCZOS
        img_final = img.resize((fig_w, fig_h), resample=resample)
    img_final.convert("RGB").save(out_path)
    return out_path


def _get_per_image_item(run: EvalRun, idx: int) -> Optional[Dict[str, Any]]:
    for it in run.per_image:
        try:
            if int(it.get("index")) == int(idx):
                return it
        except Exception:
            continue
    if 0 <= idx < len(run.per_image):
        it = run.per_image[idx]
        return it if isinstance(it, dict) else None
    return None


def _make_placeholder(w: int, h: int, text: str, *, scale: int) -> Image.Image:
    img = Image.new("RGBA", (w, h), (245, 245, 245, 255))
    draw = ImageDraw.Draw(img, "RGBA")
    font = _try_load_font(int(round(16 * float(TEXT_SCALE))) * max(1, scale))
    tw, th = _text_bbox(draw, text, font)
    draw.text(((w - tw) // 2, (h - th) // 2), text, fill=(120, 120, 120, 255), font=font)
    return img


def make_idx_row_image(
    runs: Sequence[EvalRun],
    idx: int,
    out_path: Path,
    *,
    tile_size: int = 256,
    pad: int = 10,
    scale: int = 2,
) -> Path:
    """
    Make a single-row comparison image for a given index:
      [HR] [res(run1)] [res(run2)] ...
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    runs = list(runs)
    if not runs:
        raise ValueError("runs is empty")

    idx = int(idx)

    # Gather paths (HR from the first run, res from each run).
    first_it = _get_per_image_item(runs[0], idx)
    hr_path = str(first_it.get("hr")) if isinstance(first_it, dict) and first_it.get("hr") else None

    run_labels = _suffix_labels([r.name for r in runs])

    cells: List[Tuple[str, Optional[str]]] = [("HR", hr_path)]
    for run, lab in zip(runs, run_labels):
        it = _get_per_image_item(run, idx)
        res_path = str(it.get("res")) if isinstance(it, dict) and it.get("res") else None
        cells.append((lab, res_path))

    ts = float(TEXT_SCALE)
    tile = int(tile_size)
    pad_px = int(pad)
    label_h = int(round(26 * ts))

    # Create per-cell images (image + caption). No frames/borders, no square padding.
    cell_imgs: List[Image.Image] = []
    for label, path in cells:
        font = _try_load_font(int(round(12 * ts)) * max(1, scale))

        target_h = tile * scale
        if not path:
            img_part = _make_placeholder(target_h, target_h, "missing", scale=scale)
        else:
            try:
                with Image.open(path) as im:
                    im = im.convert("RGB")
                    if im.height <= 0:
                        raise ValueError("invalid image height")
                    ratio = target_h / float(im.height)
                    new_w = max(1, int(round(im.width * ratio)))
                    resample = getattr(Image, "Resampling", Image).LANCZOS
                    im = im.resize((new_w, target_h), resample=resample)
                    img_part = im.convert("RGBA")
            except Exception:
                img_part = _make_placeholder(target_h, target_h, "error", scale=scale)

        # caption (bottom)
        cap_h = label_h * scale
        cap = Image.new("RGBA", (img_part.size[0], cap_h), (255, 255, 255, 255))
        d = ImageDraw.Draw(cap, "RGBA")
        safe = str(label)
        lw, lh = _text_bbox(d, safe, font)
        d.text(((cap.size[0] - lw) // 2, (cap_h - lh) // 2), safe, fill=(0, 0, 0, 255), font=font)

        cell = Image.new("RGBA", (img_part.size[0], img_part.size[1] + cap_h), (255, 255, 255, 255))
        cell.alpha_composite(img_part, dest=(0, 0))
        cell.alpha_composite(cap, dest=(0, img_part.size[1]))
        cell_imgs.append(cell)

    # Compose a single row.
    row_w = sum(im.size[0] for im in cell_imgs) + (len(cell_imgs) + 1) * (pad_px * scale)
    row_h = max((im.size[1] for im in cell_imgs), default=(tile + label_h) * scale) + 2 * (pad_px * scale)
    row = Image.new("RGBA", (row_w, row_h), (255, 255, 255, 255))
    x = pad_px * scale
    y = pad_px * scale
    for cimg in cell_imgs:
        row.alpha_composite(cimg, dest=(x, y))
        x += cimg.size[0] + pad_px * scale

    # Downsample for antialiasing
    img_final = row
    if scale > 1:
        resample = getattr(Image, "Resampling", Image).LANCZOS
        img_final = row.resize((row_w // scale, row_h // scale), resample=resample)

    img_final.convert("RGB").save(out_path)
    return out_path


def _metric_title(metric: str) -> str:
    return metric


def _estimate_legend_height(
    run_names: Sequence[str],
    *,
    fig_w: int,
    pad: int,
    ts: float,
) -> int:
    """
    Estimate legend height (in final/output pixels, not scaled-canvas pixels).
    """
    font_leg = _try_load_font(int(round(14 * ts)))
    tmp = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    d = ImageDraw.Draw(tmp, "RGBA")

    x0 = pad
    x1 = fig_w - pad

    box = int(round(16 * ts))
    gap = int(round(10 * ts))
    tail = int(round(22 * ts))
    header_y = int(round(14 * ts))
    first_line_y = int(round(48 * ts))
    line_step = int(round(24 * ts))
    bottom_pad = int(round(22 * ts))

    lx = x0
    lines = 1
    for name in run_names:
        tw, th = _text_bbox(d, name, font_leg)
        entry_w = box + gap + tw + tail
        if lx != x0 and lx + entry_w > x1:
            lines += 1
            lx = x0
        lx += entry_w

    # Height = header area + legend rows
    height = max(header_y + line_step, first_line_y + lines * line_step) + bottom_pad
    return max(int(round(90 * ts)), height)


def _draw_mean_bar_panel(
    img: Image.Image,
    rect: Tuple[int, int, int, int],
    metric: str,
    values: Sequence[Optional[float]],
    colors: Sequence[Tuple[int, int, int]],
    *,
    scale: int,
) -> None:
    ts = float(TEXT_SCALE)
    x0, y0, x1, y1 = rect
    draw = ImageDraw.Draw(img, "RGBA")

    font_title = _try_load_font(int(round(20 * ts)) * max(1, scale))
    font_tick = _try_load_font(int(round(13 * ts)) * max(1, scale))
    font_val = _try_load_font(int(round(12 * ts)) * max(1, scale))

    # panel frame
    draw.rectangle([x0, y0, x1, y1], outline=(220, 220, 220, 255), width=1)

    title = _metric_title(metric)
    draw.text((x0 + 10 * scale, y0 + 8 * scale), title, fill=(0, 0, 0, 255), font=font_title)

    # Keep chart large; only grow margins moderately with TEXT_SCALE.
    chart_x0 = x0 + int(round(64 + 16 * (ts - 1.0))) * scale
    chart_y0 = y0 + int(round(40 + 20 * (ts - 1.0) + 8)) * scale
    chart_x1 = x1 - 14 * scale
    chart_y1 = y1 - int(round(34 + 14 * (ts - 1.0))) * scale

    axis_col = (20, 20, 20, 255)
    grid_col = (235, 235, 235, 255)

    # axes
    draw.line([(chart_x0, chart_y1), (chart_x1, chart_y1)], fill=axis_col, width=2 * scale)
    draw.line([(chart_x0, chart_y0), (chart_x0, chart_y1)], fill=axis_col, width=2 * scale)

    vals = [v for v in values if v is not None]
    if not vals:
        msg = "N/A"
        tw, th = _text_bbox(draw, msg, font_title)
        draw.text(((chart_x0 + chart_x1 - tw) // 2, (chart_y0 + chart_y1 - th) // 2), msg, fill=(130, 130, 130, 255), font=font_title)
        return

    vmin, vmax = min(vals), max(vals)
    if vmin == vmax:
        pad = abs(vmin) * 0.1 + 1e-6
        vmin -= pad
        vmax += pad
    else:
        pad = (vmax - vmin) * 0.12
        vmin -= pad
        vmax += pad

    yticks = _nice_ticks(vmin, vmax, nticks=5)
    y_min, y_max = yticks[0], yticks[-1]

    # y ticks + grid
    for t in yticks:
        yp = int(chart_y1 - (t - y_min) / (y_max - y_min) * (chart_y1 - chart_y0))
        draw.line([(chart_x0, yp), (chart_x1, yp)], fill=grid_col, width=1)
        draw.line([(chart_x0 - int(round(4 * ts)) * scale, yp), (chart_x0, yp)], fill=axis_col, width=1)
        lab = _fmt_tick(t)
        tw, th = _text_bbox(draw, lab, font_tick)
        draw.text((chart_x0 - int(round(8 * ts)) * scale - tw, yp - th // 2), lab, fill=(0, 0, 0, 255), font=font_tick)

    n = len(values)
    if n <= 0:
        return

    cw = chart_x1 - chart_x0
    bar_w = max(6 * scale, int(cw / (n * 1.6)))
    gap = int((cw - n * bar_w) / (n + 1)) if cw > n * bar_w else 2 * scale
    if gap < 2 * scale:
        gap = 2 * scale

    for i, v in enumerate(values):
        cx = chart_x0 + gap + i * (bar_w + gap)
        if v is None:
            draw.rectangle([cx, chart_y0, cx + bar_w, chart_y1], outline=(210, 210, 210, 255), width=1)
        else:
            frac = (v - y_min) / (y_max - y_min)
            frac = min(1.0, max(0.0, frac))
            topy = int(chart_y1 - frac * (chart_y1 - chart_y0))
            r, g, b = colors[i]
            draw.rectangle([cx, topy, cx + bar_w, chart_y1], fill=(r, g, b, 200), outline=(r, g, b, 255), width=1)

            # value label
            if n <= 12:
                val_txt = _fmt_tick(v)
                tw, th = _text_bbox(draw, val_txt, font_val)
                vx = cx + (bar_w - tw) // 2
                vy = max(chart_y0, topy - th - int(round(3 * ts)) * scale)
                # white backing for readability
                draw.rectangle(
                    [
                        vx - int(round(3 * ts)) * scale,
                        vy - int(round(2 * ts)) * scale,
                        vx + tw + int(round(3 * ts)) * scale,
                        vy + th + int(round(2 * ts)) * scale,
                    ],
                    fill=(255, 255, 255, 190),
                )
                draw.text((vx, vy), val_txt, fill=(0, 0, 0, 255), font=font_val)

        # x-axis labels intentionally omitted (legend below maps colors to runs).


def make_mean_metrics_figure(
    runs: Sequence[EvalRun],
    out_path: Path,
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    cols: int = 4,
    size: Tuple[int, int] = (2050, 1400),
    scale: int = 2,
) -> Path:
    """
    One PNG containing per-metric mean bar charts (no normalization).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = list(metrics)
    n_panels = len(metrics)
    if n_panels <= 0:
        raise ValueError("metrics is empty")

    cols = max(1, int(cols))
    rows = int(math.ceil(n_panels / cols))

    fig_w, fig_h = int(size[0]), int(size[1])
    w, h = fig_w * max(1, scale), fig_h * max(1, scale)
    img = Image.new("RGBA", (w, h), (255, 255, 255, 255))

    pad = 24
    ts = float(TEXT_SCALE)
    run_labels = _suffix_labels([r.name for r in runs])
    legend_h = _estimate_legend_height(run_labels, fig_w=fig_w, pad=pad, ts=ts)

    panel_w = int((fig_w - (cols + 1) * pad) / cols)
    panel_h = int((fig_h - legend_h - (rows + 1) * pad) / rows)
    panel_w = max(260, panel_w)
    panel_h = max(220, panel_h)

    pad_s = pad * scale
    panel_w_s = panel_w * scale
    panel_h_s = panel_h * scale

    colors = _palette(len(runs))

    for idx, m in enumerate(metrics):
        r_i = idx // cols
        c_i = idx % cols
        x0 = pad_s + c_i * (panel_w_s + pad_s)
        y0 = pad_s + r_i * (panel_h_s + pad_s)
        rect = (x0, y0, x0 + panel_w_s, y0 + panel_h_s)
        values = [_as_float(run.mean.get(m)) for run in runs]
        _draw_mean_bar_panel(img, rect, m, values, colors, scale=scale)

    # legend at bottom
    draw = ImageDraw.Draw(img, "RGBA")
    font_axis = _try_load_font(int(round(18 * ts)) * max(1, scale))
    font_leg = _try_load_font(int(round(14 * ts)) * max(1, scale))

    legend_y0 = pad_s + rows * (panel_h_s + pad_s)
    x0 = pad_s
    x1 = w - pad_s
    draw.line([(x0, legend_y0), (x1, legend_y0)], fill=(210, 210, 210, 255), width=1)
    draw.text((x0, legend_y0 + int(round(14 * ts)) * scale), "Runs:", fill=(0, 0, 0, 255), font=font_axis)

    box = int(round(16 * ts)) * scale
    gap = int(round(10 * ts)) * scale
    tail = int(round(22 * ts)) * scale
    line_step = int(round(24 * ts)) * scale

    lx = x0
    ly = legend_y0 + int(round(48 * ts)) * scale
    for i, run in enumerate(runs):
        r, g, b = colors[i]
        txt = run_labels[i] if i < len(run_labels) else run.name
        tw, th = _text_bbox(draw, txt, font_leg)
        entry_w = box + gap + tw + tail
        if lx != x0 and lx + entry_w > x1:
            lx = x0
            ly += line_step

        draw.rectangle(
            [lx, ly + int(round(3 * ts)) * scale, lx + box, ly + int(round(3 * ts)) * scale + box],
            fill=(r, g, b, 255),
            outline=(0, 0, 0, 255),
            width=1,
        )
        draw.text((lx + box + int(round(10 * ts)) * scale, ly), txt, fill=(0, 0, 0, 255), font=font_leg)
        lx += entry_w

    # Downsample for antialiasing
    img_final = img
    if scale > 1:
        resample = getattr(Image, "Resampling", Image).LANCZOS
        img_final = img.resize((fig_w, fig_h), resample=resample)
    rgb = img_final.convert("RGB")

    # Trim excessive bottom whitespace (keep a small margin).
    try:
        px = rgb.load()
        ww, hh = rgb.size
        last_nonwhite = -1
        for yy in range(hh - 1, -1, -1):
            row_has = False
            for xx in range(ww):
                if px[xx, yy] != (255, 255, 255):
                    row_has = True
                    break
            if row_has:
                last_nonwhite = yy
                break
        if last_nonwhite >= 0:
            margin = 14
            new_h = min(hh, last_nonwhite + 1 + margin)
            if new_h < hh:
                rgb = rgb.crop((0, 0, ww, new_h))
    except Exception:
        pass

    rgb.save(out_path)
    return out_path
