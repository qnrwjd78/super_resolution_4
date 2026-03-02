#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_METRICS = ["niqe", "maniqa", "musiq", "clipiqa", "lpips", "dists", "psnr", "ssim"]
DEFAULT_LOWER_BETTER = {"niqe", "lpips", "dists"}
METRIC_GROUPS = [
    ("No-reference", ["niqe", "maniqa", "musiq", "clipiqa"]),
    ("Full-reference", ["lpips", "dists"]),
    ("Restoration", ["psnr", "ssim"]),
]


@dataclass
class EvalRow:
    name: str
    mean: Dict[str, float]


def _infer_name(path: Path) -> str:
    name = path.name
    if name.endswith(".eval.json"):
        return name[: -len(".eval.json")]
    if name.endswith(".json"):
        return name[: -len(".json")]
    return name


def _as_float(v: object) -> Optional[float]:
    try:
        x = float(v)  # type: ignore[arg-type]
    except Exception:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def _load_rows(input_dir: Path) -> List[EvalRow]:
    rows: List[EvalRow] = []
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
        parsed: Dict[str, float] = {}
        for k, v in mean.items():
            x = _as_float(v)
            if x is not None:
                parsed[str(k)] = x
        rows.append(EvalRow(name=_infer_name(p), mean=parsed))
    return rows


def _load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}

    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config JSON not found: {config_path}")

    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config JSON must be a JSON object.")
    return data


def _get_str_list(config: Dict[str, Any], key: str, default: Sequence[str]) -> List[str]:
    value = config.get(key)
    if value is None:
        return list(default)
    if not isinstance(value, list):
        raise ValueError(f"Config field `{key}` must be a JSON array.")
    return [str(v) for v in value]


def _get_optional_str(config: Dict[str, Any], key: str, default: Optional[str] = None) -> Optional[str]:
    value = config.get(key)
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _get_str(config: Dict[str, Any], key: str, default: str) -> str:
    value = _get_optional_str(config, key, default)
    if value is None:
        return default
    return value


def _get_int(config: Dict[str, Any], key: str, default: int) -> int:
    value = config.get(key)
    if value is None:
        return default
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(f"Config field `{key}` must be an integer.") from exc


def _get_float(config: Dict[str, Any], key: str, default: float) -> float:
    value = config.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"Config field `{key}` must be a float.") from exc


def _present_metrics(rows: Sequence[EvalRow], metrics: Sequence[str]) -> List[str]:
    out: List[str] = []
    for m in metrics:
        if any(m in r.mean for r in rows):
            out.append(m)
    return out


def _best_second_values(values: Sequence[float], lower_better: bool, eps: float) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    ordered = sorted(values, reverse=not lower_better)
    uniq: List[float] = []
    for v in ordered:
        if not uniq or abs(v - uniq[-1]) > eps:
            uniq.append(v)
    best = uniq[0] if uniq else None
    second = uniq[1] if len(uniq) >= 2 else None
    return best, second


def _rank_marks(rows: Sequence[EvalRow], metric: str, lower_better: bool, eps: float) -> Dict[str, str]:
    vals = [r.mean[metric] for r in rows if metric in r.mean]
    best, second = _best_second_values(vals, lower_better=lower_better, eps=eps)
    marks: Dict[str, str] = {}
    for r in rows:
        v = r.mean.get(metric)
        if v is None:
            continue
        if best is not None and abs(v - best) <= eps:
            marks[r.name] = "first"
        elif second is not None and abs(v - second) <= eps:
            marks[r.name] = "second"
    return marks


def _latex_escape(text: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in text)


def _metric_header(metric: str, lower_better: bool) -> str:
    arrow = r"$\downarrow$" if lower_better else r"$\uparrow$"
    return f"{metric.upper()} {arrow}"


def _metric_groups_in_order(metrics: Sequence[str]) -> List[Tuple[str, List[str]]]:
    grouped: List[Tuple[str, List[str]]] = []
    used = set()
    for gname, gmetrics in METRIC_GROUPS:
        gm = [m for m in metrics if m in gmetrics]
        if gm:
            grouped.append((gname, gm))
            used.update(gm)
    rem = [m for m in metrics if m not in used]
    if rem:
        grouped.append(("Other", rem))
    return grouped


def _family_key(model_name: str) -> str:
    return model_name.split("_", 1)[0]


def _family_display(key: str) -> str:
    return key


def _group_rows_by_family(rows: Sequence[EvalRow]) -> List[Tuple[str, List[EvalRow]]]:
    groups: List[Tuple[str, List[EvalRow]]] = []
    for r in rows:
        fam = _family_key(r.name)
        if not groups or groups[-1][0] != fam:
            groups.append((fam, [r]))
        else:
            groups[-1][1].append(r)
    return groups


def _pretty_model_name(model_name: str) -> str:
    return model_name


def _format_value(v: Optional[float], mark: str, digits: int) -> str:
    if v is None:
        return "--"
    s = f"{v:.{digits}f}"
    if mark == "first":
        return rf"\textbf{{{s}}}"
    if mark == "second":
        return rf"\underline{{{s}}}"
    return s


def render_latex_table(
    rows: Sequence[EvalRow],
    metrics: Sequence[str],
    lower_better_metrics: Sequence[str],
    *,
    style: str,
    digits: int,
    caption: str,
    label: str,
    eps: float,
) -> str:
    lower_set = set(lower_better_metrics)
    metric_groups = _metric_groups_in_order(metrics)
    family_groups = _group_rows_by_family(rows)

    rank_by_metric: Dict[str, Dict[str, str]] = {}
    for m in metrics:
        rank_by_metric[m] = _rank_marks(rows, m, lower_better=(m in lower_set), eps=eps)

    header_cells = [_metric_header(m, m in lower_set) for m in metrics]
    colspec_booktabs = "l" + "c" * len(metrics)
    colspec_grid = "|" + "|".join(["l"] + ["c"] * len(metrics)) + "|"

    lines: List[str] = []
    if style == "boxed":
        lines.append(r"% Requires: \usepackage{graphicx}")
    else:
        lines.append(r"% Requires: \usepackage{booktabs,graphicx}")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \setlength{\tabcolsep}{4.4pt}")
    lines.append(r"  \renewcommand{\arraystretch}{1.12}")
    lines.append(r"  \small")
    lines.append(f"  \\caption{{{_latex_escape(caption)}}}")
    lines.append(f"  \\label{{{_latex_escape(label)}}}")
    lines.append(r"  \resizebox{\textwidth}{!}{%")
    if style == "boxed":
        lines.append(f"  \\begin{{tabular}}{{{colspec_grid}}}")
        lines.append(r"    \hline")
        if len(metric_groups) > 1:
            group_cells = [
                rf"\multicolumn{{{len(gm)}}}{{c|}}{{{_latex_escape(gname)}}}"
                for gname, gm in metric_groups
            ]
            lines.append("    Method & " + " & ".join(group_cells) + r" \\")
            lines.append(r"    \hline")
        lines.append("    Method & " + " & ".join(header_cells) + r" \\")
        lines.append(r"    \hline")

        many_families = len(family_groups) > 1
        for fam, fam_rows in family_groups:
            if many_families:
                label = _latex_escape(_family_display(fam))
                lines.append(rf"    \multicolumn{{{len(metrics) + 1}}}{{|l|}}{{\textit{{{label}}}}} \\")
                lines.append(r"    \hline")
            for r in fam_rows:
                cells: List[str] = []
                for m in metrics:
                    mark = rank_by_metric.get(m, {}).get(r.name, "")
                    val = r.mean.get(m)
                    cells.append(_format_value(val, mark, digits))
                model_label = _latex_escape(_pretty_model_name(r.name))
                lines.append(f"    {model_label} & " + " & ".join(cells) + r" \\")
                lines.append(r"    \hline")
    else:
        lines.append(f"  \\begin{{tabular}}{{{colspec_booktabs}}}")
        lines.append(r"    \toprule")
        if len(metric_groups) > 1:
            group_cells = [rf"\multicolumn{{{len(gm)}}}{{c}}{{{_latex_escape(gname)}}}" for gname, gm in metric_groups]
            lines.append("    Method & " + " & ".join(group_cells) + r" \\")
            cmid: List[str] = []
            start = 2
            for _gname, gm in metric_groups:
                end = start + len(gm) - 1
                cmid.append(rf"\cmidrule(lr){{{start}-{end}}}")
                start = end + 1
            lines.append("    " + "".join(cmid))
            lines.append("    & " + " & ".join(header_cells) + r" \\")
        else:
            lines.append("    Method & " + " & ".join(header_cells) + r" \\")
        lines.append(r"    \midrule")

        many_families = len(family_groups) > 1
        for gi, (fam, fam_rows) in enumerate(family_groups):
            if many_families:
                label = _latex_escape(_family_display(fam))
                if style == "grid":
                    lines.append(rf"    \multicolumn{{{len(metrics) + 1}}}{{l}}{{\textbf{{{label}}}}} \\")
                else:
                    lines.append(rf"    \multicolumn{{{len(metrics) + 1}}}{{l}}{{\textit{{{label}}}}} \\")
            for r in fam_rows:
                cells: List[str] = []
                for m in metrics:
                    mark = rank_by_metric.get(m, {}).get(r.name, "")
                    val = r.mean.get(m)
                    cells.append(_format_value(val, mark, digits))
                model_label = _latex_escape(_pretty_model_name(r.name))
                lines.append(f"    {model_label} & " + " & ".join(cells) + r" \\")
            if style == "booktabs" and gi != len(family_groups) - 1:
                lines.append(r"    \midrule")

        lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}%")
    lines.append(r"  }")
    lines.append(r"  \vspace{2pt}")
    lines.append(r"  \footnotesize{\textbf{Best} in bold, \underline{second-best} underlined.}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a LaTeX table from evaluation JSON files (top-level `mean` required)."
    )
    ap.add_argument("input", type=str, help="Input directory containing evaluation JSON files.")
    ap.add_argument(
        "--config_json",
        type=str,
        default=None,
        help="Optional JSON file describing LaTeX export options.",
    )
    ap.add_argument(
        "-o",
        "--out",
        type=str,
        default=None,
        help="Output .tex path. If omitted, use `out` from config_json or print to stdout.",
    )
    ap.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Metric order for table columns. Overrides config_json when provided.",
    )
    ap.add_argument(
        "--lower_better",
        nargs="+",
        default=None,
        help="Metrics where lower value is better. Overrides config_json when provided.",
    )
    ap.add_argument("--digits", type=int, default=None, help="Number of decimal places. Overrides config_json.")
    ap.add_argument(
        "--caption",
        type=str,
        default=None,
        help="LaTeX table caption. Overrides config_json.",
    )
    ap.add_argument("--label", type=str, default=None, help="LaTeX table label. Overrides config_json.")
    ap.add_argument(
        "--style",
        type=str,
        default=None,
        help=(
            "Table style. "
            "booktabs: paper default; "
            "grid: journal-like horizontal-rule style; "
            "boxed: framed table with full row/column lines."
        ),
    )
    ap.add_argument(
        "--eps",
        type=float,
        default=None,
        help="Tie tolerance when deciding best/second-best values. Overrides config_json.",
    )
    args = ap.parse_args()

    config = _load_config(args.config_json)

    input_dir = Path(args.input).resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    rows = _load_rows(input_dir)
    if not rows:
        raise SystemExit(f"No valid evaluation JSON with top-level 'mean' found under: {input_dir}")

    metrics = args.metrics if args.metrics is not None else _get_str_list(config, "metrics", DEFAULT_METRICS)
    lower_better = (
        args.lower_better
        if args.lower_better is not None
        else _get_str_list(config, "lower_better", sorted(DEFAULT_LOWER_BETTER))
    )
    digits = args.digits if args.digits is not None else _get_int(config, "digits", 4)
    caption = (
        args.caption
        if args.caption is not None
        else _get_str(config, "caption", "Mean evaluation metrics across checkpoints.")
    )
    label = args.label if args.label is not None else _get_str(config, "label", "tab:mean-metrics")
    style = args.style if args.style is not None else _get_str(config, "style", "booktabs")
    eps = args.eps if args.eps is not None else _get_float(config, "eps", 1e-12)
    out_value = args.out if args.out is not None else _get_optional_str(config, "out")

    if style not in {"booktabs", "grid", "boxed"}:
        raise SystemExit(f"Invalid style: {style}. Expected one of: booktabs, grid, boxed.")

    metrics = _present_metrics(rows, metrics)
    if not metrics:
        raise SystemExit("No requested metrics are present in loaded eval files.")

    tex = render_latex_table(
        rows=rows,
        metrics=metrics,
        lower_better_metrics=lower_better,
        style=style,
        digits=max(0, int(digits)),
        caption=caption,
        label=label,
        eps=max(0.0, float(eps)),
    )

    if out_value:
        out_path = Path(out_value).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(tex, encoding="utf-8")
        print(f"Saved: {out_path}")
    else:
        print(tex)


if __name__ == "__main__":
    main()
