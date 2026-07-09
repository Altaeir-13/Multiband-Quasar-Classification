from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


CLASS_NAMES = ["STAR", "GALAXY", "QSO"]


def _flatten_metrics(metrics: dict) -> dict:
    return {
        "model": metrics.get("model"),
        "split": metrics.get("split"),
        "accuracy": metrics.get("accuracy"),
        "precision_macro": metrics.get("precision_macro"),
        "recall_macro": metrics.get("recall_macro"),
        "f1_macro": metrics.get("f1_macro"),
        "pretrained": metrics.get("pretrained", "not_recorded"),
        "best_epoch": metrics.get("best_epoch", "not_recorded"),
        "run_dir": metrics.get("run_dir") or str(Path(metrics.get("checkpoint", "")).parent),
    }


def _load_metric_files(runs_dir: str | Path) -> list[dict[str, Any]]:
    loaded: list[dict[str, Any]] = []
    for metrics_path in Path(runs_dir).glob("*/*/metrics*.json"):
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        metrics["metrics_path"] = str(metrics_path)
        loaded.append(metrics)
    return loaded


def collect_metrics(runs_dir: str | Path) -> pd.DataFrame:
    rows = [_flatten_metrics(metrics) for metrics in _load_metric_files(runs_dir)]
    if not rows:
        return pd.DataFrame(
            columns=[
                "model",
                "split",
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "pretrained",
                "best_epoch",
                "run_dir",
            ]
        )
    return pd.DataFrame(rows).sort_values(["split", "f1_macro"], ascending=[True, False])


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a dataframe as a GitHub-flavored Markdown table without tabulate."""
    if frame.empty:
        return ""
    columns = [str(column) for column in frame.columns]
    body = [[str(row[column]) for column in frame.columns] for _, row in frame.iterrows()]
    rows = [columns, *body]
    widths = [max(len(row[index]) for row in rows) for index in range(len(columns))]

    def render_row(row: list[str]) -> str:
        cells = [value.ljust(widths[index]) for index, value in enumerate(row)]
        return "| " + " | ".join(cells) + " |"

    separator = ["-" * width for width in widths]
    lines = [render_row(columns), render_row(separator)]
    lines.extend(render_row(row) for row in body)
    return "\n".join(lines)


def _confusion_matrix_section(metrics: dict[str, Any]) -> str:
    matrix = metrics.get("confusion_matrix")
    if not matrix:
        return ""
    frame = pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).reset_index()
    frame = frame.rename(columns={"index": "true\\pred"})
    return _markdown_table(frame)


def _dominant_confusions(metrics: dict[str, Any]) -> list[str]:
    matrix = metrics.get("confusion_matrix") or []
    notes: list[str] = []
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            if row_index != col_index and value:
                notes.append(f"{CLASS_NAMES[row_index]} -> {CLASS_NAMES[col_index]}: {value}")
    return sorted(notes, key=lambda item: int(item.rsplit(": ", 1)[1]), reverse=True)


def _report_subset(metrics: pd.DataFrame) -> pd.DataFrame:
    non_smoke = metrics[~metrics["run_dir"].astype(str).str.contains("smoke", case=False, na=False)].copy()
    return non_smoke if not non_smoke.empty else metrics


def _write_best_model(handle, metrics: pd.DataFrame) -> None:
    test_metrics = metrics[metrics["split"] == "test"].copy()
    if test_metrics.empty:
        handle.write("## Best Model\n\nNo test metrics found yet.\n\n")
        return
    best = test_metrics.sort_values("f1_macro", ascending=False).iloc[0]
    handle.write("## Best Model\n\n")
    handle.write(
        f"Best test F1 macro: `{best['model']}` with F1=`{best['f1_macro']:.4f}` "
        f"and accuracy=`{best['accuracy']:.4f}`.\n\n"
    )


def _write_confusions(handle, metric_files: list[dict[str, Any]]) -> None:
    test_runs = [metrics for metrics in metric_files if metrics.get("split") == "test"]
    if not test_runs:
        return
    handle.write("## Test Confusion Matrices\n\n")
    for metrics in sorted(test_runs, key=lambda item: str(item.get("model"))):
        handle.write(f"### {metrics.get('model')}\n\n")
        handle.write(_confusion_matrix_section(metrics))
        handle.write("\n\n")
        confusions = _dominant_confusions(metrics)[:4]
        if confusions:
            handle.write("Dominant off-diagonal confusions: " + "; ".join(confusions) + ".\n\n")


def _filter_metric_files(metric_files: list[dict[str, Any]], metrics: pd.DataFrame) -> list[dict[str, Any]]:
    allowed_run_dirs = set(metrics["run_dir"].astype(str))
    return [
        metric
        for metric in metric_files
        if _flatten_metrics(metric)["run_dir"] in allowed_run_dirs
    ]


def write_model_comparison(
    metrics: pd.DataFrame,
    *,
    output_csv: str | Path,
    output_md: str | Path,
    runs_dir: str | Path = "runs/image_benchmark",
) -> None:
    csv_path = Path(output_csv)
    md_path = Path(output_md)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(csv_path, index=False)
    metric_files = _load_metric_files(runs_dir)

    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Image Benchmark Results\n\n")
        handle.write(
            "Partial Phase 1 image benchmark on the controlled small SDSS sample. "
            "Only models with available metrics are listed; this is not the final large-scale experiment.\n\n"
        )
        handle.write(
            "All currently configured small runs use SDSS JPEG RGB cutouts and `pretrained=false` "
            "unless a metric row explicitly records otherwise. JPEG RGB may not contain enough "
            "information to perfectly separate STAR, GALAXY and QSO, especially for point-like objects.\n\n"
        )
        if metrics.empty:
            handle.write("No metrics found yet. Run training/evaluation first.\n")
            return
        report_metrics = _report_subset(metrics)
        if len(report_metrics) < len(metrics):
            handle.write(
                "Smoke-test runs are omitted from the main comparison table; "
                "they remain in metrics.csv for traceability.\n\n"
            )
        display = report_metrics.copy()
        for column in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
            display[column] = display[column].map(lambda value: f"{value:.4f}")
        handle.write("## Aggregate Metrics\n\n")
        handle.write(_markdown_table(display.reset_index(drop=True)))
        handle.write("\n\n")
        _write_best_model(handle, report_metrics)
        _write_confusions(handle, _filter_metric_files(metric_files, report_metrics))
        handle.write("## Notes\n\n")
        handle.write(
            "Inspect GALAXY->STAR and STAR<->QSO confusions before scaling. "
            "If CNN baselines remain unstable, increase sample size and consider calibrated FITS-based inputs in a later image-only refinement.\n"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect image benchmark metrics.")
    parser.add_argument("--runs-dir", default="runs/image_benchmark")
    parser.add_argument("--output-csv", default="reports/image_benchmark/metrics.csv")
    parser.add_argument("--output-md", default="reports/image_benchmark/model_comparison.md")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    metrics = collect_metrics(args.runs_dir)
    write_model_comparison(
        metrics,
        output_csv=args.output_csv,
        output_md=args.output_md,
        runs_dir=args.runs_dir,
    )


if __name__ == "__main__":
    main()
