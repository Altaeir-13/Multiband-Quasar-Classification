from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


CLASS_NAMES = ["STAR", "GALAXY", "QSO"]
METRIC_COLUMNS = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
MODEL_ORDER = ["simple_cnn", "resnet18", "densenet121"]
SEED_ORDER = ["42", "123", "13"]


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


def _seed_from_run_dir(run_dir: str) -> str:
    lowered = run_dir.lower()
    if "seed123" in lowered:
        return "123"
    if "seed13" in lowered:
        return "13"
    return "42"


def _model_sort_key(model: str) -> int:
    return MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER)


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
                "seed",
            ]
        )
    frame = pd.DataFrame(rows)
    frame["seed"] = frame["run_dir"].astype(str).map(_seed_from_run_dir)
    frame["model_order"] = frame["model"].map(_model_sort_key)
    frame["seed_order"] = frame["seed"].map(lambda seed: SEED_ORDER.index(seed) if seed in SEED_ORDER else len(SEED_ORDER))
    frame = frame.sort_values(["seed_order", "split", "model_order"])
    return frame.drop(columns=["model_order", "seed_order"])


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


def _without_smoke(metrics: pd.DataFrame) -> pd.DataFrame:
    non_smoke = metrics[~metrics["run_dir"].astype(str).str.contains("smoke", case=False, na=False)].copy()
    return non_smoke if not non_smoke.empty else metrics.copy()


def _test_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics[metrics["split"] == "test"].copy()


def _format_metric_columns(metrics: pd.DataFrame) -> pd.DataFrame:
    display = metrics.copy()
    for column in METRIC_COLUMNS:
        if column in display.columns:
            display[column] = display[column].map(lambda value: f"{value:.4f}")
    return display.reset_index(drop=True)


def _format_summary_columns(summary: pd.DataFrame) -> pd.DataFrame:
    display = summary.copy()
    for column in display.columns:
        if column != "model":
            display[column] = display[column].map(lambda value: "missing" if pd.isna(value) else f"{value:.4f}")
    return display


def _seed_table(test_metrics: pd.DataFrame, seed: str) -> pd.DataFrame:
    seed_metrics = test_metrics[test_metrics["seed"] == seed].copy()
    seed_metrics["model_order"] = seed_metrics["model"].map(_model_sort_key)
    seed_metrics = seed_metrics.sort_values("model_order")
    columns = ["model", "accuracy", "precision_macro", "recall_macro", "f1_macro", "best_epoch", "run_dir"]
    return seed_metrics[columns]


def _aggregate_table(test_metrics: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        model_metrics = test_metrics[test_metrics["model"] == model]
        values = {
            seed: model_metrics.loc[model_metrics["seed"] == seed, metric].iloc[0]
            if not model_metrics.loc[model_metrics["seed"] == seed].empty
            else pd.NA
            for seed in SEED_ORDER
        }
        present = pd.Series([value for value in values.values() if not pd.isna(value)], dtype="float64")
        rows.append(
            {
                "model": model,
                f"{metric} seed 42": values["42"],
                f"{metric} seed 123": values["123"],
                f"{metric} seed 13": values["13"],
                f"mean {metric}": present.mean() if not present.empty else pd.NA,
                f"std {metric}": present.std(ddof=0) if len(present) > 1 else 0.0 if len(present) == 1 else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def _best_and_stability_lines(test_metrics: pd.DataFrame) -> tuple[str, str]:
    f1_summary = _aggregate_table(test_metrics, "f1_macro")
    valid_mean = f1_summary.dropna(subset=["mean f1_macro"])
    if valid_mean.empty:
        return "No valid F1 metrics are available.", "No stability estimate is available."
    best = valid_mean.sort_values("mean f1_macro", ascending=False).iloc[0]
    stable = valid_mean.sort_values("std f1_macro", ascending=True).iloc[0]
    return (
        f"Best model by mean test F1 macro: `{best['model']}` with mean F1=`{best['mean f1_macro']:.4f}`.",
        f"Most stable model by lowest observed test F1 standard deviation: `{stable['model']}` with std F1=`{stable['std f1_macro']:.4f}`.",
    )


def _filter_metric_files(metric_files: list[dict[str, Any]], metrics: pd.DataFrame) -> list[dict[str, Any]]:
    allowed_run_dirs = set(metrics["run_dir"].astype(str))
    return [
        metric
        for metric in metric_files
        if _flatten_metrics(metric)["run_dir"] in allowed_run_dirs
    ]


def _write_seed_section(handle, test_metrics: pd.DataFrame, metric_files: list[dict[str, Any]], seed: str) -> None:
    handle.write(f"## Benchmark small - seed {seed}\n\n")
    table = _seed_table(test_metrics, seed)
    if table.empty:
        handle.write("No valid test metrics for this seed.\n\n")
        return
    handle.write(_markdown_table(_format_metric_columns(table)))
    handle.write("\n\n")
    seed_files = _filter_metric_files(metric_files, test_metrics[test_metrics["seed"] == seed])
    test_runs = [metrics for metrics in seed_files if metrics.get("split") == "test"]
    for metrics in sorted(test_runs, key=lambda item: _model_sort_key(str(item.get("model")))):
        run_dir = _flatten_metrics(metrics)["run_dir"]
        handle.write(f"### {metrics.get('model')} - `{Path(run_dir).name}`\n\n")
        handle.write(_confusion_matrix_section(metrics))
        handle.write("\n\n")
        confusions = _dominant_confusions(metrics)[:4]
        if confusions:
            handle.write("Dominant off-diagonal confusions: " + "; ".join(confusions) + ".\n\n")


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
    metrics = metrics.copy()
    if "seed" not in metrics.columns and "run_dir" in metrics.columns:
        metrics["seed"] = metrics["run_dir"].astype(str).map(_seed_from_run_dir)
    for optional_column in ["best_epoch", "pretrained"]:
        if optional_column not in metrics.columns:
            metrics[optional_column] = "not_recorded"
    metrics.to_csv(csv_path, index=False)
    metric_files = _load_metric_files(runs_dir)

    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Image Benchmark Results\n\n")
        handle.write(
            "Phase 1 controlled small benchmark using SDSS JPEG RGB cutouts. "
            "This report compares image classifiers across seeds 42, 123 and 13; "
            "it is still not the final large-scale experiment.\n\n"
        )
        handle.write(
            "All currently configured small runs use `pretrained=false`. JPEG RGB cutouts are useful for this initial PDI benchmark, "
            "but they may not carry enough calibrated information to separate point-like QSO and STAR objects perfectly.\n\n"
        )
        if metrics.empty:
            handle.write("No metrics found yet. Run training/evaluation first.\n")
            return

        report_metrics = _without_smoke(metrics)
        test_metrics = _test_metrics(report_metrics)
        if len(report_metrics) < len(metrics):
            handle.write(
                "Smoke-test runs are omitted from the Markdown comparison; "
                "they remain in metrics.csv for traceability.\n\n"
            )

        for seed in SEED_ORDER:
            _write_seed_section(handle, test_metrics, metric_files, seed)

        handle.write("## Aggregate F1 Macro By Model\n\n")
        f1_summary = _aggregate_table(test_metrics, "f1_macro")
        f1_summary = f1_summary.rename(
            columns={
                "f1_macro seed 42": "F1 seed 42",
                "f1_macro seed 123": "F1 seed 123",
                "f1_macro seed 13": "F1 seed 13",
                "mean f1_macro": "mean F1",
                "std f1_macro": "std F1",
            }
        )
        handle.write(_markdown_table(_format_summary_columns(f1_summary)))
        handle.write("\n\n")

        handle.write("## Aggregate Accuracy By Model\n\n")
        acc_summary = _aggregate_table(test_metrics, "accuracy")
        acc_summary = acc_summary.rename(
            columns={
                "accuracy seed 42": "acc seed 42",
                "accuracy seed 123": "acc seed 123",
                "accuracy seed 13": "acc seed 13",
                "mean accuracy": "mean acc",
                "std accuracy": "std acc",
            }
        )
        handle.write(_markdown_table(_format_summary_columns(acc_summary)))
        handle.write("\n\n")

        best_line, stable_line = _best_and_stability_lines(test_metrics)
        handle.write("## Robustness Interpretation\n\n")
        handle.write(best_line + "\n\n")
        handle.write(stable_line + "\n\n")
        handle.write(
            "`simple_cnn` remains useful as a minimum baseline, but it trails the stronger architectures in mean test F1. "
            "`resnet18` should remain the principal baseline only if its mean performance and variance stay competitive as the dataset grows. "
            "`densenet121` is competitive and should be compared on equal seeds before selecting a final image baseline.\n\n"
        )
        handle.write(
            "Recurring confusions to inspect before scaling are QSO -> STAR, QSO -> GALAXY and GALAXY -> STAR. "
            "These errors are scientifically plausible with JPEG RGB cutouts because QSO and STAR can both appear point-like, "
            "and color/photometric calibration is limited compared with FITS or tabular photometry.\n\n"
        )
        handle.write(
            "No conclusion about photometric redshift should be drawn from this report. Phase 1 is only image classification; "
            "redshift estimation is intentionally outside the current benchmark scope.\n"
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
