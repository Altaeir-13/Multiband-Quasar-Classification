from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _flatten_metrics(metrics: dict) -> dict:
    return {
        "model": metrics.get("model"),
        "split": metrics.get("split"),
        "accuracy": metrics.get("accuracy"),
        "precision_macro": metrics.get("precision_macro"),
        "recall_macro": metrics.get("recall_macro"),
        "f1_macro": metrics.get("f1_macro"),
        "run_dir": metrics.get("run_dir") or str(Path(metrics.get("checkpoint", "")).parent),
    }


def collect_metrics(runs_dir: str | Path) -> pd.DataFrame:
    rows: list[dict] = []
    for metrics_path in Path(runs_dir).glob("*/*/metrics*.json"):
        with metrics_path.open("r", encoding="utf-8") as handle:
            rows.append(_flatten_metrics(json.load(handle)))
    if not rows:
        return pd.DataFrame(
            columns=[
                "model",
                "split",
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
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


def write_model_comparison(
    metrics: pd.DataFrame,
    *,
    output_csv: str | Path,
    output_md: str | Path,
) -> None:
    csv_path = Path(output_csv)
    md_path = Path(output_md)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(csv_path, index=False)

    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Image Benchmark Results\n\n")
        if metrics.empty:
            handle.write("No metrics found yet. Run training/evaluation first.\n")
            return
        display = metrics.copy()
        for column in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
            display[column] = display[column].map(lambda value: f"{value:.4f}")
        handle.write(_markdown_table(display.reset_index(drop=True)))
        handle.write("\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect image benchmark metrics.")
    parser.add_argument("--runs-dir", default="runs/image_benchmark")
    parser.add_argument("--output-csv", default="reports/image_benchmark/metrics.csv")
    parser.add_argument("--output-md", default="reports/image_benchmark/model_comparison.md")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    metrics = collect_metrics(args.runs_dir)
    write_model_comparison(metrics, output_csv=args.output_csv, output_md=args.output_md)


if __name__ == "__main__":
    main()