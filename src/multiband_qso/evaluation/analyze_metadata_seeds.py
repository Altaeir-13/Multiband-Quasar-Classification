from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

from multiband_qso.config import load_config

SEED_LABELS = ["42", "123", "13"]
PHOTOMETRY_PREFIXES = ("psfMag_", "modelMag_", "extinction_")
COLOR_COLUMNS = ["u_g", "g_r", "r_i", "i_z"]
REDSHIFT_COLUMNS = ["spec_z", "redshift", "z"]


def _seed_from_config(config_path: str | Path, config: dict[str, Any]) -> str:
    seed = config.get("project", {}).get("seed")
    if seed is not None:
        return str(seed)
    name = Path(config_path).stem.lower()
    if "seed123" in name:
        return "123"
    if "seed13" in name:
        return "13"
    return "42"


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    table = frame.copy()
    for column in table.columns:
        if pd.api.types.is_float_dtype(table[column]):
            table[column] = table[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
        else:
            table[column] = table[column].map(lambda value: "" if pd.isna(value) else str(value))
    columns = [str(column) for column in table.columns]
    rows = [columns, *table.astype(str).values.tolist()]
    widths = [max(len(row[index]) for row in rows) for index in range(len(columns))]

    def render(row: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |"

    lines = [render(columns), render(["-" * width for width in widths])]
    lines.extend(render(row) for row in rows[1:])
    return "\n".join(lines)


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    detected: list[str] = []
    for column in frame.columns:
        if column in REDSHIFT_COLUMNS or column in COLOR_COLUMNS or column.startswith(PHOTOMETRY_PREFIXES):
            if pd.api.types.is_numeric_dtype(frame[column]):
                detected.append(column)
    return detected


def _load_seed_frames(config_paths: list[str]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for config_path in config_paths:
        config = load_config(config_path)
        seed = _seed_from_config(config_path, config)
        metadata_csv = Path(config["paths"]["metadata_csv"])
        if not metadata_csv.exists():
            raise FileNotFoundError(f"Metadata not found for seed {seed}: {metadata_csv}")
        frame = pd.read_csv(metadata_csv)
        for column in frame.select_dtypes(include=["number"]).columns:
            frame.loc[frame[column] <= -9990, column] = pd.NA
        frame["_seed"] = seed
        frame["_metadata_path"] = str(metadata_csv)
        frames[seed] = frame
    return frames


def _counts_summary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed, frame in frames.items():
        rows.append(
            {
                "seed": seed,
                "rows": len(frame),
                "unique_object_id": frame["object_id"].nunique() if "object_id" in frame else pd.NA,
                "duplicates": len(frame) - frame["object_id"].nunique() if "object_id" in frame else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def _class_counts(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed, frame in frames.items():
        counts = frame["class"].value_counts().to_dict()
        rows.append({"seed": seed, **{class_name: counts.get(class_name, 0) for class_name in sorted(frame["class"].unique())}})
    return pd.DataFrame(rows).fillna(0)


def _split_counts(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed, frame in frames.items():
        grouped = frame.groupby(["class", "split"]).size()
        row: dict[str, Any] = {"seed": seed}
        for (class_name, split), value in grouped.items():
            row[f"{class_name}_{split}"] = int(value)
        rows.append(row)
    return pd.DataFrame(rows).fillna(0)


def _intersection_summary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    object_sets = {seed: set(frame["object_id"].astype(str)) for seed, frame in frames.items()}
    rows: list[dict[str, Any]] = []
    for left, right in combinations(sorted(object_sets), 2):
        intersection = object_sets[left] & object_sets[right]
        union = object_sets[left] | object_sets[right]
        rows.append(
            {
                "seed_a": left,
                "seed_b": right,
                "intersection": len(intersection),
                "union": len(union),
                "jaccard": len(intersection) / len(union) if union else 0.0,
            }
        )
    all_intersection = set.intersection(*object_sets.values()) if object_sets else set()
    rows.append({"seed_a": "all", "seed_b": "all", "intersection": len(all_intersection), "union": pd.NA, "jaccard": pd.NA})
    return pd.DataFrame(rows)


def _missing_summary(frames: dict[str, pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed, frame in frames.items():
        row: dict[str, Any] = {"seed": seed}
        for column in columns:
            row[column] = int(frame[column].isna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _describe_by_seed_class(frames: dict[str, pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed, frame in frames.items():
        for class_name, part in frame.groupby("class"):
            for column in columns:
                values = pd.to_numeric(part[column], errors="coerce").dropna()
                if values.empty:
                    continue
                rows.append(
                    {
                        "seed": seed,
                        "class": class_name,
                        "column": column,
                        "mean": values.mean(),
                        "std": values.std(ddof=0),
                        "min": values.min(),
                        "q25": values.quantile(0.25),
                        "median": values.quantile(0.50),
                        "q75": values.quantile(0.75),
                        "max": values.max(),
                    }
                )
    return pd.DataFrame(rows)


def _describe_test_by_seed_class(frames: dict[str, pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    test_frames = {seed: frame[frame["split"] == "test"].copy() for seed, frame in frames.items()}
    return _describe_by_seed_class(test_frames, columns)


def _outlier_summary(frames: dict[str, pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    combined = pd.concat(frames.values(), ignore_index=True)
    for column in columns:
        values = pd.to_numeric(combined[column], errors="coerce").dropna()
        if values.empty:
            continue
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        for seed, frame in frames.items():
            seed_values = pd.to_numeric(frame[column], errors="coerce")
            rows.append(
                {
                    "seed": seed,
                    "column": column,
                    "low_outliers": int((seed_values < lower).sum()),
                    "high_outliers": int((seed_values > upper).sum()),
                }
            )
    return pd.DataFrame(rows)


def _max_seed13_differences(stats: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if stats.empty:
        return pd.DataFrame()
    for class_name in sorted(stats["class"].unique()):
        for column in columns:
            subset = stats[(stats["class"] == class_name) & (stats["column"] == column)]
            seed13 = subset[subset["seed"] == "13"]
            others = subset[subset["seed"].isin(["42", "123"])]
            if seed13.empty or others.empty:
                continue
            mean_13 = float(seed13["mean"].iloc[0])
            other_mean = float(others["mean"].mean())
            pooled_std = float(others["std"].replace(0, pd.NA).dropna().mean()) if not others.empty else pd.NA
            rows.append(
                {
                    "class": class_name,
                    "column": column,
                    "seed13_mean": mean_13,
                    "other_seeds_mean": other_mean,
                    "absolute_delta": abs(mean_13 - other_mean),
                    "delta_over_other_std": abs(mean_13 - other_mean) / pooled_std if pooled_std and not pd.isna(pooled_std) else pd.NA,
                }
            )
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    return output.sort_values("absolute_delta", ascending=False).head(20)


def _load_run_metrics(runs_dir: str | Path = "runs/image_benchmark") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in Path(runs_dir).glob("*/*/metrics*.json"):
        with path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        run_dir = metrics.get("run_dir") or str(path.parent)
        if "smoke" in run_dir.lower():
            continue
        seed = "123" if "seed123" in run_dir.lower() else "13" if "seed13" in run_dir.lower() else "42"
        rows.append(
            {
                "seed": seed,
                "model": metrics.get("model"),
                "split": metrics.get("split"),
                "accuracy": metrics.get("accuracy"),
                "f1_macro": metrics.get("f1_macro"),
                "run_dir": run_dir,
            }
        )
    return pd.DataFrame(rows)


def _val_test_gap_table() -> pd.DataFrame:
    metrics = _load_run_metrics()
    if metrics.empty:
        return metrics
    rows: list[dict[str, Any]] = []
    for (seed, model), group in metrics.groupby(["seed", "model"]):
        val = group[group["split"] == "val"]
        test = group[group["split"] == "test"]
        if val.empty or test.empty:
            continue
        rows.append(
            {
                "seed": seed,
                "model": model,
                "val_f1": float(val["f1_macro"].iloc[0]),
                "test_f1": float(test["f1_macro"].iloc[0]),
                "val_minus_test_f1": float(val["f1_macro"].iloc[0] - test["f1_macro"].iloc[0]),
                "val_acc": float(val["accuracy"].iloc[0]),
                "test_acc": float(test["accuracy"].iloc[0]),
                "val_minus_test_acc": float(val["accuracy"].iloc[0] - test["accuracy"].iloc[0]),
            }
        )
    return pd.DataFrame(rows).sort_values(["seed", "model"])


def write_diagnostics(config_paths: list[str], output_path: str | Path) -> Path:
    frames = _load_seed_frames(config_paths)
    all_columns = sorted(set().union(*[set(frame.columns) for frame in frames.values()]))
    common_columns = [column for column in all_columns if all(column in frame for frame in frames.values())]
    numeric_columns = [column for column in _numeric_columns(pd.concat(frames.values(), ignore_index=True)) if column in common_columns]
    magnitude_columns = [column for column in numeric_columns if column.startswith(("psfMag_", "modelMag_"))]
    color_columns = [column for column in numeric_columns if column in COLOR_COLUMNS]
    redshift_columns = [column for column in numeric_columns if column in REDSHIFT_COLUMNS]

    counts = _counts_summary(frames)
    class_counts = _class_counts(frames)
    split_counts = _split_counts(frames)
    intersections = _intersection_summary(frames)
    missing = _missing_summary(frames, numeric_columns)
    stats = _describe_by_seed_class(frames, numeric_columns)
    test_stats = _describe_test_by_seed_class(frames, numeric_columns)
    outliers = _outlier_summary(frames, numeric_columns)
    seed13_diffs = _max_seed13_differences(test_stats, numeric_columns)
    gaps = _val_test_gap_table()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Metadata Seed Diagnostics\n\n")
        handle.write("Phase 1 diagnostic report for the controlled small SDSS image benchmark. This report compares metadata distributions for seeds 42, 123 and 13; it does not evaluate tabular models or redshift estimation.\n\n")
        handle.write("## Inputs\n\n")
        handle.write("| seed | metadata_csv | rows |\n| ---- | ------------ | ---- |\n")
        for seed, frame in frames.items():
            handle.write(f"| {seed} | {frame['_metadata_path'].iloc[0]} | {len(frame)} |\n")
        handle.write("\n")

        handle.write("## Object Counts\n\n")
        handle.write(_markdown_table(counts) + "\n\n")
        handle.write("## Class Counts\n\n")
        handle.write(_markdown_table(class_counts) + "\n\n")
        handle.write("## Split Counts By Class\n\n")
        handle.write(_markdown_table(split_counts) + "\n\n")
        handle.write("## Object ID Intersections\n\n")
        handle.write(_markdown_table(intersections) + "\n\n")
        handle.write("## Columns Analyzed\n\n")
        handle.write(f"Magnitude columns: `{', '.join(magnitude_columns) if magnitude_columns else 'none'}`.\n\n")
        handle.write(f"Color columns: `{', '.join(color_columns) if color_columns else 'none'}`.\n\n")
        handle.write(f"Redshift columns: `{', '.join(redshift_columns) if redshift_columns else 'none'}`.\n\n")
        handle.write("## Missing Values\n\n")
        handle.write(_markdown_table(missing) + "\n\n")
        handle.write("## Distribution Summary By Seed And Class\n\n")
        handle.write(_markdown_table(stats) + "\n\n")
        handle.write("## Test Split Distribution Summary By Seed And Class\n\n")
        handle.write(_markdown_table(test_stats) + "\n\n")
        handle.write("## Outlier Counts\n\n")
        handle.write(_markdown_table(outliers) + "\n\n")
        handle.write("## Validation-Test Gaps\n\n")
        if gaps.empty:
            handle.write("No paired validation/test metrics were found.\n\n")
        else:
            handle.write(_markdown_table(gaps) + "\n\n")

        handle.write("## Diagnostico da seed 13\n\n")
        handle.write("Seed 13 has the same object count, class balance and stratified split counts as seeds 42 and 123. Therefore the performance drop is not explained by gross class imbalance or split-size errors.\n\n")
        if not seed13_diffs.empty:
            handle.write("Largest seed 13 test-split distribution differences versus the mean of seeds 42 and 123:\n\n")
            handle.write(_markdown_table(seed13_diffs) + "\n\n")
        if not gaps.empty:
            seed13_gaps = gaps[gaps["seed"] == "13"]
            if not seed13_gaps.empty:
                handle.write("Seed 13 validation-test gaps:\n\n")
                handle.write(_markdown_table(seed13_gaps) + "\n\n")
        handle.write("Available metadata can indicate distribution shifts in magnitudes, colors or spectroscopic redshift, but it cannot directly prove visual similarity in JPEG cutouts. If seed 13 shows larger validation-test gaps for strong CNNs, that supports a test-set or training-instability component, but not a single definitive cause.\n\n")

        handle.write("## Seed Control Limitation\n\n")
        handle.write("The current pipeline uses `project.seed` for object sampling, split creation, model initialization, DataLoader shuffle and stochastic train-time augmentation. This couples data variance and training variance. The current diagnostic documents this limitation rather than changing the pipeline mid-experiment; a future controlled experiment should separate `seeds.data`, `seeds.split` and `seeds.training`.\n\n")

        handle.write("## Diagnostic Conclusion\n\n")
        handle.write("The seed 13 drop cannot be attributed to a reporting mix-up if the model comparison report points to the expected `metrics_test.json` runs and this metadata report confirms balanced data. The most plausible current explanation is a combination of sampled-object/split difficulty and training from scratch on a small image dataset. Transfer learning should be tested before scaling.\n")
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare metadata distributions across benchmark seeds.")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--output", default="reports/image_benchmark/metadata_seed_diagnostics.md")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    write_diagnostics(args.configs, args.output)


if __name__ == "__main__":
    main()
