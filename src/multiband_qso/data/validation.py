from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, UnidentifiedImageError

from multiband_qso.config import ensure_parent, load_config
from multiband_qso.data.metadata import CLASS_MAPPING, REQUIRED_COLUMNS

DERIVED_COLOR_COLUMNS = {"u_g", "g_r", "r_i", "i_z"}
METADATA_EXTRA_COLUMNS = {"label", "split", "image_path"}
REQUIRED_METADATA_COLUMNS = REQUIRED_COLUMNS | DERIVED_COLOR_COLUMNS | METADATA_EXTRA_COLUMNS


def load_mapping(path: str | Path) -> dict[str, int]:
    mapping_path = Path(path)
    if not mapping_path.exists():
        raise ValueError(f"class_mapping.json not found: {mapping_path}")
    with mapping_path.open("r", encoding="utf-8") as handle:
        raw_mapping = json.load(handle)
    return {str(key): int(value) for key, value in raw_mapping.items()}


def _image_is_openable(path: Path) -> tuple[bool, str | None]:
    if not path.exists():
        return False, "missing"
    if not path.is_file():
        return False, "not_a_file"
    try:
        with Image.open(path) as image:
            image.verify()
        return True, None
    except (OSError, UnidentifiedImageError) as exc:
        return False, str(exc)


def _validate_splits(frame: pd.DataFrame, expected_classes: set[str]) -> list[str]:
    errors: list[str] = []
    expected_splits = {"train", "val", "test"}
    actual_splits = set(frame["split"].astype(str))
    if actual_splits != expected_splits:
        errors.append(f"Expected splits {sorted(expected_splits)}, found {sorted(actual_splits)}")
        return errors

    counts = frame.groupby(["split", "class"]).size()
    missing_pairs = [
        f"{split}/{class_name}"
        for split in sorted(expected_splits)
        for class_name in sorted(expected_classes)
        if counts.get((split, class_name), 0) == 0
    ]
    if missing_pairs:
        errors.append(f"Missing stratified split/class rows: {missing_pairs}")
    return errors


def validate_metadata(
    *,
    metadata_csv: str | Path,
    class_mapping_json: str | Path,
    output_json: str | Path,
    expected_classes: list[str] | None = None,
    require_images: bool = True,
) -> dict[str, Any]:
    """Validate generated multimodal metadata and write a JSON report."""
    metadata_path = Path(metadata_csv)
    if not metadata_path.exists():
        raise ValueError(f"metadata.csv not found: {metadata_path}")

    mapping = load_mapping(class_mapping_json)
    expected = set(expected_classes or mapping.keys() or CLASS_MAPPING.keys())
    frame = pd.read_csv(metadata_path)
    errors: list[str] = []
    warnings: list[str] = []

    missing_columns = sorted(REQUIRED_METADATA_COLUMNS.difference(frame.columns))
    if missing_columns:
        errors.append(f"Missing required metadata columns: {missing_columns}")

    if frame.empty:
        errors.append("metadata.csv is empty")

    if "object_id" in frame and frame["object_id"].duplicated().any():
        duplicate_ids = frame.loc[frame["object_id"].duplicated(), "object_id"].head(10).tolist()
        errors.append(f"Duplicate object_id values found, examples: {duplicate_ids}")

    if "class" in frame:
        actual_classes = set(frame["class"].astype(str))
        if actual_classes != expected:
            errors.append(f"Expected classes {sorted(expected)}, found {sorted(actual_classes)}")
    else:
        actual_classes = set()

    class_counts = (
        frame["class"].astype(str).value_counts().sort_index().to_dict() if "class" in frame else {}
    )
    if class_counts and len(set(class_counts.values())) != 1:
        errors.append(f"Classes are not balanced: {class_counts}")

    if {"class", "label"}.issubset(frame.columns):
        expected_labels = frame["class"].astype(str).map(mapping)
        mismatched = frame[expected_labels != frame["label"]]
        if not mismatched.empty:
            examples = mismatched[["object_id", "class", "label"]].head(10).to_dict("records")
            errors.append(f"Label values inconsistent with class_mapping.json, examples: {examples}")

    if {"split", "class"}.issubset(frame.columns) and actual_classes:
        errors.extend(_validate_splits(frame, expected))

    image_status = Counter()
    invalid_images: list[dict[str, Any]] = []
    if require_images and "image_path" in frame:
        for row in frame[["object_id", "image_path"]].itertuples(index=False):
            raw_path = str(row.image_path)
            if not raw_path or raw_path.lower() == "nan":
                image_status["invalid_path"] += 1
                invalid_images.append({"object_id": row.object_id, "image_path": raw_path, "error": "empty"})
                continue
            image_path = Path(raw_path)
            ok, error = _image_is_openable(image_path)
            if ok:
                image_status["ok"] += 1
            else:
                image_status["invalid"] += 1
                invalid_images.append(
                    {"object_id": row.object_id, "image_path": raw_path, "error": error}
                )
        if invalid_images:
            errors.append(f"{len(invalid_images)} image files are missing or invalid")
    elif require_images:
        warnings.append("Image validation skipped because image_path column is missing")

    report = {
        "metadata_csv": str(metadata_path),
        "class_mapping_json": str(class_mapping_json),
        "rows": int(len(frame)),
        "classes": sorted(actual_classes),
        "class_counts": class_counts,
        "split_counts": (
            frame.groupby(["split", "class"]).size().unstack(fill_value=0).to_dict()
            if {"split", "class"}.issubset(frame.columns)
            else {}
        ),
        "image_status": dict(image_status),
        "invalid_images": invalid_images[:50],
        "errors": errors,
        "warnings": warnings,
        "valid": not errors,
    }

    output_path = ensure_parent(output_json)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    if errors:
        joined = "; ".join(errors)
        raise ValueError(f"Metadata validation failed: {joined}")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate generated SDSS image metadata.")
    parser.add_argument("--config", default="configs/image_benchmark.yaml")
    parser.add_argument("--metadata-csv", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--skip-images", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    expected_classes = config["sdss"].get("classes", list(CLASS_MAPPING))
    output_json = args.output_json or paths.get(
        "metadata_validation_json",
        str(Path(paths["metadata_csv"]).with_name("metadata_validation.json")),
    )

    report = validate_metadata(
        metadata_csv=args.metadata_csv or paths["metadata_csv"],
        class_mapping_json=paths["class_mapping_json"],
        output_json=output_json,
        expected_classes=expected_classes,
        require_images=not args.skip_images,
    )
    print(
        "Metadata validation passed: "
        f"{report['rows']} rows, class_counts={report['class_counts']}, "
        f"image_status={report['image_status']}"
    )


if __name__ == "__main__":
    main()
