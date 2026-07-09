from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import pandas as pd
from sklearn.model_selection import train_test_split

from multiband_qso.config import ensure_parent, load_config

CLASS_MAPPING: dict[str, int] = {"STAR": 0, "GALAXY": 1, "QSO": 2}

REQUIRED_COLUMNS = {
    "object_id",
    "spec_obj_id",
    "ra",
    "dec",
    "class",
    "spec_z",
    "psfMag_u",
    "psfMag_g",
    "psfMag_r",
    "psfMag_i",
    "psfMag_z",
    "modelMag_u",
    "modelMag_g",
    "modelMag_r",
    "modelMag_i",
    "modelMag_z",
    "extinction_u",
    "extinction_g",
    "extinction_r",
    "extinction_i",
    "extinction_z",
}

CANONICAL_COLUMN_NAMES = {
    "objid": "object_id",
    "objectid": "object_id",
    "bestobjid": "object_id",
    "specobjid": "spec_obj_id",
    "specobj_id": "spec_obj_id",
    "z": "spec_z",
}


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize likely SDSS/CAS column variants into the project schema."""
    rename: dict[str, str] = {}
    for column in frame.columns:
        compact = column.replace("_", "").lower()
        if compact in CANONICAL_COLUMN_NAMES:
            rename[column] = CANONICAL_COLUMN_NAMES[compact]
    return frame.rename(columns=rename)


def validate_catalog_schema(frame: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Catalog is missing required columns: {missing}")


def add_derived_colors(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["u_g"] = output["psfMag_u"] - output["psfMag_g"]
    output["g_r"] = output["psfMag_g"] - output["psfMag_r"]
    output["r_i"] = output["psfMag_r"] - output["psfMag_i"]
    output["i_z"] = output["psfMag_i"] - output["psfMag_z"]
    return output


def add_image_paths(frame: pd.DataFrame, image_root: str | Path) -> pd.DataFrame:
    output = frame.copy()
    root = Path(image_root)
    output["image_path"] = [
        str(root / class_name / f"{object_id}.jpg")
        for class_name, object_id in zip(output["class"], output["object_id"], strict=True)
    ]
    return output


def _stratified_split(
    frame: pd.DataFrame,
    *,
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int,
) -> pd.Series:
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-8:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    train_frame, temp_frame = train_test_split(
        frame,
        train_size=train_size,
        random_state=seed,
        stratify=frame["class"],
    )
    relative_test_size = test_size / (val_size + test_size)
    val_frame, test_frame = train_test_split(
        temp_frame,
        test_size=relative_test_size,
        random_state=seed,
        stratify=temp_frame["class"],
    )

    split = pd.Series(index=frame.index, dtype="object")
    split.loc[train_frame.index] = "train"
    split.loc[val_frame.index] = "val"
    split.loc[test_frame.index] = "test"
    return split


def create_metadata_from_dataframe(
    candidates: pd.DataFrame,
    *,
    n_per_class: int,
    image_root: str | Path,
    class_mapping: Mapping[str, int] | None = None,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Create the balanced multimodal metadata table used by all phases."""
    if n_per_class <= 0:
        raise ValueError("n_per_class must be positive")

    mapping = dict(class_mapping or CLASS_MAPPING)
    frame = normalize_columns(candidates)
    validate_catalog_schema(frame)

    frame = frame.copy()
    frame["class"] = frame["class"].astype(str).str.upper()
    frame = frame[frame["class"].isin(mapping)].copy()
    frame = frame.drop_duplicates(subset=["object_id"], keep="first")

    sampled_parts: list[pd.DataFrame] = []
    for class_name in mapping:
        class_rows = frame[frame["class"] == class_name]
        if len(class_rows) < n_per_class:
            raise ValueError(
                f"Class {class_name} has only {len(class_rows)} rows; "
                f"need at least {n_per_class}."
            )
        sampled_parts.append(class_rows.sample(n=n_per_class, random_state=seed))

    metadata = pd.concat(sampled_parts, ignore_index=True)
    metadata = metadata.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    metadata["label"] = metadata["class"].map(mapping).astype(int)
    metadata["split"] = _stratified_split(
        metadata,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
    )
    metadata = add_derived_colors(metadata)
    metadata = add_image_paths(metadata, image_root=image_root)

    preferred_columns = [
        "object_id",
        "spec_obj_id",
        "ra",
        "dec",
        "class",
        "label",
        "split",
        "image_path",
        "spec_z",
        "psfMag_u",
        "psfMag_g",
        "psfMag_r",
        "psfMag_i",
        "psfMag_z",
        "modelMag_u",
        "modelMag_g",
        "modelMag_r",
        "modelMag_i",
        "modelMag_z",
        "extinction_u",
        "extinction_g",
        "extinction_r",
        "extinction_i",
        "extinction_z",
        "u_g",
        "g_r",
        "r_i",
        "i_z",
    ]
    remaining_columns = [column for column in metadata.columns if column not in preferred_columns]
    return metadata[preferred_columns + remaining_columns]


def build_metadata(
    *,
    candidates_csv: str | Path,
    metadata_csv: str | Path,
    class_mapping_json: str | Path,
    image_root: str | Path,
    n_per_class: int,
    class_mapping: Mapping[str, int] | None = None,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    candidates = pd.read_csv(candidates_csv)
    metadata = create_metadata_from_dataframe(
        candidates,
        n_per_class=n_per_class,
        image_root=image_root,
        class_mapping=class_mapping,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
    )

    metadata_path = ensure_parent(metadata_csv)
    metadata.to_csv(metadata_path, index=False)

    mapping_path = ensure_parent(class_mapping_json)
    with mapping_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(class_mapping or CLASS_MAPPING), handle, indent=2, sort_keys=True)
        handle.write("\n")

    return metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build balanced multimodal metadata.")
    parser.add_argument("--config", default="configs/image_benchmark.yaml")
    parser.add_argument("--candidates-csv", default=None)
    parser.add_argument("--metadata-csv", default=None)
    parser.add_argument("--n-per-class", type=int, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_config(args.config)
    paths_config = config["paths"]
    sample_config = config["sample"]
    project_config = config["project"]

    build_metadata(
        candidates_csv=args.candidates_csv or paths_config["candidates_csv"],
        metadata_csv=args.metadata_csv or paths_config["metadata_csv"],
        class_mapping_json=paths_config["class_mapping_json"],
        image_root=paths_config["image_root"],
        n_per_class=args.n_per_class or sample_config["n_per_class"],
        class_mapping=project_config.get("class_mapping", CLASS_MAPPING),
        train_size=sample_config.get("train_size", 0.70),
        val_size=sample_config.get("val_size", 0.15),
        test_size=sample_config.get("test_size", 0.15),
        seed=project_config.get("seed", 42),
    )


if __name__ == "__main__":
    main()
