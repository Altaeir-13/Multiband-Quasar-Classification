from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from multiband_qso.data.validation import validate_metadata


def _write_mapping(path: Path) -> None:
    path.write_text(json.dumps({"STAR": 0, "GALAXY": 1, "QSO": 2}), encoding="utf-8")


def _metadata_rows(tmp_path: Path) -> list[dict]:
    rows: list[dict] = []
    splits = ["train", "train", "val", "test"]
    for class_name, label in [("STAR", 0), ("GALAXY", 1), ("QSO", 2)]:
        class_dir = tmp_path / "images" / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for index, split in enumerate(splits):
            object_id = label * 100 + index
            image_path = class_dir / f"{object_id}.jpg"
            Image.new("RGB", (16, 16), color=(label * 40, 80, 120)).save(image_path)
            rows.append(
                {
                    "object_id": object_id,
                    "spec_obj_id": object_id + 100000,
                    "ra": 180.0,
                    "dec": 0.1,
                    "class": class_name,
                    "label": label,
                    "split": split,
                    "image_path": str(image_path),
                    "spec_z": 0.1,
                    "psfMag_u": 20.0,
                    "psfMag_g": 19.0,
                    "psfMag_r": 18.5,
                    "psfMag_i": 18.0,
                    "psfMag_z": 17.8,
                    "modelMag_u": 20.1,
                    "modelMag_g": 19.1,
                    "modelMag_r": 18.6,
                    "modelMag_i": 18.1,
                    "modelMag_z": 17.9,
                    "extinction_u": 0.1,
                    "extinction_g": 0.08,
                    "extinction_r": 0.06,
                    "extinction_i": 0.04,
                    "extinction_z": 0.03,
                    "u_g": 1.0,
                    "g_r": 0.5,
                    "r_i": 0.5,
                    "i_z": 0.2,
                }
            )
    return rows


def _write_metadata(tmp_path: Path, rows: list[dict]) -> tuple[Path, Path, Path]:
    metadata_path = tmp_path / "metadata.csv"
    mapping_path = tmp_path / "class_mapping.json"
    output_path = tmp_path / "metadata_validation.json"
    pd.DataFrame(rows).to_csv(metadata_path, index=False)
    _write_mapping(mapping_path)
    return metadata_path, mapping_path, output_path


def test_validate_metadata_accepts_valid_fixture(tmp_path: Path) -> None:
    metadata_path, mapping_path, output_path = _write_metadata(tmp_path, _metadata_rows(tmp_path))

    report = validate_metadata(
        metadata_csv=metadata_path,
        class_mapping_json=mapping_path,
        output_json=output_path,
        expected_classes=["STAR", "GALAXY", "QSO"],
    )

    assert report["valid"] is True
    assert report["rows"] == 12
    assert report["image_status"]["ok"] == 12
    assert output_path.exists()


def test_validate_metadata_rejects_duplicate_object_id(tmp_path: Path) -> None:
    rows = _metadata_rows(tmp_path)
    rows[1]["object_id"] = rows[0]["object_id"]
    metadata_path, mapping_path, output_path = _write_metadata(tmp_path, rows)

    with pytest.raises(ValueError, match="Duplicate object_id"):
        validate_metadata(
            metadata_csv=metadata_path,
            class_mapping_json=mapping_path,
            output_json=output_path,
            expected_classes=["STAR", "GALAXY", "QSO"],
        )


def test_validate_metadata_rejects_inconsistent_label(tmp_path: Path) -> None:
    rows = _metadata_rows(tmp_path)
    rows[0]["label"] = 2
    metadata_path, mapping_path, output_path = _write_metadata(tmp_path, rows)

    with pytest.raises(ValueError, match="Label values inconsistent"):
        validate_metadata(
            metadata_csv=metadata_path,
            class_mapping_json=mapping_path,
            output_json=output_path,
            expected_classes=["STAR", "GALAXY", "QSO"],
        )


def test_validate_metadata_rejects_missing_image(tmp_path: Path) -> None:
    rows = _metadata_rows(tmp_path)
    Path(rows[0]["image_path"]).unlink()
    metadata_path, mapping_path, output_path = _write_metadata(tmp_path, rows)

    with pytest.raises(ValueError, match="image files are missing or invalid"):
        validate_metadata(
            metadata_csv=metadata_path,
            class_mapping_json=mapping_path,
            output_json=output_path,
            expected_classes=["STAR", "GALAXY", "QSO"],
        )
