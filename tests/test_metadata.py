from __future__ import annotations

import pandas as pd

from multiband_qso.data.metadata import CLASS_MAPPING, create_metadata_from_dataframe


def _candidate_rows(class_name: str, start: int, count: int) -> list[dict]:
    rows = []
    for offset in range(count):
        object_id = start + offset
        rows.append(
            {
                "object_id": object_id,
                "spec_obj_id": object_id + 100000,
                "ra": 180.0 + offset * 0.001,
                "dec": -1.0 + offset * 0.001,
                "class": class_name,
                "spec_z": 0.1 + offset * 0.01,
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
            }
        )
    return rows


def test_create_metadata_balances_classes_and_preserves_multimodal_columns() -> None:
    candidates = pd.DataFrame(
        _candidate_rows("STAR", 1000, 12)
        + _candidate_rows("GALAXY", 2000, 12)
        + _candidate_rows("QSO", 3000, 12)
    )

    metadata = create_metadata_from_dataframe(
        candidates,
        n_per_class=10,
        image_root="data/raw/images",
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
        seed=7,
    )

    assert metadata["object_id"].is_unique
    assert metadata.groupby("class").size().to_dict() == {"GALAXY": 10, "QSO": 10, "STAR": 10}
    assert metadata.groupby(["split", "class"]).size().min() > 0
    assert set(metadata["label"]) == set(CLASS_MAPPING.values())
    assert {"u_g", "g_r", "r_i", "i_z", "image_path", "spec_z"}.issubset(metadata.columns)
    assert metadata["image_path"].str.endswith(".jpg").all()
