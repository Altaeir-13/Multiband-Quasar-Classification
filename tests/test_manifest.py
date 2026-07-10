from __future__ import annotations

import pandas as pd

from multiband_qso.data.manifest import MANIFEST_COLUMNS, export_image_manifest


def test_export_image_manifest_keeps_only_reproducibility_columns(tmp_path) -> None:
    metadata_csv = tmp_path / "metadata.csv"
    output_csv = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "object_id": 1,
                "spec_obj_id": 10,
                "ra": 12.3,
                "dec": -1.5,
                "class": "QSO",
                "label": 2,
                "split": "test",
                "image_path": "data/raw/images/1.jpg",
            }
        ]
    ).to_csv(metadata_csv, index=False)

    manifest = export_image_manifest(metadata_csv, output_csv)

    assert list(manifest.columns) == MANIFEST_COLUMNS
    assert list(pd.read_csv(output_csv).columns) == MANIFEST_COLUMNS
