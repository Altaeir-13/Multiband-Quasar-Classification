from __future__ import annotations

import json

import pandas as pd
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from multiband_qso.datasets import ImageMetadataDataset, build_image_transforms  # noqa: E402


def test_image_metadata_dataset_loads_images_and_labels(tmp_path) -> None:
    image_dir = tmp_path / "images" / "STAR"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "1.jpg"
    Image.new("RGB", (32, 32), color=(120, 80, 40)).save(image_path)

    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame(
        [
            {
                "object_id": 1,
                "class": "STAR",
                "label": 0,
                "split": "train",
                "image_path": str(image_path),
            }
        ]
    ).to_csv(metadata_path, index=False)
    mapping_path = tmp_path / "class_mapping.json"
    mapping_path.write_text(json.dumps({"STAR": 0, "GALAXY": 1, "QSO": 2}), encoding="utf-8")

    dataset = ImageMetadataDataset(
        metadata_path,
        split="train",
        class_mapping_json=mapping_path,
        transform=build_image_transforms(image_size=32, train=False),
    )

    sample = dataset[0]
    assert tuple(sample["image"].shape) == (3, 32, 32)
    assert sample["label"].item() == 0
    assert sample["object_id"] == 1
