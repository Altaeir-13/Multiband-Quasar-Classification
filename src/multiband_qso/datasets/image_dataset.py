from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pandas as pd
from PIL import Image

from multiband_qso.data.metadata import CLASS_MAPPING

try:
    import torch
    from torch.utils.data import Dataset
    from torchvision import transforms
except ImportError:  # pragma: no cover - covered by optional dependency environments.
    torch = None
    transforms = None

    class Dataset:  # type: ignore[no-redef]
        pass


def _require_torch() -> None:
    if torch is None or transforms is None:
        raise ImportError("torch and torchvision are required for image datasets")


def load_class_mapping(path: str | Path | None) -> dict[str, int]:
    if path is None:
        return dict(CLASS_MAPPING)
    with Path(path).open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    return {str(key): int(value) for key, value in mapping.items()}


def build_image_transforms(
    *,
    image_size: int = 128,
    train: bool = False,
    imagenet_normalize: bool = True,
) -> Callable:
    """Build the standard transforms for the image benchmark."""
    _require_torch()
    steps: list[Callable] = [transforms.Resize((image_size, image_size))]
    if train:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=15),
            ]
        )
    steps.append(transforms.ToTensor())
    if imagenet_normalize:
        steps.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )
    return transforms.Compose(steps)


class ImageMetadataDataset(Dataset):
    """Image dataset backed by the multimodal metadata CSV."""

    def __init__(
        self,
        metadata_csv: str | Path,
        *,
        split: str | None = None,
        class_mapping_json: str | Path | None = None,
        transform: Callable | None = None,
        image_root: str | Path | None = None,
    ) -> None:
        _require_torch()
        self.metadata_csv = Path(metadata_csv)
        self.metadata = pd.read_csv(self.metadata_csv)
        if split is not None:
            self.metadata = self.metadata[self.metadata["split"] == split].reset_index(drop=True)
        if self.metadata.empty:
            raise ValueError(f"No rows found for split={split!r} in {metadata_csv}")

        required = {"image_path", "class"}
        missing = required.difference(self.metadata.columns)
        if missing:
            raise ValueError(f"Metadata is missing required image dataset columns: {sorted(missing)}")

        self.class_mapping = load_class_mapping(class_mapping_json)
        self.transform = transform or build_image_transforms(train=(split == "train"))
        self.image_root = Path(image_root) if image_root is not None else None

    def __len__(self) -> int:
        return len(self.metadata)

    def _resolve_image_path(self, raw_path: str | Path) -> Path:
        image_path = Path(raw_path)
        if image_path.is_absolute() or self.image_root is None:
            return image_path
        return image_path

    def __getitem__(self, index: int):
        row = self.metadata.iloc[index]
        image_path = self._resolve_image_path(row["image_path"])
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        label = int(row["label"]) if "label" in row else self.class_mapping[str(row["class"])]
        return {
            "image": tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "object_id": row.get("object_id", ""),
            "class": row["class"],
        }
