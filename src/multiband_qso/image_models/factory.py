from __future__ import annotations

from typing import Literal

try:
    from torch import nn
    from torchvision import models
except ImportError:  # pragma: no cover
    nn = None
    models = None

from multiband_qso.image_models.simple_cnn import SimpleCNN

ModelName = Literal["simple_cnn", "resnet18", "densenet121"]


def _require_torchvision() -> None:
    if nn is None or models is None:
        raise ImportError("torch and torchvision are required for model factory")


def create_image_model(
    name: str,
    *,
    num_classes: int = 3,
    pretrained: bool = False,
):
    """Create one of the supported image classification models."""
    normalized = name.lower()
    if normalized == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)

    _require_torchvision()
    if normalized == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if normalized == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    supported = "simple_cnn, resnet18, densenet121"
    raise ValueError(f"Unsupported image model {name!r}. Supported models: {supported}")
