from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from multiband_qso.image_models import create_image_model  # noqa: E402


@pytest.mark.parametrize("model_name", ["simple_cnn", "resnet18", "densenet121"])
def test_image_models_return_three_class_logits(model_name: str) -> None:
    model = create_image_model(model_name, num_classes=3, pretrained=False)
    model.eval()
    with torch.no_grad():
        logits = model(torch.zeros(2, 3, 128, 128))
    assert tuple(logits.shape) == (2, 3)
