"""Image model factory."""

from multiband_qso.image_models.factory import create_image_model
from multiband_qso.image_models.simple_cnn import SimpleCNN

__all__ = ["SimpleCNN", "create_image_model"]
