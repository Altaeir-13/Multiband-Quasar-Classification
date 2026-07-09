from __future__ import annotations

from multiband_qso.config import load_config


def test_smoke_config_is_small_and_cpu_bound() -> None:
    config = load_config("configs/image_benchmark_smoke.yaml")

    assert config["sdss"]["classes"] == ["STAR", "GALAXY", "QSO"]
    assert config["sdss"]["candidate_rows_per_class"] == 30
    assert config["sdss"]["science_primary"] is False
    assert config["sdss"]["zwarning_zero"] is False
    assert config["sdss"]["clean_photometry"] is False
    assert config["sample"]["n_per_class"] == 10
    assert config["sample"]["train_size"] == 0.60
    assert config["sample"]["val_size"] == 0.20
    assert config["sample"]["test_size"] == 0.20
    assert config["training"]["epochs"] == 2
    assert config["training"]["batch_size"] == 8
    assert config["training"]["num_workers"] == 0
    assert config["training"]["device"] == "cpu"


def test_small_config_is_separate_and_controlled() -> None:
    config = load_config("configs/image_benchmark_small.yaml")

    assert config["paths"]["metadata_csv"] == "data/processed/image_sample_small/metadata.csv"
    assert config["paths"]["image_root"] == "data/raw/images_small"
    assert config["sdss"]["classes"] == ["STAR", "GALAXY", "QSO"]
    assert config["sdss"]["candidate_rows_per_class"] == 1000
    assert config["sample"]["n_per_class"] == 300
    assert config["sample"]["train_size"] == 0.70
    assert config["sample"]["val_size"] == 0.15
    assert config["sample"]["test_size"] == 0.15
    assert config["training"]["epochs"] == 10
    assert config["training"]["batch_size"] == 32
    assert config["training"]["num_workers"] == 0
    assert config["training"]["device"] == "auto"
    assert config["training"]["patience"] == 3