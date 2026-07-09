from __future__ import annotations

from multiband_qso.evaluation.metrics import classification_metrics


def test_classification_metrics_returns_macro_and_per_class_values() -> None:
    metrics = classification_metrics(
        [0, 0, 1, 1, 2, 2],
        [0, 1, 1, 1, 2, 0],
        labels=[0, 1, 2],
        class_names=["STAR", "GALAXY", "QSO"],
    )

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_macro"] <= 1.0
    assert set(metrics["per_class"]) == {"STAR", "GALAXY", "QSO"}
    assert metrics["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
