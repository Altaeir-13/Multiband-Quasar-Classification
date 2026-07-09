from __future__ import annotations

import pandas as pd

from multiband_qso.evaluation.reporting import write_model_comparison


def test_write_model_comparison_renders_markdown_without_tabulate(tmp_path) -> None:
    metrics = pd.DataFrame(
        [
            {
                "model": "simple_cnn",
                "split": "test",
                "accuracy": 0.5,
                "precision_macro": 0.4,
                "recall_macro": 0.45,
                "f1_macro": 0.42,
                "run_dir": "runs/image_benchmark/simple_cnn/smoke-simple-cnn",
            }
        ]
    )
    output_csv = tmp_path / "metrics.csv"
    output_md = tmp_path / "model_comparison.md"

    write_model_comparison(metrics, output_csv=output_csv, output_md=output_md)

    content = output_md.read_text(encoding="utf-8")
    assert "| model" in content
    assert "simple_cnn" in content
    assert "0.4200" in content
    assert output_csv.exists()



def test_write_model_comparison_keeps_medium_separate(tmp_path) -> None:
    metrics = pd.DataFrame(
        [
            {
                "model": "simple_cnn",
                "split": "test",
                "accuracy": 0.60,
                "precision_macro": 0.61,
                "recall_macro": 0.60,
                "f1_macro": 0.59,
                "pretrained": False,
                "run_dir": "runs/image_benchmark/simple_cnn/small-simple-cnn",
            },
            {
                "model": "simple_cnn",
                "split": "test",
                "accuracy": 0.70,
                "precision_macro": 0.71,
                "recall_macro": 0.70,
                "f1_macro": 0.69,
                "pretrained": False,
                "run_dir": "runs/image_benchmark/simple_cnn/medium-simple-cnn",
            },
            {
                "model": "resnet18",
                "split": "test",
                "accuracy": 0.80,
                "precision_macro": 0.81,
                "recall_macro": 0.80,
                "f1_macro": 0.79,
                "pretrained": True,
                "run_dir": "runs/image_benchmark/resnet18/medium-resnet18-pretrained",
            },
        ]
    )
    output_csv = tmp_path / "metrics.csv"
    output_md = tmp_path / "model_comparison.md"

    write_model_comparison(metrics, output_csv=output_csv, output_md=output_md, runs_dir=tmp_path)

    content = output_md.read_text(encoding="utf-8")
    assert "## Benchmark medium - seed 42" in content
    assert "medium-simple-cnn" in content
    assert "medium-resnet18-pretrained" in content
    assert "### Small vs medium - seed 42" in content
    assert "| simple_cnn | 0.5900" in content
