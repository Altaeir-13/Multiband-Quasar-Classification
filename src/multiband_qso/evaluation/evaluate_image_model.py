from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm

from multiband_qso.config import load_config
from multiband_qso.data.metadata import CLASS_MAPPING
from multiband_qso.datasets import ImageMetadataDataset, build_image_transforms
from multiband_qso.evaluation.metrics import classification_metrics
from multiband_qso.evaluation.plots import save_confusion_matrix_plot
from multiband_qso.image_models import create_image_model
from multiband_qso.training.utils import get_device, require_torch, save_json

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _class_names_from_mapping(mapping: dict[str, int]) -> list[str]:
    return [class_name for class_name, _ in sorted(mapping.items(), key=lambda item: item[1])]


def evaluate_checkpoint(
    *,
    config_path: str | Path,
    model_name: str,
    checkpoint_path: str | Path,
    split: str = "test",
    output_dir: str | Path | None = None,
) -> dict:
    require_torch()
    config = load_config(config_path)
    paths = config["paths"]
    training = config["training"]
    class_mapping = config["project"].get("class_mapping", CLASS_MAPPING)
    class_names = _class_names_from_mapping(class_mapping)
    labels = list(range(len(class_mapping)))
    device = get_device(training.get("device", "auto"))

    dataset = ImageMetadataDataset(
        paths["metadata_csv"],
        split=split,
        class_mapping_json=paths["class_mapping_json"],
        transform=build_image_transforms(image_size=int(training.get("image_size", 128)), train=False),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(training.get("batch_size", 64)),
        shuffle=False,
        num_workers=int(training.get("num_workers", 2)),
    )
    model = create_image_model(
        model_name,
        num_classes=len(class_mapping),
        pretrained=False,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"evaluate:{split}", leave=False):
            images = batch["image"].to(device)
            labels_tensor = batch["label"].to(device)
            logits = model(images)
            predictions = logits.argmax(dim=1)
            y_true.extend(labels_tensor.detach().cpu().tolist())
            y_pred.extend(predictions.detach().cpu().tolist())

    metrics = classification_metrics(
        y_true,
        y_pred,
        labels=labels,
        class_names=class_names,
    )
    metrics["model"] = model_name
    metrics["split"] = split
    metrics["checkpoint"] = str(checkpoint_path)

    destination = Path(output_dir) if output_dir is not None else Path(checkpoint_path).parent
    destination.mkdir(parents=True, exist_ok=True)
    save_json(metrics, destination / f"metrics_{split}.json")
    save_confusion_matrix_plot(
        metrics["confusion_matrix"],
        class_names=class_names,
        output_path=destination / f"confusion_matrix_{split}.png",
        title=f"{model_name} - {split}",
    )
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an image benchmark checkpoint.")
    parser.add_argument("--config", default="configs/image_benchmark.yaml")
    parser.add_argument("--model", required=True, choices=["simple_cnn", "resnet18", "densenet121"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    evaluate_checkpoint(
        config_path=args.config,
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        split=args.split,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
