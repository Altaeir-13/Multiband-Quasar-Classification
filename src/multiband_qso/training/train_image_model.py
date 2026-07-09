from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from multiband_qso.config import load_config
from multiband_qso.data.metadata import CLASS_MAPPING
from multiband_qso.datasets import ImageMetadataDataset, build_image_transforms
from multiband_qso.evaluation.metrics import classification_metrics
from multiband_qso.image_models import create_image_model
from multiband_qso.training.utils import EarlyStopping, get_device, require_torch, save_json, set_seed

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _class_names_from_mapping(mapping: dict[str, int]) -> list[str]:
    return [class_name for class_name, _ in sorted(mapping.items(), key=lambda item: item[1])]


def _run_epoch(model, loader, criterion, device: str, optimizer=None) -> tuple[float, list[int], list[int]]:
    is_train = optimizer is not None
    model.train(is_train)
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc="train" if is_train else "eval", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            predictions = logits.argmax(dim=1)
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(predictions.detach().cpu().tolist())

    mean_loss = sum(losses) / max(len(losses), 1)
    return mean_loss, y_true, y_pred


def train_image_model(
    *,
    config_path: str | Path,
    model_name: str,
    run_name: str | None = None,
) -> Path:
    require_torch()
    config = load_config(config_path)
    seed = int(config["project"].get("seed", 42))
    set_seed(seed)

    paths = config["paths"]
    training = config["training"]
    class_mapping = config["project"].get("class_mapping", CLASS_MAPPING)
    class_names = _class_names_from_mapping(class_mapping)

    device = get_device(training.get("device", "auto"))
    timestamp = run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(paths["runs_dir"]) / model_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    image_size = int(training.get("image_size", 128))
    train_dataset = ImageMetadataDataset(
        paths["metadata_csv"],
        split="train",
        class_mapping_json=paths["class_mapping_json"],
        transform=build_image_transforms(image_size=image_size, train=True),
    )
    val_dataset = ImageMetadataDataset(
        paths["metadata_csv"],
        split="val",
        class_mapping_json=paths["class_mapping_json"],
        transform=build_image_transforms(image_size=image_size, train=False),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training.get("batch_size", 64)),
        shuffle=True,
        num_workers=int(training.get("num_workers", 2)),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(training.get("batch_size", 64)),
        shuffle=False,
        num_workers=int(training.get("num_workers", 2)),
    )

    model = create_image_model(
        model_name,
        num_classes=len(class_mapping),
        pretrained=bool(training.get("pretrained", False)),
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 3e-4)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    stopper = EarlyStopping(patience=int(training.get("patience", 5)), mode="max")

    history: list[dict] = []
    best_checkpoint = run_dir / "checkpoint_best.pt"
    labels = list(range(len(class_mapping)))

    for epoch in range(1, int(training.get("epochs", 20)) + 1):
        train_loss, train_true, train_pred = _run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        val_loss, val_true, val_pred = _run_epoch(model, val_loader, criterion, device)
        train_f1 = f1_score(train_true, train_pred, labels=labels, average="macro", zero_division=0)
        val_f1 = f1_score(val_true, val_pred, labels=labels, average="macro", zero_division=0)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_f1_macro": train_f1,
            "val_f1_macro": val_f1,
        }
        history.append(row)
        improved = stopper.step(val_f1)
        if improved:
            torch.save(
                {
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "class_mapping": class_mapping,
                    "config": config,
                    "epoch": epoch,
                    "val_f1_macro": val_f1,
                },
                best_checkpoint,
            )
        if stopper.should_stop:
            break

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    if not best_checkpoint.exists():
        raise RuntimeError("Training finished without writing a best checkpoint")

    checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, val_true, val_pred = _run_epoch(model, val_loader, criterion, device)
    metrics = classification_metrics(
        val_true,
        val_pred,
        labels=labels,
        class_names=class_names,
    )
    metrics["model"] = model_name
    metrics["split"] = "val"
    metrics["run_dir"] = str(run_dir)
    metrics["pretrained"] = bool(training.get("pretrained", False))
    metrics["best_epoch"] = int(checkpoint["epoch"])
    save_json(metrics, run_dir / "metrics.json")
    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an image benchmark model.")
    parser.add_argument("--config", default="configs/image_benchmark.yaml")
    parser.add_argument("--model", required=True, choices=["simple_cnn", "resnet18", "densenet121"])
    parser.add_argument("--run-name", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    train_image_model(config_path=args.config, model_name=args.model, run_name=args.run_name)


if __name__ == "__main__":
    main()
