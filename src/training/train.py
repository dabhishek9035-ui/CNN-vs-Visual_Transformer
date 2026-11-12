"""Train one CNN/ViT configuration using train and validation splits only."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn

from src.config import load_yaml, resolve_path
from src.data.dataset import build_loader
from src.encoders import build_model, configure_trainable_parameters, get_classifier_head
from src.evaluation.metrics import compute_metrics
from src.evaluation.plots import plot_training_history
from src.training.utils import (
    PeakRSSSampler,
    device_description,
    parameter_counts,
    resolve_device,
    set_seed,
    synchronize,
    write_json,
)


def read_split(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}. Run `python -m src.data.split` first.")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"image_id", "image_path", "label", "group_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing split columns: {missing}")
    return frame


def evaluate_validation(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool,
    criterion: nn.Module,
    class_names: list[str],
) -> tuple[float, dict[str, Any]]:
    model.eval()
    labels: list[int] = []
    scores: list[np.ndarray] = []
    total_loss = torch.zeros((), dtype=torch.float32, device=device)
    count = 0
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for images, targets, _ in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)
            total_loss.add_(loss.detach().float() * len(targets))
            count += len(targets)
            labels.extend(targets.cpu().numpy().astype(int).tolist())
            scores.extend(torch.softmax(logits.float(), dim=1).cpu().numpy())
    synchronize(device)
    elapsed = time.perf_counter() - started
    probabilities = np.asarray(scores, dtype=np.float32)
    predictions = probabilities.argmax(axis=1)
    metric_values = compute_metrics(labels, predictions, probabilities, class_names)
    return float((total_loss / max(count, 1)).item()), {
        **metric_values,
        "validation_seconds": elapsed,
    }


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
) -> tuple[float, float, float]:
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("No trainable parameters were configured")
    total_loss = torch.zeros((), dtype=torch.float32, device=device)
    count = 0
    data_loading_wait = 0.0
    synchronize(device)
    started = time.perf_counter()
    iterator = iter(loader)
    while True:
        wait_started = time.perf_counter()
        try:
            images, targets, _ = next(iterator)
        except StopIteration:
            break
        data_loading_wait += time.perf_counter() - wait_started
        images = images.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss.add_(loss.detach().float() * len(targets))
        count += len(targets)
    synchronize(device)
    elapsed = time.perf_counter() - started
    return float((total_loss / max(count, 1)).item()), elapsed, data_loading_wait


def make_optimizer(
    model: nn.Module,
    model_name: str,
    strategy: str,
    settings: dict[str, Any],
) -> torch.optim.Optimizer:
    weight_decay = float(settings.get("weight_decay", 0.0001))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if strategy == "finetune":
        head_ids = {id(parameter) for parameter in get_classifier_head(model, model_name).parameters()}
        backbone_parameters = [parameter for parameter in trainable if id(parameter) not in head_ids]
        head_parameters = [parameter for parameter in trainable if id(parameter) in head_ids]
        groups = []
        if backbone_parameters:
            groups.append(
                {
                    "params": backbone_parameters,
                    "lr": float(settings.get("finetune_backbone_learning_rate", 1e-5)),
                }
            )
        groups.append(
            {
                "params": head_parameters,
                "lr": float(settings.get("finetune_head_learning_rate", 1e-4)),
            }
        )
    else:
        key = "frozen_head_learning_rate" if strategy == "frozen" else "scratch_learning_rate"
        groups = [{"params": trainable, "lr": float(settings.get(key, 1e-3))}]
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def set_training_mode(model: nn.Module, model_name: str, strategy: str) -> None:
    if strategy == "frozen":
        model.eval()
        get_classifier_head(model, model_name).train()
    else:
        model.train()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("resnet50", "vit_b_16"))
    parser.add_argument("--strategy", required=True, choices=("scratch", "frozen", "finetune"))
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--experiment-config", default="configs/experiments.yaml")
    parser.add_argument("--seed", type=int, help="Override the initialization/augmentation seed without changing saved splits")
    parser.add_argument("--run-id", help="Optional unique run folder name; defaults to model/strategy/UTC time")
    args = parser.parse_args()

    data_config = load_yaml(args.data_config)
    experiment_config = load_yaml(args.experiment_config)
    primary = data_config["primary"]
    seed = args.seed if args.seed is not None else int(experiment_config.get("seed", 42))
    deterministic = bool(experiment_config.get("deterministic", False))
    set_seed(seed, deterministic=deterministic)
    device = resolve_device(str(experiment_config.get("device", "auto")))
    class_names = [str(name) for name in primary["classes"]]
    input_size = int(experiment_config.get("input_size", 224))
    batch_size = int(experiment_config.get("batch_size", 4))
    num_workers = int(experiment_config.get("num_workers", 0))
    augmentation = experiment_config.get("augmentation", {})
    splits_dir = resolve_path(primary.get("splits_dir", "data/splits"))
    train_records = read_split(splits_dir / "train.csv")
    validation_records = read_split(splits_dir / "validation.csv")

    pin_memory = device.type == "cuda"
    train_loader = build_loader(
        train_records,
        primary["root"],
        class_names,
        input_size,
        batch_size,
        num_workers,
        training=True,
        augmentation=augmentation,
        seed=seed,
        pin_memory=pin_memory,
    )
    validation_loader = build_loader(
        validation_records,
        primary["root"],
        class_names,
        input_size,
        batch_size,
        num_workers,
        training=False,
        seed=seed,
        pin_memory=pin_memory,
    )

    pretrained = args.strategy != "scratch"
    requested_weights = experiment_config.get("model_weights", {}).get(args.model)
    model, loaded_weights = build_model(
        args.model,
        num_classes=len(class_names),
        pretrained=pretrained,
        weights_name=requested_weights,
    )
    configure_trainable_parameters(model, args.model, args.strategy)
    model.to(device)
    total_parameters, trainable_parameters = parameter_counts(model)

    optimizer_settings = experiment_config.get("optimizer", {})
    optimizer = make_optimizer(model, args.model, args.strategy, optimizer_settings)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.2,
        patience=int(optimizer_settings.get("scheduler_patience", 2)),
    )
    train_counts = train_records["label"].value_counts()
    absent_classes = sorted(set(class_names) - set(train_counts.index.astype(str)))
    if absent_classes:
        raise ValueError(f"Training split is missing classes: {absent_classes}")
    if bool(experiment_config.get("class_weighted_loss", True)):
        weights = [len(train_records) / (len(class_names) * train_counts[name]) for name in class_names]
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    else:
        class_weights = None
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    use_amp = bool(experiment_config.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    run_id = args.run_id or (
        f"{args.model}_{args.strategy}_seed{seed}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run-id must be a single safe folder name")
    run_dir = resolve_path(Path("results/runs") / run_id)
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "configs").mkdir(parents=True, exist_ok=True)
    (run_dir / "configs" / "data.yaml").write_text(
        resolve_path(args.data_config).read_text(encoding="utf-8"), encoding="utf-8"
    )
    (run_dir / "configs" / "experiments.yaml").write_text(
        resolve_path(args.experiment_config).read_text(encoding="utf-8"), encoding="utf-8"
    )
    write_json(
        run_dir / "run_overrides.json",
        {
            "model": args.model,
            "strategy": args.strategy,
            "seed": seed,
            "deterministic_cudnn": deterministic,
        },
    )

    max_epochs = int(experiment_config.get("epochs", 20))
    patience = int(experiment_config.get("early_stopping_patience", 4))
    best_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    total_training_seconds = 0.0
    total_training_data_wait_seconds = 0.0
    total_validation_seconds = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    print(
        f"Run {run_id}: {args.model}/{args.strategy}; device={device_description(device)}; "
        f"weights={loaded_weights or 'random initialization'}; trainable={trainable_parameters:,}/{total_parameters:,}",
        flush=True,
    )

    with PeakRSSSampler() as memory_sampler:
        for epoch in range(1, max_epochs + 1):
            set_training_mode(model, args.model, args.strategy)
            train_loss, train_seconds, train_data_wait_seconds = train_epoch(
                model, train_loader, device, optimizer, criterion, scaler, use_amp
            )
            validation_loss, validation_metrics = evaluate_validation(
                model, validation_loader, device, use_amp, criterion, class_names
            )
            validation_seconds = float(validation_metrics.pop("validation_seconds"))
            macro_f1 = float(validation_metrics["macro_f1"])
            scheduler.step(macro_f1)
            total_training_seconds += train_seconds
            total_training_data_wait_seconds += train_data_wait_seconds
            total_validation_seconds += validation_seconds
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_macro_f1": macro_f1,
                "validation_balanced_accuracy": float(validation_metrics["balanced_accuracy"]),
                "train_seconds": train_seconds,
                "train_data_loading_wait_seconds": train_data_wait_seconds,
                "validation_seconds": validation_seconds,
            }
            history.append(row)
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
            print(
                f"epoch {epoch:02d}/{max_epochs}: train_loss={train_loss:.4f} "
                f"val_loss={validation_loss:.4f} val_macro_f1={macro_f1:.4f} "
                f"train_s={train_seconds:.1f}",
                flush=True,
            )

            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                best_epoch = epoch
                epochs_without_improvement = 0
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "model": args.model,
                    "strategy": args.strategy,
                    "weights": loaded_weights,
                    "pretraining_dataset": "ImageNet-1K" if loaded_weights else "random initialization",
                    "class_names": class_names,
                    "input_size": input_size,
                    "batch_size": batch_size,
                    "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                    "use_amp": use_amp,
                    "seed": seed,
                    "run_id": run_id,
                    "best_epoch": epoch,
                    "validation_macro_f1": macro_f1,
                    "total_parameters": total_parameters,
                    "trainable_parameters": trainable_parameters,
                }
                temporary_path = checkpoint_dir / "best.pt.tmp"
                torch.save(checkpoint, temporary_path)
                os.replace(temporary_path, checkpoint_dir / "best.pt")
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after epoch {epoch}; best validation macro F1={best_macro_f1:.4f}")
                break

    best_checkpoint = checkpoint_dir / "best.pt"
    plot_training_history(pd.DataFrame(history), run_dir / "training_curves.png")
    checkpoint_size = best_checkpoint.stat().st_size if best_checkpoint.exists() else 0
    peak_gpu_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else None
    )
    summary = {
        "run_id": run_id,
        "model": args.model,
        "strategy": args.strategy,
        "pretrained_weights": loaded_weights,
        "pretraining_dataset": "ImageNet-1K" if loaded_weights else "random initialization",
        "seed": seed,
        "deterministic_cudnn": deterministic,
        "device": device_description(device),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_version": torch.version.cuda,
        "input_size": input_size,
        "batch_size": batch_size,
        "precision": "amp_fp16" if use_amp else "fp32",
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_macro_f1,
        "training_seconds": total_training_seconds,
        "training_data_loading_wait_seconds": total_training_data_wait_seconds,
        "validation_seconds": total_validation_seconds,
        "peak_gpu_memory_mb": peak_gpu_mb,
        "peak_process_rss_mb": memory_sampler.peak_mb,
        "checkpoint_size_bytes": checkpoint_size,
        "checkpoint": str(best_checkpoint),
        "test_set_used": False,
    }
    write_json(run_dir / "training_summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(f"Best checkpoint: {best_checkpoint}")


if __name__ == "__main__":
    main()
