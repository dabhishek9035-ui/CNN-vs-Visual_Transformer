"""Evaluate a selected checkpoint on validation, test, or external data."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torchvision

from src.config import load_yaml, resolve_path
from src.data.dataset import build_loader
from src.data.validate import build_records
from src.encoders import build_model
from src.evaluation.error_analysis import save_error_analysis
from src.evaluation.metrics import compute_metrics
from src.evaluation.timing import run_inference
from src.training.train import read_split
from src.training.utils import (
    PeakRSSSampler,
    device_description,
    parameter_counts,
    resolve_device,
    write_json,
)


def append_metrics_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict[str, Any]] = []
    if path.exists() and path.stat().st_size:
        with path.open("r", newline="", encoding="utf-8") as stream:
            existing_rows = list(csv.DictReader(stream))
    fieldnames = list(dict.fromkeys([*(existing_rows[0].keys() if existing_rows else []), *row.keys()]))
    existing_rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a best.pt checkpoint from a completed training run")
    parser.add_argument("--split", choices=("validation", "test", "external"), default="validation")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--experiment-config", default="configs/experiments.yaml")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision for inference")
    parser.add_argument("--warmup-batches", type=int)
    args = parser.parse_args()

    checkpoint_path = resolve_path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    # Checkpoints are created by this project and contain only tensors and primitive metadata.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    data_config = load_yaml(args.data_config)
    experiment_config = load_yaml(args.experiment_config)
    primary = data_config["primary"]
    class_names = [str(name) for name in checkpoint["class_names"]]
    device = resolve_device(str(experiment_config.get("device", "auto")))
    batch_size = args.batch_size or int(experiment_config.get("batch_size", 4))
    num_workers = int(experiment_config.get("num_workers", 0))
    input_size = int(checkpoint["input_size"])
    seed = int(checkpoint.get("seed", experiment_config.get("seed", 42)))

    if args.split == "external":
        if "external" not in data_config:
            raise ValueError("No external dataset is configured in the data config")
        dataset_config = data_config["external"]
        records, dataset_report = build_records(dataset_config)
        dataset_name = str(dataset_config.get("name", "external"))
    else:
        dataset_config = primary
        splits_dir = resolve_path(primary.get("splits_dir", "data/splits"))
        records = read_split(splits_dir / f"{args.split}.csv")
        dataset_report = None
        dataset_name = str(primary.get("name", "HAM10000"))
    if records.empty:
        raise ValueError(f"No records available for split {args.split!r}")

    loader = build_loader(
        records,
        dataset_config["root"],
        class_names,
        input_size,
        batch_size,
        num_workers,
        training=False,
        seed=seed,
        pin_memory=device.type == "cuda",
    )
    model, _ = build_model(
        str(checkpoint["model"]),
        num_classes=len(class_names),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    constructed_total, constructed_trainable = parameter_counts(model)
    total_parameters = int(checkpoint.get("total_parameters", constructed_total))
    trainable_parameters = int(checkpoint.get("trainable_parameters", constructed_trainable))
    use_amp = bool(checkpoint.get("use_amp", False)) and device.type == "cuda" and not args.no_amp
    warmup_batches = (
        int(args.warmup_batches)
        if args.warmup_batches is not None
        else int(experiment_config.get("inference_warmup_batches", 5))
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with PeakRSSSampler() as memory_sampler:
        y_true, y_pred, image_ids, timing = run_inference(
            model, loader, device, use_amp, warmup_batches=warmup_batches
        )
    probabilities = timing.pop("probabilities")
    metrics = compute_metrics(y_true, y_pred, probabilities, class_names)
    peak_gpu_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else None
    )
    run_id = str(checkpoint.get("run_id", checkpoint_path.parent.parent.name))
    output_dir = resolve_path(Path("results/evaluations") / run_id / args.split)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_error_analysis(
        output_dir,
        records,
        image_ids,
        y_true,
        y_pred,
        probabilities,
        class_names,
        metrics,
        args.split,
    )
    training_summary_path = checkpoint_path.parent.parent / "training_summary.json"
    training_summary = (
        json.loads(training_summary_path.read_text(encoding="utf-8"))
        if training_summary_path.is_file()
        else {}
    )
    summary: dict[str, Any] = {
        "run_id": run_id,
        "model": checkpoint["model"],
        "strategy": checkpoint["strategy"],
        "pretrained_weights": checkpoint.get("weights"),
        "pretraining_dataset": checkpoint.get("pretraining_dataset"),
        "dataset": dataset_name,
        "split": args.split,
        "seed": seed,
        "device": device_description(device),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_version": torch.version.cuda,
        "precision": "amp_fp16" if use_amp else "fp32",
        "batch_size": batch_size,
        "input_size": input_size,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "training_seconds": training_summary.get("training_seconds"),
        "training_peak_gpu_memory_mb": training_summary.get("peak_gpu_memory_mb"),
        "inference_peak_gpu_memory_mb": peak_gpu_mb,
        "peak_process_rss_mb": memory_sampler.peak_mb,
        **timing,
        **metrics,
        "external_dataset_audit": dataset_report,
        "checkpoint": str(checkpoint_path),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = output_dir / "summary.json"
    write_json(summary_path, summary)
    ledger_row = {
        key: summary.get(key)
        for key in (
            "run_id",
            "model",
            "strategy",
            "dataset",
            "split",
            "seed",
            "accuracy",
            "macro_f1",
            "balanced_accuracy",
            "macro_specificity_ovr",
            "macro_roc_auc_ovr",
            "training_seconds",
            "latency_ms_per_image",
            "throughput_images_per_second",
            "inference_peak_gpu_memory_mb",
            "peak_process_rss_mb",
            "checkpoint_size_bytes",
        )
    }
    ledger_row["summary_path"] = str(summary_path)
    append_metrics_row(resolve_path("results/metrics.csv"), ledger_row)
    print(json.dumps(summary, indent=2, default=str))
    print(f"Evaluation artifacts saved under {output_dir}")


if __name__ == "__main__":
    main()
