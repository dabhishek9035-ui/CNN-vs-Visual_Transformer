"""Synchronized inference timing helpers."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.training.utils import synchronize


def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    warmup_batches: int = 5,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """Predict every row and time host-to-device transfer plus model inference.

    Batch decode and data-loader wait time are excluded from the timed latency;
    total evaluation wall time is also reported separately.
    """
    model.eval()
    total_batches = len(loader)
    actual_warmup = min(max(0, warmup_batches), max(0, total_batches - 1))
    true_values: list[int] = []
    score_values: list[np.ndarray] = []
    image_ids: list[str] = []
    measured_seconds = 0.0
    measured_images = 0
    measured_batches = 0
    wall_start = time.perf_counter()
    last_batch_complete = wall_start
    data_loading_wait = 0.0

    with torch.inference_mode():
        for batch_index, (images, labels, ids) in enumerate(loader):
            batch_ready = time.perf_counter()
            data_loading_wait += batch_ready - last_batch_complete
            is_timed = batch_index >= actual_warmup
            if is_timed:
                synchronize(device)
                started = time.perf_counter()
            images = images.to(device, non_blocking=device.type == "cuda")
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(images)
                probabilities = torch.softmax(logits.float(), dim=1)
            if is_timed:
                synchronize(device)
                measured_seconds += time.perf_counter() - started
                measured_images += len(labels)
                measured_batches += 1
            true_values.extend(labels.numpy().astype(int).tolist())
            score_values.extend(probabilities.cpu().numpy())
            image_ids.extend(str(item) for item in ids)
            last_batch_complete = time.perf_counter()

    wall_seconds = time.perf_counter() - wall_start
    scores = np.asarray(score_values, dtype=np.float32)
    predictions = scores.argmax(axis=1)
    timing = {
        "warmup_batches": int(actual_warmup),
        "timed_batches": int(measured_batches),
        "timed_images": int(measured_images),
        "inference_seconds": float(measured_seconds),
        "data_loading_wait_seconds": float(data_loading_wait),
        "latency_ms_per_image": float(1000 * measured_seconds / measured_images) if measured_images else None,
        "throughput_images_per_second": float(measured_images / measured_seconds) if measured_seconds else None,
        "evaluation_wall_seconds": float(wall_seconds),
    }
    return np.asarray(true_values), predictions, image_ids, {**timing, "probabilities": scores}
