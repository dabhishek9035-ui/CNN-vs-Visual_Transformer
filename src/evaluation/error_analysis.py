"""Write per-image predictions, class metrics, and confusion matrices."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.plots import plot_confusion_matrix


def save_error_analysis(
    output_dir: str | Path,
    records: pd.DataFrame,
    image_ids: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    metrics: dict[str, Any],
    split_name: str,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = records.set_index(records["image_id"].astype(str)).loc[image_ids].reset_index(drop=True)
    predictions = pd.DataFrame(
        {
            "image_id": image_ids,
            "true_label": [class_names[index] for index in y_true],
            "predicted_label": [class_names[index] for index in y_pred],
            "correct": (y_true == y_pred),
        }
    )
    if "group_id" in rows:
        predictions["group_id"] = rows["group_id"].astype(str).to_numpy()
    for index, class_name in enumerate(class_names):
        predictions[f"prob_{class_name}"] = probabilities[:, index]
    predictions.to_csv(output / f"{split_name}_predictions.csv", index=False)

    class_metrics = pd.DataFrame(metrics["class_metrics"])
    class_metrics.to_csv(output / f"{split_name}_class_metrics.csv", index=False)
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(
        output / f"{split_name}_confusion_matrix.csv", index_label="true_label"
    )
    plot_confusion_matrix(matrix, class_names, output / f"{split_name}_confusion_matrix.png")
