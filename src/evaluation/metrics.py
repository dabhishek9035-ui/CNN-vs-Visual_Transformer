"""Class-balanced classification metrics for the HAM10000 label set."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


def compute_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    """Compute macro metrics over classes present in the evaluation labels.

    Per-class one-vs-rest AUC uses the model's softmax score for that class.
    For external evaluation, predictions outside the harmonized classes count
    as errors while only classes with true support enter macro averages.
    """
    truth = np.asarray(y_true, dtype=np.int64)
    predicted = np.asarray(y_pred, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if truth.ndim != 1 or predicted.shape != truth.shape:
        raise ValueError("y_true and y_pred must be one-dimensional arrays of equal length")
    if scores.shape != (len(truth), len(class_names)):
        raise ValueError(
            f"Expected probability matrix {(len(truth), len(class_names))}, received {scores.shape}"
        )
    if not len(truth):
        raise ValueError("Cannot compute metrics for an empty evaluation set")

    supported_indices = sorted(set(truth.tolist()))
    if any(index < 0 or index >= len(class_names) for index in supported_indices):
        raise ValueError("A true label index is outside class_names")
    labels = list(range(len(class_names)))
    matrix = confusion_matrix(truth, predicted, labels=labels)
    total = int(len(truth))
    per_class: list[dict[str, Any]] = []
    auc_values: list[float] = []

    for index in supported_indices:
        tp = int(np.sum((truth == index) & (predicted == index)))
        fn = int(np.sum((truth == index) & (predicted != index)))
        fp = int(np.sum((truth != index) & (predicted == index)))
        tn = total - tp - fn - fp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        specificity = tn / (tn + fp) if tn + fp else float("nan")
        binary_target = truth == index
        auc = float("nan")
        if binary_target.any() and (~binary_target).any():
            from sklearn.metrics import roc_auc_score

            auc = float(roc_auc_score(binary_target.astype(np.uint8), scores[:, index]))
            auc_values.append(auc)
        per_class.append(
            {
                "class": class_names[index],
                "support": int(np.sum(binary_target)),
                "precision": float(precision),
                "recall_sensitivity": float(recall),
                "f1": float(f1),
                "specificity_ovr": float(specificity),
                "roc_auc_ovr": auc,
            }
        )

    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(np.mean([item["f1"] for item in per_class])),
        "balanced_accuracy": float(np.mean([item["recall_sensitivity"] for item in per_class])),
        "macro_specificity_ovr": float(np.nanmean([item["specificity_ovr"] for item in per_class])),
        "macro_roc_auc_ovr": float(np.mean(auc_values)) if auc_values else None,
        "roc_auc_classes_included": [item["class"] for item in per_class if np.isfinite(item["roc_auc_ovr"])],
        "class_metrics": per_class,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": list(class_names),
        "evaluated_examples": total,
    }
