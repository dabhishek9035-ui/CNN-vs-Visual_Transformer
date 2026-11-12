"""Aggregate completed runs and bootstrap paired CNN-versus-ViT differences."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, f1_score

from src.config import resolve_path


SUMMARY_METRICS = (
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "macro_specificity_ovr",
    "macro_roc_auc_ovr",
    "training_seconds",
    "latency_ms_per_image",
    "throughput_images_per_second",
    "inference_peak_gpu_memory_mb",
    "checkpoint_size_bytes",
)


def load_completed_runs(split: str) -> list[dict[str, Any]]:
    root = resolve_path("results/evaluations")
    runs: list[dict[str, Any]] = []
    if not root.exists():
        return runs
    for summary_path in root.rglob("summary.json"):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("split") == split:
            summary["summary_path"] = str(summary_path)
            summary["prediction_path"] = str(summary_path.parent / f"{split}_predictions.csv")
            runs.append(summary)
    return runs


def aggregate_runs(runs: list[dict[str, Any]], split: str) -> pd.DataFrame:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[(run["dataset"], run["model"], run["strategy"])].append(run)
    rows: list[dict[str, Any]] = []
    for (dataset, model, strategy), group in sorted(grouped.items()):
        row: dict[str, Any] = {
            "dataset": dataset,
            "split": split,
            "model": model,
            "strategy": strategy,
            "run_count": len(group),
            "seeds": ",".join(str(item.get("seed", "")) for item in group),
        }
        for metric in SUMMARY_METRICS:
            values = [float(item[metric]) for item in group if item.get(metric) is not None]
            row[f"{metric}_mean"] = float(np.mean(values)) if values else None
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        rows.append(row)
    return pd.DataFrame(rows)


def prediction_frame(run: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    path = Path(run["prediction_path"])
    if not path.is_file():
        raise FileNotFoundError(f"Prediction file missing for run {run['run_id']}: {path}")
    frame = pd.read_csv(path)
    probability_columns = [column for column in frame.columns if column.startswith("prob_")]
    class_names = [column.removeprefix("prob_") for column in probability_columns]
    required = {"image_id", "true_label", "predicted_label"}
    if not required.issubset(frame.columns) or not probability_columns:
        raise ValueError(f"Prediction file has an unexpected schema: {path}")
    return frame, class_names


def paired_bootstrap(
    cnn: pd.DataFrame,
    vit: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    left = cnn.sort_values("image_id").reset_index(drop=True)
    right = vit.sort_values("image_id").reset_index(drop=True)
    if left["image_id"].tolist() != right["image_id"].tolist():
        raise ValueError("Paired bootstrap requires both models to predict the same image IDs")
    if left["true_label"].tolist() != right["true_label"].tolist():
        raise ValueError("Paired runs disagree on true labels for the same image IDs")

    truth = left["true_label"].astype(str).to_numpy()
    cnn_prediction = left["predicted_label"].astype(str).to_numpy()
    vit_prediction = right["predicted_label"].astype(str).to_numpy()
    active_labels = sorted(set(truth.tolist()))
    class_positions = [np.flatnonzero(truth == label) for label in active_labels]
    observed_f1_difference = float(
        f1_score(truth, vit_prediction, labels=active_labels, average="macro", zero_division=0)
        - f1_score(truth, cnn_prediction, labels=active_labels, average="macro", zero_division=0)
    )
    observed_balanced_accuracy_difference = float(
        balanced_accuracy_score(truth, vit_prediction)
        - balanced_accuracy_score(truth, cnn_prediction)
    )
    rng = np.random.default_rng(seed)
    f1_differences: list[float] = []
    balanced_accuracy_differences: list[float] = []

    # Stratified paired resampling keeps the observed class support fixed and
    # resamples the same held-out cases for both model families.
    for _ in range(repetitions):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_positions]
        )
        f1_cnn = f1_score(truth[sampled], cnn_prediction[sampled], labels=active_labels, average="macro", zero_division=0)
        f1_vit = f1_score(truth[sampled], vit_prediction[sampled], labels=active_labels, average="macro", zero_division=0)
        bacc_cnn = balanced_accuracy_score(truth[sampled], cnn_prediction[sampled])
        bacc_vit = balanced_accuracy_score(truth[sampled], vit_prediction[sampled])
        f1_differences.append(float(f1_vit - f1_cnn))
        balanced_accuracy_differences.append(float(bacc_vit - bacc_cnn))

    f1_interval = np.percentile(f1_differences, [2.5, 97.5])
    bacc_interval = np.percentile(balanced_accuracy_differences, [2.5, 97.5])
    return {
        "macro_f1_difference_vit_minus_cnn": observed_f1_difference,
        "macro_f1_ci95_low": float(f1_interval[0]),
        "macro_f1_ci95_high": float(f1_interval[1]),
        "balanced_accuracy_difference_vit_minus_cnn": observed_balanced_accuracy_difference,
        "balanced_accuracy_ci95_low": float(bacc_interval[0]),
        "balanced_accuracy_ci95_high": float(bacc_interval[1]),
        "bootstrap_repetitions": repetitions,
        "bootstrap_method": "paired, stratified by true class, percentile 95% interval",
        "paired_images": int(len(left)),
    }


def plot_aggregate_metrics(aggregate: pd.DataFrame, split: str) -> Path:
    output = resolve_path(Path("results/plots") / f"{split}_model_comparison.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    if aggregate.empty:
        return output
    labels = [f"{row.model}\n{row.strategy}" for row in aggregate.itertuples()]
    positions = np.arange(len(aggregate))
    figure, axes = plt.subplots(1, 2, figsize=(max(10, len(labels) * 1.4), 4.5))
    for axis, metric, title in (
        (axes[0], "macro_f1", "Macro F1"),
        (axes[1], "balanced_accuracy", "Balanced accuracy"),
    ):
        means = aggregate[f"{metric}_mean"].to_numpy(dtype=float)
        stds = aggregate[f"{metric}_std"].fillna(0).to_numpy(dtype=float)
        axis.bar(positions, means, yerr=stds, capsize=3, color="tab:blue", alpha=0.85)
        axis.set_xticks(positions, labels, rotation=30, ha="right")
        axis.set_ylim(0, 1)
        axis.set_title(title)
        axis.set_ylabel("Mean across runs; error bars show SD")
    figure.suptitle(f"{split.capitalize()} model comparison")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test", "external"), default="test")
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap_repetitions < 100:
        raise ValueError("Use at least 100 bootstrap repetitions")

    runs = load_completed_runs(args.split)
    if not runs:
        raise FileNotFoundError(f"No evaluation summaries were found for split {args.split!r}")
    table_root = resolve_path("results/tables")
    table_root.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_runs(runs, args.split)
    aggregate_path = table_root / f"{args.split}_aggregate.csv"
    aggregate.to_csv(aggregate_path, index=False)
    plot_path = plot_aggregate_metrics(aggregate, args.split)

    grouped: dict[tuple[str, str, int], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for run in runs:
        grouped[(run["dataset"], run["strategy"], int(run.get("seed", 0)))][run["model"]].append(run)

    paired_rows: list[dict[str, Any]] = []
    for (dataset, strategy, run_seed), model_runs in sorted(grouped.items()):
        if "resnet50" not in model_runs or "vit_b_16" not in model_runs:
            continue
        if len(model_runs["resnet50"]) != 1 or len(model_runs["vit_b_16"]) != 1:
            print(
                f"Skipping ambiguous pair: dataset={dataset}, strategy={strategy}, seed={run_seed}; "
                "expected one run per model and seed."
            )
            continue
        cnn_frame, cnn_classes = prediction_frame(model_runs["resnet50"][0])
        vit_frame, vit_classes = prediction_frame(model_runs["vit_b_16"][0])
        if cnn_classes != vit_classes:
            raise ValueError("CNN and ViT prediction files have different class columns")
        result = paired_bootstrap(
            cnn_frame,
            vit_frame,
            repetitions=args.bootstrap_repetitions,
            seed=args.seed + run_seed,
        )
        paired_rows.append(
            {
                "dataset": dataset,
                "split": args.split,
                "strategy": strategy,
                "seed": run_seed,
                "cnn_run_id": model_runs["resnet50"][0]["run_id"],
                "vit_run_id": model_runs["vit_b_16"][0]["run_id"],
                **result,
            }
        )

    paired_path = table_root / f"{args.split}_paired_bootstrap.csv"
    pd.DataFrame(paired_rows).to_csv(paired_path, index=False)
    print(f"Aggregate table: {aggregate_path}")
    print(f"Metric comparison plot: {plot_path}")
    print(f"Paired bootstrap table: {paired_path}")


if __name__ == "__main__":
    main()
