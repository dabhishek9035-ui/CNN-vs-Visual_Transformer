"""Create fixed lesion-grouped HAM10000 train, validation, and test splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import load_yaml, resolve_path
from src.data.validate import build_records


def make_grouped_splits(
    frame: pd.DataFrame,
    seed: int,
    test_folds: int = 7,
    validation_folds: int = 6,
) -> dict[str, pd.DataFrame]:
    """Split by group while approximately preserving class proportions.

    Nested stratified-group folds target roughly 70/15/15 partitions. Actual
    proportions vary because all images from a lesion must stay together.
    """
    if frame["group_id"].nunique() < max(test_folds, validation_folds):
        raise ValueError("There are too few unique groups for the configured split folds")
    y = frame["label"].astype(str).to_numpy()
    groups = frame["group_id"].astype(str).to_numpy()

    outer = StratifiedGroupKFold(n_splits=test_folds, shuffle=True, random_state=seed)
    outer_splits = list(outer.split(np.zeros(len(frame)), y, groups))
    train_validation_indices, test_indices = outer_splits[seed % test_folds]

    inner = StratifiedGroupKFold(
        n_splits=validation_folds,
        shuffle=True,
        random_state=seed + 1,
    )
    inner_splits = list(
        inner.split(
            np.zeros(len(train_validation_indices)),
            y[train_validation_indices],
            groups[train_validation_indices],
        )
    )
    train_local, validation_local = inner_splits[(seed + 1) % validation_folds]
    train_indices = train_validation_indices[train_local]
    validation_indices = train_validation_indices[validation_local]

    result = {
        "train": frame.iloc[train_indices].copy().reset_index(drop=True),
        "validation": frame.iloc[validation_indices].copy().reset_index(drop=True),
        "test": frame.iloc[test_indices].copy().reset_index(drop=True),
    }
    group_sets = {name: set(part["group_id"].astype(str)) for name, part in result.items()}
    if (
        group_sets["train"] & group_sets["validation"]
        or group_sets["train"] & group_sets["test"]
        or group_sets["validation"] & group_sets["test"]
    ):
        raise RuntimeError("Group leakage detected between generated partitions")
    return result


def summarize_partition(frame: pd.DataFrame, total_images: int) -> dict[str, Any]:
    return {
        "images": int(len(frame)),
        "groups": int(frame["group_id"].nunique()),
        "fraction": float(len(frame) / total_images),
        "class_counts": {
            str(label): int(count) for label, count in frame["label"].value_counts().sort_index().items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--experiment-config", default="configs/experiments.yaml")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    data_config = load_yaml(args.data_config)
    experiment_config = load_yaml(args.experiment_config)
    seed = args.seed if args.seed is not None else int(experiment_config.get("seed", 42))
    split_options = experiment_config.get("splitting", {})
    frame, data_report = build_records(data_config["primary"])
    splits = make_grouped_splits(
        frame,
        seed=seed,
        test_folds=int(split_options.get("test_folds", 7)),
        validation_folds=int(split_options.get("validation_folds", 6)),
    )

    output_dir = resolve_path(data_config["primary"].get("splits_dir", "data/splits"))
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_csv(output_dir / f"{name}.csv", index=False)

    manifest = {
        "dataset": data_config["primary"].get("name", "HAM10000"),
        "seed": seed,
        "group_column": data_config["primary"].get("group_column", "lesion_id"),
        "strategy": "nested StratifiedGroupKFold",
        "test_folds": int(split_options.get("test_folds", 7)),
        "validation_folds": int(split_options.get("validation_folds", 6)),
        "total_images": int(len(frame)),
        "total_groups": int(frame["group_id"].nunique()),
        "source_class_counts": data_report["class_counts"],
        "partitions": {name: summarize_partition(part, len(frame)) for name, part in splits.items()},
        "group_overlap": False,
        "note": "Group constraints make exact target fractions approximate; inspect counts before training.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Split files saved under {output_dir}")


if __name__ == "__main__":
    main()
