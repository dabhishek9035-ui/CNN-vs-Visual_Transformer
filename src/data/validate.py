"""Validate metadata-to-image mappings for HAM10000 and PAD-UFES-20."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, UnidentifiedImageError

from src.config import load_yaml, resolve_path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_image_index(dataset_config: dict[str, Any]) -> dict[str, str]:
    """Map image filename stems to dataset-root-relative paths."""
    root = resolve_path(dataset_config["root"])
    image_dirs = dataset_config.get("image_dirs", [])
    if not image_dirs:
        raise ValueError(f"No image_dirs configured for {dataset_config.get('name', 'dataset')}")

    index: dict[str, str] = {}
    for image_dir in image_dirs:
        directory = (root / image_dir).resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"Image directory not found: {directory}")
        for path in directory.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            key = path.stem.casefold()
            relative = path.relative_to(root).as_posix()
            if key in index:
                raise ValueError(
                    f"Multiple image files have the same image ID {path.stem!r}: "
                    f"{index[key]!r} and {relative!r}"
                )
            index[key] = relative
    if not index:
        raise ValueError(f"No image files found under {root}")
    return index


def build_records(dataset_config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read metadata and produce canonical rows with label and relative image path.

    External labels omitted by a configured label_map are excluded and counted in
    the report. This makes conservative cross-dataset harmonization explicit.
    """
    root = resolve_path(dataset_config["root"])
    metadata_path = root / dataset_config["metadata"]
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    id_column = dataset_config["image_id_column"]
    label_column = dataset_config["label_column"]
    group_column = dataset_config.get("group_column")
    required = [id_column, label_column]
    if group_column:
        required.append(group_column)
    missing_columns = [column for column in required if column not in metadata.columns]
    if missing_columns:
        raise ValueError(f"{metadata_path.name} is missing columns: {missing_columns}")
    metadata[id_column] = metadata[id_column].str.strip()
    metadata[label_column] = metadata[label_column].str.strip()
    if group_column:
        metadata[group_column] = metadata[group_column].str.strip()
    if metadata[id_column].str.strip().eq("").any():
        raise ValueError(f"Metadata contains blank values in {id_column!r}")
    if metadata[id_column].duplicated().any():
        duplicate_ids = metadata.loc[metadata[id_column].duplicated(), id_column].head(5).tolist()
        raise ValueError(f"Metadata has duplicate {id_column} values: {duplicate_ids}")
    if metadata[label_column].str.strip().eq("").any():
        raise ValueError(f"Metadata contains blank values in {label_column!r}")

    configured_map = dataset_config.get("label_map") or {}
    raw_labels = metadata[label_column].str.strip()
    if configured_map:
        label_map = {str(key): str(value) for key, value in configured_map.items()}
        mapped_labels = raw_labels.map(label_map)
        unmapped_counts = Counter(raw_labels[mapped_labels.isna()].tolist())
        keep = mapped_labels.notna()
        selected = metadata.loc[keep].copy()
        selected["label"] = mapped_labels.loc[keep].astype(str)
    else:
        selected = metadata.copy()
        selected["label"] = raw_labels
        unmapped_counts = Counter()

    expected_classes = set(str(label) for label in dataset_config.get("classes", []))
    unexpected = sorted(set(selected["label"].astype(str)) - expected_classes) if expected_classes else []
    if unexpected:
        raise ValueError(f"Labels outside configured classes: {unexpected}")
    if selected.empty:
        raise ValueError(f"No labeled examples remain for {dataset_config.get('name', 'dataset')}")

    image_index = build_image_index(dataset_config)
    id_to_path: dict[str, str] = {}
    missing_images: list[str] = []
    for image_id in selected[id_column].astype(str):
        image_key = Path(image_id).stem.casefold()
        if image_key not in image_index:
            missing_images.append(image_id)
        else:
            id_to_path[image_id] = image_index[image_key]
    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} metadata images are missing; examples: {missing_images[:10]}"
        )

    selected["image_id"] = selected[id_column].astype(str)
    selected["image_path"] = selected[id_column].map(id_to_path)
    if group_column:
        if selected[group_column].str.strip().eq("").any():
            raise ValueError(f"Metadata contains blank values in {group_column!r}")
        selected["group_id"] = selected[group_column].astype(str)
    else:
        selected["group_id"] = selected["image_id"]

    report = {
        "dataset": dataset_config.get("name", "dataset"),
        "metadata_rows": int(len(metadata)),
        "usable_rows": int(len(selected)),
        "unmapped_rows": int(len(metadata) - len(selected)),
        "unmapped_label_counts": dict(sorted(unmapped_counts.items())),
        "indexed_images": int(len(image_index)),
        "class_counts": {str(k): int(v) for k, v in selected["label"].value_counts().sort_index().items()},
        "group_count": int(selected["group_id"].nunique()),
        "image_root": str(root),
    }
    return selected.reset_index(drop=True), report


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_image_content(frame: pd.DataFrame, dataset_config: dict[str, Any]) -> list[str]:
    """Return image IDs whose files Pillow cannot decode or verify."""
    root = resolve_path(dataset_config["root"])
    failures: list[str] = []
    for row in frame[["image_id", "image_path"]].itertuples(index=False):
        try:
            with Image.open(root / row.image_path) as image:
                image.verify()
        except (OSError, UnidentifiedImageError, ValueError):
            failures.append(str(row.image_id))
    return failures


def find_exact_overlaps(
    primary_config: dict[str, Any],
    external_config: dict[str, Any],
) -> list[dict[str, str]]:
    """Find byte-identical files across datasets using SHA-256."""
    primary_root = resolve_path(primary_config["root"])
    external_root = resolve_path(external_config["root"])
    primary_hashes: dict[str, str] = {}
    for relative_path in build_image_index(primary_config).values():
        path = Path(relative_path)
        primary_hashes[sha256_file(primary_root / path)] = path.stem

    overlaps: list[dict[str, str]] = []
    for relative_path in build_image_index(external_config).values():
        path = Path(relative_path)
        digest = sha256_file(external_root / path)
        if digest in primary_hashes:
            overlaps.append({"primary_image_id": primary_hashes[digest], "external_image_id": path.stem})
    return overlaps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--dataset", choices=("primary", "external", "all"), default="primary")
    parser.add_argument("--check-external-overlap", action="store_true")
    parser.add_argument("--verify-image-content", action="store_true")
    parser.add_argument("--fail-on-overlap", action="store_true")
    parser.add_argument("--report", default="results/tables/data_validation.json")
    args = parser.parse_args()

    config = load_yaml(args.config)
    keys = ["primary", "external"] if args.dataset == "all" else [args.dataset]
    frames: dict[str, pd.DataFrame] = {}
    reports: dict[str, Any] = {}
    for key in keys:
        if key not in config:
            raise ValueError(f"Dataset {key!r} is not configured")
        frames[key], reports[key] = build_records(config[key])
        if args.verify_image_content:
            failures = verify_image_content(frames[key], config[key])
            reports[key]["unreadable_image_count"] = len(failures)
            reports[key]["unreadable_image_ids"] = failures

    overlap_report: dict[str, Any] | None = None
    if args.check_external_overlap:
        primary = frames.get("primary")
        external = frames.get("external")
        if primary is None:
            primary, reports["primary"] = build_records(config["primary"])
        if external is None:
            external, reports["external"] = build_records(config["external"])
        overlaps = find_exact_overlaps(config["primary"], config["external"])
        overlap_report = {"method": "exact SHA-256 match", "count": len(overlaps), "matches": overlaps}

    output = {"datasets": reports, "exact_overlap_check": overlap_report}
    report_path = resolve_path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"Report saved to {report_path}")
    if args.fail_on_overlap and overlap_report and overlap_report["count"]:
        raise SystemExit("Exact cross-dataset image overlap detected")


if __name__ == "__main__":
    main()
