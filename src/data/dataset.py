"""PyTorch dataset and matched image preprocessing for Experiment 1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.config import resolve_path


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SkinLesionDataset(Dataset):
    def __init__(
        self,
        records: pd.DataFrame,
        root: str | Path,
        class_names: list[str],
        transform: transforms.Compose,
    ) -> None:
        self.records = records.reset_index(drop=True)
        self.root = Path(root)
        self.class_names = list(class_names)
        self.class_to_index = {name: index for index, name in enumerate(self.class_names)}
        self.transform = transform
        unknown = sorted(set(self.records["label"].astype(str)) - set(self.class_to_index))
        if unknown:
            raise ValueError(f"Dataset contains labels not in model classes: {unknown}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        row = self.records.iloc[index]
        path = self.root / str(row["image_path"])
        with Image.open(path) as source:
            image = source.convert("RGB")
        tensor = self.transform(image)
        label = self.class_to_index[str(row["label"])]
        return tensor, label, str(row["image_id"])


def build_transforms(
    input_size: int,
    training: bool,
    augmentation: dict[str, Any] | None = None,
) -> transforms.Compose:
    augmentation = augmentation or {}
    ops: list[Any] = [transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BILINEAR)]
    if training:
        if augmentation.get("horizontal_flip", True):
            ops.append(transforms.RandomHorizontalFlip())
        if augmentation.get("vertical_flip", True):
            ops.append(transforms.RandomVerticalFlip())
        rotation = float(augmentation.get("rotation_degrees", 0))
        if rotation > 0:
            ops.append(
                transforms.RandomRotation(
                    degrees=rotation,
                    interpolation=InterpolationMode.BILINEAR,
                )
            )
        jitter = float(augmentation.get("color_jitter", 0))
        if jitter > 0:
            ops.append(
                transforms.ColorJitter(
                    brightness=jitter,
                    contrast=jitter,
                    saturation=jitter,
                )
            )
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(ops)


def build_loader(
    records: pd.DataFrame,
    dataset_root: str | Path,
    class_names: list[str],
    input_size: int,
    batch_size: int,
    num_workers: int,
    training: bool,
    augmentation: dict[str, Any] | None = None,
    seed: int = 42,
    pin_memory: bool = False,
) -> DataLoader:
    dataset = SkinLesionDataset(
        records=records,
        root=resolve_path(dataset_root),
        class_names=class_names,
        transform=build_transforms(input_size, training, augmentation),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        generator=generator,
    )
