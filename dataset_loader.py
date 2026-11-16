"""
dataset_loader.py
Dataset loader, downloader, and splitter for genuine Multi-Modal Medical Datasets.
Populates standard directory structures:
  data/<dataset_id>/train/<class_name>/
  data/<dataset_id>/val/<class_name>/
  data/<dataset_id>/test/<class_name>/
"""
from __future__ import annotations
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from PIL import Image

from dataset_registry import get_modality, ModalityMetadata
from dataset_preprocessing import get_modality_transform

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"

MEDMNIST_FLAG_MAP = {
    "chest_xray": "pneumoniamnist",
    "ultrasound": "breastmnist",
    "oct": "octmnist",
    "dermatology": "dermamnist",
    "histopathology": "pathmnist",
    "fundus": "retinamnist",
    "brain_ct": "organamnist"
}

def install_dataset_from_medmnist(
    dataset_id: str,
    max_samples_per_split: Optional[int] = None
) -> ModalityMetadata:
    """
    Downloads and extracts real clinical benchmark data from MedMNIST into genuine PNG files.
    """
    import medmnist
    from medmnist import INFO

    meta = get_modality(dataset_id)
    if not meta:
        raise ValueError(f"Unknown dataset_id: {dataset_id}")

    flag = MEDMNIST_FLAG_MAP.get(dataset_id)
    if not flag:
        raise ValueError(f"No direct MedMNIST automated fetcher for {dataset_id}. Use manual folder loader.")

    info = INFO[flag]
    DataClass = getattr(medmnist, info["python_class"])

    target_dir = DATA_DIR / dataset_id
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{dataset_id.upper()}] Initializing clinical dataset download via MedMNIST ({flag})...")

    splits = ["train", "val", "test"]
    counts = {}

    cache_dir = target_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    root_path = str(cache_dir.resolve())

    for split in splits:
        split_dir = target_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        ds = DataClass(split=split, download=True, root=root_path)
        total = len(ds)
        limit = min(total, max_samples_per_split) if max_samples_per_split else total

        saved = 0
        for i in range(limit):
            img, target = ds[i]
            label_idx = int(target[0]) if hasattr(target, "__len__") else int(target)
            label_name = meta.class_labels[label_idx] if label_idx < len(meta.class_labels) else f"class_{label_idx}"

            class_dir = split_dir / label_name
            class_dir.mkdir(parents=True, exist_ok=True)

            img_path = class_dir / f"{split}_{i:05d}.png"
            if not img_path.exists():
                if isinstance(img, Image.Image):
                    img.save(img_path)
                else:
                    Image.fromarray(img).save(img_path)
            saved += 1

        counts[split] = saved
        print(f"  Split '{split}': {saved} genuine clinical PNG images written into {split_dir}")

    counts["total"] = sum(counts.values())
    meta.is_installed = True
    meta.sample_count = counts

    # Write dataset metadata json
    meta_json = target_dir / "dataset_metadata.json"
    import json
    from dataclasses import asdict
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, indent=2)

    return meta

def bootstrap_existing_chest_xray():
    """Bootstraps existing local 40 Normal/Pneumonia images into data/chest_xray/train."""
    source_dir = ROOT_DIR / "uploaded_datasets" / "dataset_20261115_224756"
    target_dir = DATA_DIR / "chest_xray"

    if source_dir.exists():
        train_dir = target_dir / "train"
        val_dir = target_dir / "val"
        test_dir = target_dir / "test"

        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        for cls_name in ["Normal", "Pneumonia"]:
            src_cls = source_dir / cls_name
            if src_cls.exists():
                dst_train = train_dir / cls_name
                dst_train.mkdir(parents=True, exist_ok=True)
                dst_test = test_dir / cls_name
                dst_test.mkdir(parents=True, exist_ok=True)

                images = [f for f in src_cls.glob("*.*") if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]]
                for i, img_file in enumerate(images):
                    # 75% train, 25% test
                    if i < int(len(images) * 0.75):
                        shutil.copy2(img_file, dst_train / img_file.name)
                    else:
                        shutil.copy2(img_file, dst_test / img_file.name)

        print(f"[BOOTSTRAP] Copied genuine local chest X-ray images into {target_dir}")

def get_medical_dataloader(
    dataset_id: str,
    split: str = "train",
    batch_size: int = 8,
    resolution: tuple[int, int] = (28, 28),
    shuffle: bool = True,
    num_workers: int = 0
) -> Tuple[DataLoader, List[str]]:
    """
    Returns a configured DataLoader and class names for the requested modality and split.
    """
    target_dir = DATA_DIR / dataset_id / split
    if not target_dir.exists():
        # Fallback to train if val or test is requested but not partitioned separately
        target_dir = DATA_DIR / dataset_id / "train"

    if not target_dir.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {target_dir}. Install it first.")

    meta = get_modality(dataset_id)
    channels = meta.input_channels if meta else 1
    modality_name = meta.modality if meta else "X-Ray"

    transform = get_modality_transform(
        modality=modality_name,
        channels=channels,
        resolution=resolution,
        is_training=(split == "train")
    )

    dataset = datasets.ImageFolder(str(target_dir), transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False
    )
    return loader, dataset.classes

if __name__ == "__main__":
    bootstrap_existing_chest_xray()
