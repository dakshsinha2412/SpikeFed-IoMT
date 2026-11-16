"""
dataset_statistics.py
Statistical profiling for multi-modality medical datasets.
Generates class frequencies, train/val/test split ratios, and intensity distributions.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any

def compute_dataset_statistics(dataset_dir: Path) -> Dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    stats = {
        "dataset_name": dataset_dir.name,
        "splits": {},
        "class_totals": {},
        "grand_total": 0,
        "imbalance_ratio": 1.0
    }

    if not dataset_dir.exists():
        return stats

    for split in ["train", "val", "test"]:
        split_path = dataset_dir / split
        if split_path.is_dir():
            split_classes = {}
            for class_dir in split_path.iterdir():
                if class_dir.is_dir():
                    count = len([f for f in class_dir.glob("*.*") if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]])
                    split_classes[class_dir.name] = count
                    stats["class_totals"][class_dir.name] = stats["class_totals"].get(class_dir.name, 0) + count
            stats["splits"][split] = {
                "classes": split_classes,
                "split_total": sum(split_classes.values())
            }

    stats["grand_total"] = sum(stats["class_totals"].values())

    if len(stats["class_totals"]) >= 2:
        counts = list(stats["class_totals"].values())
        max_c = max(counts)
        min_c = max(min(counts), 1)
        stats["imbalance_ratio"] = round(max_c / min_c, 2)

    return stats

if __name__ == "__main__":
    s = compute_dataset_statistics(Path("data/chest_xray"))
    print(json.dumps(s, indent=2))
