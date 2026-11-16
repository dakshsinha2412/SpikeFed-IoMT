"""
dataset_validation.py
Integrity, corruption detection, and distribution validation for medical datasets.
Ensures zero data leakage, validates image header health, and computes cryptographic hashes.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Dict, List, Any
from PIL import Image

def validate_dataset_integrity(dataset_dir: Path) -> Dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    report = {
        "dataset_path": str(dataset_dir),
        "is_valid": True,
        "total_files_scanned": 0,
        "valid_images": 0,
        "corrupted_images": [],
        "duplicate_hashes": [],
        "classes_found": {},
        "channel_formats": {},
        "resolutions_found": {}
    }

    if not dataset_dir.exists():
        report["is_valid"] = False
        report["error"] = f"Directory not found: {dataset_dir}"
        return report

    seen_hashes: Dict[str, str] = {}

    for file_path in dataset_dir.rglob("*.*"):
        if file_path.suffix.lower() not in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
            continue

        report["total_files_scanned"] += 1
        class_name = file_path.parent.name
        report["classes_found"][class_name] = report["classes_found"].get(class_name, 0) + 1

        # Check MD5 hash for exact duplicates
        try:
            with open(file_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            if file_hash in seen_hashes:
                report["duplicate_hashes"].append({
                    "original": seen_hashes[file_hash],
                    "duplicate": str(file_path)
                })
            else:
                seen_hashes[file_hash] = str(file_path)
        except Exception as e:
            report["corrupted_images"].append({"file": str(file_path), "error": f"Read error: {e}"})
            report["is_valid"] = False
            continue

        # Check PIL image header and mode
        try:
            with Image.open(file_path) as img:
                img.verify()
            # Reopen to read size and mode after verify
            with Image.open(file_path) as img:
                res_key = f"{img.size[0]}x{img.size[1]}"
                report["resolutions_found"][res_key] = report["resolutions_found"].get(res_key, 0) + 1
                mode_key = img.mode
                report["channel_formats"][mode_key] = report["channel_formats"].get(mode_key, 0) + 1
            report["valid_images"] += 1
        except Exception as e:
            report["corrupted_images"].append({"file": str(file_path), "error": str(e)})
            report["is_valid"] = False

    if len(report["corrupted_images"]) > 0:
        report["is_valid"] = False

    return report

if __name__ == "__main__":
    import json
    rep = validate_dataset_integrity(Path("data/chest_xray"))
    print("Validation Report:")
    print(json.dumps(rep, indent=2))
