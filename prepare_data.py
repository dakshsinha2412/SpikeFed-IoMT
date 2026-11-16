"""
prepare_data.py – SpikeFed-IoMT Dataset Shard Preparation
===========================================================
Splits the Brain Tumor MRI Training folder into N equal shards (one per Pi).
Each shard keeps the ImageFolder class-subfolder layout:

    data_shard_1/
      glioma/
      meningioma/
      notumor/
      pituitary/

Usage
-----
    # From the Kaggle download already on disk
    python prepare_data.py --source <KAGGLE_TRAIN_DIR> --num_shards 2

    # Download automatically via kagglehub first
    python prepare_data.py --download --num_shards 2

Outputs: data_shard_1/, data_shard_2/, … in the current directory.
Copy each shard folder to the matching Pi before starting the experiment:

    scp -r data_shard_1 pi@<PI_1_IP>:~/spikefed-iomt/data
"""

import argparse
import os
import random
import shutil
import sys

from utils import get_logger, set_seed

logger = get_logger("prepare_data")


def collect_images(source_dir: str):
    """Walk an ImageFolder directory and return {class_name: [abs_paths]}."""
    class_images: dict[str, list[str]] = {}
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    for class_name in sorted(os.listdir(source_dir)):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        images = [
            os.path.join(class_path, f)
            for f in os.listdir(class_path)
            if os.path.splitext(f)[1].lower() in valid_exts
        ]
        if images:
            class_images[class_name] = sorted(images)
            logger.info(f"  Class '{class_name}': {len(images)} images")
    return class_images


def split_and_copy(class_images: dict, num_shards: int, output_prefix: str, seed: int) -> None:
    """Distribute images into N shard folders."""
    rng = random.Random(seed)
    for class_name, images in class_images.items():
        shuffled = images[:]
        rng.shuffle(shuffled)
        shard_size = len(shuffled) // num_shards
        for i in range(num_shards):
            start = i * shard_size
            end = start + shard_size if i < num_shards - 1 else len(shuffled)
            shard_images = shuffled[start:end]
            shard_dir = os.path.join(f"{output_prefix}_{i + 1}", class_name)
            os.makedirs(shard_dir, exist_ok=True)
            for img_path in shard_images:
                shutil.copy2(img_path, shard_dir)
        logger.info(
            f"  '{class_name}' split: ~{shard_size} images per shard"
        )


def main():
    parser = argparse.ArgumentParser(description="Prepare data shards for multi-Pi SpikeFed-IoMT")
    parser.add_argument("--source", type=str, default="",
                        help="Path to the Kaggle MRI Training directory (ImageFolder layout)")
    parser.add_argument("--download", action="store_true",
                        help="Download dataset from Kaggle via kagglehub first")
    parser.add_argument("--num_shards", type=int, default=2,
                        help="Number of shards (= number of Pis)")
    parser.add_argument("--output_prefix", type=str, default="data_shard",
                        help="Output directory prefix (e.g. 'data_shard' -> 'data_shard_1', 'data_shard_2')")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    source_dir = args.source

    if args.download or not source_dir:
        try:
            import kagglehub
            logger.info("Downloading Brain Tumor MRI dataset via kagglehub…")
            dataset_path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
            source_dir = os.path.join(dataset_path, "Training")
            logger.info(f"Dataset downloaded to: {source_dir}")
        except ImportError:
            logger.error("kagglehub not installed. Run: pip install kagglehub")
            sys.exit(1)

    if not os.path.isdir(source_dir):
        logger.error(f"Source directory not found: {source_dir}")
        sys.exit(1)

    logger.info(f"Scanning source: {source_dir}")
    class_images = collect_images(source_dir)

    if not class_images:
        logger.error("No class subdirectories found. Check the --source path.")
        sys.exit(1)

    total = sum(len(v) for v in class_images.values())
    logger.info(f"Total images: {total} across {len(class_images)} classes")
    logger.info(f"Creating {args.num_shards} shards with prefix '{args.output_prefix}_N/'…")

    split_and_copy(class_images, args.num_shards, args.output_prefix, args.seed)

    logger.info("Done! Shard directories created:")
    for i in range(1, args.num_shards + 1):
        shard_path = f"{args.output_prefix}_{i}"
        logger.info(f"  {shard_path}/")

    print("\nNext steps:")
    print("  Copy each shard to the matching Pi, e.g.:")
    for i in range(1, args.num_shards + 1):
        print(f"    scp -r {args.output_prefix}_{i} pi@<PI_{i}_IP>:~/spikefed-iomt/data")


if __name__ == "__main__":
    main()
