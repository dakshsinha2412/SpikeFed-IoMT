"""
evaluate.py – SpikeFed-IoMT Standalone Model Evaluator
=======================================================
Loads a saved checkpoint (saved by the server after training) and evaluates it
on a local MRI class-folder dataset.  

Usage
-----
    python evaluate.py --data_dir data --checkpoint results/model.pt

The script prints accuracy, macro-F1, and a per-class breakdown table,
and saves the results to results/evaluation_<timestamp>.json.
"""

import argparse
import os
import sys

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import SpikingCNN, measure_uncertainty
from utils import get_logger, print_metrics_table, save_json, set_seed, timestamp

logger = get_logger("evaluate")

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


def load_dataset(data_dir: str, batch_size: int = 32) -> DataLoader:
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    logger.info(f"Loaded {len(dataset)} images from '{data_dir}'")
    logger.info(f"Classes detected: {dataset.classes}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def evaluate(model: SpikingCNN, loader: DataLoader, device: torch.device):
    """Run evaluation and return a metrics dict."""
    import snntorch.functional as SF
    criterion = SF.ce_rate_loss()

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    num_classes = 4

    tp = torch.zeros(num_classes, device=device)
    fp = torch.zeros(num_classes, device=device)
    fn = torch.zeros(num_classes, device=device)

    all_uncertainties = []

    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            spk_rec, _ = model(data)

            total_loss += criterion(spk_rec, targets).item() * data.size(0)

            spk_sum = spk_rec.sum(dim=0)
            pred = spk_sum.argmax(dim=1)

            correct += pred.eq(targets).sum().item()
            total += targets.size(0)

            all_uncertainties.append(measure_uncertainty(spk_rec))

            for c in range(num_classes):
                pred_c = pred.eq(c)
                true_c = targets.eq(c)
                tp[c] += (pred_c & true_c).sum()
                fp[c] += (pred_c & ~true_c).sum()
                fn[c] += (~pred_c & true_c).sum()

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    precision = tp / (tp + fp).clamp_min(1)
    recall = tp / (tp + fn).clamp_min(1)
    f1_per_class = (2 * precision * recall / (precision + recall).clamp_min(1e-12))
    f1_macro = f1_per_class.mean().item()
    avg_uncertainty = sum(all_uncertainties) / max(len(all_uncertainties), 1)

    metr = {
        "total_samples": total,
        "correct": correct,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "avg_loss": avg_loss,
        "avg_uncertainty": avg_uncertainty,
    }

    # Per-class breakdown
    per_class = {}
    for c, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            "precision": precision[c].item(),
            "recall": recall[c].item(),
            "f1": f1_per_class[c].item(),
        }
    metr["per_class"] = per_class
    return metr


def main():
    parser = argparse.ArgumentParser(description="SpikeFed-IoMT Model Evaluator")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to MRI class-folder directory (ImageFolder layout)")
    parser.add_argument("--checkpoint", type=str, default="results/model.pt",
                        help="Path to saved model checkpoint (state dict)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    if not os.path.isdir(args.data_dir):
        logger.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    model = SpikingCNN(num_classes=4).to(device)
    if os.path.isfile(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
        logger.info(f"Loaded checkpoint: {args.checkpoint}")
    else:
        logger.warning(
            f"Checkpoint '{args.checkpoint}' not found – evaluating a fresh (untrained) model."
        )

    loader = load_dataset(args.data_dir, args.batch_size)
    metrics = evaluate(model, loader, device)

    # Print summary table
    flat_metrics = {k: v for k, v in metrics.items() if k != "per_class"}
    print_metrics_table(flat_metrics, title="SpikeFed-IoMT Evaluation Results")

    print("Per-class breakdown:")
    for cls_name, cls_m in metrics["per_class"].items():
        print(f"  {cls_name:<12} precision={cls_m['precision']:.3f}  "
              f"recall={cls_m['recall']:.3f}  f1={cls_m['f1']:.3f}")

    # Save results
    metrics["checkpoint"] = args.checkpoint
    metrics["data_dir"] = args.data_dir
    out_path = save_json(metrics, f"evaluation_{timestamp()}.json")
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
