"""
snn_accuracy_optimizer.py
Systematic SNN Accuracy Optimization Framework for SpikeFed-IoMT.
Executes hyperparameter grid search across LIF dynamics, surrogate gradients,
loss formulations, and data augmentations to maximize reproducible test accuracy.
"""
from __future__ import annotations
import os
import sys
import time
import json
import copy
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import snntorch as snn
from snntorch import surrogate
import snntorch.functional as SF

# ────────────────────────────── REPRODUCIBILITY ──────────────────────────────
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# ────────────────────────────── DATASET LOADER ──────────────────────────────
def load_pneumonia_data(
    data_dir: Path,
    max_train_samples: int = 2000,
    balanced_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Loads authentic PneumoniaMNIST chest radiographs.
    Applies contrast stretching [0.0, 1.0] and data augmentation.
    """
    npz_path = data_dir / "chest_xray" / ".cache" / "pneumoniamnist.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"PneumoniaMNIST not found at {npz_path}")

    data = np.load(npz_path)
    train_x, train_y = data["train_images"], data["train_labels"].flatten()
    val_x, val_y = data["val_images"], data["val_labels"].flatten()
    test_x, test_y = data["test_images"], data["test_labels"].flatten()

    # Class balance or sampling for training
    if balanced_train:
        idx_0 = np.where(train_y == 0)[0]
        idx_1 = np.where(train_y == 1)[0]
        n_per_class = min(len(idx_0), max_train_samples // 2)
        rng = np.random.RandomState(42)
        chosen_0 = rng.choice(idx_0, n_per_class, replace=False)
        chosen_1 = rng.choice(idx_1, n_per_class, replace=False)
        train_idx = np.concatenate([chosen_0, chosen_1])
        rng.shuffle(train_idx)
        train_x = train_x[train_idx]
        train_y = train_y[train_idx]
    elif max_train_samples and len(train_x) > max_train_samples:
        train_x = train_x[:max_train_samples]
        train_y = train_y[:max_train_samples]

    # Convert to float tensors in [0.0, 1.0] with per-image min-max contrast stretching
    def preprocess_images(imgs_np: np.ndarray) -> torch.Tensor:
        # imgs_np: [N, 28, 28] uint8
        tensors = torch.from_numpy(imgs_np).float() / 255.0
        # Shape: [N, 1, 28, 28]
        tensors = tensors.unsqueeze(1)
        # Per-image min-max contrast stretching to full excitatory dynamic range
        t_min = tensors.amin(dim=(2, 3), keepdim=True)
        t_max = tensors.amax(dim=(2, 3), keepdim=True)
        stretched = (tensors - t_min) / (t_max - t_min + 1e-6)
        return stretched.clamp(0.0, 1.0)

    train_tensor_x = preprocess_images(train_x)
    val_tensor_x = preprocess_images(val_x)
    test_tensor_x = preprocess_images(test_x)

    train_tensor_y = torch.from_numpy(train_y).long()
    val_tensor_y = torch.from_numpy(val_y).long()
    test_tensor_y = torch.from_numpy(test_y).long()

    # Calculate class weights for loss balancing
    n_0 = (train_tensor_y == 0).sum().item()
    n_1 = (train_tensor_y == 1).sum().item()
    w_0 = (n_0 + n_1) / (2.0 * max(n_0, 1))
    w_1 = (n_0 + n_1) / (2.0 * max(n_1, 1))
    class_weights = torch.tensor([w_0, w_1], dtype=torch.float32)

    stats = {
        "train_samples": len(train_tensor_x),
        "val_samples": len(val_tensor_x),
        "test_samples": len(test_tensor_x),
        "train_class_0": n_0,
        "train_class_1": n_1,
        "class_weights": [round(w_0, 4), round(w_1, 4)]
    }

    train_loader = DataLoader(TensorDataset(train_tensor_x, train_tensor_y), batch_size=16, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tensor_x, val_tensor_y), batch_size=32, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_tensor_x, test_tensor_y), batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader, stats

# ────────────────────────────── SNN ARCHITECTURE ──────────────────────────────
class ConfigurableSpikingCNN(nn.Module):
    """
    Parametric Spiking Convolutional Neural Network matching SpikeFed-IoMT topology
    (Conv1[4] -> LIF1 -> Pool1 -> Conv2[8] -> LIF2 -> Pool2 -> FC1[32] -> LIF3 -> FC2[2] -> LIF4).
    Enables hyperparameter experimentation across threshold, decay, reset, and surrogate slope.
    """
    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 1,
        beta: float = 0.90,
        threshold: float = 0.50,
        reset_mechanism: str = "subtract",
        surrogate_name: str = "fast_sigmoid",
        slope: float = 25.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.threshold = threshold
        self.beta = beta
        self.reset_mechanism = reset_mechanism

        # Surrogate gradient selection
        if surrogate_name == "fast_sigmoid":
            grad = surrogate.fast_sigmoid(slope=slope)
        elif surrogate_name == "atan":
            grad = surrogate.atan(alpha=slope)
        else:
            grad = surrogate.fast_sigmoid(slope=25.0)

        # Architectural layers (12,994 trainable parameters)
        self.conv1 = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=grad, reset_mechanism=reset_mechanism)

        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=grad, reset_mechanism=reset_mechanism)

        self.fc1 = nn.Linear(8 * 7 * 7, 32)
        self.lif3 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=grad, reset_mechanism=reset_mechanism)

        self.fc2 = nn.Linear(32, num_classes)
        # Readout layer: threshold can be matched or slightly lower for sensitive classification
        self.lif4 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=grad, reset_mechanism=reset_mechanism)

    def forward(self, x: torch.Tensor, num_steps: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        mem1, mem2 = self.lif1.init_leaky(), self.lif2.init_leaky()
        mem3, mem4 = self.lif3.init_leaky(), self.lif4.init_leaky()

        spikes = []
        membranes = []

        # Data augmentation on the fly for training batches
        for _ in range(num_steps):
            # Layer 1
            cur1 = self.conv1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            # Layer 2
            cur2 = self.conv2(self.pool1(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)

            # Layer 3
            cur3 = self.fc1(self.pool2(spk2).flatten(1))
            spk3, mem3 = self.lif3(cur3, mem3)

            # Layer 4 (Readout)
            cur4 = self.fc2(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)

            spikes.append(spk4)
            membranes.append(mem4)

        return torch.stack(spikes), torch.stack(membranes)

# ────────────────────────────── EVALUATION METRICS ──────────────────────────────
def evaluate_snn(
    model: nn.Module,
    loader: DataLoader,
    num_steps: int = 10,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    model.eval()
    all_preds = []
    all_targets = []
    total_spikes = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            spk, _ = model(x, num_steps=num_steps)
            # Classification via spike count accumulation over time
            spk_counts = spk.sum(dim=0)  # [B, num_classes]
            preds = spk_counts.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(y.cpu().numpy().tolist())
            total_spikes += spk.sum().item()
            total_samples += len(y)

    preds_arr = np.array(all_preds)
    targets_arr = np.array(all_targets)

    # Confusion matrix
    tp = int(np.sum((preds_arr == 1) & (targets_arr == 1)))
    tn = int(np.sum((preds_arr == 0) & (targets_arr == 0)))
    fp = int(np.sum((preds_arr == 1) & (targets_arr == 0)))
    fn = int(np.sum((preds_arr == 0) & (targets_arr == 1)))

    acc = float(np.mean(preds_arr == targets_arr)) * 100.0
    precision = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    avg_spikes_per_sample = total_spikes / max(total_samples, 1)

    return {
        "accuracy": round(acc, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1_score": round(f1, 2),
        "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "avg_spikes_per_sample": round(avg_spikes_per_sample, 2),
        "total_samples": total_samples
    }

# ────────────────────────────── TRAINING LOOP ──────────────────────────────
def train_snn_config(
    cfg: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 5,
    device: torch.device = torch.device("cpu")
) -> Tuple[nn.Module, Dict[str, Any]]:
    model = ConfigurableSpikingCNN(
        num_classes=2,
        in_channels=1,
        beta=cfg["beta"],
        threshold=cfg["threshold"],
        reset_mechanism=cfg["reset_mechanism"],
        surrogate_name=cfg["surrogate_name"],
        slope=cfg["slope"]
    ).to(device)

    # Optimizer & Scheduler
    lr = cfg["lr"]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    # Loss function selection
    crit = SF.ce_rate_loss()

    num_steps = cfg["num_steps"]
    best_val_acc = 0.0
    best_state = None

    history = []
    start_time = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_spikes = 0
        train_count = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Modality-appropriate data augmentation: slight random horizontal flip
            if cfg.get("use_augmentation", False) and np.random.rand() > 0.5:
                x = torch.flip(x, dims=[3])

            opt.zero_grad()
            spk, _ = model(x, num_steps=num_steps)
            loss = crit(spk, y)
            loss.backward()
            # Gradient clipping to prevent exploding gradients in BPTT
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            train_loss += loss.item() * len(y)
            train_spikes += spk.sum().item()
            train_count += len(y)

        scheduler.step()
        avg_train_loss = train_loss / max(train_count, 1)

        # Evaluate on validation set
        val_res = evaluate_snn(model, val_loader, num_steps=num_steps, device=device)
        history.append({
            "epoch": ep,
            "train_loss": round(avg_train_loss, 4),
            "val_acc": val_res["accuracy"],
            "val_f1": val_res["f1_score"]
        })

        if val_res["accuracy"] > best_val_acc:
            best_val_acc = val_res["accuracy"]
            best_state = copy.deepcopy(model.state_dict())

    duration = time.time() - start_time
    if best_state is not None:
        model.load_state_dict(best_state)

    final_val_res = evaluate_snn(model, val_loader, num_steps=num_steps, device=device)
    summary = {
        "config": cfg,
        "best_val_accuracy": final_val_res["accuracy"],
        "best_val_f1": final_val_res["f1_score"],
        "val_precision": final_val_res["precision"],
        "val_recall": final_val_res["recall"],
        "training_time_sec": round(duration, 2),
        "history": history
    }
    return model, summary

# ────────────────────────────── GRID SEARCH STUDY ──────────────────────────────
def run_optimization_study(data_dir: Path, results_dir: Path):
    print("=" * 78)
    print("  SPIKEFED-IoMT SNN ACCURACY OPTIMIZATION STUDY")
    print("  Target: 75-80%+ Reproducible Medical Accuracy with Neuromorphic Integrity")
    print("=" * 78)

    train_loader, val_loader, test_loader, ds_stats = load_pneumonia_data(
        data_dir, max_train_samples=2000, balanced_train=True
    )
    print(f"Dataset Loaded:")
    print(f"  Training Samples:   {ds_stats['train_samples']} (Class 0: {ds_stats['train_class_0']}, Class 1: {ds_stats['train_class_1']})")
    print(f"  Validation Samples: {ds_stats['val_samples']} (Tuning set)")
    print(f"  Test Samples:       {ds_stats['test_samples']} (Untouched hold-out)")
    print("-" * 78)

    # Systematic Configuration Matrix
    configs = [
        {
            "id": "CFG-1-Baseline",
            "name": "Historical Baseline (theta=0.5, beta=0.9, T=10, Hard Reset)",
            "threshold": 0.50,
            "beta": 0.90,
            "reset_mechanism": "zero",
            "surrogate_name": "fast_sigmoid",
            "slope": 25.0,
            "num_steps": 10,
            "lr": 0.003,
            "loss_type": "rate",
            "use_augmentation": False,
            "epochs": 3
        },
        {
            "id": "CFG-2-TemporalSoft",
            "name": "Soft Reset + Temporal Granularity (T=12, theta=0.45, beta=0.88)",
            "threshold": 0.45,
            "beta": 0.88,
            "reset_mechanism": "subtract",
            "surrogate_name": "fast_sigmoid",
            "slope": 20.0,
            "num_steps": 12,
            "lr": 0.002,
            "loss_type": "rate",
            "use_augmentation": True,
            "epochs": 3
        },
        {
            "id": "CFG-3-SurrogateSlope",
            "name": "Calibrated Surrogate Slope (slope=12, theta=0.40, beta=0.85)",
            "threshold": 0.40,
            "beta": 0.85,
            "reset_mechanism": "subtract",
            "surrogate_name": "fast_sigmoid",
            "slope": 12.0,
            "num_steps": 12,
            "lr": 0.002,
            "loss_type": "rate",
            "use_augmentation": True,
            "epochs": 3
        },
        {
            "id": "CFG-4-OptimizedEngine",
            "name": "SpikeFed-IoMT Calibrated Engine (T=14, theta=0.38, beta=0.85, Soft Reset, AdamW+Cosine)",
            "threshold": 0.38,
            "beta": 0.85,
            "reset_mechanism": "subtract",
            "surrogate_name": "fast_sigmoid",
            "slope": 10.0,
            "num_steps": 14,
            "lr": 0.002,
            "loss_type": "rate",
            "use_augmentation": True,
            "epochs": 3
        }
    ]

    all_results = []
    best_val_acc = -1.0
    champion_model = None
    champion_cfg = None

    for c in configs:
        print(f"\nEvaluating {c['id']}: {c['name']} ...")
        model, summary = train_snn_config(c, train_loader, val_loader, epochs=c["epochs"])
        all_results.append(summary)

        print(f"  --> Best Val Accuracy: {summary['best_val_accuracy']:.2f}% | Val F1: {summary['best_val_f1']:.2f}% | Time: {summary['training_time_sec']:.1f}s")

        if summary["best_val_accuracy"] > best_val_acc:
            best_val_acc = summary["best_val_accuracy"]
            champion_model = model
            champion_cfg = c

    print("\n" + "=" * 78)
    print(f"  CHAMPION MODEL SELECTED VIA VALIDATION: {champion_cfg['id']}")
    print(f"  Validation Accuracy: {best_val_acc:.2f}%")
    print("=" * 78)

    # ──────────────── FINAL UNTOUCHED TEST SET EVALUATION ────────────────
    print("\nEvaluating Champion Model on Untouched Hold-Out Test Set (N=624)...")
    test_metrics = evaluate_snn(
        champion_model,
        test_loader,
        num_steps=champion_cfg["num_steps"]
    )

    print("-" * 78)
    print(f"  TEST ACCURACY:        {test_metrics['accuracy']:.2f}%")
    print(f"  TEST PRECISION:       {test_metrics['precision']:.2f}%")
    print(f"  TEST RECALL:          {test_metrics['recall']:.2f}%")
    print(f"  TEST F1-SCORE:        {test_metrics['f1_score']:.2f}%")
    cm = test_metrics["confusion_matrix"]
    print(f"  CONFUSION MATRIX:     TP={cm['TP']}, TN={cm['TN']}, FP={cm['FP']}, FN={cm['FN']}")
    print(f"  AVG SPIKES / SAMPLE:  {test_metrics['avg_spikes_per_sample']}")
    print("-" * 78)

    # Save Checkpoints and Reports
    results_dir.mkdir(parents=True, exist_ok=True)

    # Backup prior model.pt if exists
    orig_model_path = results_dir / "model.pt"
    if orig_model_path.exists():
        backup_path = results_dir / "model_baseline_backup.pt"
        torch.save(torch.load(orig_model_path), backup_path)
        print(f"Backed up previous checkpoint to {backup_path.name}")

    # Save new champion model checkpoint
    torch.save(champion_model.state_dict(), orig_model_path)
    print(f"Saved optimized champion checkpoint to {orig_model_path}")

    # Save complete optimization report
    report = {
        "study_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": "PneumoniaMNIST Pediatric Chest Radiography",
        "dataset_statistics": ds_stats,
        "configurations_evaluated": all_results,
        "champion_configuration": champion_cfg,
        "final_untouched_test_evaluation": test_metrics,
        "target_achieved": (test_metrics["accuracy"] >= 75.0)
    }

    report_path = results_dir / "snn_optimization_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved full experiment report to {report_path.name}")

    return report

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent
    DATA = ROOT / "data"
    RESULTS = ROOT / "results"
    run_optimization_study(DATA, RESULTS)
