"""
benchmark.py – Performance and Resource Monitoring for SpikeFed-IoMT
====================================================================
Measures and logs:
  - CPU utilization (%)
  - RAM consumption (Peak & Current RSS in MB)
  - Execution time (Total & Per-round seconds)
  - Model complexity (Parameter count & Checkpoint size)
  - Communication overhead (Bytes exchanged across rounds)

Compatible with:
  - Raspberry Pi OS (Linux ARMv7/ARMv8)
  - Windows / macOS development machines
"""

from __future__ import annotations
import argparse
import os
import sys
import time
import torch
from pathlib import Path

# Local project imports
from dataset import load_image_folder, SyntheticMedicalImages, split_dataset, make_federated_loaders
from client import HospitalClient
from federated import FederatedServer
from model import SpikingCNN, parameter_count
from utils import get_device, set_seed, save_json, print_metrics_table

def get_ram_mb() -> float:
    """Return current process RAM usage in Megabytes."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        pass
    try:
        import resource
        # ru_maxrss is in KB on Linux (including Raspberry Pi)
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return 0.0

def get_cpu_percent(interval: float = 0.1) -> float:
    """Return CPU utilization percentage."""
    try:
        import psutil
        return psutil.cpu_percent(interval=interval)
    except ImportError:
        return 0.0

def run_benchmark(
    data_dir: str = "",
    rounds: int = 3,
    num_clients: int = 2,
    local_epochs: int = 1,
    batch_size: int = 8,
    non_iid: bool = False,
    seed: int = 42,
    label: str = "pi_benchmark",
):
    set_seed(seed)
    device = get_device()
    
    # 1. Dataset loading
    print("\n" + "=" * 55)
    print("  SpikeFed-IoMT Edge Performance & Resource Benchmark")
    print("=" * 55)
    print(f"Device: {device}")
    
    if data_dir and Path(data_dir).is_dir():
        dataset = load_image_folder(data_dir)
        print(f"Dataset : Local Medical Folder '{data_dir}' (Classes: {dataset.classes})")
        num_classes = len(dataset.classes)
    else:
        dataset = SyntheticMedicalImages(seed=seed)
        print(f"Dataset : Synthetic Medical Images (Classes: {dataset.classes})")
        num_classes = len(dataset.classes)
        
    train, val, test = split_dataset(dataset, seed=seed)
    loaders = make_federated_loaders(train, num_clients=num_clients, batch_size=batch_size, non_iid=non_iid, seed=seed)
    clients = [HospitalClient(f"Edge_Hospital_{i+1}", l, device=device, num_classes=num_classes) for i, l in enumerate(loaders)]
    server = FederatedServer(num_classes=num_classes, device=device)
    
    model_params = parameter_count(server.model)
    ram_initial = get_ram_mb()
    
    print(f"Clients : {num_clients} | Rounds: {rounds} | Local Epochs: {local_epochs}")
    print(f"Model Parameters: {model_params:,}")
    print(f"Initial RAM: {ram_initial:.2f} MB")
    print("-" * 55)
    
    # 2. Benchmarked Federated Training
    round_times = []
    round_cpu = []
    start_total = time.perf_counter()
    
    for r in range(rounds):
        cpu_before = get_cpu_percent()
        t0 = time.perf_counter()
        
        row = server.run_round(clients, local_epochs=local_epochs)
        
        t1 = time.perf_counter()
        cpu_after = get_cpu_percent()
        
        round_duration = t1 - t0
        round_times.append(round_duration)
        avg_cpu = max(cpu_before, cpu_after)
        round_cpu.append(avg_cpu)
        
        ram_now = get_ram_mb()
        print(f"Round {row['round']:02d}/{rounds:02d} | Time: {round_duration:.2f}s | RAM: {ram_now:.2f} MB | Comm: {row['communication_bytes']:,} B | Acc: {row['global_accuracy']:.3f} | Loss: {row['global_loss']:.3f}")

    total_training_time = time.perf_counter() - start_total
    ram_peak = get_ram_mb()
    
    # 3. Held-out test evaluation
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test, batch_size=batch_size)
    clients[0].set_parameters(server.global_state)
    eval_metrics = clients[0].evaluate(test_loader)
    
    # 4. Save checkpoint & measure disk size
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/global_model.pt"
    torch.save(server.model.state_dict(), ckpt_path)
    model_file_size_kb = os.path.getsize(ckpt_path) / 1024.0
    
    total_comm_bytes = sum(r["communication_bytes"] for r in server.history)
    avg_cpu_utilization = sum(round_cpu) / len(round_cpu) if round_cpu else 0.0
    
    summary = {
        "device": str(device),
        "dataset_type": "medical_folder" if (data_dir and Path(data_dir).is_dir()) else "synthetic",
        "num_classes": num_classes,
        "num_clients": num_clients,
        "rounds": rounds,
        "local_epochs": local_epochs,
        "model_parameter_count": model_params,
        "model_checkpoint_size_kb": round(model_file_size_kb, 2),
        "total_training_time_sec": round(total_training_time, 2),
        "avg_round_time_sec": round(sum(round_times) / max(len(round_times), 1), 2),
        "ram_initial_mb": round(ram_initial, 2),
        "ram_peak_mb": round(ram_peak, 2),
        "avg_cpu_percent": round(avg_cpu_utilization, 2),
        "total_communication_bytes": total_comm_bytes,
        "total_communication_kb": round(total_comm_bytes / 1024.0, 2),
        "test_accuracy": round(eval_metrics.get("accuracy", 0.0), 4),
        "test_f1_macro": round(eval_metrics.get("f1_macro", 0.0), 4),
        "test_loss": round(eval_metrics.get("loss", 0.0), 4),
    }
    
    # Display table
    print_metrics_table(
        {
            "Total Execution Time": f"{total_training_time:.2f} s",
            "Avg Round Time": f"{sum(round_times)/max(len(round_times),1):.2f} s",
            "RAM Peak Usage": f"{ram_peak:.2f} MB",
            "CPU Utilization (avg)": f"{avg_cpu_utilization:.1f}%",
            "Model Parameters": f"{model_params:,}",
            "Checkpoint Disk Size": f"{model_file_size_kb:.2f} KB",
            "Total Comm Overhead": f"{total_comm_bytes/1024.0:.2f} KB",
            "Held-Out Test Accuracy": f"{eval_metrics['accuracy']:.4f}",
            "Held-Out Macro F1": f"{eval_metrics['f1_macro']:.4f}",
        },
        title="Edge Performance & Resource Summary",
    )
    
    save_path = save_json(summary, f"{label}_summary.json")
    print(f"Full benchmark summary saved -> {save_path}")
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpikeFed-IoMT Edge Performance Benchmark")
    parser.add_argument("--data-dir", type=str, default="", help="Path to classwise medical dataset folder (optional)")
    parser.add_argument("--rounds", type=int, default=3, help="Number of federated rounds")
    parser.add_argument("--num-clients", type=int, default=2, help="Number of federated clients")
    parser.add_argument("--local-epochs", type=int, default=1, help="Epochs per client per round")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--non-iid", action="store_true", help="Partition data non-IID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--label", type=str, default="pi_benchmark", help="Output label")
    args = parser.parse_args()
    
    run_benchmark(
        data_dir=args.data_dir,
        rounds=args.rounds,
        num_clients=args.num_clients,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        non_iid=args.non_iid,
        seed=args.seed,
        label=args.label,
    )
