"""
investigate_model.py
Thorough diagnostic script investigating why the SNN training produces
constant loss = 0.6931 and accuracy = 50.0% across rounds.
"""
import torch
import snntorch as snn
import snntorch.functional as SF
from model import SpikingCNN
from dataset import load_image_folder
from torch.utils.data import DataLoader

def run_investigation():
    data_dir = "uploaded_datasets/dataset_20261115_224756"
    dataset = load_image_folder(data_dir)
    print("=" * 60)
    print("1. DATASET INSPECTION")
    print("=" * 60)
    print(f"Dataset classes: {dataset.classes}")
    print(f"Class-to-idx: {dataset.class_to_idx}")
    print(f"Total samples: {len(dataset)}")
    
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    batch_x, batch_y = next(iter(loader))
    print(f"Batch X shape: {batch_x.shape}, min: {batch_x.min():.3f}, max: {batch_x.max():.3f}, mean: {batch_x.mean():.3f}")
    print(f"Batch Y: {batch_y.tolist()}")

    print("\n" + "=" * 60)
    print("2. MODEL FORWARD PASS & SPIKE FIRING ANALYSIS")
    print("=" * 60)
    model = SpikingCNN(num_classes=2)
    spk, mem = model(batch_x, num_steps=10)
    print(f"Output spk shape: {spk.shape}  [time_steps, batch_size, num_classes]")
    print(f"Output mem shape: {mem.shape}  [time_steps, batch_size, num_classes]")
    print(f"Spike sum per sample:\n{spk.sum(dim=0)}")
    print(f"Total spikes fired across batch: {spk.sum().item()}")
    print(f"Membrane potential min: {mem.min().item():.4f}, max: {mem.max().item():.4f}, mean: {mem.mean().item():.4f}")

    print("\n" + "=" * 60)
    print("3. LOSS & PREDICTIONS ANALYSIS")
    print("=" * 60)
    criterion = SF.ce_rate_loss()
    loss = criterion(spk, batch_y)
    print(f"Calculated ce_rate_loss: {loss.item():.6f}")
    
    spike_sums = spk.sum(dim=0)
    preds = spike_sums.argmax(dim=1)
    print(f"Predicted classes: {preds.tolist()}")
    print(f"True classes:      {batch_y.tolist()}")
    correct = (preds == batch_y).sum().item()
    print(f"Batch accuracy: {correct}/{len(batch_y)} = {correct/len(batch_y):.2%}")

    print("\n" + "=" * 60)
    print("4. GRADIENTS & WEIGHT UPDATES")
    print("=" * 60)
    model.zero_grad()
    loss.backward()
    
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norm = param.grad.norm().item()
            grad_norms[name] = norm
            print(f"  {name:25s} | grad_norm: {norm:.6e} | param_shape: {list(param.shape)}")
        else:
            print(f"  {name:25s} | NO GRADIENT")

    total_grad_norm = sum(grad_norms.values())
    print(f"\nTotal Gradient Norm: {total_grad_norm:.6e}")
    if total_grad_norm == 0:
        print("CRITICAL: All gradients are ZERO! Weights CANNOT update.")
    elif total_grad_norm < 1e-4:
        print("WARNING: Vanishing gradients! Weight updates are near negligible.")
    else:
        print("Gradients are non-zero.")

    print("\n" + "=" * 60)
    print("5. SINGLE STEP OPTIMIZER TEST")
    print("=" * 60)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    p_before = model.fc2.weight.clone()
    opt.step()
    p_after = model.fc2.weight.clone()
    diff = (p_after - p_before).abs().max().item()
    print(f"Max absolute change in fc2.weight after 1 step: {diff:.6e}")

if __name__ == "__main__":
    run_investigation()
