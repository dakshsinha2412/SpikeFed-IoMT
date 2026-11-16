"""
test_snn_training.py
Tests various hyperparameters and fixes to ensure the LIF-SNN actually learns
on the medical dataset, achieving loss descent and accuracy well above 50%.
"""
import torch
import torch.nn as nn
import snntorch as snn
import snntorch.functional as SF
from dataset import load_image_folder
from torch.utils.data import DataLoader, random_split

class DiagnosticSpikingCNN(nn.Module):
    def __init__(self, num_classes=2, beta=0.85, threshold=0.5):
        super().__init__()
        grad = snn.surrogate.fast_sigmoid(slope=25)
        self.conv1, self.pool1 = nn.Conv2d(1, 8, 3, padding=1), nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=grad)
        self.conv2, self.pool2 = nn.Conv2d(8, 16, 3, padding=1), nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=grad)
        self.fc1 = nn.Linear(16 * 7 * 7, 64)
        self.lif3 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=grad)
        self.fc2 = nn.Linear(64, num_classes)
        self.lif4 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=grad)

    def forward(self, x, num_steps=15):
        mem1, mem2 = self.lif1.init_leaky(), self.lif2.init_leaky()
        mem3, mem4 = self.lif3.init_leaky(), self.lif4.init_leaky()
        spikes, membranes = [], []
        for _ in range(num_steps):
            spk1, mem1 = self.lif1(self.conv1(x), mem1)
            spk2, mem2 = self.lif2(self.conv2(self.pool1(spk1)), mem2)
            spk3, mem3 = self.lif3(self.fc1(self.pool2(spk2).flatten(1)), mem3)
            spk4, mem4 = self.lif4(self.fc2(spk3), mem4)
            spikes.append(spk4)
            membranes.append(mem4)
        return torch.stack(spikes), torch.stack(membranes)

def test_original_model():
    from model import SpikingCNN
    print("Testing original model with 10 vs 20 steps, learning rates, epochs...")
    dataset = load_image_folder("uploaded_datasets/dataset_20261115_224756")
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Try different configurations
    configs = [
        ("Original SpikingCNN, lr=1e-3, steps=10, 5 epochs", SpikingCNN(2), 0.001, 10, 5),
        ("Original SpikingCNN, lr=5e-3, steps=20, 10 epochs", SpikingCNN(2), 0.005, 20, 10),
        ("Original SpikingCNN, lr=1e-2, steps=15, 10 epochs", SpikingCNN(2), 0.01, 15, 10),
    ]
    
    for name, model, lr, steps, epochs in configs:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = SF.ce_rate_loss()
        print(f"\n--- {name} ---")
        for ep in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            spk_count = 0
            for x, y in loader:
                opt.zero_grad()
                spk, _ = model(x, num_steps=steps)
                loss = criterion(spk, y)
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(y)
                preds = spk.sum(0).argmax(1)
                correct += (preds == y).sum().item()
                total += len(y)
                spk_count += spk.sum().item()
            acc = correct / total
            print(f"Epoch {ep+1:2d} | Loss: {total_loss/total:.4f} | Acc: {acc:.2%} | Spikes fired: {spk_count}")

if __name__ == "__main__":
    test_original_model()
