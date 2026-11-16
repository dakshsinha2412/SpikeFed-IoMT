"""Compact, trainable spiking model used by every federated client."""
from __future__ import annotations
import torch
from torch import nn
import snntorch as snn
from snntorch import surrogate

class SpikingCNN(nn.Module):
    """LIF convolutional SNN with surrogate-gradient BPTT."""
    def __init__(self, num_classes: int = 2, in_channels: int = 1, beta: float = 0.9, threshold: float = 0.5):
        super().__init__()
        grad = surrogate.fast_sigmoid(slope=25)
        self.conv1, self.pool1 = nn.Conv2d(in_channels, 4, 3, padding=1), nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=grad)
        self.conv2, self.pool2 = nn.Conv2d(4, 8, 3, padding=1), nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=grad)
        self.fc1, self.lif3 = nn.Linear(8 * 7 * 7, 32), snn.Leaky(beta=beta, threshold=threshold, spike_grad=grad)
        self.fc2, self.lif4 = nn.Linear(32, num_classes), snn.Leaky(beta=beta, threshold=threshold, spike_grad=grad)

    def forward(self, x: torch.Tensor, num_steps: int = 10):
        mem1, mem2 = self.lif1.init_leaky(), self.lif2.init_leaky()
        mem3, mem4 = self.lif3.init_leaky(), self.lif4.init_leaky()
        spikes, membranes = [], []
        for _ in range(num_steps):
            spk1, mem1 = self.lif1(self.conv1(x), mem1)
            spk2, mem2 = self.lif2(self.conv2(self.pool1(spk1)), mem2)
            spk3, mem3 = self.lif3(self.fc1(self.pool2(spk2).flatten(1)), mem3)
            spk4, mem4 = self.lif4(self.fc2(spk3), mem4)
            spikes.append(spk4); membranes.append(mem4)
        return torch.stack(spikes), torch.stack(membranes)

def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_uncertainty(spike_outputs: torch.Tensor) -> float:
    probs = torch.softmax(spike_outputs.sum(dim=0).float(), dim=1)
    return float((-(probs * (probs + 1e-8).log()).sum(dim=1).mean()).item())

def get_spike_telemetry(spikes: torch.Tensor) -> dict:
    total_spikes = float(spikes.sum().item())
    steps, batch_size, classes = spikes.shape
    firing_rate = total_spikes / (steps * batch_size * classes) if (steps * batch_size * classes) > 0 else 0.0
    return {
        "total_spikes": total_spikes,
        "firing_rate": round(firing_rate, 4),
        "zero_spike_collapse": (total_spikes == 0)
    }
