import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

class SpikingCNN(nn.Module):
    """
    A Spiking Convolutional Neural Network (SNN) that processes 
    MRI images and outputs spike trains.
    """
    def __init__(self, num_classes=4):
        super(SpikingCNN, self).__init__()
        
        # Initialize surrogate gradient for Backpropagation Through Time (BPTT)
        spike_grad = surrogate.fast_sigmoid(slope=25)
        
        # Standard convolutional layers acting as feature extractors before spiking neurons
        self.conv1 = nn.Conv2d(1, 12, kernel_size=3, padding=1)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=spike_grad)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(12, 16, kernel_size=3, padding=1)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=spike_grad)
        
        # Assuming input image size is adjusted to 28x28. After two pooling layers (if we had them),
        # but here we have one pool: 28 -> 14. 
        # (7 * 7 * 16) = 784 (since pool is called twice in forward pass)
        self.fc1 = nn.Linear(784, 128)
        self.lif3 = snn.Leaky(beta=0.9, spike_grad=spike_grad)
        
        self.fc2 = nn.Linear(128, num_classes)
        self.lif4 = snn.Leaky(beta=0.9, spike_grad=spike_grad)
        
    def forward(self, x, num_steps=20):
        """
        Forward pass converting static images to spike trains over T steps.
        """
        # Initialize hidden states
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        
        # Record outputs over time
        spk_rec = []
        mem_rec = []
        
        # Simulate over time
        for step in range(num_steps):
            cur1 = self.pool(self.conv1(x))
            spk1, mem1 = self.lif1(cur1, mem1)
            
            # Using pool again to reduce dimensions 14 -> 7
            cur2 = self.pool(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)
            
            # Flatten for Linear layers
            cur_flat = cur2.view(cur2.size(0), -1)
            
            cur3 = self.fc1(cur_flat)
            spk3, mem3 = self.lif3(cur3, mem3)
            
            cur4 = self.fc2(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)
            
            spk_rec.append(spk4)
            mem_rec.append(mem4)
            
        # Return stacked lists: shape shape [num_steps, batch_size, num_classes]
        return torch.stack(spk_rec), torch.stack(mem_rec)

def measure_uncertainty(spike_outputs):
    """
    Measures the predictive uncertainty based on spike rates.
    High uncertainty -> Model is confused -> Needs to send update to Server.
    Low uncertainty -> Model is confident -> Can skip to save communication.
    """
    # Count total spikes per class for each sample
    spike_counts = spike_outputs.sum(dim=0) # Shape: [batch_size, num_classes]
    
    # Calculate probabilities from spike rates (softmax over rates)
    probs = torch.softmax(spike_counts.float(), dim=1)
    
    # Calculate entropy for each sample (handling log(0))
    entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=1)
    
    # Return mean uncertainty of the batch
    mean_uncertainty = entropy.mean().item()
    return mean_uncertainty
