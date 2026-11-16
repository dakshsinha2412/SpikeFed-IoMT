"""
dataset_preprocessing.py
Modality-aware preprocessing pipeline for biological Spiking Neural Networks.
Crucially enforces non-negative current domain [0.0, 1.0] to prevent LIF
membrane potential hyperpolarization and ensure reliable spike generation.
"""
from __future__ import annotations
import torch
from torchvision import transforms

def get_modality_transform(
    modality: str,
    channels: int = 1,
    resolution: tuple[int, int] = (28, 28),
    is_training: bool = False
) -> transforms.Compose:
    """
    Constructs a preprocessing pipeline tailored to the imaging modality and LIF-SNN requirements.

    IMPORTANT FOR SPIKING NEURAL NETWORKS:
    Standard ANN normalization (e.g. mean=0.5, std=0.5 resulting in [-1, 1]) is intentionally
    avoided because negative values act as inhibitory hyperpolarizing currents on LIF neurons,
    suppressing action potentials and causing prediction collapse.
    Here, tensors are mapped strictly into [0.0, 1.0].
    """
    transform_list = []

    # Channel handling
    if channels == 1:
        transform_list.append(transforms.Grayscale(num_output_channels=1))

    # Spatial resizing
    transform_list.append(transforms.Resize(resolution))

    # Data augmentation for training splits (modality-appropriate)
    if is_training:
        if modality in ["X-Ray", "Ultrasound", "Dermatology", "Histopathology"]:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        if modality in ["Dermatology", "Histopathology"]:
            transform_list.append(transforms.RandomRotation(degrees=15))

    # Convert to Tensor (scales raw [0, 255] pixels strictly to [0.0, 1.0])
    transform_list.append(transforms.ToTensor())

    return transforms.Compose(transform_list)

def encode_poisson_spikes(
    x: torch.Tensor,
    num_steps: int = 10,
    gain: float = 1.0
) -> torch.Tensor:
    """
    Optional Poisson rate encoder converting static image tensors [B, C, H, W]
    into binary spike trains [T, B, C, H, W] based on pixel intensities.
    """
    # x must be in [0.0, 1.0]
    scaled = (x * gain).clamp(0.0, 1.0)
    spikes = []
    for _ in range(num_steps):
        rand = torch.rand_like(scaled)
        spk = (rand < scaled).float()
        spikes.append(spk)
    return torch.stack(spikes)

if __name__ == "__main__":
    t_gray = get_modality_transform("X-Ray", channels=1, resolution=(28, 28))
    t_rgb = get_modality_transform("Dermatology", channels=3, resolution=(28, 28))
    print("Preprocessing pipelines verified:")
    print("  Grayscale (1-channel):", t_gray)
    print("  Color (3-channel):", t_rgb)
