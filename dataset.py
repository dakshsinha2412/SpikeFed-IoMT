"""Local-only image loading, preprocessing, splits, and synthetic test data."""
from __future__ import annotations
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
def image_transform(normalize: bool = False):
    """
    Transforms for SNNs. Default normalize=False maintains non-negative [0, 1] current domain
    to prevent LIF neuron hyperpolarization and ensure reliable spike generation.
    """
    ops = [transforms.Grayscale(1), transforms.Resize((28, 28)), transforms.ToTensor()]
    if normalize:
        ops.append(transforms.Normalize((0.5,), (0.5,)))
    return transforms.Compose(ops)

def load_image_folder(path, normalize: bool = False):
    path = Path(path)
    if not path.is_dir(): raise FileNotFoundError(f"Dataset directory not found: {path}")
    if not any(p.suffix.lower() in IMAGE_EXTENSIONS for p in path.rglob("*")): raise ValueError(f"No supported images under {path}")
    return datasets.ImageFolder(str(path), transform=image_transform(normalize=normalize))

class SyntheticMedicalImages(Dataset):
    """Synthetic pipeline-test images only; never clinical data or results."""
    def __init__(self, samples_per_class=24, num_classes=4, seed=42):
        gen = torch.Generator().manual_seed(seed); self.labels = torch.arange(num_classes).repeat_interleave(samples_per_class)
        self.images = torch.randn(len(self.labels), 1, 28, 28, generator=gen) * .10
        for i, y in enumerate(self.labels.tolist()):
            if y == 0: self.images[i, 0, 4:10, :] += .7
            elif y == 1: self.images[i, 0, :, 4:10] += .7
            elif y == 2: self.images[i, 0, 8:20, 8:20] += .7
            else: self.images[i, 0].diagonal().add_(.7)
        self.images = self.images.clamp(-1, 1); self.classes = [f"synthetic_class_{i}" for i in range(num_classes)]
    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return self.images[i], self.labels[i]

def split_dataset(dataset, seed=42, train_fraction=.7, val_fraction=.15):
    n = len(dataset); a, b = int(n * train_fraction), int(n * val_fraction)
    return random_split(dataset, [a, b, n-a-b], generator=torch.Generator().manual_seed(seed))
def partition_indices(dataset, num_clients, non_iid=False, seed=42):
    ids = list(range(len(dataset)))
    if non_iid:
        labels = getattr(dataset, "labels", None) if hasattr(dataset, "labels") else getattr(dataset, "targets", None)
        if labels is None and isinstance(dataset, Subset):
            parent_labels = getattr(dataset.dataset, "labels", None)
            if parent_labels is None: parent_labels = getattr(dataset.dataset, "targets", None)
            if parent_labels is not None: labels = [parent_labels[i] for i in dataset.indices]
        if labels is None: raise ValueError("Non-IID partitioning requires labels")
        ids.sort(key=lambda i: int(labels[i]))
    else: ids = torch.randperm(len(ids), generator=torch.Generator().manual_seed(seed)).tolist()
    return [ids[i::num_clients] for i in range(num_clients)]
def make_federated_loaders(dataset, num_clients=3, batch_size=8, non_iid=False, seed=42):
    return [DataLoader(Subset(dataset, ids), batch_size=batch_size, shuffle=True) for ids in partition_indices(dataset, num_clients, non_iid, seed)]
