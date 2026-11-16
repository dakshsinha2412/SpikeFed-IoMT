"""
federated_partition.py
IID and Non-IID data partitioning algorithms for federated medical imaging.
Supports balanced uniform distribution as well as pathological and Dirichlet label skew.
"""
from __future__ import annotations
import torch
import numpy as np
from torch.utils.data import Dataset, Subset, DataLoader
from typing import List, Dict, Any, Tuple

def partition_federated_dataset(
    dataset: Dataset,
    num_clients: int = 2,
    partition_type: str = "iid",
    alpha: float = 0.5,
    seed: int = 42
) -> List[List[int]]:
    """
    Partitions dataset sample indices across `num_clients`.

    - 'iid': Random uniform shuffle and partition.
    - 'non_iid_pathological': Each client receives a dominant fraction of specific classes.
    - 'non_iid_dirichlet': Dirichlet distribution Dir(alpha) across class label frequencies.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    targets = []
    if hasattr(dataset, "targets"):
        targets = dataset.targets
    elif hasattr(dataset, "labels"):
        targets = dataset.labels
    else:
        # Extract from dataset samples
        targets = [dataset[i][1] for i in range(len(dataset))]

    targets = np.array(targets)
    num_samples = len(targets)
    classes = np.unique(targets)
    num_classes = len(classes)

    client_indices = [[] for _ in range(num_clients)]

    if partition_type == "iid":
        perm = np.random.permutation(num_samples)
        client_indices = [perm[i::num_clients].tolist() for i in range(num_clients)]

    elif partition_type == "non_iid_dirichlet":
        # Dirichlet distribution
        for c in classes:
            idx_c = np.where(targets == c)[0]
            np.random.shuffle(idx_c)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            # Rescale to integer counts
            proportions = np.array([p * (len(idx_c) / proportions.sum()) for p in proportions])
            proportions = np.cumsum(proportions).astype(int)[:-1]
            split_c = np.split(idx_c, proportions)
            for client_id in range(num_clients):
                client_indices[client_id].extend(split_c[client_id].tolist())

    elif partition_type == "non_iid_pathological":
        # Sort by class label and distribute shards
        sorted_indices = np.argsort(targets)
        shards_per_client = 2
        total_shards = num_clients * shards_per_client
        shard_size = num_samples // total_shards
        shard_indices = [sorted_indices[i * shard_size:(i + 1) * shard_size] for i in range(total_shards)]
        np.random.shuffle(shard_indices)

        for client_id in range(num_clients):
            for s in range(shards_per_client):
                client_indices[client_id].extend(shard_indices[client_id * shards_per_client + s].tolist())

    return client_indices

def create_federated_client_loaders(
    dataset: Dataset,
    client_indices: List[List[int]],
    batch_size: int = 8,
    shuffle: bool = True
) -> List[DataLoader]:
    loaders = []
    for indices in client_indices:
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle)
        loaders.append(loader)
    return loaders

def get_partition_summary(dataset: Dataset, client_indices: List[List[int]]) -> List[Dict[str, Any]]:
    summary = []
    for client_id, indices in enumerate(client_indices):
        class_counts = {}
        for idx in indices:
            label = dataset[idx][1]
            class_counts[str(label)] = class_counts.get(str(label), 0) + 1
        summary.append({
            "client_id": client_id + 1,
            "sample_count": len(indices),
            "class_distribution": class_counts
        })
    return summary

if __name__ == "__main__":
    from torchvision.datasets import FakeData
    dummy = FakeData(size=40, image_size=(1, 28, 28), num_classes=2)
    parts = partition_federated_dataset(dummy, num_clients=2, partition_type="iid")
    summ = get_partition_summary(dummy, parts)
    print("Federated Partition Summary:", summ)
