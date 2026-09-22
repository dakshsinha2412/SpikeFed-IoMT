import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import kagglehub
import os
import flwr as fl
from model import SpikingCNN
from client import FlowerClient
from server import SpikeFedStrategy

def load_kaggle_dataset(num_clients=5, batch_size=16):
    """
    Downloads and loads the Kaggle Brain Tumor MRI Dataset,
    preprocesses images (grayscale, resized to 28x28), and splits them
    among the federated clients.
    """
    print("Fetching Kaggle Brain Tumor MRI Dataset...")
    dataset_path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
    train_dir = os.path.join(dataset_path, "Training")
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    full_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    print(f"Dataset loaded: {len(full_dataset)} total images across classes: {full_dataset.classes}")
    
    # Split dataset equally among clients
    total_samples = len(full_dataset)
    samples_per_client = total_samples // num_clients
    lengths = [samples_per_client] * num_clients
    lengths[-1] += total_samples - sum(lengths)
    
    client_subsets = random_split(full_dataset, lengths)
    client_loaders = [DataLoader(subset, batch_size=batch_size, shuffle=True) for subset in client_subsets]
    return client_loaders

def main():
    print("==================================================")
    print(" SpikeFed-IoMT: Communication-Efficient FL (Flower)")
    print("==================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    num_clients = 5
    client_loaders = load_kaggle_dataset(num_clients=num_clients)
    thresholds = [1.38, 1.37, 1.36, 1.38, 1.39]
    
    # Client generator function for Flower Simulation
    def client_fn(cid: str) -> fl.client.Client:
        client_idx = int(cid)
        loader = client_loaders[client_idx]
        thresh = thresholds[client_idx]
        return FlowerClient(
            client_id=client_idx + 1,
            data_loader=loader,
            device=device,
            uncertainty_threshold=thresh
        ).to_client()

    # Define custom Flower Strategy
    strategy = SpikeFedStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
    )

    num_rounds = 3
    print(f"\nStarting Flower Simulation for {num_rounds} rounds across {num_clients} clients...")
    
    # Start Flower Simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    print("\n==================================================")
    print("             Simulation Complete!                 ")
    print("==================================================")
    print(f"Total Updates in standard FedAvg: {strategy.total_possible_updates}")
    print(f"Total Updates in SpikeFed-IoMT  : {strategy.actual_received_updates}")
    final_savings = 100.0 * (1.0 - (strategy.actual_received_updates / max(strategy.total_possible_updates, 1)))
    print(f"Overall Communication Savings   : {final_savings:.2f}%")

if __name__ == "__main__":
    main()
