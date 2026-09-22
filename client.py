from collections import OrderedDict
import torch
import flwr as fl
import snntorch.functional as SF
from model import SpikingCNN, measure_uncertainty

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id, data_loader, device='cpu', uncertainty_threshold=1.0):
        self.client_id = client_id
        self.data_loader = data_loader
        self.device = device
        
        self.model = SpikingCNN(num_classes=4).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = SF.ce_rate_loss()  # Cross-entropy rate loss for spiking networks
        
        self.uncertainty_threshold = uncertainty_threshold
        self.last_uncertainty = 0.0

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v).to(self.device) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        
        epochs = config.get("epochs", 1) if config else 1
        total_uncertainty = 0.0
        batches = 0
        
        # Local training loop
        for epoch in range(epochs):
            for data, targets in self.data_loader:
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass returns spikes and membrane potential over time
                spk_rec, mem_rec = self.model(data)
                
                # Loss calculation
                loss = self.criterion(spk_rec, targets)
                loss.backward()
                self.optimizer.step()
                
                # Measure uncertainty to decide on update utility
                batch_uncert = measure_uncertainty(spk_rec)
                total_uncertainty += batch_uncert
                batches += 1
                
        # Calculate avg uncertainty over this local training pass
        if batches > 0:
            self.last_uncertainty = total_uncertainty / batches
            
        is_informative = float(self.last_uncertainty) > float(self.uncertainty_threshold)
        
        metrics = {
            "client_id": int(self.client_id),
            "uncertainty": float(self.last_uncertainty),
            "threshold": float(self.uncertainty_threshold),
            "is_informative": bool(is_informative)
        }
        
        return self.get_parameters(), len(self.data_loader.dataset), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, targets in self.data_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                spk_rec, _ = self.model(data)
                loss += self.criterion(spk_rec, targets).item() * data.size(0)
                # Count correct predictions based on highest spike count
                spk_sum = spk_rec.sum(dim=0)
                pred = spk_sum.argmax(dim=1)
                correct += pred.eq(targets).sum().item()
                total += targets.size(0)
                
        avg_loss = loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return float(avg_loss), total, {"accuracy": float(accuracy)}
