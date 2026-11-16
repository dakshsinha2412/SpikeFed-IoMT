"""Independent hospital client.  It retains all raw samples locally."""
from __future__ import annotations
from collections import OrderedDict
import time
import torch
import snntorch.functional as SF
from model import SpikingCNN
from dataset import load_image_folder
from metrics import classification_metrics

# Maximize multi-core execution throughput on edge ARM CPU
torch.set_num_threads(4)

def clone_state(state): return {k: v.detach().cpu().clone() for k, v in state.items()}

class HospitalClient:
    def __init__(self, client_id, train_loader, val_loader=None, device=None, num_classes=4):
        self.client_id, self.train_loader, self.val_loader = str(client_id), train_loader, val_loader
        self.device = device or torch.device("cpu"); self.model = SpikingCNN(num_classes).to(self.device)
        self.criterion = SF.ce_rate_loss()
    def set_parameters(self, state): self.model.load_state_dict(state, strict=True)
    def fit(self, global_state, epochs=1, lr=.003):
        self.set_parameters(global_state); self.model.train(); opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_sum = samples = total_spikes = 0; started = time.perf_counter()
        for _ in range(epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device); opt.zero_grad(); spk, _ = self.model(x)
                loss = self.criterion(spk, y); loss.backward(); opt.step()
                loss_sum += loss.item() * len(y); samples += len(y)
                total_spikes += spk.sum().item()
        state = clone_state(self.model.state_dict())
        nbytes = sum(v.numpy().nbytes for v in state.values())
        return state, len(self.train_loader.dataset), {
            "client_id": self.client_id,
            "train_loss": loss_sum/max(samples,1),
            "training_seconds": time.perf_counter()-started,
            "update_bytes": nbytes,
            "total_spikes": total_spikes
        }
    def evaluate(self, loader=None):
        loader = loader or self.val_loader or self.train_loader; self.model.eval(); loss = correct = total = total_spk = 0
        predictions=[]; targets=[]
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device); spk, _ = self.model(x); loss += self.criterion(spk,y).item()*len(y)
                pred=spk.sum(0).argmax(1); correct += (pred==y).sum().item(); total += len(y); predictions.append(pred); targets.append(y)
                total_spk += spk.sum().item()
        metrics=classification_metrics(torch.cat(predictions),torch.cat(targets),self.model.fc2.out_features)
        return {"loss": loss/max(total,1), "samples": total, "total_spikes": total_spk, **metrics}

# Flower adapter for real server/Pi runs. Parameters only; raw images never leave this process.
try:
    import flwr as fl
    class FlowerClient(fl.client.NumPyClient):
        def __init__(self, client_id, data_loader, device=torch.device("cpu"), num_classes=None, **_):
            if num_classes is None:
                num_classes = len(data_loader.dataset.classes) if hasattr(data_loader.dataset, "classes") else 4
            self.local = HospitalClient(client_id, data_loader, device=device, num_classes=num_classes)
            self.round_num = 0
        def get_parameters(self, config=None):
            return [v.numpy() for v in self.local.model.state_dict().values()]
        def set_parameters(self, parameters):
            self.local.set_parameters(OrderedDict((k, torch.tensor(v)) for k, v in zip(self.local.model.state_dict(), parameters)))
        def fit(self, parameters, config):
            self.round_num += 1
            param_bytes = sum(p.nbytes for p in parameters)
            print(f"\n[PI EDGE CLIENT] ======================================================")
            print(f"[PI EDGE CLIENT] >>> PHYSICAL FL ROUND {self.round_num}: FIT PHASE")
            print(f"[PI EDGE CLIENT] Step 3: Received global model weights from HP Victus server")
            print(f"         Tensors: {len(parameters)} | Total Payload: {param_bytes} bytes ({param_bytes/1024:.2f} KB)")
            self.set_parameters(parameters)

            epochs = int(config.get("local_epochs", 1))
            print(f"[PI EDGE CLIENT] Step 4: Training LIF-SNN locally on Raspberry Pi 3B+...")
            state, n, metric = self.local.fit(clone_state(self.local.model.state_dict()), epochs)
            spikes = metric.get("total_spikes", 0.0)
            print(f"         Local training finished: {n} images, {epochs} epoch(s)")
            print(f"         Train Loss: {metric['train_loss']:.4f} | Spikes Fired: {spikes:.0f} | Time: {metric['training_seconds']:.2f}s")

            updated_ndarrays = [v.numpy() for v in state.values()]
            uplink_bytes = sum(a.nbytes for a in updated_ndarrays)
            print(f"[PI EDGE CLIENT] Step 5: Transferring updated weights from Pi to HP Victus")
            print(f"         Tensors: {len(updated_ndarrays)} | Uplink Payload: {uplink_bytes} bytes ({uplink_bytes/1024:.2f} KB)")
            metric["is_informative"] = True
            metric["client_id"] = self.local.client_id
            return updated_ndarrays, n, metric

        def evaluate(self, parameters, config):
            print(f"\n[PI EDGE CLIENT] >>> PHYSICAL FL ROUND {self.round_num}: EVALUATE PHASE")
            print(f"[PI EDGE CLIENT] Step 7: Received updated global model from HP Victus server")
            self.set_parameters(parameters)
            m = self.local.evaluate()
            print(f"[PI EDGE CLIENT] Evaluated on local dataset: Loss = {m['loss']:.4f}, Accuracy = {m['accuracy']:.2%}, Spikes = {m.get('total_spikes', 0):.0f}")
            print(f"[PI EDGE CLIENT] Sending evaluation metrics back to HP Victus server")
            print(f"[PI EDGE CLIENT] ======================================================\n")
            return m["loss"], m["samples"], {"accuracy": m["accuracy"]}
except ImportError: FlowerClient = None

def run_client(server_address="127.0.0.1:8080", client_id=1, data_dir="data", batch_size=8, **_):
    import flwr as fl
    from torch.utils.data import DataLoader
    dataset = load_image_folder(data_dir); loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    fl.client.start_numpy_client(server_address=server_address, client=FlowerClient(client_id, loader, torch.device("cpu"), num_classes=len(dataset.classes)))

if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description="Federated SNN hospital client")
    parser.add_argument("--server",default="127.0.0.1:8080"); parser.add_argument("--client_id",type=int,default=1)
    parser.add_argument("--data_dir",required=True); parser.add_argument("--batch_size",type=int,default=8)
    args=parser.parse_args(); run_client(args.server,args.client_id,args.data_dir,args.batch_size)
