"""FedAvg coordinator used for reproducible single-computer simulations."""
from __future__ import annotations
import time, torch
from model import SpikingCNN, parameter_count
from client import clone_state
from utils import save_json

def fedavg(updates):
    """Sample-count weighted parameter average; validates matching tensors."""
    if not updates: raise ValueError("FedAvg requires at least one client update")
    reference = updates[0][0]
    for state, _, _ in updates:
        if state.keys() != reference.keys() or any(state[k].shape != reference[k].shape for k in reference): raise ValueError("Inconsistent client model parameters")
    total = sum(n for _, n, _ in updates)
    return {k: sum(state[k].float() * (n/total) for state,n,_ in updates).to(reference[k].dtype) for k in reference}

class FederatedServer:
    def __init__(self, num_classes=4, device=None):
        self.device=device or torch.device("cpu"); self.model=SpikingCNN(num_classes).to(self.device); self.history=[]
    @property
    def global_state(self): return clone_state(self.model.state_dict())
    def run_round(self, clients, local_epochs=1, lr=.001):
        start=time.perf_counter(); state=self.global_state; updates=[]; reports=[]
        for client in clients:
            update,n,report=client.fit(state, local_epochs, lr); updates.append((update,n,report)); reports.append(report)
        self.model.load_state_dict(fedavg(updates), strict=True)
        evaluations=[]
        for client in clients:
            client.set_parameters(self.global_state); evaluations.append({"client_id":client.client_id, **client.evaluate()})
        total=sum(e["samples"] for e in evaluations)
        row={"round":len(self.history)+1,"clients":reports,"client_evaluations":evaluations,"global_accuracy":sum(e["accuracy"]*e["samples"] for e in evaluations)/max(total,1),"global_loss":sum(e["loss"]*e["samples"] for e in evaluations)/max(total,1),"communication_bytes":sum(r["update_bytes"] for r in reports),"round_seconds":time.perf_counter()-start}
        self.history.append(row); return row
    def save(self, filename="federated_results.json"):
        import os
        os.makedirs("checkpoints",exist_ok=True); torch.save(self.model.state_dict(),"checkpoints/global_model.pt")
        return save_json({"model_parameter_count":parameter_count(self.model),"rounds":self.history,"privacy_note":"Only model states and aggregate training metadata are exchanged. Basic FedAvg is not differential privacy or secure aggregation."},filename)
