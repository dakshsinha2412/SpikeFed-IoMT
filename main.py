"""One-command, local federated SNN experiment.  Default uses synthetic test data."""
from __future__ import annotations
import argparse
import time
from torch.utils.data import DataLoader
from dataset import SyntheticMedicalImages, load_image_folder, split_dataset, make_federated_loaders
from client import HospitalClient
from federated import FederatedServer
from model import parameter_count
from utils import get_device, set_seed
from visualization import generate_artifacts

def run_simulation(rounds=2, num_clients=3, local_epochs=1, batch_size=8, non_iid=False, data_dir="", seed=42, label="simulation"):
    set_seed(seed); device=get_device()
    dataset=load_image_folder(data_dir) if data_dir else SyntheticMedicalImages(seed=seed)
    train, _, test = split_dataset(dataset, seed=seed)
    loaders=make_federated_loaders(train, num_clients, batch_size, non_iid, seed)
    clients=[HospitalClient(f"Hospital {i+1}", loader, device=device, num_classes=len(dataset.classes)) for i,loader in enumerate(loaders)]
    server=FederatedServer(num_classes=len(dataset.classes), device=device)
    started=time.perf_counter()
    print(f"Device: {device}; model parameters: {parameter_count(server.model)}; synthetic data: {not bool(data_dir)}", flush=True)
    for _ in range(rounds):
        row=server.run_round(clients, local_epochs); print(f"Round {row['round']}: accuracy={row['global_accuracy']:.3f}, loss={row['global_loss']:.3f}, bytes={row['communication_bytes']}", flush=True)
    test_loader=DataLoader(test,batch_size=batch_size); clients[0].set_parameters(server.global_state); final=clients[0].evaluate(test_loader)
    server.history[-1]["held_out_test"] = final
    server.history[-1]["run_configuration"] = {"num_clients":num_clients,"communication_rounds":rounds,"local_epochs":local_epochs,"batch_size":batch_size,"distribution":"non_iid" if non_iid else "iid","dataset_type":"real_image_folder" if data_dir else "synthetic","data_dir":data_dir or None,"elapsed_seconds":time.perf_counter()-started}
    output=server.save(f"{label}_results.json")
    plots=generate_artifacts(output)
    print(f"Held-out test accuracy={final['accuracy']:.3f}; saved {output}, checkpoints/global_model.pt, and {plots}", flush=True)
    return server.history

if __name__ == "__main__":
    p=argparse.ArgumentParser(description="Federated LIF-SNN medical-image pipeline")
    p.add_argument("--rounds",type=int,default=2); p.add_argument("--num-clients",type=int,default=3)
    p.add_argument("--local-epochs",type=int,default=1); p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--non-iid",action="store_true"); p.add_argument("--data-dir",default=""); p.add_argument("--seed",type=int,default=42); p.add_argument("--label",default="simulation")
    args=p.parse_args(); run_simulation(args.rounds,args.num_clients,args.local_epochs,args.batch_size,args.non_iid,args.data_dir,args.seed,args.label)
