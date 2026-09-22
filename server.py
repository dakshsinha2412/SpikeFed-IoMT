from typing import List, Tuple, Union, Optional, Dict
import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

class SpikeFedStrategy(FedAvg):
    """
    Custom Flower FedAvg strategy implementing SpikeFed-IoMT uncertainty-based 
    update filtering and communication efficiency tracking.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.total_possible_updates = 0
        self.actual_received_updates = 0

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        print(f"\n--- Starting FL Round {server_round} (Flower Framework) ---")
        
        if not results:
            print("No client fit results received in this round.")
            return None, {}
            
        informative_results = []
        for client, fit_res in results:
            self.total_possible_updates += 1
            is_informative = fit_res.metrics.get("is_informative", True)
            uncert = fit_res.metrics.get("uncertainty", 0.0)
            client_id = fit_res.metrics.get("client_id", "Unknown")
            
            if is_informative:
                informative_results.append((client, fit_res))
                self.actual_received_updates += 1
                print(f"Client {client_id} sent an update. (Uncertainty: {uncert:.3f} > Threshold)")
            else:
                print(f"Client {client_id} skipped update. (Uncertainty: {uncert:.3f} <= Threshold)")

        # Aggregate weights only from clients who sent informative updates
        if informative_results:
            aggregated_parameters, metrics = super().aggregate_fit(server_round, informative_results, failures)
            print("Model aggregated successfully.")
        else:
            print("No informative updates received. Global model unchanged.")
            # Return current global weights (from parent aggregated parameters or last round)
            aggregated_parameters = None
            metrics = {}

        savings = 100.0 * (1.0 - (self.actual_received_updates / max(self.total_possible_updates, 1)))
        metrics["communication_savings"] = savings
        print(f"Round {server_round} Communication Savings: {savings:.2f}%")
        
        return aggregated_parameters, metrics
