"""
server.py – SpikeFed-IoMT Federated Learning Server
=====================================================
Implements the custom FedAvg strategy that:
  - Filters out uninformative updates (client sent empty parameters).
  - Tracks communication savings across all rounds.
  - Logs per-round metrics to results/round_results.json.
  - Saves the final global model to results/model.pt.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import numpy as np
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

from utils import get_logger, print_metrics_table, save_json, timestamp
from model import SpikingCNN

logger = get_logger("server")


class SpikeFedStrategy(FedAvg):
    """
    Custom Flower FedAvg strategy implementing SpikeFed-IoMT
    uncertainty-based update filtering and communication-efficiency tracking.

    Clients signal whether their local update is informative by including
    model parameters in the FitRes.  If the parameters tensor list is empty
    the update is skipped, saving uplink bandwidth.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.total_possible_updates: int = 0
        self.actual_received_updates: int = 0
        self._round_log: List[Dict] = []
        # Flower keeps the current global parameters internally.  Keep the
        # latest successful aggregate as well so the checkpoint written at
        # the end of a networked run is the trained model, rather than a
        # freshly-initialised model.
        self.latest_parameters: Optional[Parameters] = kwargs.get(
            "initial_parameters"
        )

    # ──────────────────────────── aggregate_fit ────────────────────────────

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        logger.info(f"--- FL Round {server_round} ---")
        logger.info(f"[HP VICTUS SERVER] Step 5: Received {len(results)} client update(s) over physical network.")

        if not results:
            logger.warning("No client fit results received in this round.")
            return None, {}

        informative_results = []
        round_client_logs = []

        for client, fit_res in results:
            self.total_possible_updates += 1
            is_informative = fit_res.metrics.get("is_informative", True)
            uncert = fit_res.metrics.get("uncertainty", 0.0)
            client_id = fit_res.metrics.get("client_id", "?")
            threshold = fit_res.metrics.get("threshold", 0.0)
            has_parameters = bool(fit_res.parameters.tensors)

            if is_informative and has_parameters:
                informative_results.append((client, fit_res))
                self.actual_received_updates += 1
                status = "SENT   ✓"
            else:
                status = "SKIPPED ✗"

            logger.info(
                f"  Client {client_id}: {status} | "
                f"uncertainty={uncert:.4f} threshold={threshold:.4f}"
            )
            round_client_logs.append({
                "client_id": client_id,
                "uncertainty": uncert,
                "threshold": threshold,
                "sent_update": is_informative and has_parameters,
            })

        # Aggregate from informative clients only
        if informative_results:
            logger.info(f"[HP VICTUS SERVER] Step 6: Performing FedAvg aggregation on HP Victus...")
            agg_params, metrics = super().aggregate_fit(
                server_round, informative_results, failures
            )
            if agg_params is not None:
                self.latest_parameters = agg_params
            logger.info(f"[HP VICTUS SERVER] Step 6 Complete: Aggregated {len(informative_results)} / {len(results)} client update(s).")
            logger.info(f"[HP VICTUS SERVER] Step 7: Distributing updated global model back to edge client(s).")
        else:
            logger.warning("  No informative updates – global model unchanged.")
            agg_params = None
            metrics = {}

        savings = 100.0 * (
            1.0 - (self.actual_received_updates / max(self.total_possible_updates, 1))
        )
        metrics["communication_savings"] = savings
        logger.info(f"  Cumulative communication savings: {savings:.2f}%")

        # Store round log
        self._round_log.append({
            "round": server_round,
            "clients": round_client_logs,
            "updates_sent": len(informative_results),
            "updates_possible": len(results),
            "cumulative_savings_pct": round(savings, 4),
        })

        return agg_params, metrics

    # ─────────────────────────── aggregate_evaluate ───────────────────────

    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures,
    ):
        """Extend parent with per-round metric logging."""
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        if metrics:
            logger.info(
                f"  Round {server_round} eval – "
                f"accuracy={metrics.get('accuracy', 0):.4f}  "
                f"f1_macro={metrics.get('f1_macro', 0):.4f}"
            )
        return loss, metrics

    # ─────────────────────────── save_results ─────────────────────────────

    def save_results(self, history) -> None:
        """Persist round logs and final metrics to the results/ directory."""
        final_savings = 100.0 * (
            1.0 - (self.actual_received_updates / max(self.total_possible_updates, 1))
        )
        summary = {
            "timestamp": timestamp(),
            "total_possible_updates": self.total_possible_updates,
            "actual_sent_updates": self.actual_received_updates,
            "overall_communication_savings_pct": round(final_savings, 4),
            "rounds": self._round_log,
        }
        path = save_json(summary, "round_results.json")
        logger.info(f"Round results saved → {path}")

        print_metrics_table(
            {
                "Total possible updates": self.total_possible_updates,
                "Actual updates sent": self.actual_received_updates,
                "Communication savings": f"{final_savings:.2f}%",
            },
            title="SpikeFed-IoMT Communication Efficiency",
        )


# ─────────────────────────────── run_server ───────────────────────────────

def run_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    num_rounds: int = 3,
    min_clients: int = 1,
    num_classes: int = 2,
) -> None:
    """Start the Flower gRPC server and wait for Pi clients to connect."""
    import os

    server_address = f"{host}:{port}"
    strategy = SpikeFedStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        initial_parameters=ndarrays_to_parameters(
            [value.detach().cpu().numpy() for value in SpikingCNN(num_classes=num_classes).state_dict().values()]
        ),
    )

    print("=" * 52)
    print(f"  SpikeFed-IoMT FL Server  →  {server_address}")
    print(f"  Min clients required : {min_clients}")
    print(f"  Federated rounds     : {num_rounds}")
    print(f"  Model classes        : {num_classes}")
    print("=" * 52)

    history = fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    # Save the actual final global parameters, if at least one client sent an
    # update.  A run in which every update is gated is still valid, but cannot
    # produce a trained checkpoint.
    try:
        if strategy.latest_parameters is not None:
            model = SpikingCNN(num_classes=num_classes)
            os.makedirs("results", exist_ok=True)
            state_dict = model.state_dict()
            arrays = parameters_to_ndarrays(strategy.latest_parameters)
            if len(arrays) != len(state_dict):
                raise ValueError(
                    "Checkpoint parameter count does not match SpikingCNN "
                    f"({len(arrays)} received, {len(state_dict)} expected)."
                )
            final_state = {
                name: torch.as_tensor(array, device="cpu")
                for name, array in zip(state_dict.keys(), arrays)
            }
            model.load_state_dict(final_state, strict=True)
            torch.save(model.state_dict(), os.path.join("results", "model.pt"))
            logger.info("Saved final model checkpoint → results/model.pt")
        else:
            logger.warning(
                "No client update was aggregated; no model checkpoint was written."
            )
    except Exception as exc:
        logger.warning(f"Could not save model checkpoint: {exc}")

    strategy.save_results(history)


# ─────────────────────────────── __main__ ─────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SpikeFed-IoMT Federated Learning Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--min_clients", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=2)
    args = parser.parse_args()

    run_server(
        host=args.host,
        port=args.port,
        num_rounds=args.rounds,
        min_clients=args.min_clients,
        num_classes=args.num_classes,
    )
