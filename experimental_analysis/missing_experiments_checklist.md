# Missing Experiments and Measurement Checklist

## Completed (synthetic laptop simulation)
- [x] Genuine LIF-SNN source implementation and 13,044-parameter count.
- [x] Three-client IID and Non-IID FedAvg simulations, three rounds each.
- [x] Held-out synthetic metrics, per-client metrics, confusion matrices, update-byte measurements, and per-round timing.
- [x] Checkpoint saved and reloaded in automated tests.

## Partially completed
- [~] Communication: tensor upload bytes are measured. Downlink is calculated under an explicit model-distribution assumption; protocol framing, TLS, and retries are not measured.
- [~] Dataset isolation is implemented in the simulation loader. Real patient-level split validation has not been performed.

## Missing / NOT MEASURED
- [ ] Central CNN baseline.
- [ ] Central SNN baseline.
- [ ] Federated CNN baseline.
- [ ] Real medical dataset and clinical validation.
- [ ] Raspberry Pi 3 inference/training time, CPU, RAM, temperature, power, and network measurements.
- [ ] CPU utilization, peak RAM, inference latency, aggregation-only time, and validation-loss tracking.
- [ ] Cryptographic secure aggregation, differential privacy, and formal privacy guarantees.

No physical Raspberry Pi result is claimed in this report.
