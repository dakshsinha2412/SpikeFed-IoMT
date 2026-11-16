# SpikeFed-IoMT – Experiment Results

> Fill in this table after each experiment run for your project review.  
> Results are also auto-saved to `results/round_results.json`.

---

## Experiment Setup

| Parameter | Value |
|-----------|-------|
| Date | |
| Number of Pis (clients) | |
| Federated rounds | |
| Uncertainty threshold | |
| Dataset | Brain Tumor MRI (Kaggle) |
| Images per client | |
| PC device | |
| Pi model | Raspberry Pi 3 |

---

## Run 1 – Low Threshold (Baseline: all clients send updates)

> Set threshold to a very low value (e.g. 0.01) so all clients always send.

| Round | Clients sent | Accuracy | Macro F1 | Savings % |
|-------|-------------|----------|----------|-----------|
| 1     | | | | |
| 2     | | | | |
| 3     | | | | |
| **Overall** | | | | |

**Total possible updates:** ___  
**Actual updates sent:** ___  
**Overall communication savings:** ____%

---

## Run 2 – Tuned Threshold (SpikeFed-IoMT)

> Use the default threshold (1.38) or the value you tuned.

| Round | Clients sent | Accuracy | Macro F1 | Savings % |
|-------|-------------|----------|----------|-----------|
| 1     | | | | |
| 2     | | | | |
| 3     | | | | |
| **Overall** | | | | |

**Total possible updates:** ___  
**Actual updates sent:** ___  
**Overall communication savings:** ____%

---

## Comparison

| Metric | Baseline (low threshold) | SpikeFed-IoMT (tuned) |
|--------|--------------------------|----------------------|
| Final accuracy | | |
| Final macro F1 | | |
| Total updates sent | | |
| Communication savings | 0% | |
| Accuracy drop vs baseline | — | |

---

## Per-class Breakdown (Run 2)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| glioma | | | |
| meningioma | | | |
| notumor | | | |
| pituitary | | | |

---

## Notes & Observations

_Write any qualitative observations here: convergence behaviour, effect of threshold on savings, model robustness, etc._

---

## How to reproduce

```powershell
# Run 1 – low threshold baseline
.\run_experiment.ps1 -NumPis 2 -Rounds 3
# On each Pi: bash pi_setup/start_client.sh <PC_IP> <ID> 0.01

# Run 2 – tuned threshold
.\run_experiment.ps1 -NumPis 2 -Rounds 3
# On each Pi: bash pi_setup/start_client.sh <PC_IP> <ID> 1.38
```
