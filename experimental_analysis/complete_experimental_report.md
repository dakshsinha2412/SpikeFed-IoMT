# Complete Experimental Report

## Scope and evidence
This report extracts source-defined and saved-run information from the local project. No retraining was performed. The conference paper itself was not present in the supplied workspace; Tables I-V follow the requested structure. All evaluated outcomes below are **synthetic pipeline-validation results**, not medical diagnostic or clinical validation.

## Table I - Actual SNN Architecture
| Component | Actual implementation / value | Evidence category |
| Input | 1 x 28 x 28 grayscale image | Source code |
| Conv1 + LIF | Conv2d 1->4, 3x3 padding 1; Leaky LIF | Source code |
| Pool1 | MaxPool2d 2 | Source code |
| Conv2 + LIF | Conv2d 4->8, 3x3 padding 1; Leaky LIF | Source code |
| Pool2 | MaxPool2d 2; flattened 8 x 7 x 7 = 392 | Source code |
| Dense + LIF | Linear 392->32; Leaky LIF | Source code |
| Output + LIF | Linear 32->4; Leaky LIF | Source code |
| Temporal steps | 10 | Source code |
| Membrane decay beta | 0.9 | Source code |
| Threshold theta | Library default (not explicitly configured) | Source audit |
| Surrogate gradient | snnTorch fast_sigmoid(slope=25) | Source code |
| Encoding | Static normalized image presented each time step | Source code |
| Reset mechanism | snnTorch Leaky library default; not explicitly configured | Source audit |
| Trainable parameters | 13,044 | Measured/source |
| Checkpoint size | 60017 bytes | Measured filesystem |

## Table II - Expected Effects of Parameter Changes
| Parameter change | Theoretical expectation | Observed evidence in this project |
| --- | --- | --- |
| More temporal steps | Greater temporal compute and memory; may change spike-rate decisions | Not experimentally varied. |
| Higher beta | Longer membrane retention | Not experimentally varied; beta fixed at 0.9. |
| More local epochs | More local computation and possible client drift | Not experimentally varied; one epoch used. |
| More clients/rounds | More parameter traffic and computation | Three clients/three rounds measured only. |
| Non-IID partition | Distribution skew can impede convergence | Round-3 global local-eval accuracy: IID 0.2687; Non-IID 0.2985. |

## Table III - Client Deployment Configurations
| Configuration | Status | Evidence |
| --- | --- | --- |
| Laptop simulation | Executed | CPU, three local simulated clients, synthetic data. |
| Hybrid server + Pi client | Prepared, not executed | Flower server/client and Pi scripts exist; no hardware measurements. |
| Physical Raspberry Pi deployment | Not measured | Hardware test unavailable. |

## Table IV - Actual Experimental Settings
| Setting | Actual value | Evidence category |
| Dataset | SyntheticMedicalImages (pipeline-test only) | Source/results |
| Total images | 96 (24 per synthetic class x 4) | Source code |
| Split | 70% train = 67; 15% validation = 14; 15% test = 15 | Source code/calculated |
| Classes | 4 synthetic classes | Source code |
| Image processing | grayscale; resize 28x28; ToTensor; Normalize(mean=0.5,std=0.5) | Source code |
| Clients | 3 | Saved IID/non-IID records |
| Rounds | 3 | Saved IID/non-IID records |
| Local epochs | 1 | main.py default/config |
| Batch size | 8 | Saved run configuration/default |
| Optimizer | Adam | Source code |
| Learning rate | 0.001 | Source code |
| Loss | snnTorch ce_rate_loss | Source code |
| Seed | 42 | main.py default |
| Device | CPU in saved runs | Run output/config |
| Precision | PyTorch default float32; not explicitly logged | Source audit |
| FedAvg | sample-count weighted mean | Source code |

## Table V - Actual Experimental Results
| Method | IID held-out accuracy | Non-IID held-out accuracy | Status |
| --- | --- | --- | --- |
| Federated SNN | 0.2000 | 0.2000 | Executed synthetic runs |
| Central CNN | N/A | N/A | Not implemented/evaluated |
| Central SNN | N/A | N/A | Not implemented/evaluated |
| Federated CNN | N/A | N/A | Not implemented/evaluated |

## IID versus Non-IID comparison
IID final local-evaluation global accuracy was 0.2687; Non-IID was 0.2985. Both held-out synthetic accuracy values were 0.2000. Measured sums of communication-round duration were 1.535s (IID) and 6.620s (Non-IID). Each configuration recorded 156,768 measured upload bytes per round (three 52,256-byte state updates). The result files do not isolate aggregation time or validation loss, so those are NOT MEASURED.

## Communication calculation
FedAvg uses local sample counts as weights: W = sum(n_k W_k) / sum(n_k). For each three-round run, measured parameter uploads are 156,768 x 3 = 470,304 bytes. If the global state is redistributed to each of the three clients with the same tensor size, calculated parameter downloads are also 470,304 bytes; combined parameter exchange is 940,608 bytes. These calculations exclude network/protocol overhead.

## Computational performance
Checkpoint size is 60017 bytes. Per-round time and local training time are saved. CPU utilization, RAM/peak memory, inference latency, and Raspberry Pi metrics were not recorded and are explicitly NOT MEASURED.

## Required next experiments
See `missing_experiments_checklist.md` for the evidence-based gap list.
