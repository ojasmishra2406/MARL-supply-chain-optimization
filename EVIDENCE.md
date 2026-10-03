# FINAL RESEARCH EVIDENCE (MACHINE-VERIFIED)

**Generated At:** 2026-10-02T07:44:57.414791Z
**Git Commit:** `c9dcc3ef307603c4465f601342143de691890fc3`

## 1. System Environment
- **OS:** Windows-10-10.0.26200-SP0
- **CPU:** Intel64 Family 6 Model 154 Stepping 3, GenuineIntel (16 logical cores)
- **RAM:** 16.85 GB
- **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (4.29 GB)
- **PyTorch:** 2.3.1+cu121 (CUDA: 12.1)

## 2. Model Architecture & Forward Latency (100 passes)
| Architecture | Parameters | Forward MS (Mean) | Forward MS (P99) |
|---|---|---|---|
| IPPO | 34,819 | 0.1179 | 0.1543 |
| MAPPO | 36,739 | 0.1287 | 0.1728 |
| IPPO_COMM | 36,867 | 0.1372 | 0.2183 |
| MAPPO_COMM | 41,859 | 0.1582 | 0.2611 |
| GNN_MAPPO | 26,316 | 0.0306 | 0.0426 |

## 3. Simulator Performance
- **Steps Measured:** 5000
- **Steps per Second:** 30642.2
- **Mean Step Time:** 0.0326 ms

## 4. Forecasting Benchmarks (Test Demand OOD)
| Model | Fit Time (s) | Predict (mean ms) | MAE | RMSE | sMAPE |
|---|---|---|---|---|---|
| Naive | N/A | 0.0003 | 4.5106 | 5.3753 | 23.0918 |
| MovingAverage | N/A | 0.0083 | 3.4723 | 4.1995 | 17.9424 |
| ExpSmoothing | N/A | 0.1043 | 3.4209 | 4.1648 | 17.6524 |
| XGBoost | 0.302 | 0.4212 | 1.2778 | 1.5667 | 6.8933 |
| LSTM | 2.929 | 0.9393 | 15.4634 | 15.9625 | 126.1789 |

## 5. Artifact Validation (Phase 6 / 10)
- **Phase 6 Checkpoints Discovered:** 40
- **Phase 6 Checkpoints Valid:** 40 / 40
- **Hash Mismatches:** 40
- **Phase 10 Checkpoints Discovered:** 10
- **Phase 10 Checkpoints Valid:** 10 / 10

## 6. Evaluations (Phase 7)
- **Total Evaluation JSONs:** 1230
- **RL Evaluations Found:** 1200 (Expected: 1200)
- **Unique Checkpoints Referenced:** 40
- **Hash Mismatches:** 1200

## 7. Statistical Independence (Phase 8)
- **Pseudoreplication Check:** PASS
### Key Comparisons
- **IPPO vs MAPPO (Baseline, In-Distribution)**
  - N1=5, N2=5
  - Welch t: -3.234852
  - Cohen's d: -2.0459
  - BH-Adjusted p: 0.06736626
- **MAPPO vs MAPPO_COMM (Baseline, In-Distribution)**
  - N1=5, N2=5
  - Welch t: -0.0761
  - Cohen's d: -0.04813
  - BH-Adjusted p: 0.94129647
- **IPPO vs IPPO_COMM (Baseline, In-Distribution)**
  - N1=5, N2=5
  - Welch t: 0.206427
  - Cohen's d: 0.130556
  - BH-Adjusted p: 0.94129647
- **MAPPO vs MAPPO_COMM (Baseline, Combined Shift)**
  - N1=5, N2=5
  - Welch t: -0.499544
  - Cohen's d: -0.315939
  - BH-Adjusted p: 0.94129647

## 8. Hard Failures
Found 1261 Hard Failures. Below is a sample (up to 20):
- X `[MODEL INVENTORY] [phase6] HASH MISMATCH phase6_ippo_baseline_s0: actual=3943793934351097... stored=f16fb9336ecf82ec...`
- X `[MODEL INVENTORY] [phase6] HASH MISMATCH phase6_ippo_baseline_s1: actual=19f69757de3e7622... stored=8b5e843f43f2c36b...`
- X `[MODEL INVENTORY] [phase6] HASH MISMATCH phase6_ippo_baseline_s2: actual=3edf09e8640063fb... stored=9b82569dd28357d2...`
- X `[MODEL INVENTORY] [phase6] HASH MISMATCH phase6_ippo_baseline_s3: actual=4745b14b26fb2ca5... stored=08213a3d92fb50fe...`
- X `[MODEL INVENTORY] [phase6] HASH MISMATCH phase6_ippo_baseline_s4: actual=13eb4c06f4963b2f... stored=5958c4f84bf055ac...`
- X `[MODEL INVENTORY] [phase6] HASH MISMATCH phase6_ippo_comm_baseline_s0: actual=98312f2b92ecf34d... stored=99f191123f90a7b0...`
- X `[MODEL INVENTORY] [phase6] HASH MISMATCH phase6_ippo_comm_baseline_s1: actual=7703d912df4f5cc8... stored=84e9b56a69d36b5f...`
- X `[MODEL INVENTORY] [phase6] HASH MISMATCH phase6_ippo_comm_baseline_s2: actual=23aa903901027c8f... stored=bfb7a837816bed75...`
- X `[MODEL INVENTORY] [phase6] HASH MISMATCH phase6_ippo_comm_baseline_s3: actual=f1d10fb484e551cb... stored=8e222980e6374bdf...`
- X `[MODEL INVENTORY] [phase6] HASH MISMATCH phase6_ippo_comm_baseline_s4: actual=f64cbeb54b3f3dfa... stored=e70f6dafbc1d00d8...`
- X `[MODEL INVENTORY] [phase6] HASH MISMATCH phase6_ippo_comm_high_variance_s0: actual=eeda02b032b1b18c... stored=cce527320cedc298...`
- X `[MODEL INVENTORY] [phase6] HASH MISMATCH phase6_ippo_comm_high_variance_s1: actual=ac0a754cd7139bca... stored=29f21ed202e23a45...`
- X `[MODEL INVENTORY] [phase6] HASH MISMATCH phase6_ippo_comm_high_variance_s2: actual=409c161340fa7950... stored=70e607562982631a...`
- X `[MODEL INVENTORY] [phase6] HASH MISMATCH phase6_ippo_comm_high_variance_s3: actual=ca41f88eddacfeb7... stored=4b915265fa1ad4ee...`
- X `[MODEL INVENTORY] [phase6] HASH MISMATCH phase6_ippo_comm_high_variance_s4: actual=126a8ef242e67a27... stored=5ede38c43b38bf12...`
- X `[MODEL INVENTORY] [phase6] HASH MISMATCH phase6_ippo_high_variance_s0: actual=132a96476529ff22... stored=a12f2a1ec9b19f9a...`
- X `[MODEL INVENTORY] [phase6] HASH MISMATCH phase6_ippo_high_variance_s1: actual=d8434e039b2c8ce9... stored=3f95a9589195c966...`
- X `[MODEL INVENTORY] [phase6] HASH MISMATCH phase6_ippo_high_variance_s2: actual=26eebc98fab54c33... stored=78ad3db52de27abf...`
- X `[MODEL INVENTORY] [phase6] HASH MISMATCH phase6_ippo_high_variance_s3: actual=e49499803a497659... stored=15af8c90d3983eeb...`
- X `[MODEL INVENTORY] [phase6] HASH MISMATCH phase6_ippo_high_variance_s4: actual=7e0c0d9c72abdc34... stored=8a6206110e2a07d2...`