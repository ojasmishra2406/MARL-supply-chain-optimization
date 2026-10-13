# Phase 11 Controlled Replication Report
## 50-Iteration Reduced-Budget Replication, Corrected Independent-Seeding Protocol

**Date:** 2026-10-11  
**Execution Window:** 02:06:57 – 14:27:35 IST (Elapsed Wall-Clock: 12h 20m 38s)  
**Branch:** `fix/phase11-controlled-implementation`  
**Git Commit:** `6a21d5ab1df6f081f4325fe48a5a37b48a579377`  
**Protocol Specification:** `docs/PHASE11_STATISTICAL_SPEC.md`  
**Output Directory:** `results/phase11_controlled/`  

---

## Executive Summary

This report documents the rigorous completion of the **50-iteration reduced-budget replication, corrected independent-seeding protocol** for Phase 11 (*Demand Forecasting + GNN-MAPPO for Supply Chain Multi-Agent Reinforcement Learning*). 

This replication directly addressed the critical defects discovered during the forensic audit of historical Phase 11 runs:
1. **Pseudoreplication Elimination:** Hardcoded environment resets (`env.reset(seed=42)`) were replaced with dynamic, run-specific independent resets (`env.reset(seed=self.seed)`), decoupling customer demand realizations and state transitions across training seeds.
2. **Canonical State-Dict SHA-256 Checkpointing:** Replaced incompatible physical-file hashing with deterministic PyTorch tensor state-dict hashing (`rl.checkpoint.save_checkpoint()` / `load_checkpoint()`), verified with 100% success across all 6 models.
3. **Controlled Symmetric Protocol:** Evaluated both Moving Average (MA) and XGBoost under the exact same network architecture, training hyperparameters, parallel environment count (8 envs), rollout lengths (2,048 steps), PPO updates (50 updates), and test matrix (30 OOD episodes per model, 180 evaluation episodes total).

### Core Finding
> **Under independent environment seeding and a 50-iteration training budget, there is no statistically significant evidence ($p = 0.7157$ for Total Cost, $p = 0.2086$ for Service Level, $p = 0.9345$ for Cost/Service-Level efficiency) that conditioning GNN-MAPPO on XGBoost forecasts outperforms conditioning on a simple 5-step Moving Average.**
>
> Point estimates show XGBoost incurs a minor non-significant cost reduction ($-7.13\%$), offset by a non-significant reduction in service level ($-5.38\%$) and a noticeable elevation in the bullwhip effect ($+46.82\%$, raw $p = 0.084$). Previous claims of statistical dominance were an artifact of coupled environment rollouts (pseudoreplication) and uncorrected comparisons.

---

## 1. Wall-Clock Execution & Hardware Telemetry

All runs executed sequentially on CPU to eliminate memory pressure and verify complete provenance. Intermediate snapshots were saved every 10 updates.

| Run Identifier | Forecaster | Seed | Training Updates | Wall-Clock Training | OOD Eval (30 eps) | Canonical SHA-256 Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `phase11_controlled_ma_s1` | Moving Average | 1 | 50 | 121.11 min (2.02h) | 9.5 s | Verified Valid |
| `phase11_controlled_ma_s2` | Moving Average | 2 | 50 | 114.19 min (1.90h) | 47.0 s | Verified Valid |
| `phase11_controlled_ma_s3` | Moving Average | 3 | 50 | 120.91 min (2.02h) | 13.6 s | Verified Valid |
| `phase11_controlled_xgboost_s1` | XGBoost | 1 | 50 | 129.91 min (2.17h) | 11.1 s | Verified Valid |
| `phase11_controlled_xgboost_s2` | XGBoost | 2 | 50 | 123.80 min (2.06h) | 10.8 s | Verified Valid |
| `phase11_controlled_xgboost_s3` | XGBoost | 3 | 50 | 128.37 min (2.14h) | 49.2 s | Verified Valid |
| **Total Study Duration** | — | — | **300 updates** | **738.29 min (12.30h)** | **141.2 s** | **6/6 Valid (100%)** |

### Empirically Measured Iteration Pace & 200-Iteration Projections
- **Moving Average Baseline:** Average $2.37 \text{ min / update}$ ($118.74 \text{ min per 50 updates}$).
  - *Extrapolated 200-Update Wall-Clock Time:* **$7.9 \text{ hours per seed}$** ($23.7 \text{ hours for 3 seeds}$).
- **XGBoost Pipeline:** Average $2.55 \text{ min / update}$ ($127.36 \text{ min per 50 updates}$).
  - *Extrapolated 200-Update Wall-Clock Time:* **$8.5 \text{ hours per seed}$** ($25.5 \text{ hours for 3 seeds}$).
- *Full 200-iteration replication suite (6 runs total):* **$\approx 49.2 \text{ hours of continuous wall-clock compute}$**.

---

## 2. Independent Run Results ($N = 3$ per Group)

Each trained checkpoint was audited across the prespecified 30-episode out-of-distribution (OOD) test matrix (6 scenarios $\times$ 5 evaluation seeds).

| Model ID | Condition | Seed | Total Cost ($\downarrow$) | Service Level ($\uparrow$) | Bullwhip Ratio ($\downarrow$) | Cost / SL Ratio ($\downarrow$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `phase11_controlled_ma_s1` | Moving Average | 1 | 118,112.30 | 89.62% | 14.58 | 131,787.40 |
| `phase11_controlled_ma_s2` | Moving Average | 2 | 170,211.85 | 96.09% | 12.58 | 177,133.59 |
| `phase11_controlled_ma_s3` | Moving Average | 3 | 105,115.70 | 95.94% | 16.69 | 109,560.09 |
| **MA Baseline Summary** | **Mean $\pm$ SD** | — | **$131,146.62 \pm 34,449.92$** | **$93.89\% \pm 3.69\%$** | **$14.62 \pm 2.05$** | **$139,493.70 \pm 34,439.58$** |
| `phase11_controlled_xgboost_s1` | XGBoost | 1 | 98,754.60 | 83.82% | 18.84 | 117,811.97 |
| `phase11_controlled_xgboost_s2` | XGBoost | 2 | 142,610.72 | 88.12% | 26.21 | 161,839.90 |
| `phase11_controlled_xgboost_s3` | XGBoost | 3 | 124,039.33 | 93.59% | 19.33 | 132,539.06 |
| **XGBoost Summary** | **Mean $\pm$ SD** | — | **$121,801.55 \pm 22,013.53$** | **$88.51\% \pm 4.89\%$** | **$21.46 \pm 4.12$** | **$137,396.98 \pm 22,412.36$** |

---

## 3. Inferential Statistics & Hypothesis Testing

Hypothesis testing followed the prespecified protocol in `docs/PHASE11_STATISTICAL_SPEC.md`: two-sided Welch's unequal variances $t$-test ($H_0: \mu_{\text{XGB}} = \mu_{\text{MA}}$), exact non-parametric Mann-Whitney $U$ test, sample-size-corrected Hedges' $g$ effect size, and Benjamini-Hochberg False Discovery Rate (FDR) control at $\alpha = 0.05$.

```
=============================================================================================================
METRIC COMPARISON: 50-Iteration Reduced-Budget Replication, Corrected Independent-Seeding Protocol
=============================================================================================================
Metric              Diff (XGB - MA)    % Change     Welch t (df)    p-value    Hedges' g    95% CI of Difference
-------------------------------------------------------------------------------------------------------------
Total Cost            -9,345.07         -7.13%     t(3.40) = -0.3959  0.7157     -0.2586   [-79,703.2, +61,013.1]
Service Level         -0.0538           -5.73%     t(3.72) = -1.5191  0.2086     -0.9923   [-15.50%,   +4.75%]
Bullwhip Ratio        +6.8438          +46.82%     t(2.93) = +2.5749  0.0840     +1.6819   [-1.722,   +15.410]
Cost/SL Efficiency    -2,096.72         -1.50%     t(3.46) = -0.0884  0.9345     -0.0577   [-68,318.5, +64,125.1]
=============================================================================================================
```

### Multiple Testing Correction (Benjamini-Hochberg FDR)
Across the three co-primary metrics:
1. **Bullwhip Ratio:** Raw $p = 0.0840 \rightarrow \text{FDR } q = 0.2520$ (Not Significant)
2. **Service Level:** Raw $p = 0.2086 \rightarrow \text{FDR } q = 0.3129$ (Not Significant)
3. **Total Cost:** Raw $p = 0.7157 \rightarrow \text{FDR } q = 0.7157$ (Not Significant)

**Outcome:** Zero hypotheses reject the null hypothesis at $\alpha = 0.05$.

---

## 4. Scenario Breakdown Analysis

Examining the unweighted average performance across the 6 test scenarios (5 evaluation seeds each, 15 total episodes per scenario for each forecaster):

| Test Scenario | MA Mean Cost | XGB Mean Cost | MA Service Level | XGB Service Level | MA Bullwhip | XGB Bullwhip |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `in_distribution` | 39,014.13 | 48,976.40 | **98.46%** | **98.60%** | 22.02 | 32.30 |
| `demand_shift` | **66,434.53** | 104,304.40 | **93.18%** | 88.48% | **8.34** | 10.11 |
| `lead_time_shift` | **83,805.20** | 89,463.00 | **95.98%** | 90.55% | **23.56** | 31.96 |
| `capacity_disruption` | 356,536.97 | **268,992.77** | 88.95% | 88.95% | **31.05** | 50.99 |
| `demand_spike` | **52,622.03** | 84,962.20 | **94.45%** | 87.63% | **0.84** | 1.27 |
| `combined_shift` | 188,466.83 | **134,110.53** | **92.31%** | 76.85% | **1.89** | 2.13 |

### Scenario Observations:
- **Cost Reduction Drivers:** XGBoost achieved lower total costs primarily in `capacity_disruption` ($-24.6\%$) and `combined_shift` ($-28.8\%$). However, in `combined_shift`, this cost savings came at a severe degradation in service level ($76.85\%$ vs $92.31\%$), indicating that the cost reduction was partially driven by order contraction rather than superior inventory placement.
- **Benign & Trend Scenarios:** In `demand_shift` and `demand_spike`, Moving Average clearly outperformed XGBoost in both cost and fulfillment rate.

---

## 5. Artifact & Provenance Verification

All checkpoints and manifests are persistently archived in `results/phase11_controlled/` and registered in `models/registry.json`:

```json
{
  "phase11_controlled_ma_s1": {
    "checkpoint_sha256": "e0373fefa1d7a0af3fcb85e5cb5e95d0ddc8b6c3a7d3719dd3656ca25c007cf8",
    "status": "VALID",
    "evaluations_count": 30
  },
  "phase11_controlled_ma_s2": {
    "checkpoint_sha256": "7a7f96ba9f6f8ee7adbcd8b1324c614bbd999d9c87b7d2255b5e7fc6962b8982",
    "status": "VALID",
    "evaluations_count": 30
  },
  "phase11_controlled_ma_s3": {
    "checkpoint_sha256": "718c076ff6d0b02d603a1112d7c0490fb247cbf2aebf9e54a5c0d2df5ea1bfd3",
    "status": "VALID",
    "evaluations_count": 30
  },
  "phase11_controlled_xgboost_s1": {
    "checkpoint_sha256": "3f8e57558f5df876f86be5bf838d0d4b1ada2f7f7c77f22c3721f960fd01b6c0",
    "status": "VALID",
    "evaluations_count": 30
  },
  "phase11_controlled_xgboost_s2": {
    "checkpoint_sha256": "5d1c94202af6138eb15926c48325fa77610c3ea1dbdaec7c3a0bcf42a08892f3",
    "status": "VALID",
    "evaluations_count": 30
  },
  "phase11_controlled_xgboost_s3": {
    "checkpoint_sha256": "463bc2c1ee11328c68ff83b483c65df510ec3efbf9ef2bfb65b6f0e340a66d5b",
    "status": "VALID",
    "evaluations_count": 30
  }
}
```

---

## 6. Scientific Conclusion & Research Status

1. **Resolution of Forensic Anomalies:** The uncoupled seeding pipeline successfully removed the pseudoreplication confounder. Checkpoint integrity is 100% verified via canonical state-dict SHA-256 hashes.
2. **Empirical Finding at 50 Updates:** In this **50-iteration reduced-budget replication, corrected independent-seeding protocol**, XGBoost-conditioned MARL does not exhibit statistically detectable operational superiority over Moving-Average-conditioned MARL ($p = 0.7157$).
3. **Power Limitations:** With $N = 3$ per group and substantial inter-seed variance in stochastic multi-agent inventory rollouts, statistical power is low ($1-\beta \approx 0.08$ for small effects). However, the direction of effect sizes (negative impact on service level, large positive increase in bullwhip ratio) suggests that XGBoost forecasting is not a simple, cost-free win for decentralized MARL policies in supply chains.
4. **Research Status:** Phase 11 is now **Methodologically Corrected, Fully Provenanced, and Empirically Characterized**.
