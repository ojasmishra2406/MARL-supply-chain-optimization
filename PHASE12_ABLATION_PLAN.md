# PHASE 12: COMPREHENSIVE ABLATION AND ROBUSTNESS STUDY

## 1. Research Questions
This ablation study systematically isolates the independent and interaction effects of the key supply-chain MARL mechanisms designed in Phases 6-11:
- **RQ1 (Graph Integration):** Does explicitly modeling supply chain topology via a GNN Critic (CTDE) improve global cost optimization compared to a standard MLP centralized critic?
- **RQ2 (Demand Forecasting):** Does augmenting local observations with statistical or ML-based demand forecasts improve proactive inventory placement?
- **RQ3 (Synergy):** Does the combination of Forecasting + GNN-MAPPO (Phase 11) yield synergistic performance greater than the sum of their individual contributions?
- **RQ4 (Communication):** Does explicit neighbor-to-neighbor communication (`mappo_comm`) significantly reduce the Bullwhip Effect compared to independent observations?
- **RQ5 (Centralization):** Does centralized training (MAPPO) outperform decentralized training (IPPO) under localized supply chain constraints?

## 2. Primary Comparisons
Primary experiments directly answer the core research questions.
- **Family A (Architecture):** MAPPO vs GNN-MAPPO
- **Family B (Forecasting):** MAPPO vs Forecasting+MAPPO, GNN-MAPPO vs Forecasting+GNN-MAPPO (Forecast type: Moving Average, Horizon: 2)
- **Family C (Communication):** IPPO vs IPPO_COMM, MAPPO vs MAPPO_COMM
- **Family D (Critic):** IPPO vs MAPPO

## 3. Secondary Comparisons
Secondary experiments optimize internal components and hyperparameter sensitivity.
- **Family E (Forecast Model Sensitivity):** Forecasting+MAPPO using (Naive vs. Moving Average vs. XGBoost)

## 4. Experimental Matrix
Each configuration runs under:
- **Regimes:** 2 (`baseline`, `high_variance`)
- **Training Seeds:** 5 (0, 1, 2, 3, 4)
- **Evaluation Scenarios:** 6 (`in_distribution`, `demand_shift`, `demand_spike`, `lead_time_shift`, `capacity_disruption`, `combined_shift`)
- **Evaluation Seeds:** 5

*Note: Phase 6 has already generated 35 of the 40 required models for Families C and D (IPPO, MAPPO, IPPO_COMM, MAPPO_COMM).*

## 5. Number of Runs
**Required Training Runs:**
- MAPPO/IPPO variants: 40 runs (35 already completed)
- GNN-MAPPO variants: 10 runs (2 regimes x 5 seeds)
- Forecasting+MAPPO: 10 runs
- Forecasting+GNN-MAPPO: 10 runs
- Forecast Model Sensitivity: 20 runs (Naive, XGBoost)
**Total New Training Runs:** 55 runs

## 6. Number of Evaluations
55 new models * 6 scenarios * 5 evaluation seeds = **1,650 new evaluations**

## 7. Metrics
Identical to established metrics from previous phases:
1. **Global Supply Chain Cost:** Mean accumulated cost per episode.
2. **Fill Rate:** Percentage of demand met immediately from stock.
3. **Bullwhip Effect:** Variance of orders / Variance of demand (across nodes).

## 8. Statistical Methodology
- **Experimental Unit:** The *Training Seed* ($N=5$). Multiple evaluation seeds of the same trained model are aggregated via `mean()` prior to significance testing to strictly prevent pseudoreplication.
- **Test:** Welch's t-test (two-sided, unequal variance) comparing metric distributions across $N=5$ matched training seeds.
- **Effect Size:** Cohen's $d$.
- **Significance Level:** $\alpha = 0.05$ (post-correction).

## 9. BH-FDR Families
To control the False Discovery Rate without over-penalizing, Benjamini-Hochberg FDR is applied *within* predefined comparison families (A through E). A global FDR correction is not used, as the research questions test independent orthogonal hypotheses.

## 10. Resource Requirements
- **Compute:** At ~4.5 hours per GPU training run (RTX 3050 4GB), 55 runs will require ~250 hours of raw compute time.
- **VRAM limit:** 3 concurrent jobs maximum.
- **Storage:** ~55 checkpoints * 700 KB = ~40 MB. 1,650 eval JSONs = ~10 MB.

## 11. Tests
The ablation framework includes CPU-based tests in `tests/test_phase12_framework.py`:
- Config and ID generation determinism
- Duplicate detection mechanism
- Statistical aggregation correctness and BH-FDR group parsing

## 12. Execution Status
- **IMPLEMENTED:** Ablation plan, configurations, statistical groupings.
- **TESTED:** CPU framework unit tests.
- **EXECUTED:** Pending (Framework only, training halted).
- **EXPERIMENTALLY VALIDATED:** Pending.
