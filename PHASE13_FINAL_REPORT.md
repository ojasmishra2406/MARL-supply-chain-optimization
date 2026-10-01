# 📈 PHASE 13 FINAL REPORT: ABLATION & STATISTICAL FINDINGS
**Date:** September 27, 2026
**Topic:** Baseline GNN-MAPPO vs. Forecasting-Augmented GNN-MAPPO

## 1. Executive Summary
We successfully completed the statistical analysis (Welch's t-test with Benjamini-Hochberg FDR correction) comparing our **Phase 10 Baseline GNN-MAPPO** agents against our **Phase 11 Forecasting-Augmented GNN-MAPPO** agents across 300 highly stochastic test environments.

**The core finding is a classic supply chain trade-off:** Integrating demand forecasting directly into the MARL observation space resulted in significantly higher Service Levels and lower operating costs, but at the direct expense of massively amplifying the Bullwhip Effect.

## 2. Global Metric Shifts
Aggregated across all 300 evaluations (In-Distribution, Demand Shifts, Spikes, Disruptions):

| Metric | Phase 10 (Baseline) | Phase 11 (Forecasting) | Improvement |
| :--- | :--- | :--- | :--- |
| **Total Cost** | 105,897 | 104,498 | **+ 1.32%** |
| **Service Level** | 85.61% | 93.75% | **+ 9.50%** |
| **Bullwhip Ratio** | 7.94 | 19.24 | **- 142.1%** |

## 3. Scenario-Specific Statistical Significance (α = 0.05)
The Forecasting architecture dramatically outperformed the baseline in several specific regimes, proving that the agents successfully learned to condition their policies on the exogenous predictions.

### 🌟 Successes (Where Forecasting Won)
* **In-Distribution:** Forecasting reduced total costs by an incredible **29.3%** ($46k -> $33k), achieving extreme efficiency when the environment matched training data (adj. p-value = 1.86e-05).
* **Demand Spikes:** When sudden surges occurred, Forecasting agents reduced costs by **14.1%** and maintained a near-perfect **97.7% Service Level** (compared to the baseline's 87.6%). The agents effectively used the forecast to "pre-position" inventory.
* **Lead Time Shifts:** Forecasting reduced costs by **11.6%** during logistical delays (adj. p-value = 0.0028).

### ⚠️ Trade-offs (The Bullwhip Problem)
In every single scenario, the **Bullwhip Ratio was statistically significantly worse** for the Forecasting agents (p-values < 1e-12). 
* **Why?** By feeding the agents an exogenous Moving Average forecast, they became hyper-responsive to recent trends. If demand slightly trended up, the retailer ordered heavily to protect its service level, the wholesaler multiplied that order, and the factory experienced massive demand volatility.
* **Conclusion:** The AI traded upstream stability for downstream customer satisfaction.

## 4. Engineering Bug Note
During Phase 11 execution, a configuration fallback in `Phase11Trainer` caused the `xgboost` parameter to default to the `MovingAverageForecaster`. Therefore, all Phase 11 results explicitly represent the performance of GNN-MAPPO augmented with an endogenous Moving Average.

## 5. Next Steps (Phase 14)
The data is finalized, and the statistical tests are complete. The final step of the entire project is to integrate these findings directly into the Streamlit Dashboard (`dashboard.py`).
