# Phase 11 — Prespecified Statistical and Evaluation Protocol

## 1. Objective and Problem Formulation
In supply chain management, multi-agent reinforcement learning (MARL) policies balance conflicting objectives:
1. Minimizing inventory holding, pipeline, and shortage backlog costs.
2. Maintaining high customer demand fulfillment (Service Level / Fill Rate).

Naive cost minimization can exploit order suppression to lower holding costs while inducing catastrophic stockouts. Therefore, the primary objective must jointly account for financial cost and delivery volume.

## 2. Prespecified Metrics

### Primary Metric: Cost-per-Fulfilled-Unit (CPFU)
$$\text{CPFU} = \frac{\text{Total Supply Chain Cost}}{\text{Total Fulfilled Demand}}$$
CPFU represents the operational unit cost required to successfully deliver goods to the consumer. A lower CPFU indicates superior operational efficiency without rewarding stockouts.

### Co-Primary Metrics:
- **Total Cost:** Cumulative supply chain cost across all four echelons over 52 weeks (holding + backlog + ordering).
- **Service Level (Fill Rate):** $\frac{\sum \text{Fulfilled Demand}}{\sum \text{Customer Demand}} \times 100\%$.
- **Bullwhip Ratio:** $\frac{\operatorname{Var}(\text{Retailer Orders})}{\operatorname{Var}(\text{Customer Demand})}$.

## 3. Experimental Design and Units
- **Independent Experimental Unit:** An independently initialized and trained MARL policy (training seed $s \in \mathcal{S}$).
- **Evaluation Protocol:** Each trained policy is evaluated on a fixed benchmark matrix of 30 deterministic test episodes:
  - 6 Scenarios: `in_distribution`, `demand_shift`, `lead_time_shift`, `capacity_disruption`, `demand_spike`, `combined_shift`.
  - 5 Evaluation Seeds per scenario: $\{0, 1, 2, 3, 4\}$.
  - 52 discrete simulation steps (weeks) per episode.
- **Aggregation:** The run metric is the unweighted arithmetic mean across all 30 episodes.

## 4. Hypothesis Testing and Statistical Rules
1. **Hypothesis:** Two-sided test comparing XGBoost-conditioned MARL against Moving-Average-conditioned MARL.
2. **Test Statistic:** Welch's two-sample $t$-test (unequal variances, Satterthwaite degrees of freedom).
3. **Sign Convention:** Difference defined as $\text{XGBoost} - \text{MovingAverage}$.
4. **Multiple Testing Correction:** Benjamini–Hochberg False Discovery Rate (FDR) control at $\alpha = 0.05$ across evaluated metrics.
5. **Effect Size:** Sample Cohen's $d$ and sample-size-corrected Hedges' $g$ with 95% confidence intervals.
6. **Interpretation Rule:** A non-significant $p$-value ($p \ge 0.05$) indicates insufficient evidence to reject the null hypothesis; it is NEVER interpreted as proof of equivalence.
