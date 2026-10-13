import json
import glob
import numpy as np
import pandas as pd
from scipy import stats

def get_seed_metrics(pattern):
    files = glob.glob(pattern)
    assert len(files) == 30, f"Expected 30 files for {pattern}, got {len(files)}"
    costs = []
    service_levels = []
    bullwhip_ratios = []
    for f in sorted(files):
        with open(f, 'r') as fh:
            data = json.load(fh)
            costs.append(data['total_cost'])
            service_levels.append(data['service_level'])
            bullwhip_ratios.append(data['bullwhip_ratio'])
    return {
        'mean_cost': float(np.mean(costs)),
        'mean_service_level': float(np.mean(service_levels)),
        'mean_bullwhip': float(np.mean(bullwhip_ratios))
    }

print("=" * 60)
print("PHASE 11 MULTI-SEED FINAL EVALUATION (PART C)")
print("=" * 60)

# Evaluate XGBoost seeds
xgb_seeds = ['s999', 's100', 's101']
xgb_results = {}
for s in xgb_seeds:
    pat = f"results/phase11/phase11_gnn_mappo_forecast_xgboost_{s}_*eval*.json"
    xgb_results[s] = get_seed_metrics(pat)

# Evaluate MA seeds
ma_seeds = ['s0', 's1', 's2']
ma_results = {}
for s in ma_seeds:
    pat = f"results/phase11/phase11_gnn_mappo_forecast_ma_{s}_*eval*.json"
    ma_results[s] = get_seed_metrics(pat)

print("\n--- PER-SEED PERFORMANCE (Mean over 30 OOD Test Episodes) ---")
print("XGBoost:")
for s, res in xgb_results.items():
    print(f"  Seed {s:5s} -> Cost: {res['mean_cost']:10.2f} | Service Level: {res['mean_service_level']*100:6.2f}% | Bullwhip: {res['mean_bullwhip']:6.2f}")

print("\nMoving Average:")
for s, res in ma_results.items():
    print(f"  Seed {s:5s} -> Cost: {res['mean_cost']:10.2f} | Service Level: {res['mean_service_level']*100:6.2f}% | Bullwhip: {res['mean_bullwhip']:6.2f}")

xgb_costs = [res['mean_cost'] for res in xgb_results.values()]
xgb_sls = [res['mean_service_level'] for res in xgb_results.values()]

ma_costs = [res['mean_cost'] for res in ma_results.values()]
ma_sls = [res['mean_service_level'] for res in ma_results.values()]

print("\n--- AGGREGATED METRICS (Across N=3 Seeds) ---")
print(f"XGBoost Mean Cost:          {np.mean(xgb_costs):10.2f} +/- {np.std(xgb_costs, ddof=1):.2f}")
print(f"XGBoost Mean Service Level: {np.mean(xgb_sls)*100:9.2f}% +/- {np.std(xgb_sls, ddof=1)*100:.2f}%")
print()
print(f"MA Mean Cost:               {np.mean(ma_costs):10.2f} +/- {np.std(ma_costs, ddof=1):.2f}")
print(f"MA Mean Service Level:      {np.mean(ma_sls)*100:9.2f}% +/- {np.std(ma_sls, ddof=1)*100:.2f}%")

# Statistical significance tests
t_cost, p_cost = stats.ttest_ind(xgb_costs, ma_costs, equal_var=False)
t_sl, p_sl = stats.ttest_ind(xgb_sls, ma_sls, equal_var=False)

print("\n--- STATISTICAL SIGNIFICANCE TESTS (Welch's Two-Sample t-test, N=3 vs N=3) ---")
print(f"Cost Difference:          t = {t_cost:8.4f}, p-value = {p_cost:8.4f}")
print(f"Service Level Difference: t = {t_sl:8.4f}, p-value = {p_sl:8.4f}")

# Training Curve Variance across 200 iterations
print("\n--- TRAINING CURVE TELEMETRY & VARIANCE (200 Iterations) ---")
try:
    df_xgb = pd.read_csv("results/phase11/training_metrics_phase11_gnn_mappo_forecast_xgboost_s100.csv")
    df_ma = pd.read_csv("results/phase11/training_metrics_phase11_gnn_mappo_forecast_ma_s1.csv")
    print(f"XGBoost (s100, {len(df_xgb)} iters) - Reward Std: {df_xgb['retailer_reward'].std():.2f}, Value Loss Std: {df_xgb['retailer_value_loss'].std():.4f}")
    print(f"MA      (s1,   {len(df_ma)} iters) - Reward Std: {df_ma['retailer_reward'].std():.2f}, Value Loss Std: {df_ma['retailer_value_loss'].std():.4f}")
except Exception as e:
    print(f"Telemetry comparison error: {e}")

print("=" * 60)
