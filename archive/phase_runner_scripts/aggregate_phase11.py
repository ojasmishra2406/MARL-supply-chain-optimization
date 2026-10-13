import json
import glob
import numpy as np

def agg(pattern):
    files = glob.glob(pattern)
    costs = []
    sls = []
    for f in files:
        with open(f, "r") as fh:
            data = json.load(fh)
            costs.append(data["total_cost"])
            sls.append(data["service_level"])
    return np.mean(costs), np.mean(sls)

xgb_cost, xgb_sl = agg("results/phase11/phase11_gnn_mappo_forecast_xgboost_s999_*_evals*.json")
ma_cost, ma_sl = agg("results/phase11/phase11_gnn_mappo_forecast_ma_s0_*_evals*.json")

print("-" * 50)
print("PHASE 11 FINAL COMPARISON (XGBoost vs Moving Average)")
print("Same evaluation scenarios (30 episodes: 6 shifts x 5 seeds), same eval script, same wrapper")
print("-" * 50)
print(f"XGBoost-Informed Policy (s999) Mean Cost: {xgb_cost:.2f}")
print(f"XGBoost-Informed Policy (s999) Service Level: {xgb_sl*100:.2f}%")
print()
print(f"MA-Informed Policy (s0) Mean Cost: {ma_cost:.2f}")
print(f"MA-Informed Policy (s0) Service Level: {ma_sl*100:.2f}%")
print("-" * 50)
