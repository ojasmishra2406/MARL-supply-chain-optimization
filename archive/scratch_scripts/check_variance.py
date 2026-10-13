import pandas as pd
import os
import sys

xgb_path = "results/phase11/training_metrics_phase11_gnn_mappo_forecast_xgboost_s123.csv"
ma_path = "results/phase11/training_metrics_phase11_gnn_mappo_forecast_ma_s124.csv"

if os.path.exists(xgb_path):
    df_xgb = pd.read_csv(xgb_path)
    print("=== XGBoost (First {} Iters) ===".format(len(df_xgb)))
    print(f"Reward StdDev:     {df_xgb['retailer_reward'].std():.4f}")
    print(f"Value Loss StdDev: {df_xgb['retailer_value_loss'].std():.4f}")
    print(f"Policy Loss StdDev:{df_xgb['retailer_policy_loss'].std():.4f}")
else:
    print("XGBoost CSV not ready.")

print()

if os.path.exists(ma_path):
    df_ma = pd.read_csv(ma_path)
    print("=== Moving Average (First {} Iters) ===".format(len(df_ma)))
    print(f"Reward StdDev:     {df_ma['retailer_reward'].std():.4f}")
    print(f"Value Loss StdDev: {df_ma['retailer_value_loss'].std():.4f}")
    print(f"Policy Loss StdDev:{df_ma['retailer_policy_loss'].std():.4f}")
else:
    print("MA CSV not ready.")
