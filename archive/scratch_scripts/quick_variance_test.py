import os
import torch
import numpy as np
import pandas as pd
from rl.phase11_trainer import Phase11Trainer

def run_test(forecaster, iters=20):
    print(f"--- Starting {forecaster} test for {iters} iterations ---")
    trainer = Phase11Trainer(config_path="configs/phase4_ippo.yaml", forecast_horizon=2, forecaster_type=forecaster)
    
    metrics_list = []
    for i in range(iters):
        m = trainer.train(total_updates=1)
        m["iteration"] = i + 1
        metrics_list.append(m)
        print(f"[{forecaster}] Iteration {i+1}/{iters} | Reward: {m['retailer_reward']:.4f} | V-Loss: {m['retailer_value_loss']:.4f}")
        
    df = pd.DataFrame(metrics_list)
    print(f"\n=== {forecaster.upper()} VARIANCE (First {iters} iters) ===")
    print(f"Reward StdDev:     {df['retailer_reward'].std():.4f}")
    print(f"Value Loss StdDev: {df['retailer_value_loss'].std():.4f}")
    print(f"Policy Loss StdDev:{df['retailer_policy_loss'].std():.4f}\n")

if __name__ == "__main__":
    # Run MA first because it's fast (~5 mins)
    run_test("ma", 20)
    # Run XGBoost second because it's slow (~50 mins)
    run_test("xgboost", 20)
