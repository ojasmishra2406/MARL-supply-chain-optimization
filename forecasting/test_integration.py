import os
import sys
import numpy as np
import pandas as pd
import supersuit as ss
import torch

from envs.supply_chain_env import SupplyChainParallelEnv
from forecasting.models import XGBoostForecaster
from forecasting.wrapper import DemandForecastWrapper

def test_integration():
    print("Testing Forecasting -> MARL Integration...")
    
    # 1. Train Forecaster on historical data
    train_df = pd.read_csv("data/forecasting/train_demand.csv")
    forecaster = XGBoostForecaster(window=5)
    forecaster.fit(train_df["demand"].values)
    print("Forecaster trained.")
    
    # 2. Setup Environment with Wrapper
    base_env = SupplyChainParallelEnv("configs/phase1_simulator.yaml", comm_enabled=False)
    env = DemandForecastWrapper(base_env, forecaster, forecast_horizon=2)
    
    # Apply supersuit wrapper to make it vec_env compatible just to test API
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 2) # num_envs = 2
    
    obs, info = env.reset(seed=42)
    print(f"Observation shape after reset: {obs.shape}")
    # Expected: (2 * 4, 5 + 2) = (8, 7)
    
    assert obs.shape == (8, 7), f"Expected shape (8, 7), got {obs.shape}"
    
    # Run a few steps
    for _ in range(5):
        actions = np.zeros(8, dtype=np.int64)
        obs, rews, terms, truncs, infos = env.step(actions)
        
    print(f"Observation shape after steps: {obs.shape}")
    print("Integration test passed!")

if __name__ == "__main__":
    test_integration()
