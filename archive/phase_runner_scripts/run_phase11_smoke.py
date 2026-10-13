import argparse
import torch
import os
from rl.phase11_trainer import Phase11Trainer

def main():
    print("=" * 50)
    print("PHASE 11: DEMAND FORECASTING + GNN-MAPPO SMOKE TEST")
    print("=" * 50)
    
    # 1. Initialize Phase 11 Trainer with Forecast Horizon = 2
    print("[1] Initializing Combined Phase 11 Trainer...")
    trainer = Phase11Trainer(
        config_path="configs/phase4_ippo.yaml", 
        forecast_horizon=2, 
        forecaster_type="ma"
    )
    
    print(f"    - Original Local Obs Dim: 5")
    print(f"    - Forecast Enhanced Obs Dim: {trainer.obs_dim}")
    print(f"    - GNN Global Obs Dim: {trainer.global_obs_dim}")
    print(f"    - Forecaster: MovingAverageForecaster")
    
    # 2. Run small training loop
    print("[2] Running Forward/Backward pass (2 iterations)...")
    metrics = trainer.train(total_updates=2)
    print(f"    - Training successful. Latest metrics: {metrics}")
    
    # 3. Save checkpoint
    os.makedirs("results/phase11", exist_ok=True)
    save_path = "results/phase11/smoke_test_checkpoint.pt"
    print(f"[3] Saving combined model checkpoint to {save_path}...")
    trainer.save(save_path)
    
    # 4. Load checkpoint
    print(f"[4] Reloading checkpoint to verify integrity...")
    trainer2 = Phase11Trainer("configs/phase4_ippo.yaml", forecast_horizon=2, forecaster_type="ma")
    trainer2.load(save_path)
    
    print("=" * 50)
    print("SMOKE TEST PASSED: Integration is successful.")
    print("=" * 50)

if __name__ == "__main__":
    main()
