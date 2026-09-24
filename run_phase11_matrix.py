import argparse
import os
import yaml
from rl.phase11_trainer import Phase11Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/phase4_ippo.yaml")
    parser.add_argument("--forecast_horizon", type=int, default=2)
    parser.add_argument("--forecaster_type", type=str, default="ma", choices=["naive", "ma", "xgboost", "lstm"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    
    print("=" * 50)
    print("PHASE 11: DEMAND FORECASTING + GNN-MAPPO EXPERIMENT")
    print(f"Seed: {args.seed} | Horizon: {args.forecast_horizon} | Forecaster: {args.forecaster_type}")
    print("=" * 50)
    
    # 1. Initialize
    trainer = Phase11Trainer(
        config_path=args.config, 
        forecast_horizon=args.forecast_horizon, 
        forecaster_type=args.forecaster_type
    )
    
    # 2. Train
    print("Beginning matrix training loop...")
    total_updates = 200  # Matching Phase 6 baseline
    for i in range(total_updates):
        metrics = trainer.train(total_updates=1)
        if (i + 1) % 10 == 0:
            print(f"Update {i+1}/{total_updates} | Reward: {metrics.get('retailer_reward', 0):.4f}")
            
    # 3. Save
    os.makedirs("results/phase11", exist_ok=True)
    save_path = f"results/phase11/phase11_gnn_mappo_forecast_{args.forecaster_type}_s{args.seed}.pt"
    trainer.save(save_path)
    print(f"Training complete. Checkpoint saved to {save_path}")

if __name__ == "__main__":
    main()
