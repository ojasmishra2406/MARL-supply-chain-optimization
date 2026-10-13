import argparse
import os
import sys
import json
import torch
import datetime
import subprocess
from rl.phase11_trainer import Phase11Trainer

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "unknown"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecaster_type", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--total_updates", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="results/phase11_controlled")
    parser.add_argument("--run_id", type=str, default=None)
    args = parser.parse_args()

    # Explicitly seed all RNG sources before creating any models or envs
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run_id = args.run_id or f"phase11_controlled_gnn_mappo_forecast_{args.forecaster_type}_s{args.seed}"
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    
    temp_path = f"{out_dir}/{run_id}_temp.pt"
    final_path = f"{out_dir}/{run_id}.pt"
    manifest_path = f"{out_dir}/{run_id}_manifest.json"

    print(f"[{run_id}] Starting at {datetime.datetime.now().isoformat()} with seed {args.seed}...")

    trainer = Phase11Trainer(
        config_path="configs/phase4_ippo.yaml",
        forecast_horizon=2,
        forecaster_type=args.forecaster_type,
        seed=args.seed
    )

    # Load from intermediate checkpoint if it exists
    start_iteration = 0
    if os.path.exists(temp_path):
        print(f"[{run_id}] Found intermediate checkpoint! Resuming...")
        try:
            state = torch.load(temp_path, map_location="cpu")
            for name, agent in trainer.agents.items():
                agent.load_state_dict(state["agents"][name])
            start_iteration = state.get("iteration", 0)
            print(f"[{run_id}] Resumed from iteration {start_iteration}.")
        except Exception as e:
            print(f"[{run_id}] Failed to load temp checkpoint: {e}. Starting fresh.")

    total_updates = args.total_updates
    import pandas as pd
    training_metrics = []
    
    for i in range(start_iteration, total_updates):
        metrics = trainer.train(total_updates=1)
        
        # Capture metrics
        metrics["iteration"] = i + 1
        training_metrics.append(metrics)
        
        if (i + 1) % 10 == 0 or (i + 1) == total_updates:
            print(f"[{run_id}] Iteration {i+1}/{total_updates}")
            
            # Save intermediate checkpoint
            state = {
                "iteration": i + 1,
                "agents": {name: agent.state_dict() for name, agent in trainer.agents.items()}
            }
            torch.save(state, temp_path)
            
            # Save intermediate metrics
            pd.DataFrame(training_metrics).to_csv(f"{out_dir}/training_metrics_{run_id}.csv", index=False)

    # Save final model via canonical save_checkpoint
    print(f"[{run_id}] Training complete! Saving canonical final model...")
    state = {name: agent.state_dict() for name, agent in trainer.agents.items()}
    
    from rl.checkpoint import save_checkpoint
    import hashlib
    
    extra_metadata = {
        "experiment_id": run_id,
        "algorithm": "gnn_mappo",
        "forecaster": args.forecaster_type,
        "seed": args.seed,
        "config_path": "configs/phase4_ippo.yaml",
        "total_updates": total_updates,
        "software_version": "2.0_controlled",
        "git_commit": get_git_commit(),
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    save_checkpoint(state, final_path, manifest_path, extra_metadata=extra_metadata)
    
    # Save training metrics to CSV
    metrics_df = pd.DataFrame(training_metrics)
    csv_path = f"{out_dir}/training_metrics_{run_id}.csv"
    metrics_df.to_csv(csv_path, index=False)

    # Clean up temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)

    print(f"[{run_id}] Worker finished successfully.")

if __name__ == "__main__":
    main()
