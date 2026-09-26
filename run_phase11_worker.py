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
    args = parser.parse_args()

    run_id = f"phase11_gnn_mappo_forecast_{args.forecaster_type}_s{args.seed}"
    temp_path = f"results/phase11/{run_id}_temp.pt"
    final_path = f"results/phase11/{run_id}.pt"
    manifest_path = f"results/phase11/{run_id}_manifest.json"

    print(f"[{run_id}] Starting at {datetime.datetime.now().isoformat()}...")
    os.makedirs("results/phase11", exist_ok=True)

    trainer = Phase11Trainer(
        config_path="configs/phase4_ippo.yaml",
        forecast_horizon=2,
        forecaster_type=args.forecaster_type
    )

    # Load from intermediate checkpoint if it exists
    start_iteration = 0
    if os.path.exists(temp_path):
        print(f"[{run_id}] Found intermediate checkpoint! Resuming...")
        try:
            state = torch.load(temp_path)
            for name, agent in trainer.agents.items():
                agent.load_state_dict(state["agents"][name])
            start_iteration = state.get("iteration", 0)
            print(f"[{run_id}] Resumed from iteration {start_iteration}.")
        except Exception as e:
            print(f"[{run_id}] Failed to load temp checkpoint: {e}. Starting fresh.")

    total_updates = 200
    for i in range(start_iteration, total_updates):
        metrics = trainer.train(total_updates=1)
        
        if (i + 1) % 10 == 0:
            print(f"[{run_id}] Iteration {i+1}/{total_updates}")
            
            # Save intermediate checkpoint
            state = {
                "iteration": i + 1,
                "agents": {name: agent.state_dict() for name, agent in trainer.agents.items()}
            }
            torch.save(state, temp_path)

    # Save final model
    print(f"[{run_id}] Training complete! Saving final model...")
    state = {name: agent.state_dict() for name, agent in trainer.agents.items()}
    torch.save(state, final_path)

    # Write manifest
    import hashlib
    with open(final_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    manifest = {
        "experiment_id": run_id,
        "algorithm": "gnn_mappo",
        "forecaster": args.forecaster_type,
        "seed": args.seed,
        "config_path": "configs/phase4_ippo.yaml",
        "checkpoint_path": final_path,
        "checkpoint_sha256": file_hash,
        "software_version": "1.0",
        "git_commit": get_git_commit(),
        "timestamp": datetime.datetime.now().isoformat()
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Clean up temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)

    print(f"[{run_id}] Worker finished successfully.")

if __name__ == "__main__":
    main()
