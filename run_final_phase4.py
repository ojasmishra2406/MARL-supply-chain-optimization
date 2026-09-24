import os
import json
import torch
import numpy as np
import time
import subprocess
import hashlib

from rl.pilot import PilotTrainer
from evaluator import evaluate_policy

def get_git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"

def hash_tensor_dict(state_dict):
    m = hashlib.sha256()
    for k, v in sorted(state_dict.items()):
        m.update(k.encode())
        m.update(v.cpu().numpy().tobytes())
    return m.hexdigest()

def run_training():
    config_path = "configs/phase4_ippo.yaml"
    seeds = [0, 1, 2]
    num_updates = 50
    eval_updates = [0, 1, 5, 10, 20, 50]
    
    final_results = {}
    learning_curves = {}
    
    os.makedirs("results", exist_ok=True)
    
    for seed in seeds:
        print(f"\n{'='*40}")
        print(f"Starting Training for Seed {seed}")
        print(f"{'='*40}")
        
        trainer = PilotTrainer(config_path)
        trainer.seed = seed
        
        # Ensure random initialization
        seed_curve = {}
        for update in range(num_updates + 1):
            if update in eval_updates:
                res = evaluate_policy(trainer.agents["retailer"].actor, seeds=[0,1,2], num_episodes_per_seed=10)
                mean_cost = res["mean_cost"]
                seed_curve[update] = mean_cost
                print(f"[Update {update:3d}] Mean Cost: {mean_cost:8.2f} | Fill Rate: {res['mean_fill_rate']:.4f} | Bullwhip: {res['mean_bullwhip']:.4f}")
            
            if update < num_updates:
                metrics = trainer.train(1, use_wandb=False)
                if update % 5 == 0:
                    print(f"  > [Train {update:3d}] P-Loss: {metrics.get('retailer_policy_loss',0):.4f} | V-Loss: {metrics.get('retailer_value_loss',0):.4f} | Ent: {metrics.get('retailer_entropy',0):.4f}")
                
        # Final detailed eval
        final_res = evaluate_policy(trainer.agents["retailer"].actor, seeds=[0,1,2], num_episodes_per_seed=30)
        final_results[seed] = final_res
        learning_curves[seed] = seed_curve
        
        # Save checkpoint
        ckpt_path = f"results/phase4_final_seed{seed}.pt"
        torch.save(trainer.agents["retailer"].actor.state_dict(), ckpt_path)
        
    # --- Generate Manifest & Artifacts ---
    commit = get_git_commit()
    
    manifest = {
        "experiment_id": "phase4_final_random_init",
        "git_commit": commit,
        "seeds": seeds,
        "configuration_path": config_path,
        "architecture": "ActorNetwork(obs_dim=5, action_dim=101, hidden_size=128, ordinal_gaussian_surrogate)",
        "observation_mode": "aggregate",
        "reward_scaling": "100.0",
        "initialization": "random",
        "number_of_updates": num_updates,
        "evaluation_protocol": "evaluator.py (deterministic=True, seeds=[0,1,2], 30 eps/seed for final)",
        "checkpoints": {seed: f"results/phase4_final_seed{seed}.pt" for seed in seeds},
        "checkpoint_hashes": {seed: hash_tensor_dict(torch.load(f"results/phase4_final_seed{seed}.pt")) for seed in seeds}
    }
    
    with open("results/phase4_final_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    with open("results/phase4_final_learning_curves.json", "w") as f:
        json.dump(learning_curves, f, indent=2)
        
    with open("results/phase4_final_results.json", "w") as f:
        # Numpy types are not json serializable
        clean_res = {}
        for s, res in final_results.items():
            clean_res[s] = {
                "mean_cost": float(res["mean_cost"]),
                "std_cost": float(res["std_cost"]),
                "mean_fill_rate": float(res["mean_fill_rate"]),
                "mean_bullwhip": float(res["mean_bullwhip"]),
                "all_costs": [float(c) for c in res["all_costs"]]
            }
        json.dump(clean_res, f, indent=2)

if __name__ == "__main__":
    run_training()
