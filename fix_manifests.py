import os
import json
import torch
import hashlib
import datetime
from rl.phase6_trainer import Phase6Trainer
from rl.phase5_evaluator import evaluate_phase5_policy
from run_phase6_matrix import generate_manifest

def fix_manifests():
    runs = [
        ("phase6_ippo_comm_baseline_s0", "ippo_comm", "baseline", 0),
        ("phase6_ippo_comm_baseline_s1", "ippo_comm", "baseline", 1),
        ("phase6_ippo_comm_baseline_s2", "ippo_comm", "baseline", 2),
        ("phase6_ippo_comm_baseline_s3", "ippo_comm", "baseline", 3),
        ("phase6_ippo_comm_baseline_s4", "ippo_comm", "baseline", 4),
        ("phase6_ippo_comm_high_variance_s0", "ippo_comm", "high_variance", 0),
        ("phase6_ippo_comm_high_variance_s1", "ippo_comm", "high_variance", 1),
        ("phase6_ippo_comm_high_variance_s2", "ippo_comm", "high_variance", 2)
    ]
    
    for run_id, algo, condition, seed in runs:
        manifest_path = f"results/phase6/{run_id}_manifest.json"
        if os.path.exists(manifest_path):
            continue
            
        print(f"Fixing {run_id}...")
        ckpt_path = f"results/phase6/{run_id}.pt"
        
        ablation = {}
        ablation["centralized_critic"] = False
        ablation["communication"] = True
            
        trainer = Phase6Trainer("configs/phase4_ippo.yaml", ablation)
        
        # Load weights
        state_dict = torch.load(ckpt_path)
        if "retailer" not in state_dict:
            for name in trainer.agents_names:
                trainer.agents[name].load_state_dict(state_dict)
        else:
            for name in trainer.agents_names:
                trainer.agents[name].load_state_dict(state_dict[name])
                
        # Hash
        from rl.checkpoint import _hash_state_dict
        hash_val = _hash_state_dict(state_dict)
            
        eval_res = evaluate_phase5_policy(trainer.agents, "configs/phase4_ippo.yaml", ablation, seeds=[0], num_episodes_per_seed=1, deterministic=True)
        
        generate_manifest(run_id, algo, condition, seed, "configs/phase4_ippo.yaml", ckpt_path, hash_val, eval_res)
        print(f"Fixed {run_id}")

if __name__ == "__main__":
    fix_manifests()
