import os
import json
import torch
import time
import numpy as np
from rl.pilot import PilotTrainer

def hash_tensor_dict(state_dict):
    import hashlib
    m = hashlib.sha256()
    for k, v in sorted(state_dict.items()):
        m.update(k.encode())
        m.update(v.cpu().numpy().tobytes())
    return m.hexdigest()

def run_phase4d_micro():
    config_path = "configs/phase4_ippo.yaml"
    checkpoint_path = "results/phase4b_step1_imitation_actor_recreated.pt"
    
    print("Starting ONE-UPDATE MICRO-DIAGNOSTIC...")
    results = {"seeds": {}}
    
    for seed in [0, 1, 2]:
        print(f"  Seed {seed}...")
        trainer = PilotTrainer(config_path)
        trainer.seed = seed
        
        # Load imitation weights
        state_dict = torch.load(checkpoint_path, map_location=trainer.device)
        trainer.agents["retailer"].actor.load_state_dict(state_dict)
        
        # Hash before
        hash_before = hash_tensor_dict(trainer.agents["retailer"].actor.state_dict())
        
        # To evaluate "before update", we can do a frozen step
        original_backward = torch.Tensor.backward
        torch.Tensor.backward = lambda self, *args, **kwargs: None
        
        for agent in trainer.agents_names:
            trainer.optimizers[agent].step = lambda: None
            trainer.optimizers[agent].zero_grad = lambda: None
            
        metrics_before = trainer.train(1, use_wandb=False)
        episodes_per_rollout = 2048.0 / 52.0
        cost_before = -metrics_before["retailer_reward"] / episodes_per_rollout
        
        # Restore normal operation
        torch.Tensor.backward = original_backward
        # Re-initialize trainer to reset optimizers
        trainer = PilotTrainer(config_path)
        trainer.seed = seed
        trainer.agents["retailer"].actor.load_state_dict(state_dict)
        
        # Perform exactly 1 real update
        metrics_after = trainer.train(1, use_wandb=False)
        cost_after = -metrics_after["retailer_reward"] / episodes_per_rollout
        
        # Hash after
        hash_after = hash_tensor_dict(trainer.agents["retailer"].actor.state_dict())
        
        results["seeds"][seed] = {
            "hash_before": hash_before,
            "hash_after": hash_after,
            "cost_before": float(cost_before),
            "cost_after": float(cost_after),
            "policy_loss": float(metrics_after.get("retailer_policy_loss", 0.0)),
            "value_loss": float(metrics_after.get("retailer_value_loss", 0.0)),
            "entropy": float(metrics_after.get("retailer_entropy", 0.0)),
            "explained_var": float(metrics_after.get("retailer_explained_var", 0.0)),
            "adv_var": float(metrics_after.get("retailer_adv_var", 0.0)),
        }
        
        print(f"    Cost Before: {cost_before:.2f} -> Cost After: {cost_after:.2f}")
        print(f"    Hash changed: {hash_before != hash_after}")
        
    with open("results/phase_4d_micro_diagnostic.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_phase4d_micro()
