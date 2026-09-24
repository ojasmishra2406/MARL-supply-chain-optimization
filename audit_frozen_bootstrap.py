import os
import json
import torch
import time
import numpy as np
from rl.pilot import PilotTrainer
from evaluator import evaluate_policy
import hashlib

def hash_tensor_dict(state_dict):
    m = hashlib.sha256()
    for k, v in sorted(state_dict.items()):
        m.update(k.encode())
        m.update(v.cpu().numpy().tobytes())
    return m.hexdigest()

def audit_frozen():
    print("=== Condition A: Frozen Imitation ===")
    config_path = "configs/phase4_ippo.yaml"
    checkpoint_path = "results/phase4b_step1_imitation_actor_recreated.pt"
    
    for seed in [0, 1, 2]:
        trainer = PilotTrainer(config_path)
        trainer.seed = seed
        state_dict = torch.load(checkpoint_path, map_location=trainer.device)
        trainer.agents["retailer"].actor.load_state_dict(state_dict)
        
        hash_start = hash_tensor_dict(trainer.agents["retailer"].actor.state_dict())
        
        # We don't train, we just evaluate it N times to show variance.
        # But wait, the instruction says "run the frozen policy through the live pilot environment."
        # I will evaluate it 5 times and print results.
        evals = []
        for _ in range(5):
            res = evaluate_policy(trainer.agents["retailer"].actor, seeds=[seed], num_episodes_per_seed=10)
            evals.append(res["mean_cost"])
            
        hash_end = hash_tensor_dict(trainer.agents["retailer"].actor.state_dict())
        
        print(f"Seed {seed}:")
        print(f"  Hash Start: {hash_start}")
        print(f"  Hash End:   {hash_end}")
        print(f"  Identical:  {hash_start == hash_end}")
        print(f"  Eval Costs: {['%.2f' % e for e in evals]}")
        print(f"  Mean:       {np.mean(evals):.2f}")

def audit_ppo_bootstrap():
    print("=== Condition B: PPO Bootstrap ===")
    config_path = "configs/phase4_ippo.yaml"
    checkpoint_path = "results/phase4b_step1_imitation_actor_recreated.pt"
    
    results = {seed: {} for seed in [0,1,2]}
    eval_updates = [0, 1, 5, 10, 20, 30, 40, 50]
    
    for seed in [0, 1, 2]:
        print(f"Starting Seed {seed}...")
        trainer = PilotTrainer(config_path)
        trainer.seed = seed
        
        # Initialize Critic naturally. Initialize Actor from Imitation.
        state_dict = torch.load(checkpoint_path, map_location=trainer.device)
        trainer.agents["retailer"].actor.load_state_dict(state_dict)
        
        hash_init = hash_tensor_dict(trainer.agents["retailer"].actor.state_dict())
        
        for update in range(51):
            if update in eval_updates:
                res = evaluate_policy(trainer.agents["retailer"].actor, seeds=[0,1,2], num_episodes_per_seed=10)
                results[seed][update] = res["mean_cost"]
                print(f"  Update {update}: {res['mean_cost']:.2f}")
                
            if update < 50:
                trainer.train(1, use_wandb=False)
                
        hash_final = hash_tensor_dict(trainer.agents["retailer"].actor.state_dict())
        print(f"  Hash Init:  {hash_init}")
        print(f"  Hash Final: {hash_final}")
        
    with open("results/audit_ppo_bootstrap.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    audit_frozen()
    audit_ppo_bootstrap()
