import os
import json
import torch
import time
import numpy as np
from rl.pilot import PilotTrainer
from baselines.out import OUTPolicy
from rl.pilot_env import PilotOneEchelonEnv

def run_phase4d():
    config_path = "configs/phase4_ippo.yaml"
    checkpoint_path = "results/phase4b_step1_imitation_actor_recreated.pt"
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Condition A: Frozen Imitation Control
    # 2. Condition B: PPO Bootstrap
    # We will run both for 50 updates on seeds 0, 1, 2
    
    # Get OUT Baseline costs for reference
    out_costs_per_seed = {}
    env = PilotOneEchelonEnv()
    policy = OUTPolicy(z_scores=[0.5], lead_times=[2], capacity=100)
    for seed in [0, 1, 2]:
        env.reset(seed=seed)
        done = False
        total_cost = 0.0
        while not done:
            fake_state = env.get_fake_simulator_state()
            action = policy.get_actions(fake_state)[0]
            _, rews, terms, truncs, _ = env.step({"retailer": action})
            total_cost += -rews["retailer"]
            done = terms["retailer"] or truncs["retailer"]
        out_costs_per_seed[seed] = total_cost

    conditions = ["frozen", "ppo_bootstrap"]
    
    for condition in conditions:
        print(f"Starting {condition}...")
        results = {"seeds": {}}
        
        for seed in [0, 1, 2]:
            print(f"  Seed {seed}...")
            trainer = PilotTrainer(config_path)
            trainer.seed = seed
            
            # Load imitation weights into the actor!
            state_dict = torch.load(checkpoint_path, map_location=trainer.device)
            trainer.agents["retailer"].actor.load_state_dict(state_dict)
            
            # Record initial cost
            # By running one rollout without updates? No, the first loop in train() gathers rollouts.
            
            best_cost = float("inf")
            worst_cost = float("-inf")
            best_update = 0
            worst_update = 0
            trajectory = []
            
            start_time = time.time()
            for update in range(1, 51):
                # If frozen, we bypass the backward and step
                if condition == "frozen":
                    # Patch the optimizer to do nothing
                    for agent in trainer.agents_names:
                        trainer.optimizers[agent].step = lambda: None
                        trainer.optimizers[agent].zero_grad = lambda: None
                        
                    # Also need to monkeypatch torch.Tensor.backward to avoid graph accumulation?
                    # Easiest way: wrap the compute_gae and update loop in no_grad? 
                    # But the train function has hardcoded loss.backward()
                    # Let's just monkey-patch the backward of loss.
                    original_backward = torch.Tensor.backward
                    torch.Tensor.backward = lambda self, *args, **kwargs: None
                    
                try:
                    metrics = trainer.train(1, use_wandb=False)
                finally:
                    if condition == "frozen":
                        torch.Tensor.backward = original_backward
                
                cost = metrics.get("total_cost", 0) # wait, PilotTrainer doesn't return total_cost if we don't modify it.
                # Actually trainer.train returns metrics, but wait, does it have total_cost?
                # Convert from sum over rollout (2048 steps) to per-episode (52 steps) cost
                episodes_per_rollout = 2048.0 / 52.0
                cost = float(-metrics["retailer_reward"] / episodes_per_rollout)
                
                if cost < best_cost:
                    best_cost = cost
                    best_update = update
                if cost > worst_cost:
                    worst_cost = cost
                    worst_update = update
                    
                trajectory.append({
                    "update": update,
                    "cost": cost,
                    "policy_loss": metrics.get("retailer_policy_loss", 0),
                    "value_loss": metrics.get("retailer_value_loss", 0),
                    "entropy": metrics.get("retailer_entropy", 0),
                    "elapsed_time": time.time() - start_time,
                })
                
                print(f"    Update {update}: Cost={cost:.2f}", flush=True)
                
            results["seeds"][seed] = {
                "out_cost": out_costs_per_seed[seed],
                "initial_cost": trajectory[0]["cost"],
                "best_cost": best_cost,
                "best_update": best_update,
                "final_cost": trajectory[-1]["cost"],
                "worst_cost": worst_cost,
                "worst_update": worst_update,
                "trajectory": trajectory
            }
            
            with open(os.path.join(results_dir, f"phase_4d_{condition}.json"), "w") as f:
                json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_phase4d()
