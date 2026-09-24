import json
import uuid
import yaml
import time
from datetime import datetime, timezone
import subprocess
import os

from rl.pilot import run_pilot

def run_condition(mode):
    run_id = str(uuid.uuid4())
    start_dt = datetime.now(timezone.utc).isoformat()
    start_time = time.time()
    
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except:
        commit_hash = "unknown"
        
    print(f"Running Phase 4C: {mode}. Commit: {commit_hash}")
    
    config_path = f"configs/phase4c_{mode}.yaml"
    with open('configs/phase4_ippo.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    config['observation_mode'] = mode
    config['reward_scale'] = 100.0  # From trusted baseline
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
        
    # We must patch pilot.py to use our config file.
    # pilot.py's run_pilot reads 'configs/phase4_ippo.yaml'.
    # We'll patch run_pilot to accept config_override.
    
    # Actually, we can just call IPPOTrainer directly, but pilot.py's OUT baseline is nice.
    # Instead, we'll temporarily replace phase4_ippo.yaml, run, and restore!
    # No, that's not thread-safe for parallel execution.
    # Let's just write a custom runner here.
    
    from rl.pilot_env import PilotOneEchelonEnv
    from baselines.out import OUTPolicy
    from rl.pilot import PilotTrainer
    import numpy as np
    
    # OUT reference
    env = PilotOneEchelonEnv()
    out_costs_per_seed = {}
    policy = OUTPolicy(z_scores=[0.5], lead_times=[2], capacity=100)
    for seed in [0, 1, 2]:
        env.reset(seed=seed)
        done = False
        total_cost = 0.0
        while not done:
            fake_state = env.get_fake_simulator_state()
            actions_list = policy.get_actions(fake_state)
            action = actions_list[0]
            _, rews, terms, truncs, _ = env.step({"retailer": action})
            total_cost += -rews["retailer"]
            done = terms["retailer"] or truncs["retailer"]
        out_costs_per_seed[seed] = total_cost

    mean_out_cost = np.mean(list(out_costs_per_seed.values()))
    
    results = {"seeds": {}, "out_reference_aggregate": mean_out_cost, "config": config}
    results_dir = "results"
    output_file = f"phase_4c_{mode}.json"
    
    updates = 50
    for seed in [0, 1, 2]:
        trainer = PilotTrainer(config_path)
        trainer.seed = seed
        
        best_cost = float("inf")
        trajectory = []
        
        start_time_seed = time.time()
        for update in range(1, updates + 1):
            metrics = trainer.train(1, use_wandb=False)
            cost = metrics["total_cost"]
            if cost < best_cost:
                best_cost = cost
            
            trajectory.append({
                "update": update,
                "cost": cost,
                "policy_loss": metrics.get("retailer_policy_loss", 0),
                "value_loss": metrics.get("retailer_value_loss", 0),
                "entropy": metrics.get("retailer_entropy", 0),
                "elapsed_time": time.time() - start_time_seed,
            })
            print(f"Seed {seed}, Update {update}: Cost={cost:.2f}, ValueLoss={trajectory[-1]['value_loss']:.4f}, Entropy={trajectory[-1]['entropy']:.4f}", flush=True)
            
            # Write partially so we can monitor
            results["seeds"][seed] = {
                "out_cost": out_costs_per_seed[seed],
                "initial_cost": trajectory[0]["cost"],
                "best_cost": best_cost,
                "final_cost": trajectory[-1]["cost"],
                "trajectory": trajectory,
            }
            with open(os.path.join(results_dir, output_file), "w") as f:
                json.dump(results, f, indent=2)
            
    manifest = {
        "phase": "4c",
        "condition": mode,
        "seed": "[0, 1, 2]",
        "configuration_identity": config_path,
        "reward_scale": 100.0,
        "ppo_hyperparameters": "lr=3e-4, gamma=0.99, gae=0.95, clip=0.2, epochs=10, min_batch=64",
        "observation_specification": "aggregated" if mode == "aggregate" else "itemized",
        "update_count": updates,
        "commit_hash": commit_hash,
        "timestamp": start_dt,
        "change_justification": "Test whether arrival-timed pipeline observations reduce critic state aliasing identified in Phase 4B." if mode == "itemized" else "Control condition."
    }
    
    with open(f"results/manifest_phase_4c_{mode}.json", "w") as f:
        json.dump(manifest, f, indent=2)

if __name__ == "__main__":
    import sys
    run_condition(sys.argv[1])
