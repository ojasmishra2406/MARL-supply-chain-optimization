import os
import torch
import numpy as np
import yaml
from rl.pilot import PilotTrainer
from rl.pilot_env import PilotOneEchelonEnv
from baselines.out import OUTPolicy
import json

def main():
    print("Running Action Distribution Diagnostics (Step 4)...")
    config = yaml.safe_load(open('configs/phase4_ippo.yaml'))
    config['simulator_config'] = 'configs/phase1_simulator.yaml'
    config['seeds'] = [42] # Use a separate seed to diagnose
    
    with open('configs/test_diagnostics.yaml', 'w') as f:
        yaml.dump(config, f)
        
    trainer = PilotTrainer('configs/test_diagnostics.yaml')
    trainer.seed = 42
    
    updates = 25
    stats = []
    
    print("Training and tracking distribution...")
    for update in range(1, updates + 1):
        metrics = trainer.train(1, use_wandb=False)
        
        # We need to extract the actor's current mean and std over a batch of states.
        # Let's just generate some states using OUT to get a representative set of states.
        env = PilotOneEchelonEnv()
        env.reset(seed=42)
        obs_list = []
        for _ in range(100):
            obs = env._get_observations()["retailer"]
            obs_list.append(obs)
            _, _, terms, truncs, _ = env.step({"retailer": 60}) # just sample random actions
            if terms["retailer"] or truncs["retailer"]:
                env.reset()
                
        obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32)
        with torch.no_grad():
            dist = trainer.agents["retailer"].actor(obs_tensor)
            mean_vals = trainer.agents["retailer"].actor.net(obs_tensor / 100.0) * 100.0
            std_val = trainer.agents["retailer"].actor.log_std.exp().item()
            entropy = dist.entropy().mean().item()
            
            stats.append({
                "update": update,
                "cost": metrics["total_cost"],
                "avg_mean": mean_vals.mean().item(),
                "std": std_val,
                "entropy": entropy
            })
        print(f"Update {update}: Cost={metrics['total_cost']:.0f}, Mean Action={stats[-1]['avg_mean']:.2f}, Std={std_val:.2f}, Entropy={entropy:.2f}")

    print("Generating Action Histogram...")
    env.reset(seed=100)
    ppo_actions = []
    out_actions = []
    out_policy = OUTPolicy(z_scores=[0.5], lead_times=[2], capacity=100)
    
    done = False
    while not done:
        obs = env._get_observations()["retailer"]
        fake_state = env.get_fake_simulator_state()
        
        # PPO action
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            dist = trainer.agents["retailer"].actor(obs_tensor)
            ppo_a = dist.sample().item()
        
        # OUT action
        out_a = out_policy.get_actions(fake_state)[0]
        
        ppo_actions.append(ppo_a)
        out_actions.append(out_a)
        
        _, _, terms, truncs, _ = env.step({"retailer": ppo_a})
        done = terms["retailer"] or truncs["retailer"]
        
    with open("results/step4_diagnostics.json", "w") as f:
        json.dump({
            "training_stats": stats,
            "ppo_histogram": ppo_actions,
            "out_histogram": out_actions
        }, f, indent=2)

if __name__ == "__main__":
    main()
