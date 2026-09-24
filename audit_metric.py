import os
import json
import torch
import numpy as np
import time
from rl.networks import ActorNetwork, CriticNetwork
from rl.pilot_env import PilotOneEchelonEnv
from baselines.out import OUTPolicy
from rl.pilot import PilotTrainer
from evaluator import evaluate_policy
import hashlib

def hash_tensor_dict(state_dict):
    m = hashlib.sha256()
    for k, v in sorted(state_dict.items()):
        m.update(k.encode())
        m.update(v.cpu().numpy().tobytes())
    return m.hexdigest()

def step1_metric_reconstruction():
    print("=== 1. Metric Reconstruction ===")
    config_path = "configs/phase4_ippo.yaml"
    from rl.pilot import SyncPilotVectorEnv
    from rl.networks import ActorNetwork
    import yaml
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    num_envs = config.get("num_envs", 8)
    rollout_steps = config.get("rollout_length", 2048)
    scale_factor = config.get("reward_scale", 100.0)
    horizon = 52
    
    vec_env = SyncPilotVectorEnv(num_envs=num_envs, config_path=config_path)
    actor = ActorNetwork(obs_dim=5, action_dim=101, hidden_size=128)
    actor.load_state_dict(torch.load("results/phase4b_step1_imitation_actor_recreated.pt"))
    actor.eval()
    
    obs_list = vec_env.reset(seed=0)
    
    raw_costs = np.zeros((rollout_steps, num_envs), dtype=np.float32)
    dones = np.zeros((rollout_steps, num_envs), dtype=bool)
    
    for step in range(rollout_steps):
        agent = "retailer"
        
        # Convert obs_list to tensor
        obs_tensor = torch.tensor(np.array([o[agent] for o in obs_list]), dtype=torch.float32)
        
        with torch.no_grad():
            dist = actor(obs_tensor)
            actions = dist.sample()
            
        actions_list = [{agent: a.item()} for a in actions]
        next_obs_list, rews_list, terms_list, truncs_list, infos_list = vec_env.step(actions_list)
        
        for env_idx in range(num_envs):
            raw_costs[step, env_idx] = infos_list[env_idx][agent]["cost"]
            dones[step, env_idx] = terms_list[env_idx][agent] or truncs_list[env_idx][agent]
            
        obs_list = next_obs_list
    
    total_complete_episodes = 0
    sum_complete_episode_costs = 0.0
    partial_steps = 0
    sum_partial_episode_costs = 0.0
    
    for env_idx in range(num_envs):
        current_ep_cost = 0.0
        current_ep_steps = 0
        for step_idx in range(rollout_steps):
            current_ep_cost += raw_costs[step_idx, env_idx]
            current_ep_steps += 1
            if dones[step_idx, env_idx]:
                total_complete_episodes += 1
                sum_complete_episode_costs += current_ep_cost
                current_ep_cost = 0.0
                current_ep_steps = 0
                
        # Whatever is left is a partial episode
        if current_ep_steps > 0:
            partial_steps += current_ep_steps
            sum_partial_episode_costs += current_ep_cost
            
    aggregate_rollout_cost = raw_costs.sum() / num_envs
    mean_complete_episode_cost = sum_complete_episode_costs / total_complete_episodes if total_complete_episodes > 0 else 0
    
    print(f"Rollout steps: {rollout_steps}")
    print(f"Environments: {num_envs}")
    print(f"Episode horizon: {horizon}")
    print(f"Complete episodes total (across all envs): {total_complete_episodes}")
    print(f"Complete episodes/env: {total_complete_episodes / num_envs}")
    print(f"Partial steps (across all envs): {partial_steps}")
    print(f"Aggregate rollout cost (mean across envs): {aggregate_rollout_cost:.2f}")
    print(f"Sum of complete episode costs (mean across envs): {(sum_complete_episode_costs / num_envs):.2f}")
    print(f"Partial episode cost (mean across envs): {(sum_partial_episode_costs / num_envs):.2f}")
    print(f"Mean complete episode cost: {mean_complete_episode_cost:.2f}")
    
    approx_formula = aggregate_rollout_cost / (rollout_steps / horizon)
    print(f"aggregate / (2048/52): {approx_formula:.2f}")
    
    return mean_complete_episode_cost

if __name__ == "__main__":
    step1_metric_reconstruction()
