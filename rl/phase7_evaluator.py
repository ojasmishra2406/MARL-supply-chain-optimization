import os
import yaml
import torch
import numpy as np
import datetime
import subprocess

from envs.scenario_generators import create_scenario_env
from rl.checkpoint import load_checkpoint
from rl.mappo_trainer import Phase5Agent, flatten_obs
from baselines.out import OUTPolicy

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "unknown"

def evaluate_scenario(algorithm, checkpoint_path, config_path, scenario_id, seed, manifest_path=None):
    """
    Evaluates a policy on a Phase 7 held-out scenario.
    """
    # Create scenario environment (this prevents leakage by dynamically injecting)
    env = create_scenario_env(scenario_id, canonical_config_path="configs/phase1_simulator.yaml", comm_enabled=("comm" in algorithm))
    
    if "out" in algorithm.lower():
        return _evaluate_out(env, config_path, scenario_id, seed)
    else:
        return _evaluate_rl(algorithm, checkpoint_path, manifest_path, config_path, scenario_id, seed, env)

def _evaluate_out(env, config_path, scenario_id, seed):
    with open(config_path, "r") as f:
        out_cfg = yaml.safe_load(f)
        
    z_scores = [0.0, 0.0, 0.0, 0.0]
    if "z_scores" in out_cfg:
        z_scores = out_cfg["z_scores"]
    elif "optimal_z" in out_cfg:
        z_scores = out_cfg["optimal_z"]
        
    policy = OUTPolicy(
        z_scores=z_scores,
        lead_times=env.simulator.lead_times,
        capacity=env.simulator.capacity,
        mu=env.simulator.demand_mean,
        sigma=env.simulator.demand_std,
    )
    
    # 52-step evaluation loop
    env.simulator.reset(seed=seed)
    
    total_cost = 0.0
    seed_demands = []
    seed_orders = []
    seed_fulfilled = []
    
    for _ in range(52):
        ret_echelon = env.simulator.state.echelons[0]
        pre_backlog = ret_echelon.backlog
        
        # Calculate OUT actions
        actions = policy.get_actions(env.simulator.state)
        
        env.simulator.step(actions)
        
        ret_echelon = env.simulator.state.echelons[0]
        demand = ret_echelon.demand_history[-1] if len(ret_echelon.demand_history) > 0 else 0
        fulfilled = demand - (ret_echelon.backlog - pre_backlog)
        
        seed_demands.append(demand)
        seed_orders.append(actions[0])
        seed_fulfilled.append(fulfilled)
        
    total_cost = env.simulator.state.total_cost
        
    var_orders = np.var(seed_orders)
    var_demands = np.var(seed_demands)
    bullwhip = var_orders / var_demands if var_demands > 0 else 1.0
    
    sum_demands = np.sum(seed_demands)
    sum_fulfilled = np.sum(seed_fulfilled)
    fill_rate = sum_fulfilled / sum_demands if sum_demands > 0 else 1.0
    
    return _format_result("out", None, None, scenario_id, seed, total_cost, fill_rate, bullwhip)

def _evaluate_rl(algorithm, checkpoint_path, manifest_path, config_path, scenario_id, seed, env):
    with open(config_path, "r") as f:
        rl_cfg = yaml.safe_load(f)
        
    comm_enabled = env.comm_enabled
    obs_dim = 13 if comm_enabled else 5
    num_agents = len(env.possible_agents)
    
    centralized_critic = "mappo" in algorithm
    global_obs_dim = obs_dim * num_agents if centralized_critic else obs_dim
    
    agents = {}
    for name in env.possible_agents:
        agent = Phase5Agent(obs_dim, global_obs_dim, 101, comm_dim=4 if comm_enabled else 0, centralized_critic=centralized_critic)
        agent.eval() # Evaluation mode
        agents[name] = agent
        
    state_dict = load_checkpoint(checkpoint_path, manifest_path)
    # Check if parameter_sharing is true based on state_dict keys
    if "retailer" not in state_dict:
        # Shared params
        for name in env.possible_agents:
            agents[name].load_state_dict(state_dict)
    else:
        for name in env.possible_agents:
            agents[name].load_state_dict(state_dict[name])
            
    env.reset(seed=seed)
    
    total_cost = 0.0
    seed_demands = []
    seed_orders = []
    seed_fulfilled = []
    
    for _ in range(52):
        obs_dict = env._get_observations()
        actions = {}
        for a in env.possible_agents:
            flat_obs = flatten_obs(obs_dict[a])
            device = next(agents[a].parameters()).device
            obs_t = torch.tensor(flat_obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            with torch.no_grad():
                action_dist = agents[a].actor(obs_t)
                action = action_dist.probs.argmax(dim=-1).item() # deterministic
                    
                if comm_enabled:
                    comm_mean = agents[a].comm_net(obs_t / 100.0)
                    comm = comm_mean.squeeze(0).cpu().numpy()
                    actions[a] = {"action": action, "message": comm}
                else:
                    actions[a] = action
                    
        ret_echelon = env.simulator.state.echelons[0]
        pre_backlog = ret_echelon.backlog
        
        _, rews, _, _, _ = env.step(actions)
        
        ret_echelon = env.simulator.state.echelons[0]
        for a in env.possible_agents:
            if a in rews:
                total_cost += -rews[a]
                
        demand = ret_echelon.demand_history[-1] if len(ret_echelon.demand_history) > 0 else 0
        fulfilled = demand - (ret_echelon.backlog - pre_backlog)
        order = actions[env.possible_agents[0]]
        if isinstance(order, dict):
            order = order["action"]
            
        seed_demands.append(demand)
        seed_orders.append(order)
        seed_fulfilled.append(fulfilled)
        
    var_orders = np.var(seed_orders)
    var_demands = np.var(seed_demands)
    bullwhip = var_orders / var_demands if var_demands > 0 else 1.0
    
    sum_demands = np.sum(seed_demands)
    sum_fulfilled = np.sum(seed_fulfilled)
    fill_rate = sum_fulfilled / sum_demands if sum_demands > 0 else 1.0
    
    import hashlib
    with open(f"configs/eval_scenarios/{scenario_id}.yaml", "rb") as f:
        scenario_hash = hashlib.sha256(f.read()).hexdigest()
        
    # Re-read checkpoint manifest for hash
    import json
    with open(manifest_path, "r") as f:
        ckpt_hash = json.load(f)["checkpoint_sha256"]
        
    return _format_result(algorithm, checkpoint_path, ckpt_hash, scenario_id, seed, total_cost, fill_rate, bullwhip, scenario_hash=scenario_hash)

def _format_result(algo, ckpt, ckpt_hash, scenario, seed, cost, fill_rate, bullwhip, scenario_hash=None):
    return {
        "experiment_id": f"eval_{scenario}_{algo}_s{seed}",
        "scenario": scenario,
        "algorithm": algo,
        "seed": seed,
        "checkpoint": ckpt,
        "checkpoint_hash": ckpt_hash,
        "scenario_hash": scenario_hash or "N/A",
        "episode_count": 1,
        "cost": float(cost),
        "fill_rate": float(fill_rate),
        "bullwhip": float(bullwhip),
        "runtime": datetime.datetime.now().isoformat(),
        "reproducibility": {
            "git_commit": get_git_commit()
        }
    }
