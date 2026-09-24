import numpy as np
import torch
import os
import yaml
from envs.supply_chain_env import SupplyChainParallelEnv
from rl.mappo_trainer import flatten_obs

def evaluate_phase5_policy(agent_networks, config_path, ablation_config, seeds=[0, 1, 2, 3, 4], num_episodes_per_seed=10, deterministic=True):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sim_config_path = os.path.join(repo_root, config["simulator_config"])
    
    with open(sim_config_path, "r") as f:
        sim_cfg = yaml.safe_load(f)
        
    comm_enabled = ablation_config.get("communication", False)
    sim_cfg["communication_enabled"] = comm_enabled
    if "reward_weights" in ablation_config:
        sim_cfg["cost"] = ablation_config["reward_weights"]
        
    os.makedirs("configs", exist_ok=True)
    import os as _os
    temp_sim_config = f"configs/temp_eval_sim_{_os.getpid()}.yaml"
    with open(temp_sim_config, "w") as f:
        yaml.dump(sim_cfg, f)
        
    env = SupplyChainParallelEnv(temp_sim_config, comm_enabled=comm_enabled)
    
    all_costs = []
    all_fill_rates = []
    all_bullwhips = []
    
    seed_results = {}
    
    for seed in seeds:
        seed_costs = []
        seed_demands = []
        seed_orders = []
        seed_fulfilled = []
        
        env.reset(seed=seed)
        
        episodes_completed = 0
        current_ep_cost = 0.0
        
        while episodes_completed < num_episodes_per_seed:
            obs_dict = env._get_observations()
            actions = {}
            for a in env.agents:
                flat_obs = flatten_obs(obs_dict[a])
                device = next(agent_networks[a].parameters()).device
                obs_t = torch.tensor(flat_obs, dtype=torch.float32, device=device).unsqueeze(0)
                
                with torch.no_grad():
                    action_dist = agent_networks[a].actor(obs_t)
                    if deterministic:
                        action = action_dist.probs.argmax(dim=-1).item()
                    else:
                        action = action_dist.sample().item()
                        
                    if comm_enabled:
                        comm_mean = agent_networks[a].comm_net(obs_t / 100.0)
                        if deterministic:
                            comm = comm_mean.squeeze(0).cpu().numpy()
                        else:
                            comm_std = torch.clamp(agent_networks[a].comm_log_std.exp(), min=1e-3, max=10.0)
                            comm = torch.distributions.Normal(comm_mean, comm_std).sample().squeeze(0).cpu().numpy()
                        actions[a] = {"action": action, "message": comm}
                    else:
                        actions[a] = action
            
            ret_echelon = env.simulator.state.echelons[0]
            pre_backlog = ret_echelon.backlog
            
            _, rews, terms, truncs, infos = env.step(actions)
            
            ret_echelon = env.simulator.state.echelons[0]
            
            for a in env.possible_agents:
                if a in rews:
                    current_ep_cost += -rews[a]
            
            demand = ret_echelon.demand_history[-1] if len(ret_echelon.demand_history) > 0 else 0
            fulfilled = demand - (ret_echelon.backlog - pre_backlog)
            order = actions[env.possible_agents[0]]
            if isinstance(order, dict):
                order = order["action"]
                
            seed_demands.append(demand)
            seed_orders.append(order)
            seed_fulfilled.append(fulfilled)
            
            if any(terms.values()) or any(truncs.values()):
                seed_costs.append(current_ep_cost)
                current_ep_cost = 0.0
                episodes_completed += 1
                env.reset()
                
        var_orders = np.var(seed_orders)
        var_demands = np.var(seed_demands)
        bullwhip = var_orders / var_demands if var_demands > 0 else 1.0
        
        sum_demands = np.sum(seed_demands)
        sum_fulfilled = np.sum(seed_fulfilled)
        fill_rate = sum_fulfilled / sum_demands if sum_demands > 0 else 1.0
        
        all_costs.extend(seed_costs)
        all_fill_rates.append(fill_rate)
        all_bullwhips.append(bullwhip)
        
        seed_results[seed] = {
            "costs": seed_costs,
            "mean_cost": float(np.mean(seed_costs)),
            "fill_rate": float(fill_rate),
            "bullwhip": float(bullwhip)
        }
        
    return {
        "seeds": seed_results,
        "mean_cost": float(np.mean(all_costs)),
        "std_cost": float(np.std(all_costs)),
        "mean_fill_rate": float(np.mean(all_fill_rates)),
        "mean_bullwhip": float(np.mean(all_bullwhips)),
        "all_costs": [float(c) for c in all_costs]
    }
