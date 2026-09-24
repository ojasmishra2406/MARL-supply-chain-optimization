import numpy as np
import torch
from rl.pilot_env import PilotOneEchelonEnv
from typing import Dict, Any

def evaluate_policy(actor_network, seeds=[0, 1, 2], num_episodes_per_seed=10, deterministic=True) -> Dict[str, Any]:
    env = PilotOneEchelonEnv()
    
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
            obs = env._get_observations()["retailer"]
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                dist = actor_network(obs_t)
                if deterministic:
                    action = dist.probs.argmax(dim=-1).item()
                else:
                    action = dist.sample().item()
                
            _, rews, terms, truncs, infos = env.step({"retailer": float(action)})
            
            info = infos["retailer"]
            current_ep_cost += info["cost"]
            seed_demands.append(info["demand"])
            seed_orders.append(info["order"])
            seed_fulfilled.append(info["fulfilled"])
            
            if terms["retailer"] or truncs["retailer"]:
                seed_costs.append(current_ep_cost)
                current_ep_cost = 0.0
                episodes_completed += 1
                env.reset()
                
        # Calculate bullwhip and fill rate for this seed
        # Bullwhip = var(orders) / var(demands)
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
            "mean_cost": np.mean(seed_costs),
            "fill_rate": fill_rate,
            "bullwhip": bullwhip
        }
        
    return {
        "seeds": seed_results,
        "mean_cost": np.mean(all_costs),
        "std_cost": np.std(all_costs),
        "mean_fill_rate": np.mean(all_fill_rates),
        "mean_bullwhip": np.mean(all_bullwhips),
        "all_costs": all_costs
    }
