"""
Worker script for evaluating a Phase 10 checkpoint.
Usage: python run_phase10_eval_worker.py <run_id>
"""
import os
import sys
import json
import traceback

run_id = sys.argv[1]
manifest_path = f"results/phase10/{run_id}_manifest.json"

if not os.path.exists(manifest_path):
    print(f"Manifest not found for {run_id}")
    sys.exit(1)

with open(manifest_path, "r") as f:
    manifest = json.load(f)

algo = manifest["algorithm"]
condition = manifest["condition"]
seed = manifest["seed"]
ckpt_path = manifest["checkpoint_path"]

print(f"Evaluating {run_id} ({algo}, {condition}, seed {seed})")

from rl.phase7_evaluator import evaluate_scenario
from rl.phase10_agent import Phase10Agent
import torch
import yaml

ablation = {
    "centralized_critic": True,
    "communication": False,
    "parameter_sharing": False,
    "seed": seed
}

with open(manifest["config_path"], "r") as f:
    config = yaml.safe_load(f)

# Initialize agents
from envs.supply_chain_env import SupplyChainParallelEnv
import supersuit as ss
base_env = SupplyChainParallelEnv(config["simulator_config"])
agents_names = base_env.agents
agents = {name: Phase10Agent(obs_dim=5, action_dim=base_env.action_space(name).n, agent_index=i) for i, name in enumerate(agents_names)}

# Load checkpoint
state_dict = torch.load(ckpt_path)
for name, agent in agents.items():
    agent.load_state_dict(state_dict[name])

scenarios = [
    "in_distribution",
    "demand_shift",
    "lead_time_shift",
    "capacity_disruption",
    "demand_spike",
    "combined_shift"
]

eval_seeds = [0, 1, 2, 3, 4]
completed = 0

for scenario in scenarios:
    for e_seed in eval_seeds:
        out_file = f"results/phase10/{run_id}_{scenario}_evals{e_seed}.json"
        if os.path.exists(out_file):
            completed += 1
            continue
            
        print(f"Evaluating {scenario} seed {e_seed}...")
        try:
            from envs.scenario_generators import create_scenario_env
            eval_env = create_scenario_env(scenario, canonical_config_path=config["simulator_config"], comm_enabled=False)
            obs_dict, _ = eval_env.reset(seed=e_seed)
            total_cost = 0.0
            seed_demands = []
            seed_orders = []
            seed_fulfilled = []
            
            import numpy as np
            for _ in range(52):
                actions = {}
                for a in eval_env.possible_agents:
                    # In phase 10, obs_dict already flat? Actually yes.
                    obs = obs_dict[a]
                    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        logits = agents[a].actor(obs_t)
                        action = logits.argmax(dim=-1).item()
                        actions[a] = action
                        
                ret_echelon = eval_env.simulator.state.echelons[0]
                pre_backlog = ret_echelon.backlog
                
                obs_dict, rews, _, _, _ = eval_env.step(actions)
                
                ret_echelon = eval_env.simulator.state.echelons[0]
                for a in eval_env.possible_agents:
                    if a in rews:
                        total_cost += -rews[a]
                        
                demand = ret_echelon.demand_history[-1] if len(ret_echelon.demand_history) > 0 else 0
                fulfilled = demand - (ret_echelon.backlog - pre_backlog)
                order = actions[eval_env.possible_agents[0]]
                
                seed_demands.append(demand)
                seed_orders.append(order)
                seed_fulfilled.append(fulfilled)
                
            var_orders = np.var(seed_orders)
            var_demands = np.var(seed_demands)
            bullwhip = var_orders / var_demands if var_demands > 0 else 1.0
            
            sum_demands = np.sum(seed_demands)
            sum_fulfilled = np.sum(seed_fulfilled)
            fill_rate = sum_fulfilled / sum_demands if sum_demands > 0 else 1.0
            
            res = {
                "experiment_id": f"eval_{scenario}_{run_id}_{e_seed}",
                "scenario": scenario,
                "algorithm": algo,
                "seed": e_seed,
                "checkpoint": ckpt_path,
                "checkpoint_hash": manifest.get("checkpoint_sha256", "unknown"),
                "total_cost": float(total_cost),
                "service_level": float(fill_rate),
                "bullwhip_ratio": float(bullwhip)
            }
            
            with open(out_file, "w") as f:
                json.dump(res, f, indent=2)
            completed += 1
        except Exception as e:
            print(f"Failed {scenario} {e_seed}: {e}")
            traceback.print_exc()

print(f"Evaluations completed for {run_id}: {completed}/30")
