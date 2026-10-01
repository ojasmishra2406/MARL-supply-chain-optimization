import os
import sys
import json
import traceback
import torch
import yaml
import numpy as np

run_id = sys.argv[1]
manifest_path = f"results/phase11/{run_id}_manifest.json"

if not os.path.exists(manifest_path):
    print(f"Manifest not found for {run_id}")
    sys.exit(1)

with open(manifest_path, "r") as f:
    manifest = json.load(f)

algo = manifest["algorithm"]
forecaster_type = manifest.get("forecaster", "ma")
seed = manifest["seed"]
ckpt_path = manifest["checkpoint_path"]

print(f"Evaluating {run_id} ({algo}, {forecaster_type}, seed {seed})")

from rl.phase10_agent import Phase10Agent
from envs.scenario_generators import create_scenario_env
from forecasting.wrapper import DemandForecastWrapper
from forecasting.models import NaiveForecaster, MovingAverageForecaster, XGBoostForecaster

with open(manifest["config_path"], "r") as f:
    config = yaml.safe_load(f)

if forecaster_type == "xgboost":
    forecaster = XGBoostForecaster(window=5)
    import pandas as pd
    df = pd.read_csv("data/forecasting/train_demand.csv")
    forecaster.fit(df["demand"].values)
else:
    forecaster = MovingAverageForecaster(window=5)

# Initialize dummy environment just to get agents and observation shapes
base_dummy = create_scenario_env("in_distribution", canonical_config_path=config["simulator_config"], comm_enabled=False)
wrapped_dummy = DemandForecastWrapper(base_dummy, forecaster, forecast_horizon=2)

agents_names = wrapped_dummy.agents
obs_dim = wrapped_dummy.observation_space(agents_names[0]).shape[0]

device = torch.device("cpu")
agents = {}
for i, name in enumerate(agents_names):
    agents[name] = Phase10Agent(
        obs_dim=obs_dim, 
        action_dim=wrapped_dummy.action_space(name).n, 
        num_nodes=len(agents_names),
        hidden_dim=64,
        agent_index=i,
        lr=0.0003
    ).to(device)

# Load checkpoint
state_dicts = torch.load(ckpt_path)
for name, agent in agents.items():
    agent.load_state_dict(state_dicts[name])

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
        out_file = f"results/phase11/{run_id}_{scenario}_evals{e_seed}.json"
        if os.path.exists(out_file):
            completed += 1
            continue
            
        print(f"Evaluating {scenario} seed {e_seed}...")
        try:
            base_env = create_scenario_env(scenario, canonical_config_path=config["simulator_config"], comm_enabled=False)
            eval_env = DemandForecastWrapper(base_env, forecaster, forecast_horizon=2)
            
            obs_dict, _ = eval_env.reset(seed=e_seed)
            total_cost = 0.0
            seed_demands = []
            seed_orders = []
            seed_fulfilled = []
            
            for _ in range(52):
                actions = {}
                for a in eval_env.possible_agents:
                    obs = obs_dict[a]
                    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                    # We don't need global_obs for action inference
                    with torch.no_grad():
                        logits = agents[a].actor(obs_t)
                        action = logits.argmax(dim=-1).item()
                        actions[a] = action
                        
                ret_echelon = eval_env.unwrapped.simulator.state.echelons[0]
                pre_backlog = ret_echelon.backlog
                
                obs_dict, rews, _, _, _ = eval_env.step(actions)
                
                ret_echelon = eval_env.unwrapped.simulator.state.echelons[0]
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
                "forecaster": forecaster_type,
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
