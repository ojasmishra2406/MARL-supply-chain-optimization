import os
import json
import pandas as pd
import streamlit as st
import torch
import sys

# Ensure project root is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

@st.cache_data
def load_registry(registry_path="models/registry.json"):
    """Load the Phase 13 model registry."""
    if not os.path.exists(registry_path):
        return {}
    try:
        with open(registry_path, "r") as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Failed to load registry: {e}")
        return {}

@st.cache_data
def load_evaluations():
    """Load all Phase 7, 10, and 11 evaluation JSONs into a flat Pandas DataFrame."""
    records = []
    
    dirs_to_check = ["results/phase7", "results/phase10", "results/phase11"]
    
    for eval_dir in dirs_to_check:
        if not os.path.exists(eval_dir):
            continue
            
        for file in os.listdir(eval_dir):
            if file.endswith(".json") and "evals" in file or "eval_" in file:
                try:
                    with open(os.path.join(eval_dir, file), "r") as f:
                        data = json.load(f)
                        
                    # Extract checkpoint ID from the checkpoint path
                    ckpt_path = data.get("checkpoint", "")
                    if ckpt_path:
                        model_id = os.path.basename(ckpt_path).replace(".pt", "")
                    else:
                        model_id = data.get("experiment_id", "unknown")
                        
                    records.append({
                        "experiment_id": data.get("experiment_id"),
                        "model_id": model_id,
                        "scenario": data.get("scenario"),
                        "algorithm": data.get("algorithm", data.get("condition")),
                        "forecaster": data.get("forecaster", "none"),
                        "seed": data.get("seed"),
                        "cost": data.get("cost", data.get("total_cost")),
                        "fill_rate": data.get("fill_rate", data.get("service_level")),
                        "bullwhip": data.get("bullwhip", data.get("bullwhip_ratio")),
                        "runtime": data.get("runtime", 0.0)
                    })
                except Exception as e:
                    # Silently skip corrupted single files to avoid dashboard crashes
                    pass
                
    return pd.DataFrame(records)

@st.cache_data
def load_statistical_analysis(filepath="PHASE8_STATISTICAL_ANALYSIS.md"):
    """Load the Phase 8 markdown report."""
    if not os.path.exists(filepath):
        return "Statistical analysis not found. Please run Phase 8."
    with open(filepath, "r") as f:
        return f.read()
        
@st.cache_data
def load_phase13_report(filepath="PHASE13_FINAL_REPORT.md"):
    """Load the Phase 13 final report."""
    if not os.path.exists(filepath):
        return "Phase 13 Final Report not found. Please run Phase 12 & 13."
    with open(filepath, "r") as f:
        return f.read()

@st.cache_data
def load_ablation_plan(filepath="PHASE12_ABLATION_PLAN.md"):
    if not os.path.exists(filepath):
        return "Ablation plan not found."
    with open(filepath, "r") as f:
        return f.read()

def generate_trajectory(model_id, scenario, seed, registry):
    # Dynamically loads a model and runs a single episode to extract trajectories.
    from rl.phase7_evaluator import _evaluate_rl
    from envs.scenario_generators import create_scenario_env
    from rl.checkpoint import load_checkpoint
    
    if model_id not in registry:
        raise ValueError("Model not found in registry")
        
    reg = registry[model_id]
    if reg["status"] not in ["TRAINED", "EVALUATED", "VALID"]:
        raise ValueError("Model checkpoint not available for replay.")
        
    config_path = reg.get("config_path", reg.get("metadata", {}).get("config_path", "configs/phase4_ippo.yaml"))
    ckpt_path = reg.get("checkpoint_path", reg.get("actual_checkpoint"))
    algorithm = reg.get("algorithm", reg.get("metadata", {}).get("algorithm", "ippo"))
    
    env = create_scenario_env(scenario, config_path, comm_enabled="comm" in algorithm)
    
    import torch
    import numpy as np
    
    # We must construct agents and load checkpoint.
    # Due to complexity of loading MAPPO vs IPPO vs GNN vs Forecasting dynamically here without
    # full ablation params, we will try to load it. If it fails, we return a DataFrame with an error message
    # rather than faking data.
    
    trajectory = []
    
    try:
        from rl.mappo_agent import MAPPOAgent
        from rl.ippo_agent import IPPOAgent
        
        ablation = {"centralized_critic": "mappo" in algorithm, "communication": "comm" in algorithm}
        
        agents = {}
        agents_names = env.possible_agents
        for i, name in enumerate(agents_names):
            if "mappo" in algorithm:
                agents[name] = MAPPOAgent(env.observation_space(name).shape[0], env.action_space(name).n, agent_index=i, num_agents=len(agents_names), hidden_dim=128)
            else:
                agents[name] = IPPOAgent(env.observation_space(name).shape[0], env.action_space(name).n, hidden_dim=128)
                
        state_dicts = torch.load(ckpt_path, map_location="cpu")
        for name, agent in agents.items():
            agent.load_state_dict(state_dicts[name])
            
        obs_dict, _ = env.reset(seed=seed)
        
        for step in range(52):
            actions = {}
            for i, a in enumerate(agents_names):
                obs = obs_dict[a]
                if "mappo" in algorithm:
                    act, _, _, _, _ = agents[a].get_action_and_value(torch.tensor(obs, dtype=torch.float32).unsqueeze(0), torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                else:
                    act, _, _, _ = agents[a].get_action_and_value(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                actions[a] = act.item()
                
            ret_echelon = env.unwrapped.simulator.state.echelons[0]
            for idx, echelon in enumerate(env.unwrapped.simulator.state.echelons):
                trajectory.append({
                    "step": step,
                    "agent": agents_names[idx],
                    "inventory": echelon.inventory,
                    "backlog": echelon.backlog,
                    "action": actions.get(agents_names[idx], 0),
                    "demand": echelon.demand_history[-1] if len(echelon.demand_history)>0 else 0,
                    "reward": 0
                })
                
            obs_dict, rews, _, _, _ = env.step(actions)
            
        return pd.DataFrame(trajectory)
    except Exception as e:
        print(f"Failed to generate real trajectory: {e}")
        return pd.DataFrame([{"step": 0, "agent": "Error", "inventory": 0, "backlog": 0, "action": 0, "demand": 0, "reward": 0, "error": "Real trajectory data unavailable."}])
