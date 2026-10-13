import os
import json
import glob
import pandas as pd
import streamlit as st
import torch
import sys
from pathlib import Path

# Ensure project root is in path for imports
WORKSPACE_ROOT = str(Path(__file__).resolve().parent.parent)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

@st.cache_data
def load_registry(registry_path="models/registry.json"):
    """Load the Phase 13 model registry safely."""
    full_path = os.path.join(WORKSPACE_ROOT, registry_path) if not os.path.isabs(registry_path) else registry_path
    if not os.path.exists(full_path):
        return {}
    try:
        with open(full_path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

@st.cache_data
def load_registry_dataframe(registry_path="models/registry.json"):
    """Load model registry as a structured Pandas DataFrame with orthogonal status columns."""
    reg = load_registry(registry_path)
    if not reg:
        return pd.DataFrame()
    
    rows = []
    for model_id, data in reg.items():
        rows.append({
            "model_id": model_id,
            "phase": data.get("phase", "unknown"),
            "algorithm": data.get("algorithm", "unknown"),
            "architecture": data.get("architecture", "unknown"),
            "training_regime": data.get("training_regime", data.get("condition", "baseline")),
            "training_seed": data.get("training_seed", -1),
            "training_updates": data.get("training_updates", 0),
            "training_status": data.get("training_status", "UNKNOWN"),
            "integrity_status": data.get("integrity_status", "UNKNOWN"),
            "evaluation_status": data.get("evaluation_status", "NONE"),
            "research_status": data.get("research_status", "UNKNOWN"),
            "status": data.get("status", "UNKNOWN"),
            "checkpoint_path": data.get("checkpoint_path", ""),
            "canonical_sha256": data.get("canonical_state_dict_sha256", ""),
            "provenance_notes": data.get("provenance_notes", "")
        })
    return pd.DataFrame(rows)

@st.cache_data
def load_evaluations():
    """Load Phase 7, 10, 11 Controlled, and 11 Historical evaluation JSONs into a flat DataFrame."""
    records = []
    
    dirs_to_check = [
        ("results/phase7", "phase6_7", False),
        ("results/phase10", "phase10", False),
        ("results/phase11_controlled", "phase11_controlled", True),
        ("results/phase11", "phase11_historical", False)
    ]
    
    scenarios = ['in_distribution', 'demand_shift', 'lead_time_shift', 'capacity_disruption', 'demand_spike', 'combined_shift']
    
    for rel_dir, phase_label, is_controlled in dirs_to_check:
        eval_dir = os.path.join(WORKSPACE_ROOT, rel_dir)
        if not os.path.exists(eval_dir):
            continue
            
        for file in os.listdir(eval_dir):
            if file.endswith(".json") and ("evals" in file or "eval_" in file or "_eval" in file):
                filepath = os.path.join(eval_dir, file)
                try:
                    with open(filepath, "r") as f:
                        data = json.load(f)
                        
                    ckpt_path = data.get("checkpoint", "")
                    if ckpt_path:
                        model_id = os.path.basename(ckpt_path).replace(".pt", "")
                    else:
                        model_id = data.get("experiment_id", file.replace(".json", ""))
                    
                    sc = data.get("scenario")
                    if not sc:
                        for s in scenarios:
                            if f"_{s}_" in file:
                                sc = s
                                break
                    if not sc:
                        sc = "unknown"
                        
                    algo = data.get("algorithm", data.get("condition", "unknown"))
                    cost_val = data.get("cost", data.get("total_cost", 0.0))
                    fill_val = data.get("fill_rate", data.get("service_level", 0.0))
                    bw_val = data.get("bullwhip", data.get("bullwhip_ratio", 0.0))
                    
                    cost = float(cost_val) if cost_val is not None else 0.0
                    fill_rate = float(fill_val) if fill_val is not None else 0.0
                    bullwhip = float(bw_val) if bw_val is not None else 0.0
                    cost_per_sl = (cost / fill_rate) if fill_rate > 0 else 0.0
                    
                    runtime_val = data.get("runtime", 0.0)
                    try:
                        runtime_float = float(runtime_val)
                    except (ValueError, TypeError):
                        runtime_float = 0.0

                    records.append({
                        "experiment_id": data.get("experiment_id", file.replace(".json", "")),
                        "model_id": model_id,
                        "phase": phase_label,
                        "scenario": sc,
                        "algorithm": algo,
                        "forecaster": data.get("forecaster", "none"),
                        "seed": data.get("seed", data.get("eval_seed", 0)),
                        "cost": cost,
                        "fill_rate": fill_rate,
                        "bullwhip": bullwhip,
                        "cost_per_sl": cost_per_sl,
                        "is_controlled": is_controlled,
                        "is_smoke": "smoke" in file or "s42" in file,
                        "runtime": runtime_float
                    })
                except Exception:
                    pass
                
    df = pd.DataFrame(records)
    if df.empty:
        # Fallback empty dataframe with required test columns
        return pd.DataFrame(columns=["experiment_id", "scenario", "algorithm", "model_id", "cost", "fill_rate", "bullwhip"])
    return df

@st.cache_data
def load_phase12_ablation_summary():
    """Load authoritative Phase 12 run-level summary CSV."""
    path = os.path.join(WORKSPACE_ROOT, "results", "phase12", "ablation_summary.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_phase12_scenario_breakdown():
    """Load authoritative Phase 12 scenario breakdown CSV."""
    path = os.path.join(WORKSPACE_ROOT, "results", "phase12", "ablation_scenario_breakdown.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_phase12_statistical_tests():
    """Load authoritative Phase 12 statistical tests JSON and flatten to DataFrame."""
    path = os.path.join(WORKSPACE_ROOT, "results", "phase12", "ablation_statistical_tests.json")
    if not os.path.exists(path):
        return pd.DataFrame(), {}
    
    with open(path, "r") as f:
        data = json.load(f)
        
    rows = []
    for fam_name, tests in data.items():
        for t in tests:
            rows.append({
                "Family": fam_name.replace("Family_", "").replace("_", " "),
                "Comparison": t.get("comparison", ""),
                "Metric": t.get("metric", ""),
                "nx": t.get("nx", 0),
                "ny": t.get("ny", 0),
                "Mean A": t.get("mean_x", 0.0),
                "SD A": t.get("sd_x", 0.0),
                "Mean B": t.get("mean_y", 0.0),
                "SD B": t.get("sd_y", 0.0),
                "Diff": t.get("diff", 0.0),
                "Pct Diff": t.get("pct_diff", 0.0),
                "Welch t": t.get("welch_t", 0.0),
                "df": t.get("df", 0.0),
                "Raw p-value": t.get("welch_p", 1.0),
                "MWU U": t.get("mwu_u", 0.0),
                "MWU p-value": t.get("mwu_p", 1.0),
                "Hedges g": t.get("hedges_g", 0.0),
                "CI Lower": t.get("ci_lower", 0.0),
                "CI Upper": t.get("ci_upper", 0.0),
                "FDR q-value": t.get("fdr_q", 1.0),
                "Nominal Sig (p < 0.05)": bool(t.get("welch_p", 1.0) < 0.05),
                "FDR Sig (q < 0.05)": bool(t.get("significant_fdr", False))
            })
    return pd.DataFrame(rows), data

@st.cache_data
def load_forecasting_metrics():
    """Load Phase 9 offline forecasting accuracy benchmark metrics."""
    path = os.path.join(WORKSPACE_ROOT, "results", "phase9", "forecasting_metrics.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_reproducibility_data():
    """Load reproducibility checker and checkpoint integrity reports."""
    rep_path = os.path.join(WORKSPACE_ROOT, "results", "phase13_verification", "reproducibility_verification.json")
    int_path = os.path.join(WORKSPACE_ROOT, "results", "phase13_verification", "checkpoint_integrity_report.json")
    
    rep_data = {}
    if os.path.exists(rep_path):
        with open(rep_path, "r") as f:
            rep_data = json.load(f)
            
    int_data = {}
    if os.path.exists(int_path):
        with open(int_path, "r") as f:
            int_data = json.load(f)
            
    return rep_data, int_data

def generate_trajectory(model_id, scenario, seed, registry):
    """
    Dynamically replay a single simulation episode using an evaluated model checkpoint.
    Fails gracefully with an error DataFrame if checkpoint or environment cannot be loaded.
    """
    if model_id not in registry:
        return pd.DataFrame([{"step": 0, "agent": "Error", "inventory": 0, "backlog": 0, "action": 0, "demand": 0, "reward": 0, "error": "Real trajectory data unavailable."}])
        
    reg = registry[model_id]
    status = reg.get("status")
    if status not in ["TRAINED", "EVALUATED", "VALID", "RESEARCH_VALIDATED"]:
        return pd.DataFrame([{"step": 0, "agent": "Error", "inventory": 0, "backlog": 0, "action": 0, "demand": 0, "reward": 0, "error": "Real trajectory data unavailable."}])
        
    config_path = reg.get("config_path", "configs/phase4_ippo.yaml")
    full_config_path = os.path.join(WORKSPACE_ROOT, config_path) if not os.path.isabs(config_path) else config_path
    
    ckpt_path = reg.get("checkpoint_path", "")
    full_ckpt_path = os.path.join(WORKSPACE_ROOT, ckpt_path) if not os.path.isabs(ckpt_path) else ckpt_path
    
    if not os.path.exists(full_ckpt_path):
        return pd.DataFrame([{"step": 0, "agent": "Error", "inventory": 0, "backlog": 0, "action": 0, "demand": 0, "reward": 0, "error": "Real trajectory data unavailable."}])
        
    algorithm = reg.get("algorithm", "ippo")
    
    try:
        from envs.scenario_generators import create_scenario_env
        from rl.mappo_trainer import Phase5Agent, flatten_obs
        from rl.phase10_agent import Phase10Agent
        
        comm_enabled = "comm" in algorithm
        env = create_scenario_env(scenario, canonical_config_path="configs/phase1_simulator.yaml", comm_enabled=comm_enabled)
        agents_names = env.possible_agents
        obs_dim = 13 if comm_enabled else 5
        num_agents = len(agents_names)
        centralized_critic = "mappo" in algorithm
        global_obs_dim = obs_dim * num_agents if centralized_critic else obs_dim
        
        agents = {}
        for name in agents_names:
            if "gnn" in algorithm:
                agents[name] = Phase10Agent(obs_dim, 101, num_nodes=num_agents)
            else:
                agent = Phase5Agent(obs_dim, global_obs_dim, 101, comm_dim=4 if comm_enabled else 0, centralized_critic=centralized_critic)
                agent.eval()
                agents[name] = agent
                
        state_dict = torch.load(full_ckpt_path, map_location="cpu")
        if "retailer" not in state_dict:
            for name in agents_names:
                agents[name].load_state_dict(state_dict)
        else:
            for name in agents_names:
                agents[name].load_state_dict(state_dict[name])
                
        obs_dict, _ = env.reset(seed=int(seed))
        trajectory = []
        
        for step in range(52):
            actions = {}
            for a in agents_names:
                flat_obs = flatten_obs(obs_dict[a])
                obs_t = torch.tensor(flat_obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    if "gnn" in algorithm:
                        act, _, _, _, _ = agents[a].get_action_and_value(obs_t, obs_t)
                        action = act.item()
                    else:
                        action_dist = agents[a].actor(obs_t)
                        action = action_dist.probs.argmax(dim=-1).item()
                        
                    if comm_enabled and hasattr(agents[a], "comm_net"):
                        comm_mean = agents[a].comm_net(obs_t / 100.0)
                        comm = comm_mean.squeeze(0).cpu().numpy()
                        actions[a] = {"action": action, "message": comm}
                    else:
                        actions[a] = action
                        
            sim_state = env.unwrapped.simulator.state
            for idx, echelon in enumerate(sim_state.echelons):
                act_val = actions[agents_names[idx]]["action"] if isinstance(actions[agents_names[idx]], dict) else actions[agents_names[idx]]
                trajectory.append({
                    "step": step,
                    "agent": agents_names[idx],
                    "inventory": echelon.inventory,
                    "backlog": echelon.backlog,
                    "action": act_val,
                    "demand": echelon.demand_history[-1] if len(echelon.demand_history) > 0 else 0,
                    "reward": 0
                })
                
            obs_dict, _, _, _, _ = env.step(actions)
            
        return pd.DataFrame(trajectory)
    except Exception as e:
        return pd.DataFrame([{"step": 0, "agent": "Error", "inventory": 0, "backlog": 0, "action": 0, "demand": 0, "reward": 0, "error": "Real trajectory data unavailable."}])
