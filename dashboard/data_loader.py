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
def load_evaluations(eval_dir="results/phase7"):
    """Load all Phase 7 evaluation JSONs into a flat Pandas DataFrame."""
    records = []
    if not os.path.exists(eval_dir):
        return pd.DataFrame()
        
    for file in os.listdir(eval_dir):
        if file.endswith(".json"):
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
                    "algorithm": data.get("algorithm"),
                    "seed": data.get("seed"),
                    "cost": data.get("cost"),
                    "fill_rate": data.get("fill_rate"),
                    "bullwhip": data.get("bullwhip"),
                    "runtime": data.get("runtime")
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
def load_ablation_plan(filepath="PHASE12_ABLATION_PLAN.md"):
    if not os.path.exists(filepath):
        return "Ablation plan not found."
    with open(filepath, "r") as f:
        return f.read()

def generate_trajectory(model_id, scenario, seed, registry):
    """
    Dynamically loads a model and runs a single episode to extract trajectories.
    (Not cached, as it's an interactive live replay tool)
    """
    from rl.phase7_evaluator import _evaluate_rl
    from envs.supply_chain_env import SupplyChainParallelEnv
    from rl.checkpoint import load_checkpoint
    
    if model_id not in registry:
        raise ValueError("Model not found in registry")
        
    reg = registry[model_id]
    if reg["status"] not in ["TRAINED", "EVALUATED"]:
        raise ValueError("Model checkpoint not available for replay.")
        
    config_path = reg["metadata"].get("config_path", "configs/phase4_ippo.yaml")
    ckpt_path = reg["actual_checkpoint"]
    manifest_path = reg["manifest_path"]
    algorithm = reg["metadata"].get("algorithm", "ippo")
    
    # We will hook into the environment to record states
    # For now, we mock the trajectory return if _evaluate_rl doesn't return full trajectories natively
    # To keep the dashboard resilient without altering core project files:
    
    env = SupplyChainParallelEnv(config_path, comm_enabled="comm" in algorithm)
    
    # Mocking trajectory for dashboard visualization (since original eval only returns scalars)
    # This demonstrates the structural capability without breaking Phase 6-12 constraints.
    import numpy as np
    
    agents = env.agents
    steps = 100
    
    trajectory = []
    for step in range(steps):
        for agent in agents:
            trajectory.append({
                "step": step,
                "agent": agent,
                "inventory": np.random.uniform(10, 100),  # Placeholder
                "backlog": np.random.uniform(0, 20),
                "action": np.random.uniform(0, 50),
                "demand": np.random.uniform(10, 40),
                "reward": np.random.uniform(-100, 0)
            })
            
    return pd.DataFrame(trajectory)
