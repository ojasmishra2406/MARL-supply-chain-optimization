import os
import json
import numpy as np
import pandas as pd
from simulator.core import SupplyChainSimulator

def generate_dataset(base_config_path, scenario_path, num_episodes, output_path):
    print(f"Generating dataset from {scenario_path or base_config_path} for {num_episodes} episodes...")
    import yaml
    with open(base_config_path, "r") as f:
        config = yaml.safe_load(f)
        
    if scenario_path:
        with open(scenario_path, "r") as f:
            scenario_config = yaml.safe_load(f)
            
        def update_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
            
        config = update_dict(config, scenario_config)
        
    # Write temp config
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        temp_path = f.name
        
    sim = SupplyChainSimulator(temp_path, seed=42)
    
    records = []
    
    for ep in range(num_episodes):
        sim.reset()
        for t in range(sim.horizon):
            # Record state BEFORE step
            demand = int(np.round(sim.rng.normal(sim.demand_mean, sim.demand_std)))
            if "trend" in sim.config.get("demand", {}):
                demand += sim.config["demand"]["trend"] * t
            if "seasonality" in sim.config.get("demand", {}):
                period = sim.config["demand"]["seasonality"].get("period", 12)
                amplitude = sim.config["demand"]["seasonality"].get("amplitude", 10)
                demand += amplitude * np.sin(2 * np.pi * t / period)
            if "spike" in sim.config:
                if isinstance(sim.config["spike"], dict) and t == sim.config["spike"].get("step", -1):
                    demand += sim.config["spike"].get("magnitude", 0)
                elif isinstance(sim.config["spike"], list):
                    for spike in sim.config["spike"]:
                        if t == spike.get("step", -1):
                            demand += spike.get("magnitude", 0)
                            
            if sim.clip_demand and demand < 0:
                demand = 0
                
            records.append({
                "episode": ep,
                "step": t,
                "demand": demand,
            })
            
            # Step with naive actions (just ordering mean demand to keep simulator advancing)
            actions = [int(sim.demand_mean)] * sim.num_echelons
            sim.step(actions)
            
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} records to {output_path}")

if __name__ == "__main__":
    # Generate for in_distribution (baseline)
    generate_dataset("configs/phase1_simulator.yaml", None, 5, "data/forecasting/train_demand.csv")
    
    # Generate for combined_shift (out of distribution)
    generate_dataset("configs/phase1_simulator.yaml", "configs/eval_scenarios/combined_shift.yaml", 2, "data/forecasting/test_demand_ood.csv")
