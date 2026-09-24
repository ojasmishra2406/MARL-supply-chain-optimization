import yaml
import copy
import numpy as np
import os
from envs.supply_chain_env import SupplyChainParallelEnv
from simulator.core import SupplyChainSimulator

class ScenarioSimulator(SupplyChainSimulator):
    """
    Extends the canonical simulator to support Phase 7 specific perturbations like Demand Spikes.
    """
    def __init__(self, config):
        # We need to bypass the strict file-loading of the original simulator, 
        # but the original simulator expects a config file path or a dict?
        # In core.py: def __init__(self, config_path):
        # Wait, if it strictly expects a file path, we might need to write the merged config to a temp file.
        super().__init__(config)
        
        # Check for spike in the YAML loaded config
        self.spike_config = self.config.get("spike", None)
        
    def step(self, actions):
        original_mean = self.demand_mean
        
        # Apply demand spike if configured
        if self.spike_config:
            if (self.current_step + 1) == self.spike_config["step"]:
                self.demand_mean += self.spike_config["magnitude"]
                
        res = super().step(actions)
        
        self.demand_mean = original_mean
        return res

class ScenarioParallelEnv(SupplyChainParallelEnv):
    """
    Wraps the ParallelEnv to use the ScenarioSimulator.
    """
    def __init__(self, config_path, comm_enabled=False, comm_dim=4):
        super().__init__(config_path, comm_enabled, comm_dim)
        # Replace canonical simulator with scenario simulator
        self.simulator = ScenarioSimulator(config_path)

def generate_scenario_config(canonical_config_path, scenario_config_path, output_path):
    """
    Loads the canonical config and overrides it with the scenario config.
    Saves the merged config to output_path.
    """
    with open(canonical_config_path, "r") as f:
        config = yaml.safe_load(f)
        
    with open(scenario_config_path, "r") as f:
        scenario = yaml.safe_load(f)
        
    def recursive_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                recursive_update(d[k], v)
            else:
                d[k] = v
                
    recursive_update(config, scenario)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f)
        
def create_scenario_env(scenario_id, canonical_config_path="configs/phase1_simulator.yaml", comm_enabled=False):
    """
    Creates an environment configured for a specific evaluation scenario.
    """
    scenario_file = f"configs/eval_scenarios/{scenario_id}.yaml"
    temp_config = f"configs/temp_scenario_{scenario_id}.yaml"
    
    generate_scenario_config(canonical_config_path, scenario_file, temp_config)
    
    return ScenarioParallelEnv(temp_config, comm_enabled=comm_enabled)
