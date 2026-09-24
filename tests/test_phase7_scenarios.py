import pytest
import os
import yaml
from envs.scenario_generators import create_scenario_env

SCENARIOS = [
    "in_distribution",
    "demand_shift",
    "lead_time_shift",
    "capacity_disruption",
    "demand_spike",
    "combined_shift"
]

@pytest.mark.parametrize("scenario_id", SCENARIOS)
def test_scenario_loads(scenario_id):
    # Ensure config file exists
    config_path = f"configs/eval_scenarios/{scenario_id}.yaml"
    assert os.path.exists(config_path)
    
    # Ensure YAML schema is valid
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    assert isinstance(config, dict)
    
    # Ensure it loads into an env without crashing
    env = create_scenario_env(scenario_id)
    obs, info = env.reset(seed=42)
    assert len(obs) > 0

def test_deterministic_behavior():
    env1 = create_scenario_env("demand_shift")
    env2 = create_scenario_env("demand_shift")
    
    env1.reset(seed=123)
    env2.reset(seed=123)
    
    assert env1.simulator.demand_mean == 30
    
    # Run 10 steps to extract demand from the state
    demands1 = []
    demands2 = []
    actions = {a: {"action": 0, "message": [0,0,0,0]} if env1.comm_enabled else 0 for a in env1.possible_agents}
    for _ in range(10):
        env1.step(actions)
        env2.step(actions)
        demands1.append(env1.simulator.state.echelons[0].demand_history[-1])
        demands2.append(env2.simulator.state.echelons[0].demand_history[-1])
    
    assert demands1 == demands2

def test_stochastic_differences():
    env1 = create_scenario_env("in_distribution")
    env2 = create_scenario_env("in_distribution")
    
    env1.reset(seed=1)
    env2.reset(seed=2)
    
    demands1 = []
    demands2 = []
    actions = {a: 0 for a in env1.possible_agents}
    for _ in range(10):
        env1.step(actions)
        env2.step(actions)
        demands1.append(env1.simulator.state.echelons[0].demand_history[-1])
        demands2.append(env2.simulator.state.echelons[0].demand_history[-1])
    
    assert demands1 != demands2

def test_demand_spike():
    env = create_scenario_env("demand_spike")
    env.reset(seed=42)
    
    spike_step = env.simulator.spike_config["step"]
    magnitude = env.simulator.spike_config["magnitude"]
    
    # Step up to the spike
    actions = {a: 0 for a in env.possible_agents}
    for _ in range(spike_step):
        env.step(actions)
        
    demand = env.simulator.state.echelons[0].demand_history[-1]

    
    # The demand should be roughly mean (20) + magnitude (150)
    assert demand > magnitude

def test_in_distribution_matches_canonical():
    env_canonical = create_scenario_env("in_distribution")
    
    with open("configs/phase1_simulator.yaml", "r") as f:
        canonical_config = yaml.safe_load(f)
        
    assert env_canonical.simulator.cost_backlog == canonical_config["cost"]["backlog"]
    assert env_canonical.simulator.lead_times == canonical_config["lead_time"]
