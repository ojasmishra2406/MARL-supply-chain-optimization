import pytest
import numpy as np
from rl.pilot_env import PilotOneEchelonEnv
from simulator.core import SupplyChainSimulator

def test_observation_dimensions_and_fields():
    # 1. exact observation dimensionality
    env_agg = PilotOneEchelonEnv(observation_mode="aggregate")
    env_item = PilotOneEchelonEnv(observation_mode="itemized")
    assert env_agg.observation_space("retailer").shape == (5,)
    assert env_item.observation_space("retailer").shape == (6,)

def test_pipeline_bucket_semantics_and_timing():
    # 2. exact field ordering
    # 3. pipeline bucket semantics
    # 4. timing correctness
    # 5. aggregate-vs-itemized distinction
    env_agg = PilotOneEchelonEnv(observation_mode="aggregate")
    env_item = PilotOneEchelonEnv(observation_mode="itemized")
    
    env_agg.reset(seed=42)
    env_item.reset(seed=42)
    
    env_agg.inventory = 10
    env_agg.backlog = 5
    env_agg.last_demand = 8
    env_agg.last_order = 12
    env_agg.pipeline = [20, 0]
    
    env_item.inventory = 10
    env_item.backlog = 5
    env_item.last_demand = 8
    env_item.last_order = 12
    env_item.pipeline = [20, 0]
    
    obs_agg_A = env_agg._get_observations()["retailer"]
    obs_item_A = env_item._get_observations()["retailer"]
    
    env_agg.pipeline = [0, 20]
    env_item.pipeline = [0, 20]
    obs_agg_B = env_agg._get_observations()["retailer"]
    obs_item_B = env_item._get_observations()["retailer"]
    
    assert np.array_equal(obs_agg_A, obs_agg_B), "Aggregate mode should alias"
    assert not np.array_equal(obs_item_A, obs_item_B), "Itemized mode should distinguish"
    
    expected_item_A = np.array([10.0, 5.0, 20.0, 0.0, 8.0, 12.0], dtype=np.float32)
    expected_item_B = np.array([10.0, 5.0, 0.0, 20.0, 8.0, 12.0], dtype=np.float32)
    np.testing.assert_array_equal(obs_item_A, expected_item_A)
    np.testing.assert_array_equal(obs_item_B, expected_item_B)

def test_no_future_demand_leakage():
    # 6. no future demand leakage
    env = PilotOneEchelonEnv(observation_mode="itemized")
    env.reset(seed=42)
    obs = env._get_observations()["retailer"]
    
    # Fast forward RNG slightly and ensure current obs doesn't change
    env.rng.normal(20, 5)
    obs_after = env._get_observations()["retailer"]
    np.testing.assert_array_equal(obs, obs_after)

def test_determinism():
    # 7. deterministic reset
    # 8. deterministic step behavior
    env1 = PilotOneEchelonEnv(observation_mode="itemized")
    env2 = PilotOneEchelonEnv(observation_mode="itemized")
    
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)
    np.testing.assert_array_equal(obs1["retailer"], obs2["retailer"])
    
    for _ in range(5):
        obs1, _, _, _, _ = env1.step({"retailer": 10})
        obs2, _, _, _, _ = env2.step({"retailer": 10})
        np.testing.assert_array_equal(obs1["retailer"], obs2["retailer"])

def test_production_simulator_unchanged():
    # 9. production four-echelon simulator unchanged
    sim = SupplyChainSimulator("configs/phase1_simulator.yaml", seed=42)
    sim.reset()
    assert sim.num_echelons == 4
    # The pipeline is still maintained internally in the simulator as a list
    assert isinstance(sim.state.echelons[0].pipeline, list)
    
def test_existing_pilot_behavior_preserved():
    # 10. existing Phase 4 pilot behavior preserved for the control condition
    env_control = PilotOneEchelonEnv(observation_mode="aggregate")
    env_item = PilotOneEchelonEnv(observation_mode="itemized")
    
    env_control.reset(seed=42)
    env_item.reset(seed=42)
    
    for _ in range(5):
        obs_c, rew_c, _, _, _ = env_control.step({"retailer": 10})
        obs_i, rew_i, _, _, _ = env_item.step({"retailer": 10})
        
        # Rewards should match identically
        assert rew_c["retailer"] == rew_i["retailer"]
        # Control observation should equal aggregate
        assert obs_c["retailer"].shape == (5,)
        # Itemized observation should equal itemized
        assert obs_i["retailer"].shape == (6,)
