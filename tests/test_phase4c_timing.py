import pytest
import numpy as np
from rl.pilot_env import PilotOneEchelonEnv

def test_critical_timing_aliasing():
    """Verify that aggregate observation aliases pipeline timings, while itemized distinguishes them."""
    env_agg = PilotOneEchelonEnv(observation_mode="aggregate")
    env_item = PilotOneEchelonEnv(observation_mode="itemized")
    
    # State A: 20 arriving tomorrow, 0 day after
    env_agg.reset()
    env_agg.inventory = 10
    env_agg.backlog = 5
    env_agg.last_demand = 8
    env_agg.last_order = 12
    env_agg.pipeline = [20, 0]
    
    env_item.reset()
    env_item.inventory = 10
    env_item.backlog = 5
    env_item.last_demand = 8
    env_item.last_order = 12
    env_item.pipeline = [20, 0]
    
    obs_agg_A = env_agg._get_observations()["retailer"]
    obs_item_A = env_item._get_observations()["retailer"]
    
    # State B: 0 arriving tomorrow, 20 day after
    env_agg.pipeline = [0, 20]
    env_item.pipeline = [0, 20]
    
    obs_agg_B = env_agg._get_observations()["retailer"]
    obs_item_B = env_item._get_observations()["retailer"]
    
    # 1. Aggregate observation should ALIAS State A and State B
    assert np.array_equal(obs_agg_A, obs_agg_B), "Aggregate mode should have aliased these states, but it didn't!"
    
    # 2. Itemized observation should DISTINGUISH State A and State B
    assert not np.array_equal(obs_item_A, obs_item_B), "Itemized mode failed to distinguish states!"
    
    # 3. Verify exact dimensionality
    assert obs_agg_A.shape == (5,), f"Expected aggregate obs dim 5, got {obs_agg_A.shape}"
    assert obs_item_A.shape == (6,), f"Expected itemized obs dim 6, got {obs_item_A.shape}"
    
    # 4. Verify exact field ordering for itemized
    # [inventory, backlog, pipeline_0, pipeline_1, last_demand, last_order]
    expected_item_A = np.array([10.0, 5.0, 20.0, 0.0, 8.0, 12.0], dtype=np.float32)
    expected_item_B = np.array([10.0, 5.0, 0.0, 20.0, 8.0, 12.0], dtype=np.float32)
    np.testing.assert_array_equal(obs_item_A, expected_item_A)
    np.testing.assert_array_equal(obs_item_B, expected_item_B)

