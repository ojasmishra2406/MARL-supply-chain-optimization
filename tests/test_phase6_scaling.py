import pytest
import numpy as np
import supersuit as ss
from envs.supply_chain_env import SupplyChainParallelEnv

def test_supersuit_vectorization():
    env = SupplyChainParallelEnv('configs/phase1_simulator.yaml')
    
    # SuperSuit conversion
    vec_env = ss.pettingzoo_env_to_vec_env_v1(env)
    assert vec_env.num_envs == 4
    
    # Concatenate environments (e.g., scale to 2 parallel envs)
    concat_env = ss.concat_vec_envs_v1(vec_env, 2)
    assert concat_env.num_envs == 8
    
    # Check observation shape and action shape
    obs, info = concat_env.reset(seed=42)
    assert len(obs) == 8
    
    # Observations should be identical between the two environments if seeded identically
    # Env 0 and Env 1 are copies, but wait, concat_env resets with a single seed or multiple?
    # Actually, PettingZoo supersuit wraps step and reset.
    
    # Take a step
    actions = [0] * 8
    obs, rews, terms, truncs, infos = concat_env.step(actions)
    
    assert len(obs) == 8
    assert len(rews) == 8
    assert len(terms) == 8
    assert len(truncs) == 8
    assert len(infos) == 8

def test_supersuit_with_communication():
    env = SupplyChainParallelEnv('configs/phase1_simulator.yaml', comm_enabled=True)
    
    vec_env = ss.pettingzoo_env_to_vec_env_v1(env)
    concat_env = ss.concat_vec_envs_v1(vec_env, 2)
    
    obs, info = concat_env.reset()
    assert isinstance(obs, dict)
    assert len(obs['state']) == 8
    
    # Take a step
    # Actions must be dicts for communication.
    actions = [{"action": 0, "message": np.zeros(4, dtype=np.float32)} for _ in range(8)]
    # Actually, SuperSuit also expects dict of batched actions if action space is Dict.
    # Let's check what action space is.
    # SupplyChainParallelEnv action_space is Dict if comm_enabled.
    # So actions should be passed as dict of batched actions.
    batched_actions = {
        "action": np.zeros(8, dtype=np.int64),
        "message": np.zeros((8, 4), dtype=np.float32)
    }
    obs, rews, terms, truncs, infos = concat_env.step(batched_actions)
    assert len(rews) == 8
