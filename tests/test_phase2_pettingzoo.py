import json
import os

import numpy as np
import pytest
from pettingzoo.test import parallel_api_test

from envs.supply_chain_env import SupplyChainParallelEnv


def get_config_path():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(repo_root, "configs", "phase1_simulator.yaml")


def test_pettingzoo_api_compliance():
    env = SupplyChainParallelEnv(get_config_path())
    parallel_api_test(env, num_cycles=100)


def test_agent_set_and_ordering():
    env = SupplyChainParallelEnv(get_config_path())
    expected = ["retailer", "wholesaler", "distributor", "manufacturer"]
    assert env.possible_agents == expected
    assert env.agents == expected


def test_action_and_observation_spaces():
    env = SupplyChainParallelEnv(get_config_path())
    for agent in env.possible_agents:
        act_space = env.action_space(agent)
        # Should be Discrete(101) for capacity 100
        assert act_space.n == 101

        obs_space = env.observation_space(agent)
        assert obs_space.shape == (5,)


def test_reset_and_step_semantics():
    env = SupplyChainParallelEnv(get_config_path())
    obs, infos = env.reset(seed=42)
    assert set(obs.keys()) == set(env.possible_agents)
    assert set(infos.keys()) == set(env.possible_agents)

    # 1 step
    actions = {a: 10 for a in env.possible_agents}
    next_obs, rewards, term, trunc, next_infos = env.step(actions)

    assert set(next_obs.keys()) == set(env.possible_agents)
    assert set(rewards.keys()) == set(env.possible_agents)
    assert set(term.keys()) == set(env.possible_agents)
    assert set(trunc.keys()) == set(env.possible_agents)
    assert set(next_infos.keys()) == set(env.possible_agents)

    assert not any(term.values())
    assert not any(trunc.values())


def test_horizon_truncation():
    env = SupplyChainParallelEnv(get_config_path())
    env.reset()
    horizon = env.horizon
    actions = {a: 0 for a in env.possible_agents}

    for _ in range(horizon - 1):
        _, _, term, trunc, _ = env.step(actions)
        assert not any(trunc.values())

    # Step `horizon`
    _, _, term, trunc, _ = env.step(actions)
    assert all(trunc.values())
    assert not any(term.values())
    assert len(env.agents) == 0


def test_reward_generation():
    env = SupplyChainParallelEnv(get_config_path())
    env.reset(seed=42)
    # Retailer gets 0 inventory, generates demand
    actions = {"retailer": 10, "wholesaler": 0, "distributor": 0, "manufacturer": 0}
    _, rewards, _, _, _ = env.step(actions)

    assert rewards["retailer"] < 0
    # Wholesaler gets order of 10 from retailer, so its backlog becomes 10. Cost = 10 * 2.0 = 20
    assert rewards["wholesaler"] == -20.0
    # Distributor gets 0 order from wholesaler
    assert rewards["distributor"] == 0
    assert rewards["manufacturer"] == 0


def test_observation_isolation():
    env = SupplyChainParallelEnv(get_config_path())
    env.reset(seed=42)

    # Change manufacturer inventory
    env.simulator.state.echelons[3].inventory = 500

    # Get obs
    obs = env._get_observations()

    # Retailer obs should have 0 inventory
    assert obs["retailer"][0] == 0.0
    # Manufacturer obs should have 500 inventory
    assert obs["manufacturer"][0] == 500.0


def test_same_seed_determinism():
    env1 = SupplyChainParallelEnv(get_config_path())
    env2 = SupplyChainParallelEnv(get_config_path())

    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)

    def serialize_obs(obs):
        return json.dumps({k: v.tolist() for k, v in obs.items()}, sort_keys=True)

    assert serialize_obs(obs1) == serialize_obs(obs2)

    actions = {a: 10 for a in env1.possible_agents}
    next_obs1, rewards1, _, _, _ = env1.step(actions)
    next_obs2, rewards2, _, _, _ = env2.step(actions)

    assert serialize_obs(next_obs1) == serialize_obs(next_obs2)
    assert json.dumps(rewards1, sort_keys=True) == json.dumps(rewards2, sort_keys=True)


def test_different_seed_behavior():
    env1 = SupplyChainParallelEnv(get_config_path())
    env2 = SupplyChainParallelEnv(get_config_path())

    obs1, _ = env1.reset(seed=1)
    obs2, _ = env2.reset(seed=999)

    actions = {a: 0 for a in env1.possible_agents}
    next_obs1, _, _, _, _ = env1.step(actions)
    next_obs2, _, _, _, _ = env2.step(actions)

    # Next obs should differ due to different demand generation
    def serialize_obs(obs):
        return json.dumps({k: v.tolist() for k, v in obs.items()}, sort_keys=True)

    assert serialize_obs(next_obs1) != serialize_obs(next_obs2)


def test_comm_spaces():
    env = SupplyChainParallelEnv(get_config_path(), comm_enabled=True, comm_dim=4)
    for agent in env.possible_agents:
        act_space = env.action_space(agent)
        assert "action" in act_space.spaces
        assert "message" in act_space.spaces
        assert act_space["message"].shape == (4,)

        obs_space = env.observation_space(agent)
        assert "state" in obs_space.spaces
        assert "message_upstream" in obs_space.spaces
        assert "message_downstream" in obs_space.spaces


def test_comm_delay_and_topology():
    env = SupplyChainParallelEnv(get_config_path(), comm_enabled=True, comm_dim=4)
    obs, _ = env.reset(seed=42)

    # t=0 observations: no messages
    for agent in env.possible_agents:
        assert np.allclose(obs[agent]["message_upstream"], np.zeros(4))
        assert np.allclose(obs[agent]["message_downstream"], np.zeros(4))

    # Retailer sends msg1, Wholesaler msg2, Dist msg3, Mfg msg4
    msg_r = np.array([1, 1, 1, 1], dtype=np.float32)
    msg_w = np.array([2, 2, 2, 2], dtype=np.float32)
    msg_d = np.array([3, 3, 3, 3], dtype=np.float32)
    msg_m = np.array([4, 4, 4, 4], dtype=np.float32)

    actions = {
        "retailer": {"action": 0, "message": msg_r},
        "wholesaler": {"action": 0, "message": msg_w},
        "distributor": {"action": 0, "message": msg_d},
        "manufacturer": {"action": 0, "message": msg_m},
    }

    obs, _, _, _, _ = env.step(actions)

    # Topology:
    # Retailer receives msg_w on message_upstream
    assert np.allclose(obs["retailer"]["message_upstream"], msg_w)
    assert np.allclose(
        obs["retailer"]["message_downstream"], np.zeros(4)
    )  # No downstream of retailer

    # Wholesaler receives msg_r on message_downstream, msg_d on message_upstream
    assert np.allclose(obs["wholesaler"]["message_downstream"], msg_r)
    assert np.allclose(obs["wholesaler"]["message_upstream"], msg_d)

    # Distributor receives msg_w on message_downstream, msg_m on message_upstream
    assert np.allclose(obs["distributor"]["message_downstream"], msg_w)
    assert np.allclose(obs["distributor"]["message_upstream"], msg_m)

    # Manufacturer receives msg_d on message_downstream
    assert np.allclose(obs["manufacturer"]["message_downstream"], msg_d)
    assert np.allclose(
        obs["manufacturer"]["message_upstream"], np.zeros(4)
    )  # No upstream of mfg


def test_performance_sanity():
    import time

    env = SupplyChainParallelEnv(get_config_path())
    env.reset(seed=42)
    actions = {a: 0 for a in env.possible_agents}

    start = time.perf_counter()
    for _ in range(100):
        _, _, _, trunc, _ = env.step(actions)
        if trunc and trunc.get("retailer"):
            env.reset()
    elapsed = time.perf_counter() - start

    steps_per_sec = 100 / elapsed
    print(f"Env Performance: {steps_per_sec:.2f} steps/sec")
    assert steps_per_sec > 1000  # meaningful sanity limit


def test_invalid_inputs():
    env = SupplyChainParallelEnv(get_config_path())
    env.reset()
    assert not env.action_space("retailer").contains(-1)
    with pytest.raises(ValueError):
        env.step({"retailer": -1})
