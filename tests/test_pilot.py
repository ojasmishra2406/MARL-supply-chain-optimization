import os

import numpy as np
import pytest

from baselines.out import OUTPolicy
from rl.pilot import PilotTrainer
from rl.pilot_env import PilotOneEchelonEnv
from simulator.core import SupplyChainSimulator


def test_one_echelon_construction():
    env = PilotOneEchelonEnv()
    assert len(env.possible_agents) == 1
    assert env.possible_agents[0] == "retailer"


def test_determinism():
    env1 = PilotOneEchelonEnv()
    env1.reset(seed=42)
    obs1, _, _, _, _ = env1.step({"retailer": 10})

    env2 = PilotOneEchelonEnv()
    env2.reset(seed=42)
    obs2, _, _, _, _ = env2.step({"retailer": 10})

    assert np.allclose(obs1["retailer"], obs2["retailer"])


def test_no_future_demand():
    env = PilotOneEchelonEnv()
    obs, _ = env.reset()
    # obs only has 5 dims: inv, back, pipeline, last_demand, last_order
    assert obs["retailer"].shape == (5,)
    with open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "rl", "pilot_env.py"
        )
    ) as f:
        assert "future" not in f.read()


def test_action_validity():
    env = PilotOneEchelonEnv()
    assert env.action_space("retailer").n == 101  # 0 to 100


def test_lead_time():
    env = PilotOneEchelonEnv()
    env.reset(seed=0)
    env.step({"retailer": 50})
    assert sum(env.pipeline) == 50
    env.step({"retailer": 0})
    env.step({"retailer": 0})
    assert sum(env.pipeline) == 0
    assert env.inventory >= 50 or env.backlog < 50


def test_cost_accounting():
    env = PilotOneEchelonEnv()
    env.reset(seed=0)
    env.inventory = 10
    env.backlog = 5
    _, rews, _, _, _ = env.step({"retailer": 20})
    assert "retailer" in rews


def test_out_compatibility():
    env = PilotOneEchelonEnv()
    env.reset(seed=0)
    policy = OUTPolicy([0.5], [2], 100)
    state = env.get_fake_simulator_state()
    actions = policy.get_actions(state)
    assert len(actions) == 1


def test_ppo_compatibility():
    import yaml

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(repo_root, "configs", "phase4_ippo.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config["simulator_config"] = "configs/phase1_simulator.yaml"
    pilot_config = os.path.join(
        repo_root, "configs", "phase4_1_echelon_pilot_test.yaml"
    )
    with open(pilot_config, "w") as f:
        yaml.dump(config, f)
    trainer = PilotTrainer(pilot_config)
    assert trainer.vec_env.envs[0].possible_agents == ["retailer"]


def test_critical_isolation():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    import yaml

    config_path = os.path.join(repo_root, "configs", "phase1_simulator.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config["echelons"] = ["retailer"]
    config["lead_time"] = [2]
    config["num_echelons"] = 1
    test_config = os.path.join(repo_root, "configs", "test_1_echelon.yaml")
    with open(test_config, "w") as f:
        yaml.dump(config, f)

    with pytest.raises(ValueError, match="Expected exactly 4 modeled echelons"):
        SupplyChainSimulator(test_config)


def test_pilot_smoke():
    from rl.pilot import run_pilot

    run_pilot(updates=1, seeds=[0])


def test_pilot_artifact():
    import json
    import os

    artifact_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "results",
        "pilot_convergence.json",
    )
    if os.path.exists(artifact_path):
        with open(artifact_path, "r") as f:
            data = json.load(f)
            assert "out_reference_aggregate" in data
            for seed in [0, 1, 2]:
                if str(seed) in data["seeds"]:
                    seed_data = data["seeds"][str(seed)]
                    assert "initial_cost" in seed_data
                    assert "final_cost" in seed_data
                    assert "trajectory" in seed_data
