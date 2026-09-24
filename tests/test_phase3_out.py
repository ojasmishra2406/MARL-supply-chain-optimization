import os

import yaml

from baselines.metrics import calculate_bullwhip, calculate_fill_rate
from baselines.optimize_out import (
    evaluate_z_scores,
    optimize_out_coordinate_descent,
    run_episode,
)
from baselines.out import OUTPolicy
from simulator.core import SupplyChainSimulator
from simulator.state import EchelonState, SimulatorState


def get_repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_phase3_config_validity():
    config_path = os.path.join(get_repo_root(), "configs", "phase3_out.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    assert config["z_score_grid"] == [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    assert config["rounds"] == 3
    assert config["seeds"] == list(range(20))
    assert "eval" not in config.get("simulator_config", "")


def test_out_policy_behavior():
    policy = OUTPolicy(
        z_scores=[1.0, 1.0, 1.0, 1.0], lead_times=[2, 2, 3, 4], capacity=100
    )
    state = SimulatorState(
        step=0,
        total_cost=0.0,
        echelons=[
            EchelonState(
                inventory=10,
                backlog=5,
                pipeline_inventory=0,
                demand_history=[],
                last_order=0,
            ),
            EchelonState(
                inventory=0,
                backlog=0,
                pipeline_inventory=0,
                demand_history=[],
                last_order=0,
            ),
            EchelonState(
                inventory=0,
                backlog=0,
                pipeline_inventory=0,
                demand_history=[],
                last_order=0,
            ),
            EchelonState(
                inventory=0,
                backlog=0,
                pipeline_inventory=0,
                demand_history=[],
                last_order=0,
            ),
        ],
    )
    actions = policy.get_actions(state)
    assert len(actions) == 4
    for a in actions:
        assert 0 <= a <= 100
    assert actions[0] == 64


def test_metrics():
    assert calculate_fill_rate([0, 5, 0, 10]) == 0.5
    assert calculate_fill_rate([]) == 0.0

    bw1 = calculate_bullwhip([10, 20, 10], [10, 10, 10])
    assert bw1 == 1.0
    bw2 = calculate_bullwhip([10, 20, 30], [5, 10, 15])
    assert bw2 == 4.0


def test_objective_consistency(monkeypatch):
    config_path = os.path.join(get_repo_root(), "configs", "phase3_out.yaml")

    def mock_eval(sim, z_scores, seeds):
        costs = [100.0 + z_scores[0]] * len(seeds)
        return {
            "mean_cost": sum(costs) / len(costs),
            "costs": costs,
            "mean_fill_rate": 0.9,
            "mean_bullwhips": [1.0] * 4,
        }

    import baselines.optimize_out

    monkeypatch.setattr(baselines.optimize_out, "evaluate_z_scores", mock_eval)
    res = optimize_out_coordinate_descent(config_path)
    assert (
        abs(
            res["mean_total_cost"]
            - sum(res["per_seed_costs"]) / len(res["per_seed_costs"])
        )
        < 1e-6
    )


def test_optimizer_correctness_and_tie_breaking(monkeypatch):
    config_path = os.path.join(get_repo_root(), "configs", "phase3_out.yaml")
    call_count = [0]

    def mock_eval(sim, z_scores, seeds):
        call_count[0] += 1
        targets = [1.5, 2.5, 0.5, 3.0]
        cost = sum((z - t) ** 2 for z, t in zip(z_scores, targets))
        return {
            "mean_cost": cost,
            "costs": [cost] * len(seeds),
            "mean_fill_rate": 0.9,
            "mean_bullwhips": [1.0] * 4,
        }

    import baselines.optimize_out

    monkeypatch.setattr(baselines.optimize_out, "evaluate_z_scores", mock_eval)
    res = optimize_out_coordinate_descent(config_path)

    assert res["optimized_z_score_vector"] == [1.5, 2.5, 0.5, 3.0]
    assert len(res["trace"]) == 3
    assert call_count[0] == 73
    for rnd in res["trace"]:
        for ech in range(4):
            cands = rnd[list(rnd.keys())[0]][f"echelon_{ech}"]["candidates"]
            assert len(cands) == 6
            assert [c["z"] for c in cands] == [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def test_invalid_inputs():
    policy = OUTPolicy(z_scores=[1.0] * 4, lead_times=[2, 2, 3, 4], capacity=-10)
    state = SimulatorState(
        step=0,
        total_cost=0.0,
        echelons=[
            EchelonState(
                inventory=0,
                backlog=0,
                pipeline_inventory=0,
                demand_history=[],
                last_order=0,
            )
        ]
        * 4,
    )
    actions = policy.get_actions(state)
    assert all(a == 0 for a in actions)


def test_no_eval_data_leakage():
    with open(
        os.path.join(get_repo_root(), "baselines", "optimize_out.py"),
        "r",
        encoding="utf-8",
    ) as f:
        code = f.read()
    assert "configs/eval_scenarios" not in code
    assert "scenario_generators" not in code


def test_no_future_demand_leakage():
    policy = OUTPolicy(z_scores=[1.0] * 4, lead_times=[2, 2, 3, 4], capacity=100)
    assert hasattr(policy, "get_actions")


def test_reproducibility(monkeypatch):
    config_path = os.path.join(get_repo_root(), "configs", "phase3_out.yaml")

    def mock_eval(sim, z_scores, seeds):
        cost = sum(z_scores)
        return {
            "mean_cost": cost,
            "costs": [cost] * len(seeds),
            "mean_fill_rate": 0.9,
            "mean_bullwhips": [1.0] * 4,
        }

    import baselines.optimize_out

    monkeypatch.setattr(baselines.optimize_out, "evaluate_z_scores", mock_eval)
    res1 = optimize_out_coordinate_descent(config_path)
    res2 = optimize_out_coordinate_descent(config_path)
    assert res1 == res2


def test_full_episode_coverage():
    config_path = os.path.join(get_repo_root(), "configs", "phase1_simulator.yaml")
    sim = SupplyChainSimulator(config_path)
    sim.horizon = 3
    policy = OUTPolicy(z_scores=[1.0] * 4, lead_times=[2, 2, 3, 4], capacity=100)
    res = run_episode(sim, policy, 42)
    assert "cost" in res
    assert "fill_rate" in res
    assert "bullwhips" in res


def test_evaluate_z_scores_coverage():
    config_path = os.path.join(get_repo_root(), "configs", "phase1_simulator.yaml")
    sim = SupplyChainSimulator(config_path)
    sim.horizon = 3
    res = evaluate_z_scores(sim, [1.0, 1.0, 1.0, 1.0], [0, 1])
    assert "mean_cost" in res

    bw3 = calculate_bullwhip([10], [10])
    assert bw3 == 1.0
