import pytest
import os
import json
import torch
from rl.phase7_evaluator import evaluate_scenario
from rl.checkpoint import save_checkpoint

SCENARIOS = [
    "in_distribution",
    "demand_shift",
    "lead_time_shift",
    "capacity_disruption",
    "demand_spike",
    "combined_shift"
]

@pytest.fixture
def mock_checkpoint(tmp_path):
    ckpt_path = str(tmp_path / "mock.pt")
    manifest_path = str(tmp_path / "mock_manifest.json")
    
    # We need to save a state dict matching Phase5Agent
    # But Phase5Agent has many weights.
    # We can just create one, extract state_dict, and save it.
    from rl.mappo_trainer import Phase5Agent
    agent = Phase5Agent(5, 5, 101, comm_dim=0, centralized_critic=False)
    state_dict = {"retailer": agent.state_dict(), "wholesaler": agent.state_dict(), "distributor": agent.state_dict(), "manufacturer": agent.state_dict()}
    
    save_checkpoint(state_dict, ckpt_path, manifest_path)
    return ckpt_path, manifest_path

@pytest.mark.parametrize("scenario", SCENARIOS)
def test_evaluate_out_baseline(scenario):
    res = evaluate_scenario("out", None, "configs/phase4_ippo.yaml", scenario, seed=42)
    assert res["cost"] > 0
    assert 0 <= res["fill_rate"] <= 1.0
    assert res["bullwhip"] >= 0

def test_evaluate_rl_policy(mock_checkpoint):
    ckpt_path, manifest_path = mock_checkpoint
    res = evaluate_scenario("ippo", ckpt_path, "configs/phase4_ippo.yaml", "in_distribution", seed=42, manifest_path=manifest_path)
    assert res["cost"] > 0
    assert res["checkpoint_hash"] is not None
    assert res["scenario_hash"] is not None

def test_invalid_checkpoint(mock_checkpoint):
    ckpt_path, manifest_path = mock_checkpoint
    # Corrupt manifest
    with open(manifest_path, "w") as f:
        json.dump({"checkpoint_sha256": "wrong_hash"}, f)
        
    with pytest.raises(ValueError, match="Corrupted checkpoint detected"):
        evaluate_scenario("ippo", ckpt_path, "configs/phase4_ippo.yaml", "in_distribution", seed=42, manifest_path=manifest_path)

def test_reproducibility(mock_checkpoint):
    ckpt_path, manifest_path = mock_checkpoint
    res1 = evaluate_scenario("ippo", ckpt_path, "configs/phase4_ippo.yaml", "demand_shift", seed=100, manifest_path=manifest_path)
    res2 = evaluate_scenario("ippo", ckpt_path, "configs/phase4_ippo.yaml", "demand_shift", seed=100, manifest_path=manifest_path)
    
    assert res1["cost"] == res2["cost"]
    assert res1["fill_rate"] == res2["fill_rate"]
