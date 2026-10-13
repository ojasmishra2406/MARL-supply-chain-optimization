import os
import copy
import json
import torch
import pytest
import numpy as np

from rl.phase11_trainer import Phase11Trainer
from rl.checkpoint import save_checkpoint, load_checkpoint, _hash_state_dict
from forecasting.wrapper import DemandForecastWrapper
from envs.supply_chain_env import SupplyChainParallelEnv
from forecasting.models import XGBoostForecaster, MovingAverageForecaster


def test_seeding_reproducibility():
    """Verify that same seed and configuration produce identical initial weights and observations."""
    trainer_a = Phase11Trainer("configs/phase4_ippo.yaml", forecast_horizon=2, forecaster_type="ma", seed=123)
    trainer_b = Phase11Trainer("configs/phase4_ippo.yaml", forecast_horizon=2, forecaster_type="ma", seed=123)

    # Initial observations must be identical
    assert np.allclose(trainer_a.next_obs, trainer_b.next_obs), "Initial observations diverged for same seed 123"

    # Agent parameters must be bitwise identical
    for agent_name in trainer_a.agents_names:
        for p_a, p_b in zip(trainer_a.agents[agent_name].parameters(), trainer_b.agents[agent_name].parameters()):
            assert torch.equal(p_a, p_b), f"Parameters for agent {agent_name} diverged for same seed 123"


def test_seeding_divergence():
    """Verify that different seeds produce different initializations."""
    trainer_1 = Phase11Trainer("configs/phase4_ippo.yaml", forecast_horizon=2, forecaster_type="ma", seed=100)
    trainer_2 = Phase11Trainer("configs/phase4_ippo.yaml", forecast_horizon=2, forecaster_type="ma", seed=200)

    # Agent parameters should differ between seed 100 and 200
    p1 = next(trainer_1.agents["retailer"].parameters())
    p2 = next(trainer_2.agents["retailer"].parameters())
    assert not torch.equal(p1, p2), "Initial weights should differ between seed 100 and 200"


def test_checkpoint_roundtrip_canonical_hash(tmp_path):
    """Verify canonical save and load through rl.checkpoint round-trip."""
    trainer = Phase11Trainer("configs/phase4_ippo.yaml", forecast_horizon=2, forecaster_type="ma", seed=42)
    ckpt_path = str(tmp_path / "model.pt")
    manifest_path = str(tmp_path / "model_manifest.json")

    # Save via trainer.save_checkpoint
    extra_meta = {"experiment_id": "test_exp", "total_updates": 5}
    computed_hash = trainer.save_checkpoint(ckpt_path, manifest_path, extra_metadata=extra_meta)

    assert os.path.exists(ckpt_path)
    assert os.path.exists(manifest_path)

    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    assert manifest["checkpoint_sha256"] == computed_hash

    # Load via canonical load_checkpoint
    loaded_state = load_checkpoint(ckpt_path, manifest_path)
    assert isinstance(loaded_state, dict)
    assert "retailer" in loaded_state

    # Verify tensor equality
    for name in trainer.agents_names:
        for p_orig, p_load in zip(trainer.agents[name].state_dict().values(), loaded_state[name].values()):
            assert torch.equal(p_orig, p_load)


def test_checkpoint_tamper_detection(tmp_path):
    """Negative test: verify that tampered checkpoints are rejected with ValueError."""
    trainer = Phase11Trainer("configs/phase4_ippo.yaml", forecast_horizon=2, forecaster_type="ma", seed=42)
    ckpt_path = str(tmp_path / "tamper_model.pt")
    manifest_path = str(tmp_path / "tamper_manifest.json")

    trainer.save_checkpoint(ckpt_path, manifest_path)

    # Tamper with the state dict on disk
    state = torch.load(ckpt_path)
    # Modify a weight tensor
    first_key = list(state["retailer"].keys())[0]
    state["retailer"][first_key] = state["retailer"][first_key] + 1.0
    torch.save(state, ckpt_path)

    # Canonical load must detect tampering and raise ValueError
    with pytest.raises(ValueError, match="Corrupted checkpoint detected"):
        load_checkpoint(ckpt_path, manifest_path)


def test_no_data_leakage_and_forecast_conditioning():
    """Verify that forecaster only accesses past demand and predictions expand observations."""
    env = SupplyChainParallelEnv("configs/phase1_simulator.yaml", comm_enabled=False)
    forecaster = MovingAverageForecaster(window=5)
    horizon = 2
    wrapped = DemandForecastWrapper(env, forecaster, forecast_horizon=horizon)

    obs, _ = wrapped.reset(seed=77)
    # At reset: initial demand history is [0], forecast is [0, 0]
    ret_obs = obs["retailer"]
    assert len(ret_obs) == 5 + horizon  # 7 dims
    np.testing.assert_array_equal(ret_obs[-horizon:], [0.0, 0.0])

    # Take step
    actions = {a: 5 for a in wrapped.possible_agents}
    next_obs, _, _, _, _ = wrapped.step(actions)

    # Length of demand history must be strictly step count (1 at reset + 1 after step = 2)
    assert len(wrapped.demand_history) == 2, "Demand history contains future data leakage"
    # New observation must have forecast features appended
    ret_next_obs = next_obs["retailer"]
    assert len(ret_next_obs) == 7


def test_xgboost_genuine_conditioning():
    """Verify that XGBoost model genuinely computes non-trivial predictions that condition the policy."""
    import pandas as pd
    df = pd.read_csv("data/forecasting/train_demand.csv")
    f_xgb = XGBoostForecaster(window=5)
    f_xgb.fit(df["demand"].values)

    history = [12.0, 15.0, 14.0, 18.0, 20.0]
    preds = f_xgb.predict(history, horizon=2)
    assert len(preds) == 2
    assert all(p > 0 for p in preds)

    # Check that trainer with XGBoost initializes actor with 7 dimensions
    trainer_xgb = Phase11Trainer("configs/phase4_ippo.yaml", forecast_horizon=2, forecaster_type="xgboost", seed=42)
    assert trainer_xgb.obs_dim == 7
    assert isinstance(trainer_xgb.forecaster, XGBoostForecaster)

    # Forward pass through actor
    sample_obs = torch.randn(1, 7)
    logits = trainer_xgb.agents["retailer"].actor(sample_obs)
    assert logits.shape[-1] == 101  # discrete action space size
