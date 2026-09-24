import os
import torch
import numpy as np
import pytest
from rl.phase11_trainer import Phase11Trainer

def test_phase11_initialization_and_dimensions():
    # A. Forecast dimension & B. Graph Input
    trainer = Phase11Trainer("configs/phase4_ippo.yaml", forecast_horizon=2, forecaster_type="naive")
    # Base is 5. Horizon is 2. Total should be 7.
    assert trainer.obs_dim == 7, f"Expected obs_dim 7, got {trainer.obs_dim}"
    assert trainer.global_obs_dim == 28, f"Expected global_obs_dim 28, got {trainer.global_obs_dim}"
    
    # C. GNN forward pass & D. Action generation
    obs, _ = trainer.env.reset(seed=42)
    obs_tensor = trainer.flatten_obs(obs)
    global_obs_tensor = trainer._get_global_obs(obs_tensor)
    
    assert obs_tensor.shape[1] == 7
    assert global_obs_tensor.shape[1] == 28
    
    agent = trainer.agents["retailer"]
    agent_obs = obs_tensor[0:1] # batch size 1
    agent_gobs = global_obs_tensor[0:1]
    
    act, comm_act, logprob, entropy, val = agent.get_action_and_value(agent_obs, agent_gobs)
    
    assert act is not None
    assert val.shape == (1, 1)

def test_phase11_gradient_flow():
    # E. Gradient flow
    trainer = Phase11Trainer("configs/phase4_ippo.yaml", forecast_horizon=1, forecaster_type="ma")
    
    agent = trainer.agents["retailer"]
    # Verify requires_grad
    for param in agent.critic.gnn.parameters():
        assert param.requires_grad
        
    metrics = trainer.train(total_updates=1)
    
    # After 1 update, gradients should exist and weights should be updated (handled by optimizer)
    has_grad = False
    for param in agent.critic.gnn.parameters():
        if param.grad is not None:
            has_grad = True
            break
    assert has_grad, "Gradients did not flow into the GNN parameters"
    
def test_phase11_save_load(tmp_path):
    # H. Checkpoint save/load
    trainer1 = Phase11Trainer("configs/phase4_ippo.yaml", forecast_horizon=2)
    save_path = str(tmp_path / "model.pt")
    
    # Take step
    metrics1 = trainer1.train(total_updates=1)
    trainer1.save(save_path)
    
    trainer2 = Phase11Trainer("configs/phase4_ippo.yaml", forecast_horizon=2)
    trainer2.load(save_path)
    
    # Verify weights match
    for p1, p2 in zip(trainer1.agents["retailer"].parameters(), trainer2.agents["retailer"].parameters()):
        assert torch.allclose(p1, p2)
        
def test_phase11_no_data_leakage():
    # K. No Data Leakage
    # We must ensure the forecast wrapper only uses historical demand and the trained forecaster,
    # NOT the future demand from the simulator environment itself.
    from forecasting.wrapper import DemandForecastWrapper
    from envs.supply_chain_env import SupplyChainParallelEnv
    from forecasting.models import NaiveForecaster
    import supersuit as ss
    
    env = SupplyChainParallelEnv("configs/phase1_simulator.yaml", comm_enabled=False)
    forecaster = NaiveForecaster()
    wrapped = DemandForecastWrapper(env, forecaster, forecast_horizon=3)
    
    obs, _ = wrapped.reset(seed=42)
    retailer_obs = obs["retailer"]
    # The naive forecaster just repeats the last demand. 
    # Since initial demand history is [0], forecast should be [0, 0, 0]
    np.testing.assert_array_equal(retailer_obs[-3:], [0, 0, 0])
    
    # Take a step, the simulator advances. The forecast wrapper only uses history.
    actions = {a: 5 for a in wrapped.possible_agents}
    obs, _, _, _, _ = wrapped.step(actions)
    
    # The wrapper relies strictly on `self.demand_history`. 
    # Ensure the length of demand history is only up to current step.
    assert len(wrapped.demand_history) == 2, "Demand history should only contain step 0 and 1"

def test_phase11_determinism():
    # G. Determinism
    trainer1 = Phase11Trainer("configs/phase4_ippo.yaml")
    torch.manual_seed(42)
    obs1, _ = trainer1.env.reset(seed=42)
    obs_tensor1 = trainer1.flatten_obs(obs1)
    
    trainer2 = Phase11Trainer("configs/phase4_ippo.yaml")
    torch.manual_seed(42)
    obs2, _ = trainer2.env.reset(seed=42)
    obs_tensor2 = trainer2.flatten_obs(obs2)
    
    assert torch.allclose(obs_tensor1, obs_tensor2)
