import pytest
import torch
import json
import os
import yaml
from rl.mappo_trainer import Phase5Agent, Phase5Trainer
from envs.supply_chain_env import SupplyChainParallelEnv
from rl.phase5_evaluator import evaluate_phase5_policy

def test_centralized_critic_vs_decentralized_actor():
    obs_dim = 5
    global_obs_dim = 20
    action_dim = 101
    
    agent_mappo = Phase5Agent(obs_dim, global_obs_dim, action_dim, centralized_critic=True)
    assert agent_mappo.actor.net[0].in_features == obs_dim
    assert agent_mappo.critic.net[0].in_features == global_obs_dim
    
    agent_ippo = Phase5Agent(obs_dim, global_obs_dim, action_dim, centralized_critic=False)
    assert agent_ippo.actor.net[0].in_features == obs_dim
    assert agent_ippo.critic.net[0].in_features == obs_dim

def test_parameter_sharing_switch():
    ablation_shared = {"parameter_sharing": True}
    trainer_shared = Phase5Trainer("configs/phase4_ippo.yaml", ablation_shared)
    # Check if all agents point to the same object
    agent_0 = trainer_shared.agents[trainer_shared.agents_names[0]]
    agent_1 = trainer_shared.agents[trainer_shared.agents_names[1]]
    assert agent_0 is agent_1
    
    ablation_indep = {"parameter_sharing": False}
    trainer_indep = Phase5Trainer("configs/phase4_ippo.yaml", ablation_indep)
    agent_0_indep = trainer_indep.agents[trainer_indep.agents_names[0]]
    agent_1_indep = trainer_indep.agents[trainer_indep.agents_names[1]]
    assert agent_0_indep is not agent_1_indep

def test_communication_switch():
    obs_dim = 13
    global_obs_dim = 52
    action_dim = 101
    comm_dim = 4
    
    agent_comm = Phase5Agent(obs_dim, global_obs_dim, action_dim, comm_dim=comm_dim)
    assert hasattr(agent_comm, "comm_net")
    assert agent_comm.comm_net[-1].out_features == comm_dim
    
    agent_no_comm = Phase5Agent(5, 20, action_dim, comm_dim=0)
    assert not hasattr(agent_no_comm, "comm_net")

def test_reward_coefficients():
    # Make sure we can set them in config and it gets to the env
    ablation = {"reward_weights": {"holding": 1, "backlog": 5, "ordering": 0.5}}
    trainer = Phase5Trainer("configs/phase4_ippo.yaml", ablation)
    assert trainer.vec_env.envs[0].simulator.cost_backlog == 5
    assert trainer.vec_env.envs[0].simulator.cost_holding == 1

def test_divergence_diagnostics():
    # Train 1 update with divergence config and ensure metrics are returned
    ablation = {"learning_rate": 3e-3, "reward_scale": 1.0}
    trainer = Phase5Trainer("configs/phase4_ippo.yaml", ablation)
    metrics = trainer.train(1, use_wandb=False)
    assert "retailer_reward" in metrics
    assert "retailer_policy_loss" in metrics
    assert "retailer_value_loss" in metrics
    assert "retailer_entropy" in metrics
    assert "retailer_clip_frac" in metrics
    assert "retailer_explained_var" in metrics

def test_checkpoints_save_load(tmp_path):
    ablation = {"centralized_critic": True, "seed": 42}
    trainer = Phase5Trainer("configs/phase4_ippo.yaml", ablation)
    ckpt_path = tmp_path / "test.pt"
    
    state_dicts = {a: trainer.agents[a].state_dict() for a in trainer.agents_names}
    torch.save(state_dicts, ckpt_path)
    
    for p in trainer.agents["retailer"].parameters():
        p.data.fill_(0)
        
    loaded = torch.load(ckpt_path)
    trainer.agents["retailer"].load_state_dict(loaded["retailer"])
    assert torch.sum(list(trainer.agents["retailer"].parameters())[0]).item() != 0

def test_full_training_execution():
    # Hit parameter sharing and comms logic in training
    ablation_shared = {"parameter_sharing": True, "communication": True}
    trainer_shared = Phase5Trainer("configs/phase4_ippo.yaml", ablation_shared)
    metrics_shared = trainer_shared.train(1, use_wandb=False)
    assert "retailer_reward" in metrics_shared
    
    # Hit evaluator logic
    res = evaluate_phase5_policy(trainer_shared.agents, "configs/phase4_ippo.yaml", ablation_shared, seeds=[0], num_episodes_per_seed=1, deterministic=False)
    assert "mean_cost" in res


def test_deterministic_behavior():
    ablation = {"seed": 42}
    trainer1 = Phase5Trainer("configs/phase4_ippo.yaml", ablation)
    res1 = evaluate_phase5_policy(trainer1.agents, "configs/phase4_ippo.yaml", ablation, seeds=[0], num_episodes_per_seed=1, deterministic=True)
    
    trainer2 = Phase5Trainer("configs/phase4_ippo.yaml", ablation)
    res2 = evaluate_phase5_policy(trainer2.agents, "configs/phase4_ippo.yaml", ablation, seeds=[0], num_episodes_per_seed=1, deterministic=True)
    
    assert res1["mean_cost"] == res2["mean_cost"]

def test_no_eval_imports():
    with open("rl/mappo_trainer.py", "r") as f:
        content = f.read()
    assert "eval_scenarios" not in content
    
    with open("run_phase5.py", "r") as f:
        content = f.read()
    assert "eval_scenarios" not in content
