import pytest
import os
from rl.phase6_trainer import Phase6Trainer

def test_phase6_trainer_initialization():
    ablation = {"centralized_critic": True, "communication": False}
    trainer = Phase6Trainer("configs/phase4_ippo.yaml", ablation)
    assert trainer.num_envs == 8
    
    # We shouldn't do a full train in tests to keep it fast, but 1 update is fine
    metrics = trainer.train(1, use_wandb=False)
    assert "retailer_policy_loss" in metrics
    
def test_phase6_trainer_communication():
    ablation = {"centralized_critic": True, "communication": True}
    trainer = Phase6Trainer("configs/phase4_ippo.yaml", ablation)
    metrics = trainer.train(1, use_wandb=False)
    assert "retailer_policy_loss" in metrics

def test_phase6_trainer_shared_params():
    ablation = {"centralized_critic": False, "communication": False, "parameter_sharing": True}
    trainer = Phase6Trainer("configs/phase4_ippo.yaml", ablation)
    metrics = trainer.train(1, use_wandb=False)
    assert "retailer_policy_loss" in metrics
