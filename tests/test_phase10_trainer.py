import os
import torch
import pytest
import tempfile
from rl.phase10_trainer import Phase10Trainer

def test_phase10_basic_training_step():
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Phase10Trainer(config_path="configs/phase4_ippo.yaml", ablation_config={})
        # Ensure it can do 1 update safely
        metrics = trainer.train(total_updates=1)
        assert "retailer_reward" in metrics
        assert metrics["retailer_reward"] != 0

def test_phase10_checkpoint_save_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Phase10Trainer(config_path="configs/phase4_ippo.yaml", ablation_config={})
        metrics = trainer.train(total_updates=1)
        
        ckpt_path = os.path.join(tmpdir, "test_ckpt.pt")
        # Save manually as done in scripts
        state = {name: agent.state_dict() for name, agent in trainer.agents.items()}
        torch.save(state, ckpt_path)
        
        # Load
        new_trainer = Phase10Trainer(config_path="configs/phase4_ippo.yaml", ablation_config={})
        loaded_state = torch.load(ckpt_path)
        for name, agent in new_trainer.agents.items():
            agent.load_state_dict(loaded_state[name])
            
        # Verify equality
        for name in trainer.agents.keys():
            orig = list(trainer.agents[name].parameters())[0]
            loaded = list(new_trainer.agents[name].parameters())[0]
            assert torch.allclose(orig, loaded)

def test_phase10_failure_mode_invalid_config():
    with pytest.raises(Exception):
        # Pass a missing config
        Phase10Trainer(config_path="configs/nonexistent.yaml")
