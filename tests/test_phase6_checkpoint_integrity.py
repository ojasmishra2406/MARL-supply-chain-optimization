import os
import torch
import pytest
from rl.checkpoint import save_checkpoint, load_checkpoint

def test_checkpoint_integrity(tmp_path):
    ckpt_path = str(tmp_path / "model.pt")
    manifest_path = str(tmp_path / "manifest.json")
    
    # 1. Valid Checkpoint
    state_dict = {
        "layer1.weight": torch.randn(10, 10),
        "layer1.bias": torch.randn(10)
    }
    
    hash1 = save_checkpoint(state_dict, ckpt_path, manifest_path, {"config": "test"})
    assert hash1 is not None
    
    loaded_state_dict = load_checkpoint(ckpt_path, manifest_path)
    assert torch.equal(loaded_state_dict["layer1.weight"], state_dict["layer1.weight"])
    
    # 2. Corrupted Checkpoint
    # Modify the saved checkpoint directly to simulate corruption
    corrupted_state_dict = state_dict.copy()
    corrupted_state_dict["layer1.bias"] = torch.randn(10)
    torch.save(corrupted_state_dict, ckpt_path)
    
    with pytest.raises(ValueError, match="Corrupted checkpoint detected"):
        load_checkpoint(ckpt_path, manifest_path)
        
    # 3. Missing Checkpoint
    os.remove(ckpt_path)
    with pytest.raises(FileNotFoundError, match="Missing checkpoint file"):
        load_checkpoint(ckpt_path, manifest_path)
        
    # 4. Incompatible Checkpoint (Missing Hash in Manifest)
    torch.save(state_dict, ckpt_path)
    import json
    with open(manifest_path, 'w') as f:
        json.dump({"checkpoint_path": ckpt_path}, f) # Missing hash
        
    with pytest.raises(ValueError, match="Incompatible checkpoint manifest"):
        load_checkpoint(ckpt_path, manifest_path)
