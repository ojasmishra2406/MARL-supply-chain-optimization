import torch
import hashlib
import json
import os
import io

def _hash_state_dict(state_dict):
    """Deterministically serialize state_dict and return SHA-256 hash."""
    hash_obj = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        hash_obj.update(key.encode('utf-8'))
        tensor = state_dict[key]
        if isinstance(tensor, torch.Tensor):
            hash_obj.update(tensor.cpu().numpy().tobytes())
        elif isinstance(tensor, dict):
            # Recursively hash nested dicts
            hash_obj.update(_hash_state_dict(tensor).encode('utf-8'))
        else:
            hash_obj.update(str(tensor).encode('utf-8'))
            
    return hash_obj.hexdigest()

def save_checkpoint(state_dict, checkpoint_path, manifest_path, extra_metadata=None):
    """Save checkpoint and write manifest with SHA-256."""
    torch.save(state_dict, checkpoint_path)
    
    sha256_hash = _hash_state_dict(state_dict)
    
    manifest = {
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": sha256_hash,
    }
    if extra_metadata:
        manifest.update(extra_metadata)
        
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
        
    return sha256_hash

def load_checkpoint(checkpoint_path, manifest_path):
    """Load checkpoint and verify SHA-256 against manifest."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint file: {checkpoint_path}")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Missing manifest file: {manifest_path}")
        
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    expected_hash = manifest.get("checkpoint_sha256")
    if not expected_hash:
        raise ValueError(f"Incompatible checkpoint manifest: missing checkpoint_sha256 in {manifest_path}")
        
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    actual_hash = _hash_state_dict(state_dict)
    
    if actual_hash != expected_hash:
        raise ValueError(f"Corrupted checkpoint detected! Expected SHA-256: {expected_hash}, Actual: {actual_hash}")
        
    return state_dict
