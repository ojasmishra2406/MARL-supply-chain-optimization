import os
import io
import json
import hashlib
import torch

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

def compute_file_sha256(filepath):
    """Compute physical-file SHA-256 hash."""
    if not os.path.exists(filepath):
        return None
    hash_obj = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()

def save_checkpoint(state_dict, checkpoint_path, manifest_path, extra_metadata=None):
    """Save checkpoint and write manifest with both canonical and physical SHA-256."""
    os.makedirs(os.path.dirname(checkpoint_path) if os.path.dirname(checkpoint_path) else ".", exist_ok=True)
    os.makedirs(os.path.dirname(manifest_path) if os.path.dirname(manifest_path) else ".", exist_ok=True)
    
    torch.save(state_dict, checkpoint_path)
    
    canonical_hash = _hash_state_dict(state_dict)
    physical_hash = compute_file_sha256(checkpoint_path)
    
    manifest = {
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": canonical_hash,
        "canonical_state_dict_sha256": canonical_hash,
        "physical_file_sha256": physical_hash,
        "hash_algorithm": "canonical_state_dict_sha256",
    }
    if extra_metadata:
        manifest.update(extra_metadata)
        
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
        
    return canonical_hash

def verify_checkpoint_integrity(checkpoint_path, manifest_path, expected_agents=None, allow_historical_physical=True):
    """
    Audits checkpoint and manifest integrity without modifying files.
    Returns a structured verification report dict.
    """
    if not os.path.exists(checkpoint_path):
        return {
            "is_valid": False,
            "status": "MISSING_CHECKPOINT",
            "error_message": f"Missing checkpoint file: {checkpoint_path}"
        }
    if not os.path.exists(manifest_path):
        return {
            "is_valid": False,
            "status": "MISSING_MANIFEST",
            "error_message": f"Missing manifest file: {manifest_path}"
        }
        
    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
    except Exception as e:
        return {
            "is_valid": False,
            "status": "MALFORMED_MANIFEST",
            "error_message": f"Failed to parse manifest JSON: {e}"
        }
        
    expected_hash = manifest.get("checkpoint_sha256") or manifest.get("canonical_state_dict_sha256")
    if not expected_hash and "checkpoint_hashes" in manifest:
        basename = os.path.basename(checkpoint_path)
        for s_key, s_hash in manifest["checkpoint_hashes"].items():
            if f"seed{s_key}" in basename or manifest.get("checkpoints", {}).get(s_key) == checkpoint_path:
                expected_hash = s_hash
                break
        if not expected_hash and len(manifest["checkpoint_hashes"]) > 0:
            expected_hash = list(manifest["checkpoint_hashes"].values())[0]
            
    if not expected_hash:
        return {
            "is_valid": False,
            "status": "INCOMPATIBLE_MANIFEST",
            "error_message": f"Incompatible checkpoint manifest: missing checkpoint_sha256 in {manifest_path}"
        }
        
    physical_hash = compute_file_sha256(checkpoint_path)
    expected_physical = manifest.get("physical_file_sha256")
    
    try:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    except Exception as e:
        return {
            "is_valid": False,
            "status": "UNLOADABLE_CHECKPOINT",
            "error_message": f"Failed to load torch checkpoint: {e}"
        }
        
    actual_sd_hash = _hash_state_dict(state_dict)
    is_canonical_manifest = ("canonical_state_dict_sha256" in manifest or 
                             manifest.get("hash_algorithm") == "canonical_state_dict_sha256")
    
    # 1. Primary check: canonical state-dict hash matches
    if actual_sd_hash == expected_hash:
        # If manifest also has recorded physical hash, ensure physical bytes are not corrupt
        if expected_physical and physical_hash != expected_physical:
            return {
                "is_valid": False,
                "status": "CORRUPTED",
                "canonical_sha256": actual_sd_hash,
                "physical_file_sha256": physical_hash,
                "expected_sha256": expected_hash,
                "manifest": manifest,
                "error_message": f"Corrupted checkpoint physical bytes detected! Expected physical SHA-256: {expected_physical}, Actual: {physical_hash}"
            }
        # Check architecture if requested
        if expected_agents:
            missing = [a for a in expected_agents if a not in state_dict]
            if missing:
                return {
                    "is_valid": False,
                    "status": "ARCHITECTURE_MISMATCH",
                    "error_message": f"Architecture mismatch: missing expected agents {missing}",
                    "canonical_sha256": actual_sd_hash,
                    "physical_file_sha256": physical_hash,
                    "manifest": manifest
                }
        return {
            "is_valid": True,
            "status": "VALID",
            "hash_mode": "canonical_state_dict",
            "canonical_sha256": actual_sd_hash,
            "actual_sha256": actual_sd_hash,
            "physical_file_sha256": physical_hash,
            "expected_sha256": expected_hash,
            "manifest": manifest,
            "error_message": None
        }
        
    # 2. Historical fallback check: only permitted for historical manifests lacking canonical state-dict hashes
    if allow_historical_physical and not is_canonical_manifest:
        if physical_hash == expected_hash or (expected_physical and physical_hash == expected_physical):
            return {
                "is_valid": True,
                "status": "VALID_HISTORICAL_PHYSICAL",
                "hash_mode": "historical_physical_file",
                "canonical_sha256": actual_sd_hash,
                "actual_sha256": physical_hash,
                "physical_file_sha256": physical_hash,
                "expected_sha256": expected_hash,
                "manifest": manifest,
                "error_message": None
            }
        
    return {
        "is_valid": False,
        "status": "CORRUPTED",
        "hash_mode": None,
        "canonical_sha256": actual_sd_hash,
        "actual_sha256": actual_sd_hash,
        "physical_file_sha256": physical_hash,
        "expected_sha256": expected_hash,
        "manifest": manifest,
        "error_message": f"Corrupted checkpoint detected! Expected SHA-256: {expected_hash}, Actual: {actual_sd_hash}"
    }

def load_checkpoint(checkpoint_path, manifest_path, expected_agents=None, allow_historical_physical=False):
    """Load checkpoint and verify SHA-256 against manifest with strict integrity checks."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint file: {checkpoint_path}")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Missing manifest file: {manifest_path}")
        
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    expected_hash = manifest.get("checkpoint_sha256") or manifest.get("canonical_state_dict_sha256")
    if not expected_hash and "checkpoint_hashes" in manifest:
        basename = os.path.basename(checkpoint_path)
        for s_key, s_hash in manifest["checkpoint_hashes"].items():
            if f"seed{s_key}" in basename or manifest.get("checkpoints", {}).get(s_key) == checkpoint_path:
                expected_hash = s_hash
                break
        if not expected_hash and len(manifest["checkpoint_hashes"]) > 0:
            expected_hash = list(manifest["checkpoint_hashes"].values())[0]
            
    if not expected_hash:
        raise ValueError(f"Incompatible checkpoint manifest: missing checkpoint_sha256 in {manifest_path}")
        
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    actual_hash = _hash_state_dict(state_dict)
    
    if actual_hash != expected_hash:
        # Check historical physical hash fallback if permitted
        is_canonical_manifest = ("canonical_state_dict_sha256" in manifest or 
                                 manifest.get("hash_algorithm") == "canonical_state_dict_sha256")
        if allow_historical_physical and not is_canonical_manifest:
            phys_hash = compute_file_sha256(checkpoint_path)
            if phys_hash == expected_hash or (manifest.get("physical_file_sha256") and phys_hash == manifest.get("physical_file_sha256")):
                pass # Acceptable historical artifact
            else:
                raise ValueError(f"Corrupted checkpoint detected! Expected SHA-256: {expected_hash}, Actual: {actual_hash}")
        else:
            raise ValueError(f"Corrupted checkpoint detected! Expected SHA-256: {expected_hash}, Actual: {actual_hash}")
            
    if expected_agents:
        missing = [a for a in expected_agents if a not in state_dict]
        if missing:
            raise ValueError(f"Architecture mismatch: missing expected agent(s) {missing} in state_dict")
            
    return state_dict
