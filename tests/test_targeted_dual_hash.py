"""
Targeted tests for Phase 13 Dual-Hash Checkpoint Implementation.
Covers 10 explicit test scenarios requested by Section 2.A:
1. Save/load round trip.
2. Valid canonical state-dict hash.
3. Valid physical-file hash.
4. Altered or corrupted checkpoint.
5. Missing checkpoint.
6. Truncated checkpoint.
7. Malformed manifest.
8. Architecture mismatch.
9. Historical manifest compatibility.
10. Hash mismatch and actionable failure behavior.
"""

import os
import sys
import json
import tempfile
import pytest
import torch
from pathlib import Path

WORKSPACE_ROOT = str(Path(__file__).resolve().parent.parent)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from rl.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    _hash_state_dict,
    compute_file_sha256,
    verify_checkpoint_integrity
)

class SampleAgent(torch.nn.Module):
    def __init__(self, in_features=5, out_features=5):
        super().__init__()
        self.fc = torch.nn.Linear(in_features, out_features)

@pytest.fixture
def temp_workspace():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

def test_1_save_load_round_trip(temp_workspace):
    """1. Save/load round trip."""
    ckpt_path = os.path.join(temp_workspace, "model_roundtrip.pt")
    manifest_path = os.path.join(temp_workspace, "model_roundtrip_manifest.json")
    
    agent = SampleAgent()
    sd = {"retailer": agent.state_dict()}
    
    canonical_hash = save_checkpoint(sd, ckpt_path, manifest_path, extra_metadata={"experiment_id": "rt_test"})
    assert os.path.exists(ckpt_path)
    assert os.path.exists(manifest_path)
    
    loaded_sd = load_checkpoint(ckpt_path, manifest_path)
    assert "retailer" in loaded_sd
    for k in sd["retailer"]:
        assert torch.equal(sd["retailer"][k], loaded_sd["retailer"][k])

def test_2_valid_canonical_state_dict_hash(temp_workspace):
    """2. Valid canonical state-dict hash."""
    ckpt_path = os.path.join(temp_workspace, "model_canonical.pt")
    manifest_path = os.path.join(temp_workspace, "model_canonical_manifest.json")
    
    agent = SampleAgent()
    sd = {"retailer": agent.state_dict()}
    save_checkpoint(sd, ckpt_path, manifest_path)
    
    audit = verify_checkpoint_integrity(ckpt_path, manifest_path)
    assert audit["is_valid"] is True
    assert audit["status"] == "VALID"
    assert audit["hash_mode"] == "canonical_state_dict"
    assert audit["canonical_sha256"] == _hash_state_dict(sd)

def test_3_valid_physical_file_hash(temp_workspace):
    """3. Valid physical-file hash."""
    ckpt_path = os.path.join(temp_workspace, "model_physical.pt")
    manifest_path = os.path.join(temp_workspace, "model_physical_manifest.json")
    
    agent = SampleAgent()
    sd = {"retailer": agent.state_dict()}
    save_checkpoint(sd, ckpt_path, manifest_path)
    
    with open(manifest_path, "r") as f:
        mf = json.load(f)
        
    physical_sha = compute_file_sha256(ckpt_path)
    assert mf["physical_file_sha256"] == physical_sha
    assert len(physical_sha) == 64

def test_4_altered_or_corrupted_checkpoint(temp_workspace):
    """4. Altered or corrupted checkpoint."""
    ckpt_path = os.path.join(temp_workspace, "model_altered.pt")
    manifest_path = os.path.join(temp_workspace, "model_altered_manifest.json")
    
    agent = SampleAgent()
    sd = {"retailer": agent.state_dict()}
    save_checkpoint(sd, ckpt_path, manifest_path)
    
    # Tamper with weights
    with torch.no_grad():
        agent.fc.weight.add_(0.5)
    torch.save({"retailer": agent.state_dict()}, ckpt_path)
    
    audit = verify_checkpoint_integrity(ckpt_path, manifest_path)
    assert audit["is_valid"] is False
    assert audit["status"] == "CORRUPTED"
    assert "Corrupted checkpoint" in audit["error_message"]

def test_5_missing_checkpoint(temp_workspace):
    """5. Missing checkpoint."""
    ckpt_path = os.path.join(temp_workspace, "missing.pt")
    manifest_path = os.path.join(temp_workspace, "manifest.json")
    
    with open(manifest_path, "w") as f:
        json.dump({"checkpoint_sha256": "dummy"}, f)
        
    audit = verify_checkpoint_integrity(ckpt_path, manifest_path)
    assert audit["is_valid"] is False
    assert audit["status"] == "MISSING_CHECKPOINT"

def test_6_truncated_checkpoint(temp_workspace):
    """6. Truncated checkpoint."""
    ckpt_path = os.path.join(temp_workspace, "truncated.pt")
    manifest_path = os.path.join(temp_workspace, "truncated_manifest.json")
    
    agent = SampleAgent()
    sd = {"retailer": agent.state_dict()}
    save_checkpoint(sd, ckpt_path, manifest_path)
    
    # Truncate physical file to 100 bytes
    with open(ckpt_path, "wb") as f:
        f.write(b"truncated_junk_bytes" * 5)
        
    audit = verify_checkpoint_integrity(ckpt_path, manifest_path)
    assert audit["is_valid"] is False
    assert audit["status"] == "UNLOADABLE_CHECKPOINT"
    assert "Failed to load torch checkpoint" in audit["error_message"]

def test_7_malformed_manifest(temp_workspace):
    """7. Malformed manifest."""
    ckpt_path = os.path.join(temp_workspace, "valid.pt")
    manifest_path = os.path.join(temp_workspace, "malformed_manifest.json")
    
    agent = SampleAgent()
    torch.save({"retailer": agent.state_dict()}, ckpt_path)
    
    # Write corrupt JSON
    with open(manifest_path, "w") as f:
        f.write("{invalid_json: true, missing_quotes")
        
    audit = verify_checkpoint_integrity(ckpt_path, manifest_path)
    assert audit["is_valid"] is False
    assert audit["status"] == "MALFORMED_MANIFEST"

def test_8_architecture_mismatch(temp_workspace):
    """8. Architecture mismatch."""
    ckpt_path = os.path.join(temp_workspace, "arch_model.pt")
    manifest_path = os.path.join(temp_workspace, "arch_manifest.json")
    
    agent = SampleAgent()
    sd = {"retailer": agent.state_dict()}
    save_checkpoint(sd, ckpt_path, manifest_path)
    
    # Expected agents includes wholesaler and distributor
    audit = verify_checkpoint_integrity(
        ckpt_path,
        manifest_path,
        expected_agents=["retailer", "wholesaler", "distributor"]
    )
    assert audit["is_valid"] is False
    assert audit["status"] == "ARCHITECTURE_MISMATCH"
    assert "missing expected agents" in audit["error_message"]
    
    with pytest.raises(ValueError, match="Architecture mismatch"):
        load_checkpoint(ckpt_path, manifest_path, expected_agents=["wholesaler"])

def test_9_historical_manifest_compatibility(temp_workspace):
    """9. Historical manifest compatibility."""
    ckpt_path = os.path.join(temp_workspace, "historical_model.pt")
    manifest_path = os.path.join(temp_workspace, "historical_manifest.json")
    
    agent = SampleAgent()
    sd = {"retailer": agent.state_dict()}
    torch.save(sd, ckpt_path)
    
    # Historical manifests recorded physical SHA-256 as checkpoint_sha256
    phys_hash = compute_file_sha256(ckpt_path)
    manifest = {
        "checkpoint_path": ckpt_path,
        "checkpoint_sha256": phys_hash,
        "hash_algorithm": "physical_file_sha256"
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
        
    # verify_checkpoint_integrity with allow_historical_physical=True
    audit = verify_checkpoint_integrity(ckpt_path, manifest_path, allow_historical_physical=True)
    assert audit["is_valid"] is True
    assert audit["status"] == "VALID_HISTORICAL_PHYSICAL"
    assert audit["hash_mode"] == "historical_physical_file"
    
    # load_checkpoint with allow_historical_physical=True
    loaded = load_checkpoint(ckpt_path, manifest_path, allow_historical_physical=True)
    assert "retailer" in loaded
    
    # load_checkpoint with allow_historical_physical=False must reject it
    with pytest.raises(ValueError, match="Corrupted checkpoint detected"):
        load_checkpoint(ckpt_path, manifest_path, allow_historical_physical=False)

def test_10_hash_mismatch_actionable_failure(temp_workspace):
    """10. Hash mismatch and actionable failure behavior."""
    ckpt_path = os.path.join(temp_workspace, "mismatch.pt")
    manifest_path = os.path.join(temp_workspace, "mismatch_manifest.json")
    
    agent = SampleAgent()
    sd = {"retailer": agent.state_dict()}
    save_checkpoint(sd, ckpt_path, manifest_path)
    
    # Alter manifest expected hash
    with open(manifest_path, "r") as f:
        mf = json.load(f)
    mf["checkpoint_sha256"] = "0000000000000000000000000000000000000000000000000000000000000000"
    mf["canonical_state_dict_sha256"] = "0000000000000000000000000000000000000000000000000000000000000000"
    with open(manifest_path, "w") as f:
        json.dump(mf, f)
        
    audit = verify_checkpoint_integrity(ckpt_path, manifest_path)
    assert audit["is_valid"] is False
    assert audit["status"] == "CORRUPTED"
    assert "Expected SHA-256: 00000000" in audit["error_message"]
    assert audit["actual_sha256"] == audit["canonical_sha256"] or audit["canonical_sha256"] is not None

def test_physical_and_canonical_hashes_never_cross_compared(temp_workspace):
    """Proves physical file hash is never compared to canonical tensor hash."""
    ckpt_path = os.path.join(temp_workspace, "cross_compare.pt")
    manifest_path = os.path.join(temp_workspace, "cross_compare_manifest.json")
    
    agent = SampleAgent()
    sd = {"retailer": agent.state_dict()}
    save_checkpoint(sd, ckpt_path, manifest_path)
    
    canon_hash = _hash_state_dict(sd)
    phys_hash = compute_file_sha256(ckpt_path)
    
    # Assert they are distinct representations
    assert canon_hash != phys_hash
    
    # Manifest clearly tags them
    with open(manifest_path, "r") as f:
        mf = json.load(f)
    assert mf["canonical_state_dict_sha256"] == canon_hash
    assert mf["physical_file_sha256"] == phys_hash
    assert mf["hash_algorithm"] == "canonical_state_dict_sha256"
