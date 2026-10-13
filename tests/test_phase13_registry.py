"""
Phase 13 Comprehensive Unit & Regression Test Suite
Tests Artifact Registry operations, dual-hash verification, quarantine workflows,
corruption detection, and automated reproducibility checker.
"""

import os
import sys
import json
import tempfile
import pytest
import torch
import numpy as np
from pathlib import Path

# Ensure root in path
WORKSPACE_ROOT = str(Path(__file__).resolve().parent.parent)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from tools.registry import (
    ArtifactRegistry,
    IntegrityStatus,
    EvaluationStatus,
    ResearchStatus,
    OverallStatus
)
from rl.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    _hash_state_dict,
    compute_file_sha256,
    verify_checkpoint_integrity
)
from tools.reproducibility.check_reproducibility import (
    check_environment,
    check_registry_integrity,
    check_seed_propagation,
    replay_deterministic_evaluation
)

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(5, 5)

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp

def test_registry_save_load_atomic(temp_dir):
    reg_path = os.path.join(temp_dir, "test_registry.json")
    reg = ArtifactRegistry(reg_path)
    
    entry = {
        "model_id": "test_model_001",
        "phase": "phase13_test",
        "algorithm": "mappo",
        "training_seed": 42,
        "status": OverallStatus.RESEARCH_VALIDATED.value
    }
    assert reg.register_entry(entry) is True
    reg.save(atomic=True, backup=False)
    
    assert os.path.exists(reg_path)
    
    # Reload
    reg2 = ArtifactRegistry(reg_path)
    assert "test_model_001" in reg2.data
    assert reg2.data["test_model_001"]["algorithm"] == "mappo"
    assert reg2.data["test_model_001"]["training_seed"] == 42

def test_registry_filters_and_queries(temp_dir):
    reg_path = os.path.join(temp_dir, "test_registry.json")
    reg = ArtifactRegistry(reg_path)
    
    reg.register_entry({"model_id": "m1", "phase": "phase6", "algorithm": "ippo", "training_seed": 1, "status": "RESEARCH_VALIDATED"})
    reg.register_entry({"model_id": "m2", "phase": "phase6", "algorithm": "mappo", "training_seed": 2, "status": "RESEARCH_VALIDATED"})
    reg.register_entry({"model_id": "m3", "phase": "phase11", "algorithm": "gnn_mappo", "training_seed": 1, "status": "HISTORICAL_SUPERSEDED"})
    
    # Query phase6
    res_p6 = reg.list_runs(phase="phase6")
    assert len(res_p6) == 2
    
    # Query seed 1
    res_s1 = reg.list_runs(seed=1)
    assert len(res_s1) == 2
    
    # Query algorithm mappo
    res_mappo = reg.list_runs(algorithm="mappo")
    assert len(res_mappo) == 1
    assert res_mappo[0]["model_id"] == "m2"
    
    # Query status HISTORICAL_SUPERSEDED
    res_hist = reg.list_runs(status="HISTORICAL_SUPERSEDED")
    assert len(res_hist) == 1
    assert res_hist[0]["model_id"] == "m3"

def test_checkpoint_dual_hash_saving_and_verification(temp_dir):
    ckpt_path = os.path.join(temp_dir, "model.pt")
    manifest_path = os.path.join(temp_dir, "model_manifest.json")
    
    model = DummyModel()
    state_dict = {"retailer": model.state_dict()}
    
    # Save checkpoint
    saved_hash = save_checkpoint(
        state_dict=state_dict,
        checkpoint_path=ckpt_path,
        manifest_path=manifest_path,
        extra_metadata={"notes": "dual-hash test", "experiment_id": "test_dual_hash", "phase": "phase13", "seed": 123}
    )
    
    assert os.path.exists(ckpt_path)
    assert os.path.exists(manifest_path)
    with open(manifest_path, "r") as f:
        mf = json.load(f)
    assert "canonical_state_dict_sha256" in mf
    assert "physical_file_sha256" in mf
    
    # Verify integrity via verification engine
    audit = verify_checkpoint_integrity(ckpt_path, manifest_path, expected_agents=["retailer"])
    assert audit["is_valid"] is True
    assert audit["status"] == "VALID"
    assert audit["hash_mode"] == "canonical_state_dict"
    
    # Load checkpoint safely via canonical load_checkpoint
    loaded = load_checkpoint(ckpt_path, manifest_path)
    assert "retailer" in loaded

def test_corrupted_checkpoint_detection(temp_dir):
    ckpt_path = os.path.join(temp_dir, "tampered.pt")
    manifest_path = os.path.join(temp_dir, "tampered_manifest.json")
    
    model = DummyModel()
    state_dict = {"retailer": model.state_dict()}
    
    save_checkpoint(
        state_dict=state_dict,
        checkpoint_path=ckpt_path,
        manifest_path=manifest_path,
        extra_metadata={"experiment_id": "test_tamper", "phase": "phase13", "seed": 777}
    )
    
    # Tamper with the model weights (modify weights and resave file under same manifest)
    with torch.no_grad():
        model.fc.weight.add_(1.0)
    torch.save({"retailer": model.state_dict()}, ckpt_path)
    
    audit = verify_checkpoint_integrity(ckpt_path, manifest_path)
    assert audit["is_valid"] is False
    assert audit["status"] == "CORRUPTED"
    assert "Corrupted checkpoint" in audit["error_message"]

def test_missing_files_handling(temp_dir):
    ckpt_path = os.path.join(temp_dir, "nonexistent.pt")
    manifest_path = os.path.join(temp_dir, "nonexistent_manifest.json")
    
    audit = verify_checkpoint_integrity(ckpt_path, manifest_path)
    assert audit["is_valid"] is False
    assert audit["status"] == "MISSING_CHECKPOINT"

def test_quarantine_workflow(temp_dir):
    reg_path = os.path.join(temp_dir, "quarantine_reg.json")
    reg = ArtifactRegistry(reg_path)
    
    reg.register_entry({
        "model_id": "suspect_model",
        "phase": "phase11",
        "algorithm": "gnn_mappo",
        "status": OverallStatus.RESEARCH_VALIDATED.value,
        "integrity_status": IntegrityStatus.ARTIFACT_VALID.value
    })
    
    assert reg.quarantine_entry("suspect_model", "Identified RNG non-independence during audit") is True
    entry = reg.data["suspect_model"]
    assert entry["integrity_status"] == IntegrityStatus.QUARANTINED.value
    assert entry["status"] == OverallStatus.QUARANTINED.value
    assert "RNG non-independence" in entry["quarantine_reason"]
    assert "quarantined_at" in entry

def test_duplicate_checkpoint_and_hash_detection(temp_dir):
    reg_path = os.path.join(temp_dir, "dup_reg.json")
    reg = ArtifactRegistry(reg_path)
    
    reg.register_entry({
        "model_id": "model_alpha",
        "checkpoint_path": "results/p6/same_path.pt",
        "canonical_state_dict_sha256": "hash_12345"
    })
    reg.register_entry({
        "model_id": "model_beta",
        "checkpoint_path": "results/p6/same_path.pt",
        "canonical_state_dict_sha256": "hash_12345"
    })
    
    dups = reg.detect_duplicates()
    assert len(dups["duplicate_checkpoints"]) == 1
    assert len(dups["identical_state_dict_hashes"]) == 1
    assert "model_alpha" in dups["duplicate_checkpoints"]["results\\p6\\same_path.pt" if os.name == 'nt' else "results/p6/same_path.pt"]

def test_export_provenance_summary(temp_dir):
    reg_path = os.path.join(temp_dir, "export_reg.json")
    json_export = os.path.join(temp_dir, "export.json")
    csv_export = os.path.join(temp_dir, "export.csv")
    
    reg = ArtifactRegistry(reg_path)
    reg.register_entry({"model_id": "m1", "phase": "phase6", "algorithm": "ippo", "training_updates": 200, "status": "RESEARCH_VALIDATED"})
    reg.register_entry({"model_id": "m2", "phase": "phase10", "algorithm": "gnn_mappo", "training_updates": 200, "status": "RESEARCH_VALIDATED"})
    
    summary = reg.export_provenance_summary(json_path=json_export, csv_path=csv_export)
    assert summary["total_models"] == 2
    assert os.path.exists(json_export)
    assert os.path.exists(csv_export)

def test_reproducibility_checker_runtime():
    env_pass, env_data = check_environment()
    assert env_pass is True
    assert env_data["python"]["passed"] is True
    assert env_data["torch"]["passed"] is True

def test_seed_propagation_determinism():
    seed_pass, seed_data = check_seed_propagation()
    assert seed_pass is True
    assert seed_data["deterministic_reproducibility"] is True
    assert seed_data["seed_entropy_independence"] is True

def test_evaluation_replay_canonical_model():
    replay_pass, replay_data = replay_deterministic_evaluation()
    assert replay_pass is True
    assert replay_data["cost_match"] is True
    assert replay_data["fill_rate_match"] is True
    assert replay_data["bullwhip_match"] is True
