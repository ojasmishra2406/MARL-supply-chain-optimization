"""
Tests for Section 2.C: Separation of Artifact Integrity from Research Validity.
Verifies that hash-valid checkpoints cannot accidentally become RESEARCH_VALIDATED,
and tests status enums and quarantine transitions.
"""

import os
import sys
import tempfile
import pytest
from pathlib import Path

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

@pytest.fixture
def temp_reg_workspace():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

def test_hash_valid_model_retains_unvalidated_or_superseded_status(temp_reg_workspace):
    """Proves that a hash-valid checkpoint does NOT automatically become RESEARCH_VALIDATED."""
    reg_path = os.path.join(temp_reg_workspace, "registry.json")
    reg = ArtifactRegistry(reg_path)
    
    # 1. Historical superseded model with valid hash
    entry_hist = {
        "model_id": "historical_model_01",
        "checkpoint_path": "results/phase11/phase11_gnn_mappo_forecast_xgboost_s0.pt",
        "manifest_path": "results/phase11/phase11_gnn_mappo_forecast_xgboost_s0_manifest.json",
        "checkpoint_sha256": "46a7fc90a01cc0d77a66623959f7d6202be08d0d7fa2c8c7d272c8849174e74c",
        "integrity_status": IntegrityStatus.ARTIFACT_VALID.value,
        "evaluation_status": EvaluationStatus.EVALUATION_VALID.value,
        "research_status": ResearchStatus.HISTORICAL_SUPERSEDED.value,
        "status": OverallStatus.HISTORICAL_SUPERSEDED.value
    }
    reg.register_entry(entry_hist)
    
    # Verify entry on disk
    res = reg.verify_entry("historical_model_01")
    assert res["is_valid"] is True # Disk integrity is physically intact
    
    # Check that registry record was NOT mutated to RESEARCH_VALIDATED
    assert reg.data["historical_model_01"]["research_status"] == ResearchStatus.HISTORICAL_SUPERSEDED.value
    assert reg.data["historical_model_01"]["status"] == OverallStatus.HISTORICAL_SUPERSEDED.value

def test_provisional_status_preservation(temp_reg_workspace):
    """Proves that provisional models stay PROVISIONAL_NEEDS_CORRECTION despite valid hashes."""
    reg_path = os.path.join(temp_reg_workspace, "registry.json")
    reg = ArtifactRegistry(reg_path)
    
    entry_prov = {
        "model_id": "phase12_ablation_model",
        "integrity_status": IntegrityStatus.ARTIFACT_VALID.value,
        "evaluation_status": EvaluationStatus.EVALUATION_VALID.value,
        "research_status": ResearchStatus.PROVISIONAL_NEEDS_CORRECTION.value,
        "status": OverallStatus.PROVISIONAL_NEEDS_CORRECTION.value
    }
    reg.register_entry(entry_prov)
    
    # Assert status cannot silently convert
    assert reg.data["phase12_ablation_model"]["research_status"] == ResearchStatus.PROVISIONAL_NEEDS_CORRECTION.value
    assert reg.data["phase12_ablation_model"]["status"] == OverallStatus.PROVISIONAL_NEEDS_CORRECTION.value

def test_training_incomplete_preservation(temp_reg_workspace):
    """Proves that diagnostic models stay TRAINING_INCOMPLETE despite valid hashes."""
    reg_path = os.path.join(temp_reg_workspace, "registry.json")
    reg = ArtifactRegistry(reg_path)
    
    entry_inc = {
        "model_id": "phase12_ablation_mappo_comm_2iters",
        "training_updates": 2,
        "integrity_status": IntegrityStatus.ARTIFACT_VALID.value,
        "evaluation_status": EvaluationStatus.NONE.value,
        "research_status": ResearchStatus.TRAINING_INCOMPLETE.value,
        "status": OverallStatus.TRAINING_INCOMPLETE.value
    }
    reg.register_entry(entry_inc)
    assert reg.data["phase12_ablation_mappo_comm_2iters"]["research_status"] == ResearchStatus.TRAINING_INCOMPLETE.value

def test_quarantine_transition(temp_reg_workspace):
    """Tests quarantining transition behavior and reason documentation."""
    reg_path = os.path.join(temp_reg_workspace, "registry.json")
    reg = ArtifactRegistry(reg_path)
    
    entry = {
        "model_id": "model_to_quarantine",
        "integrity_status": IntegrityStatus.ARTIFACT_VALID.value,
        "research_status": ResearchStatus.RESEARCH_VALIDATED.value,
        "status": OverallStatus.RESEARCH_VALIDATED.value
    }
    reg.register_entry(entry)
    
    success = reg.quarantine_entry("model_to_quarantine", reason="Suspected environment seed leakage")
    assert success is True
    
    saved_entry = reg.data["model_to_quarantine"]
    assert saved_entry["integrity_status"] == IntegrityStatus.QUARANTINED.value
    assert saved_entry["status"] == OverallStatus.QUARANTINED.value
    assert "quarantine_reason" in saved_entry
    assert saved_entry["quarantine_reason"] == "Suspected environment seed leakage"
    assert "quarantined_at" in saved_entry
