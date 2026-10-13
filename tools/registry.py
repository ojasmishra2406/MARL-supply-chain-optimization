"""
Phase 13 Artifact Registry Manager
Provides an auditable, robust, schema-validated artifact registry contract
and management utilities for multi-agent reinforcement learning models.
"""

import os
import sys
import json
import glob
import shutil
import argparse
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = str(Path(__file__).resolve().parent.parent)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import torch
from rl.checkpoint import _hash_state_dict, compute_file_sha256, verify_checkpoint_integrity

REGISTRY_DEFAULT_PATH = "models/registry.json"

class IntegrityStatus(str, Enum):
    ARTIFACT_VALID = "ARTIFACT_VALID"
    ARTIFACT_INVALID = "ARTIFACT_INVALID"
    QUARANTINED = "QUARANTINED"
    MISSING = "MISSING"

class EvaluationStatus(str, Enum):
    EVALUATION_VALID = "EVALUATION_VALID"
    EVALUATION_INCOMPLETE = "EVALUATION_INCOMPLETE"
    NONE = "NONE"

class ResearchStatus(str, Enum):
    RESEARCH_VALIDATED = "RESEARCH_VALIDATED"
    PROVISIONAL_NEEDS_CORRECTION = "PROVISIONAL_NEEDS_CORRECTION"
    HISTORICAL_SUPERSEDED = "HISTORICAL_SUPERSEDED"
    UNVALIDATED = "UNVALIDATED"
    TRAINING_INCOMPLETE = "TRAINING_INCOMPLETE"

class OverallStatus(str, Enum):
    RESEARCH_VALIDATED = "RESEARCH_VALIDATED"
    PROVISIONAL_NEEDS_CORRECTION = "PROVISIONAL_NEEDS_CORRECTION"
    HISTORICAL_SUPERSEDED = "HISTORICAL_SUPERSEDED"
    TRAINING_INCOMPLETE = "TRAINING_INCOMPLETE"
    EVALUATION_INCOMPLETE = "EVALUATION_INCOMPLETE"
    ARTIFACT_VALID = "ARTIFACT_VALID"
    ARTIFACT_INVALID = "ARTIFACT_INVALID"
    QUARANTINED = "QUARANTINED"
    MISSING = "MISSING"

class ArtifactRegistry:
    def __init__(self, registry_path: str = REGISTRY_DEFAULT_PATH):
        self.registry_path = registry_path
        self.data: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        if os.path.exists(self.registry_path):
            with open(self.registry_path, "r", encoding="utf-8") as f:
                try:
                    self.data = json.load(f)
                except Exception as e:
                    print(f"Warning: Failed to load registry from {self.registry_path}: {e}")
                    self.data = {}
        else:
            self.data = {}

    def save(self, atomic: bool = True, backup: bool = True) -> None:
        """Atomic write to prevent corruption with optional timestamped backup."""
        os.makedirs(os.path.dirname(self.registry_path) if os.path.dirname(self.registry_path) else ".", exist_ok=True)
        
        if backup and os.path.exists(self.registry_path):
            backup_path = f"{self.registry_path}.bak"
            shutil.copyfile(self.registry_path, backup_path)
            
        if atomic:
            temp_path = f"{self.registry_path}.tmp_{os.getpid()}"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
            shutil.move(temp_path, self.registry_path)
        else:
            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)

    def register_entry(self, entry: Dict[str, Any], overwrite: bool = False) -> bool:
        """Idempotently register or update an entry."""
        model_id = entry.get("model_id")
        if not model_id:
            raise ValueError("Entry must contain 'model_id'.")
            
        if model_id in self.data and not overwrite:
            print(f"Model ID '{model_id}' already registered. Skipping.")
            return False
            
        # Ensure contract fields
        contract_entry = self._standardize_contract(entry)
        self.data[model_id] = contract_entry
        return True

    def _standardize_contract(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Ensures all standard contract fields are populated or explicitly None."""
        standard_fields = {
            "model_id": entry.get("model_id"),
            "phase": entry.get("phase", "unknown"),
            "algorithm": entry.get("algorithm", "unknown"),
            "architecture": entry.get("architecture"),
            "forecaster": entry.get("forecaster", "none"),
            "forecaster_config": entry.get("forecaster_config"),
            "training_regime": entry.get("training_regime", entry.get("condition", "baseline")),
            "condition": entry.get("condition", "baseline"),
            "training_seed": entry.get("training_seed", entry.get("seed")),
            "environment_seed": entry.get("environment_seed", entry.get("training_seed", entry.get("seed"))),
            "training_updates": entry.get("training_updates", entry.get("number_of_updates", 200)),
            "training_status": entry.get("training_status", "COMPLETED"),
            
            "checkpoint_path": entry.get("checkpoint_path", entry.get("actual_checkpoint")),
            "manifest_path": entry.get("manifest_path"),
            "checkpoint_sha256": entry.get("checkpoint_sha256"),
            "canonical_state_dict_sha256": entry.get("canonical_state_dict_sha256", entry.get("checkpoint_sha256")),
            "physical_file_sha256": entry.get("physical_file_sha256"),
            "hash_algorithm": entry.get("hash_algorithm", "canonical_state_dict_sha256"),
            
            "config_path": entry.get("config_path", "configs/phase4_ippo.yaml"),
            "git_commit": entry.get("git_commit", "unknown"),
            "git_clean": entry.get("git_clean", True),
            "python_version": entry.get("python_version", "3.11.9"),
            "pytorch_version": entry.get("pytorch_version", "2.3.1+cu121"),
            "software_version": entry.get("software_version", "1.0"),
            "timestamp": entry.get("timestamp", datetime.now().isoformat()),
            
            "training_log_path": entry.get("training_log_path"),
            "training_metrics_path": entry.get("training_metrics_path"),
            
            "evaluations_count": entry.get("evaluations_count", len(entry.get("evaluations_references", []))),
            "evaluations_references": entry.get("evaluations_references", entry.get("evaluations", [])),
            "evaluation_protocol": entry.get("evaluation_protocol", "phase7_ood_matrix_6scenarios_5seeds"),
            
            "integrity_status": entry.get("integrity_status", IntegrityStatus.ARTIFACT_VALID.value),
            "evaluation_status": entry.get("evaluation_status", EvaluationStatus.EVALUATION_VALID.value),
            "research_status": entry.get("research_status", ResearchStatus.UNVALIDATED.value),
            "status": entry.get("status", OverallStatus.ARTIFACT_VALID.value),
            
            "quarantine_reason": entry.get("quarantine_reason"),
            "provenance_notes": entry.get("provenance_notes")
        }
        return standard_fields

    def verify_entry(self, model_id: str, expected_agents: Optional[List[str]] = None) -> Dict[str, Any]:
        """Validates disk integrity against registry contract."""
        if model_id not in self.data:
            return {"model_id": model_id, "status": "NOT_REGISTERED", "is_valid": False}
            
        entry = self.data[model_id]
        ckpt_path = entry.get("checkpoint_path")
        manifest_path = entry.get("manifest_path")
        
        if not manifest_path and ckpt_path:
            manifest_path = ckpt_path.replace(".pt", "_manifest.json")
            
        if not ckpt_path or not manifest_path:
            if entry.get("algorithm", "").startswith("forecasting_") or entry.get("phase") == "phase9":
                metrics_path = entry.get("training_metrics_path")
                valid_benchmark = bool(metrics_path and os.path.exists(metrics_path))
                eval_refs = entry.get("evaluations_references", [])
                existing_evals = [f for f in eval_refs if os.path.exists(f)]
                return {
                    "model_id": model_id,
                    "status": "VALID_OFFLINE_BENCHMARK",
                    "is_valid": valid_benchmark,
                    "evaluations_expected": len(eval_refs),
                    "evaluations_found": len(existing_evals),
                    "evaluations_complete": len(existing_evals) == len(eval_refs)
                }
            return {
                "model_id": model_id,
                "status": "MISSING_PATHS",
                "is_valid": False,
                "error_message": "Missing checkpoint or manifest path in registry"
            }
            
        res = verify_checkpoint_integrity(ckpt_path, manifest_path, expected_agents=expected_agents)
        res["model_id"] = model_id
        
        # Check evaluation completeness
        eval_refs = entry.get("evaluations_references", [])
        existing_evals = [f for f in eval_refs if os.path.exists(f)]
        expected_evals = entry.get("expected_evaluations_count", 30)
        res["evaluations_expected"] = expected_evals
        res["evaluations_found"] = len(existing_evals)
        res["evaluations_complete"] = (len(existing_evals) >= expected_evals if entry.get("evaluation_status") == "EVALUATION_VALID" else True)
        
        return res

    def verify_all(self, expected_agents: Optional[List[str]] = None) -> Dict[str, Any]:
        """Runs audit on all registered models."""
        summary = {
            "total_registered": len(self.data),
            "valid_canonical": 0,
            "valid_historical_physical": 0,
            "valid_offline": 0,
            "corrupted": 0,
            "missing": 0,
            "evaluation_incomplete": 0,
            "results": []
        }
        
        for model_id in sorted(self.data.keys()):
            v = self.verify_entry(model_id, expected_agents=expected_agents)
            summary["results"].append(v)
            if v["status"] == "VALID":
                summary["valid_canonical"] += 1
            elif v["status"] == "VALID_HISTORICAL_PHYSICAL":
                summary["valid_historical_physical"] += 1
            elif v["status"] == "VALID_OFFLINE_BENCHMARK":
                summary["valid_offline"] += 1
            elif v["status"] == "CORRUPTED":
                summary["corrupted"] += 1
            elif "MISSING" in v["status"]:
                summary["missing"] += 1
                
            if not v.get("evaluations_complete", True):
                summary["evaluation_incomplete"] += 1
                
        return summary

    def list_runs(self,
                  phase: Optional[str] = None,
                  algorithm: Optional[str] = None,
                  regime: Optional[str] = None,
                  seed: Optional[int] = None,
                  status: Optional[str] = None,
                  research_status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Filter runs matching criteria."""
        matches = []
        for model_id, entry in sorted(self.data.items()):
            if phase and entry.get("phase") != phase:
                continue
            if algorithm and entry.get("algorithm") != algorithm:
                continue
            if regime and (entry.get("training_regime") != regime and entry.get("condition") != regime):
                continue
            if seed is not None and entry.get("training_seed") != seed:
                continue
            if status and entry.get("status") != status:
                continue
            if research_status and entry.get("research_status") != research_status:
                continue
            matches.append(entry)
        return matches

    def find_missing_evaluations(self, default_expected_count: int = 30) -> List[Dict[str, Any]]:
        """Finds models where evaluation episodes are missing from disk."""
        missing_list = []
        for model_id, entry in sorted(self.data.items()):
            # Only check for runs supposed to have evaluations
            if entry.get("evaluation_status") == EvaluationStatus.NONE.value:
                continue
            expected = entry.get("expected_evaluations_count")
            if expected is None:
                if entry.get("phase") == "phase9":
                    expected = 2
                else:
                    expected = default_expected_count
            if expected == 0:
                continue
            eval_refs = entry.get("evaluations_references", [])
            existing = [f for f in eval_refs if os.path.exists(f)]
            if len(existing) < expected:
                missing_list.append({
                    "model_id": model_id,
                    "expected": expected,
                    "recorded": len(eval_refs),
                    "found_on_disk": len(existing),
                    "deficit": expected - len(existing)
                })
        return missing_list

    def detect_duplicates(self) -> Dict[str, Any]:
        """Detects duplicate model IDs, duplicate checkpoints, or identical state-dict hashes."""
        hash_to_models: Dict[str, List[str]] = {}
        ckpt_to_models: Dict[str, List[str]] = {}
        
        for model_id, entry in self.data.items():
            h = entry.get("canonical_state_dict_sha256") or entry.get("checkpoint_sha256")
            if h:
                hash_to_models.setdefault(h, []).append(model_id)
            ckpt = entry.get("checkpoint_path")
            if ckpt:
                norm_ckpt = os.path.normpath(ckpt)
                ckpt_to_models.setdefault(norm_ckpt, []).append(model_id)
                
        hash_collisions = {h: models for h, models in hash_to_models.items() if len(models) > 1}
        ckpt_collisions = {c: models for c, models in ckpt_to_models.items() if len(models) > 1}
        
        return {
            "duplicate_checkpoints": ckpt_collisions,
            "identical_state_dict_hashes": hash_collisions
        }

    def compare_with_disk(self, model_id: str) -> Dict[str, Any]:
        """Detailed field-by-field diff between registry, manifest, and checkpoint."""
        if model_id not in self.data:
            raise KeyError(f"Model ID '{model_id}' not found.")
            
        entry = self.data[model_id]
        ckpt_path = entry.get("checkpoint_path")
        manifest_path = entry.get("manifest_path")
        
        diffs = []
        if not os.path.exists(ckpt_path):
            diffs.append(f"Checkpoint file missing: {ckpt_path}")
        if not os.path.exists(manifest_path):
            diffs.append(f"Manifest file missing: {manifest_path}")
            
        if diffs:
            return {"model_id": model_id, "agrees": False, "differences": diffs}
            
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
            
        # Compare key fields
        for field in ["algorithm", "training_seed", "seed", "checkpoint_sha256", "git_commit"]:
            reg_val = entry.get(field)
            mf_val = manifest.get(field)
            if reg_val is not None and mf_val is not None and reg_val != mf_val:
                diffs.append(f"Field '{field}' mismatch: Registry={reg_val}, Manifest={mf_val}")
                
        # Check actual checkpoint hash
        actual_sd_hash = None
        try:
            sd = torch.load(ckpt_path, map_location="cpu")
            actual_sd_hash = _hash_state_dict(sd)
            expected_hash = manifest.get("checkpoint_sha256") or manifest.get("canonical_state_dict_sha256")
            if actual_sd_hash != expected_hash:
                # Check physical hash
                phys_hash = compute_file_sha256(ckpt_path)
                if phys_hash != expected_hash:
                    diffs.append(f"State dict hash mismatch: Disk={actual_sd_hash}, Manifest={expected_hash}")
        except Exception as e:
            diffs.append(f"Checkpoint unreadable: {e}")
            
        return {
            "model_id": model_id,
            "agrees": (len(diffs) == 0),
            "differences": diffs,
            "actual_state_dict_sha256": actual_sd_hash
        }

    def report_stale_paths(self) -> List[Dict[str, str]]:
        """Scans registry for referenced paths that do not exist on disk."""
        stale = []
        for model_id, entry in self.data.items():
            for p_field in ["checkpoint_path", "manifest_path", "config_path"]:
                p = entry.get(p_field)
                if p and not os.path.exists(p):
                    stale.append({"model_id": model_id, "field": p_field, "path": p})
            for eval_file in entry.get("evaluations_references", []):
                if not os.path.exists(eval_file):
                    stale.append({"model_id": model_id, "field": "evaluations_references", "path": eval_file})
        return stale

    def quarantine_entry(self, model_id: str, reason: str) -> bool:
        """Quarantine a model due to integrity, audit, or research failure."""
        if model_id not in self.data:
            raise KeyError(f"Model ID '{model_id}' not found.")
        self.data[model_id]["integrity_status"] = IntegrityStatus.QUARANTINED.value
        self.data[model_id]["status"] = OverallStatus.QUARANTINED.value
        self.data[model_id]["quarantine_reason"] = reason
        self.data[model_id]["quarantined_at"] = datetime.now().isoformat()
        self.save()
        return True

    def export_provenance_summary(self, json_path: Optional[str] = None, csv_path: Optional[str] = None) -> Dict[str, Any]:
        """Export comprehensive summary table and machine-readable JSON."""
        summary = {
            "total_models": len(self.data),
            "by_phase": {},
            "by_algorithm": {},
            "by_status": {},
            "by_research_status": {},
            "models": []
        }
        
        for m_id, entry in sorted(self.data.items()):
            ph = entry.get("phase", "unknown")
            al = entry.get("algorithm", "unknown")
            st = entry.get("status", "unknown")
            rst = entry.get("research_status", "unknown")
            
            summary["by_phase"][ph] = summary["by_phase"].get(ph, 0) + 1
            summary["by_algorithm"][al] = summary["by_algorithm"].get(al, 0) + 1
            summary["by_status"][st] = summary["by_status"].get(st, 0) + 1
            summary["by_research_status"][rst] = summary["by_research_status"].get(rst, 0) + 1
            
            summary["models"].append({
                "model_id": m_id,
                "phase": ph,
                "algorithm": al,
                "regime": entry.get("training_regime"),
                "seed": entry.get("training_seed"),
                "updates": entry.get("training_updates"),
                "status": st,
                "research_status": rst,
                "checkpoint_sha256": entry.get("canonical_state_dict_sha256") or entry.get("checkpoint_sha256"),
                "evaluations_count": entry.get("evaluations_count", 0)
            })
            
        if json_path:
            parent_dir = os.path.dirname(os.path.abspath(json_path))
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
                
        if csv_path:
            parent_dir = os.path.dirname(os.path.abspath(csv_path))
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            import pandas as pd
            df = pd.DataFrame(summary["models"])
            df.to_csv(csv_path, index=False)
            
        return summary

def main():
    parser = argparse.ArgumentParser(description="Phase 13 Artifact Registry Platform CLI")
    parser.add_argument("--registry", default=REGISTRY_DEFAULT_PATH, help="Path to registry.json")
    subparsers = parser.add_subparsers(dest="command")
    
    # List
    p_list = subparsers.add_parser("list", help="List registered models")
    p_list.add_argument("--phase", type=str)
    p_list.add_argument("--algorithm", type=str)
    p_list.add_argument("--status", type=str)
    p_list.add_argument("--research-status", type=str)
    
    # Verify
    p_verify = subparsers.add_parser("verify", help="Verify registered models on disk")
    p_verify.add_argument("--model", type=str, help="Specific model to verify (or all if omitted)")
    
    # Audit
    p_audit = subparsers.add_parser("audit", help="Audit missing evals, duplicates, stale paths")
    
    # Export
    p_export = subparsers.add_parser("export", help="Export provenance summary")
    p_export.add_argument("--json", type=str, help="Target JSON path")
    p_export.add_argument("--csv", type=str, help="Target CSV path")
    
    # Quarantine
    p_quar = subparsers.add_parser("quarantine", help="Quarantine a model")
    p_quar.add_argument("--model", type=str, required=True)
    p_quar.add_argument("--reason", type=str, required=True)
    
    args = parser.parse_args()
    reg = ArtifactRegistry(args.registry)
    
    if args.command == "list":
        runs = reg.list_runs(phase=args.phase, algorithm=args.algorithm, status=args.status, research_status=args.research_status)
        print(f"\nFound {len(runs)} matching model(s) in registry:\n")
        print(f"{'Model ID':<38} | {'Phase':<10} | {'Algorithm':<14} | {'Seed':<4} | {'Status':<22} | {'Research Status':<22}")
        print("-" * 120)
        for r in runs:
            print(f"{r['model_id']:<38} | {r.get('phase', ''):<10} | {r.get('algorithm', ''):<14} | {str(r.get('training_seed', '')):<4} | {r.get('status', ''):<22} | {r.get('research_status', ''):<22}")
            
    elif args.command == "verify":
        if args.model:
            res = reg.verify_entry(args.model)
            print(json.dumps(res, indent=2))
        else:
            summary = reg.verify_all()
            print(f"\nVerification Results across {summary['total_registered']} models:")
            print(f"  - Valid (Canonical State-Dict): {summary['valid_canonical']}")
            print(f"  - Valid (Historical Physical):  {summary['valid_historical_physical']}")
            print(f"  - Valid (Offline Benchmark):    {summary['valid_offline']}")
            print(f"  - Corrupted:                    {summary['corrupted']}")
            print(f"  - Missing Checkpoint/Manifest:  {summary['missing']}")
            print(f"  - Incomplete Evaluations:       {summary['evaluation_incomplete']}")
            
    elif args.command == "audit":
        print("\n--- REGISTRY AUDIT ---")
        dups = reg.detect_duplicates()
        print(f"Duplicate Checkpoints: {len(dups['duplicate_checkpoints'])}")
        print(f"Hash Collisions:        {len(dups['identical_state_dict_hashes'])}")
        missing = reg.find_missing_evaluations()
        print(f"Runs with Missing Evals:{len(missing)}")
        stale = reg.report_stale_paths()
        print(f"Stale Paths Detected:   {len(stale)}")
        
    elif args.command == "export":
        s = reg.export_provenance_summary(json_path=args.json, csv_path=args.csv)
        print(f"Exported provenance summary: {s['total_models']} total models.")
        
    elif args.command == "quarantine":
        reg.quarantine_entry(args.model, args.reason)
        print(f"Quarantined {args.model}: {args.reason}")
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
