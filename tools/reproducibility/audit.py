import os
import json
import hashlib
from pathlib import Path

import torch
import sys
sys.path.append(".")
from rl.checkpoint import _hash_state_dict

def compute_sha256(filepath):
    try:
        state_dict = torch.load(filepath, map_location="cpu")
        return _hash_state_dict(state_dict)
    except Exception as e:
        print(f"[ERROR Loading Checkpoint] {filepath}: {e}")
        return None

def audit_artifacts(results_dir="results", report_only=True):
    print("="*50)
    print("ARTIFACT AUDIT REPORT")
    print("="*50)
    
    registry = {}
    orphans = []
    missing_checkpoints = []
    corrupted_hashes = []
    
    # 1. Scan for Checkpoints and Manifests
    for root, _, files in os.walk(results_dir):
        for file in files:
            if file.endswith("_manifest.json"):
                manifest_path = os.path.join(root, file)
                with open(manifest_path, "r") as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        print(f"[CORRUPT MANIFEST] {manifest_path}")
                        continue
                        
                exp_id = data.get("experiment_id")
                if not exp_id:
                    continue
                    
                registry[exp_id] = {
                    "manifest_path": manifest_path,
                    "metadata": data,
                    "status": "UNVERIFIED",
                    "evaluations": []
                }
                
    # 2. Verify Checkpoints
    for root, _, files in os.walk(results_dir):
        for file in files:
            if file.endswith(".pt"):
                ckpt_path = os.path.join(root, file)
                # Infer exp_id
                exp_id = file.replace(".pt", "")
                
                if exp_id not in registry:
                    orphans.append(ckpt_path)
                else:
                    reg = registry[exp_id]
                    expected_hash = reg["metadata"].get("checkpoint_sha256")
                    actual_hash = compute_sha256(ckpt_path)
                    
                    if expected_hash and actual_hash != expected_hash:
                        corrupted_hashes.append((ckpt_path, expected_hash, actual_hash))
                        reg["status"] = "CORRUPT"
                    else:
                        reg["status"] = "TRAINED"
                        reg["actual_checkpoint"] = ckpt_path

    # 3. Mark missing checkpoints
    for exp_id, reg in registry.items():
        if "actual_checkpoint" not in reg and reg["status"] == "UNVERIFIED":
            missing_checkpoints.append(reg["manifest_path"])
            reg["status"] = "MISSING"

    # 4. Map Evaluations
    for root, _, files in os.walk(results_dir):
        if "phase7" in root:
            for file in files:
                if file.endswith(".json"):
                    eval_path = os.path.join(root, file)
                    with open(eval_path, "r") as f:
                        eval_data = json.load(f)
                    
                    ckpt_path = eval_data.get("checkpoint")
                    if ckpt_path:
                        exp_id = os.path.basename(ckpt_path).replace(".pt", "")
                    else:
                        exp_id = eval_data.get("experiment_id")
                        
                    if exp_id in registry:
                        registry[exp_id]["evaluations"].append(eval_path)
                        if registry[exp_id]["status"] == "TRAINED":
                            registry[exp_id]["status"] = "EVALUATED"

    # Save Registry
    os.makedirs("models", exist_ok=True)
    with open("models/registry.json", "w") as f:
        json.dump(registry, f, indent=4)
        
    print(f"Total Registered Models: {len(registry)}")
    print(f"Models EVALUATED: {len([r for r in registry.values() if r['status'] == 'EVALUATED'])}")
    print(f"Models TRAINED (no eval): {len([r for r in registry.values() if r['status'] == 'TRAINED'])}")
    print(f"Orphan Checkpoints: {len(orphans)}")
    for o in orphans: print(f"  - {o}")
    print(f"Missing Checkpoints: {len(missing_checkpoints)}")
    for m in missing_checkpoints: print(f"  - {m}")
    print(f"Corrupted Hashes: {len(corrupted_hashes)}")
    for c in corrupted_hashes: print(f"  - {c[0]}")
    
    print("\nRegistry saved to models/registry.json")
    if not report_only:
        print("Cleanup operations requested (NOT IMPLEMENTED IN THIS SAFE VERSION).")
    
    return registry

if __name__ == "__main__":
    audit_artifacts()
