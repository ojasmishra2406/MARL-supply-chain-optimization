import json
import os
import sys
import glob
sys.path.append(".")
from rl.checkpoint import load_checkpoint

REGISTRY_PATH = "models/registry.json"
RESULTS_DIR = "results/phase11_controlled"

def register_controlled_models():
    if not os.path.exists(REGISTRY_PATH):
        print(f"Registry not found at {REGISTRY_PATH}")
        return
        
    with open(REGISTRY_PATH, "r") as f:
        registry = json.load(f)
        
    controlled_models = [
        ("phase11_controlled_ma_s1", "ma", 1),
        ("phase11_controlled_ma_s2", "ma", 2),
        ("phase11_controlled_ma_s3", "ma", 3),
        ("phase11_controlled_xgboost_s1", "xgboost", 1),
        ("phase11_controlled_xgboost_s2", "xgboost", 2),
        ("phase11_controlled_xgboost_s3", "xgboost", 3),
        ("phase11_controlled_xgboost_s42_smoke", "xgboost", 42),
        ("phase11_controlled_ma_s42_smoke", "ma", 42),
    ]
    
    registered_count = 0
    for model_id, forecaster, seed in controlled_models:
        ckpt_path = f"{RESULTS_DIR}/{model_id}.pt"
        manifest_path = f"{RESULTS_DIR}/{model_id}_manifest.json"
        
        if not os.path.exists(ckpt_path) or not os.path.exists(manifest_path):
            print(f"Skipping {model_id}: artifacts missing.")
            continue
            
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
            
        # Verify canonical load
        try:
            load_checkpoint(ckpt_path, manifest_path)
            status = "VALID"
        except Exception as e:
            print(f"Validation failed for {model_id}: {e}")
            status = "CORRUPTED"
            
        eval_refs = sorted(glob.glob(f"{RESULTS_DIR}/{model_id}_*eval*.json"))
        
        entry = {
            "model_id": model_id,
            "algorithm": "gnn_mappo",
            "forecaster": forecaster,
            "architecture": "GNN_MAPPO(node_features=7, num_nodes=4, hidden_dim=64)",
            "condition": f"forecast_{forecaster}",
            "training_seed": seed,
            "config_path": manifest.get("config_path", "configs/phase4_ippo.yaml"),
            "checkpoint_path": ckpt_path,
            "checkpoint_sha256": manifest.get("checkpoint_sha256"),
            "timestamp": manifest.get("timestamp"),
            "software_version": manifest.get("software_version", "2.0_controlled"),
            "git_commit": manifest.get("git_commit", "unknown"),
            "status": status,
            "evaluations_count": len(eval_refs),
            "evaluations_references": eval_refs
        }
        
        registry[model_id] = entry
        registered_count += 1
        print(f"Successfully registered {model_id} (status: {status}, evals: {len(eval_refs)}/30)")
        
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=4)
        
    print(f"Updated registry with {registered_count} controlled Phase 11 models.")

if __name__ == "__main__":
    register_controlled_models()
