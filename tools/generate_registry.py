import os
import json
import glob
import hashlib

def get_file_hash(filepath):
    if not os.path.exists(filepath):
        return None
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def main():
    registry_path = "models/registry.json"
    if os.path.exists(registry_path):
        with open(registry_path, "r") as f:
            registry = json.load(f)
    else:
        registry = {}

    manifests = glob.glob("results/phase6/*manifest.json")
    print(f"Found {len(manifests)} manifests in results/phase6/")

    for mf in manifests:
        if "ci_test" in mf:
            continue
            
        with open(mf, "r") as f:
            data = json.load(f)
            
        model_id = data.get("experiment_id", "")
        if not model_id:
            continue
            
        # Update or create registry entry
        ckpt_path = data.get("checkpoint_path", "")
        
        # Verify checkpoint exists
        if not os.path.exists(ckpt_path):
            print(f"Warning: Checkpoint {ckpt_path} not found for {model_id}")
            status = "MISSING_CHECKPOINT"
            ckpt_hash = None
        else:
            status = "VALID"
            ckpt_hash = get_file_hash(ckpt_path)
            
        # Find evaluation results
        # Look for Phase 7 evals matching this model_id
        evals = glob.glob(f"results/phase7/{model_id}_*.json")
        
        entry = {
            "model_id": model_id,
            "algorithm": data.get("algorithm", ""),
            "architecture": data.get("architecture", ""),
            "condition": data.get("condition", ""),
            "training_seed": data.get("seed", 0),
            "config_path": data.get("config_path", ""),
            "checkpoint_path": ckpt_path,
            "checkpoint_sha256": ckpt_hash,
            "timestamp": data.get("timestamp", ""),
            "reproducibility": data.get("reproducibility", {}),
            "status": status,
            "evaluations_count": len(evals),
            "evaluations_references": evals
        }
        
        registry[model_id] = entry
        
    os.makedirs("models", exist_ok=True)
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=4)
        
    print(f"Registry updated. Total models: {len(registry)}")

if __name__ == "__main__":
    main()
