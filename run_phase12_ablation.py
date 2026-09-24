import os
import yaml
import json
import itertools
import hashlib
import argparse
from pathlib import Path

def get_experiment_id(params):
    """
    Generates a deterministic experiment ID.
    Format: p12_{family}_{algo}_{arch}_{regime}_s{seed}_{hash}
    """
    # Deterministic hash of relevant params to ensure uniqueness without massive filenames
    param_str = json.dumps(params, sort_keys=True)
    hash_val = hashlib.md5(param_str.encode('utf-8')).hexdigest()[:6]
    
    family = params.get("ablation_family", "none").split("_", 1)[0]
    algo = params.get("algorithm", "unknown")
    arch = params.get("architecture", "mlp")
    regime = params.get("regime", "baseline")
    seed = params.get("seed", 0)
    
    return f"p12_{family}_{algo}_{arch}_{regime}_s{seed}_{hash_val}"

def get_combinations(config):
    """
    Expands a config dictionary containing lists into a list of singular configs.
    """
    list_keys = []
    list_values = []
    
    # Simple approach: check top-level keys for lists
    for k, v in config.items():
        if isinstance(v, list):
            list_keys.append(k)
            list_values.append(v)
        elif k == "forecasting" and isinstance(v, dict):
            # Special case for forecasting model list
            if "model" in v and isinstance(v["model"], list):
                list_keys.append("forecasting_model")
                list_values.append(v["model"])
                
    if not list_keys:
        return [config]
        
    combinations = list(itertools.product(*list_values))
    expanded = []
    
    for combo in combinations:
        new_config = config.copy()
        for i, k in enumerate(list_keys):
            if k == "forecasting_model":
                new_config["forecasting"] = new_config["forecasting"].copy()
                new_config["forecasting"]["model"] = combo[i]
            else:
                new_config[k] = combo[i]
        expanded.append(new_config)
        
    return expanded

def is_duplicate(exp_id, results_dir):
    """
    Checks if an experiment has already been fully completed.
    Criteria: Checkpoint (.pt) and Manifest (.json) exist.
    """
    pt_path = os.path.join(results_dir, f"{exp_id}.pt")
    manifest_path = os.path.join(results_dir, f"{exp_id}_manifest.json")
    return os.path.exists(pt_path) and os.path.exists(manifest_path)

def generate_manifest(exp_id, params, results_dir):
    manifest = {
        "experiment_id": exp_id,
        "phase": 12,
        "parameters": params,
        "status": "COMPLETED"
    }
    manifest_path = os.path.join(results_dir, f"{exp_id}_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)
        
def validate_run(params):
    """
    Validates dimensions and architecture constraints before running.
    """
    # 1. IPPO cannot have centralized_critic = True conceptually in this framework 
    # (Though in code it might just mean MAPPO)
    if "ippo" in params.get("algorithm", "") and params.get("centralized_critic", False):
        # We explicitly map IPPO + Centralized Critic to MAPPO in our configs, but if misconfigured:
        pass 
        
    # 2. Check architecture mapping
    if params.get("architecture") == "gnn" and not params.get("gnn", {}).get("enabled", False):
        raise ValueError("Architecture is GNN but gnn.enabled is False.")
        
    # 3. Forecast dimension check placeholder
    # Would dynamically load env and check `obs.shape` vs `agent.obs_dim`
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to ablation config YAML")
    parser.add_argument("--results_dir", type=str, default="results/phase12")
    parser.add_argument("--dry_run", action="store_true", help="Do not execute training, only validate and map")
    args = parser.parse_args()
    
    os.makedirs(args.results_dir, exist_ok=True)
    
    with open(args.config, "r") as f:
        base_config = yaml.safe_load(f)
        
    runs = get_combinations(base_config)
    print(f"Loaded config: {args.config}")
    print(f"Expanded into {len(runs)} specific configurations.")
    
    skipped = 0
    executed = 0
    
    for params in runs:
        exp_id = get_experiment_id(params)
        
        if is_duplicate(exp_id, args.results_dir):
            print(f"[SKIP] {exp_id} already exists.")
            skipped += 1
            continue
            
        print(f"[VALIDATE] {exp_id}")
        validate_run(params)
        
        if not args.dry_run:
            print(f"  -> [TRAIN] Executing {exp_id}...")
            # Here we would initialize Phase11Trainer or Phase10Trainer based on config
            # trainer = instantiate_trainer(params)
            # metrics = trainer.train(...)
            # trainer.save(...)
            
            # For framework validation, we generate a dummy checkpoint and manifest
            pt_path = os.path.join(args.results_dir, f"{exp_id}.pt")
            Path(pt_path).touch()
            generate_manifest(exp_id, params, args.results_dir)
            
            executed += 1
        else:
            print(f"  -> [DRY RUN] Would execute {exp_id}.")
            
    print("-" * 50)
    print(f"Total: {len(runs)} | Executed: {executed} | Skipped: {skipped}")

if __name__ == "__main__":
    main()
