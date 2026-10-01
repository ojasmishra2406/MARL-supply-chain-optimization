import os
import sys
import json
import argparse
import torch
import numpy as np

sys.path.append(".")
from rl.checkpoint import load_checkpoint
from rl.phase7_evaluator import evaluate_scenario

def check_env():
    import platform
    import gymnasium
    import pettingzoo
    
    print("\n--- Environment Verification ---")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Gymnasium: {gymnasium.__version__}")
    print(f"PettingZoo: {pettingzoo.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("--------------------------------\n")

def reproduce_model(model_id, registry_path="models/registry.json", smoke_test=True):
    with open(registry_path, "r") as f:
        registry = json.load(f)
        
    if model_id not in registry:
        raise ValueError(f"Model ID '{model_id}' not found in registry.")
        
    reg = registry[model_id]
    
    if reg["status"] not in ["TRAINED", "EVALUATED", "VALID"]:
        raise ValueError(f"Cannot reproduce model with status: {reg['status']}")
        
    ckpt_path = reg.get("actual_checkpoint", reg.get("checkpoint_path"))
    manifest_path = reg.get("manifest_path", reg.get("metadata", {}).get("manifest_path"))
    if manifest_path is None and ckpt_path:
        manifest_path = ckpt_path.replace(".pt", "_manifest.json")
    config_path = reg.get("config_path", reg.get("metadata", {}).get("config_path", "configs/phase4_ippo.yaml"))
    algo = reg.get("algorithm", reg.get("metadata", {}).get("algorithm", "ippo"))
    
    print(f"Reproducing {model_id} (Algorithm: {algo})...")
    
    # Verify checkpoint hashes using the trusted checkpoint module
    try:
        print("[1] Validating Checkpoint Hash...")
        state_dict = load_checkpoint(ckpt_path, manifest_path)
        print("    -> Hash MATCHES.")
    except Exception as e:
        print(f"    -> [FAIL] Hash mismatch or load error: {e}")
        sys.exit(1)
        
    print(f"[2] Validating Execution Determinism...")
    
    # We run the in_distribution scenario for eval_seed 0 to see if it matches recorded json
    test_scenario = "in_distribution"
    test_seed = 0
    
    print(f"    -> Running Evaluation: Scenario '{test_scenario}', Seed {test_seed}")
    
    # Execute reproduction eval
    res = evaluate_scenario(
        algorithm=algo, 
        checkpoint_path=ckpt_path, 
        config_path=config_path, 
        scenario_id=test_scenario, 
        seed=test_seed, 
        manifest_path=manifest_path
    )
    
    actual_cost = res["cost"]
    
    # Compare with existing eval if available
    eval_match = False
    for eval_file in reg.get("evaluations", []):
        if "in_distribution" in eval_file and f"s{test_seed}.json" in eval_file:
            with open(eval_file, "r") as f:
                historical_eval = json.load(f)
                historical_cost = historical_eval.get("cost")
                
                print(f"    -> Historical Cost: {historical_cost}")
                print(f"    -> Reproduced Cost: {actual_cost}")
                
                if np.isclose(historical_cost, actual_cost, rtol=1e-5):
                    print("    -> [SUCCESS] Outputs match perfectly deterministically.")
                    eval_match = True
                else:
                    print("    -> [FAIL] Outputs deviated!")
            break
            
    if not eval_match and len(reg.get("evaluations", [])) > 0:
        print("[WARNING] Could not find the specific historical evaluation file to compare against, or output deviated.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model ID from registry")
    parser.add_argument("--smoke-test", action="store_true", help="Run lightweight reproducibility check")
    parser.add_argument("--full-training", action="store_true", help="Retrain model completely (WARNING: Expensive)")
    args = parser.parse_args()
    
    check_env()
    
    if args.full_training:
        print("[WARNING] Full training reproduction requested.")
        print("This requires launching the correct training matrix command.")
        print("Not executed for safety during infrastructure phase.")
        sys.exit(0)
        
    if args.smoke_test:
        reproduce_model(args.model, smoke_test=True)

if __name__ == "__main__":
    main()
