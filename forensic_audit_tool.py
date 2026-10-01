import os
import json
import glob
from collections import defaultdict
import hashlib

def main():
    print("=== ARTIFACT FORENSIC AUDIT ===")
    
    # Phase 6
    p6_ckpts = glob.glob("results/phase6/*.pt")
    p6_manifests = glob.glob("results/phase6/*manifest.json")
    print(f"Phase 6: {len(p6_ckpts)} checkpoints, {len(p6_manifests)} manifests")
    
    # Check Phase 6 manifest configs
    expected_p6 = 40
    valid_p6 = 0
    p6_corrupted = 0
    for mf in p6_manifests:
        try:
            with open(mf, 'r') as f:
                d = json.load(f)
                if "checkpoint_path" in d and os.path.exists(d["checkpoint_path"]):
                    valid_p6 += 1
        except:
            p6_corrupted += 1
            
    print(f"  Valid Phase 6 Models: {valid_p6} / {expected_p6}")
    
    # Phase 7 Evals
    p7_evals = glob.glob("results/phase7/*_evals*.json") + glob.glob("results/phase7/eval_*.json")
    print(f"Phase 7 Evals total found: {len(p7_evals)}")
    valid_p7 = 0
    p7_corrupted = 0
    for f in p7_evals:
        try:
            with open(f, 'r') as f_in:
                d = json.load(f_in)
                if "cost" in d or "total_cost" in d:
                    valid_p7 += 1
        except:
            p7_corrupted += 1
            
    # Phase 10
    p10_ckpts = glob.glob("results/phase10/*.pt")
    p10_manifests = glob.glob("results/phase10/*manifest.json")
    print(f"Phase 10: {len(p10_ckpts)} checkpoints, {len(p10_manifests)} manifests")
    p10_evals = glob.glob("results/phase10/*_evals*.json") + glob.glob("results/phase10/eval_*.json")
    print(f"Phase 10 Evals total found: {len(p10_evals)}")
    
    # Phase 11
    p11_ckpts = glob.glob("results/phase11/*.pt")
    p11_manifests = glob.glob("results/phase11/*manifest.json")
    print(f"Phase 11: {len(p11_ckpts)} checkpoints, {len(p11_manifests)} manifests")
    p11_evals = glob.glob("results/phase11/*_evals*.json") + glob.glob("results/phase11/eval_*.json")
    print(f"Phase 11 Evals total found: {len(p11_evals)}")
    
    # Phase 13 Registry
    if os.path.exists("models/registry.json"):
        with open("models/registry.json", "r") as f:
            registry = json.load(f)
        print(f"Phase 13 Registry: {len(registry)} models")
        valid_reg = sum(1 for m in registry.values() if m.get("status") in ["TRAINED", "EVALUATED"])
        placeholders = sum(1 for m in registry.values() if m.get("status") == "PLACEHOLDER")
        print(f"  Valid: {valid_reg}, Placeholders: {placeholders}")
    else:
        print("Phase 13 Registry: Not found")

if __name__ == "__main__":
    main()
