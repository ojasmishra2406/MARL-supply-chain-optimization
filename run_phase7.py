import os
import json
import glob
from rl.phase7_evaluator import evaluate_scenario

def main():
    os.makedirs("results/phase7", exist_ok=True)
    
    scenarios = [
        "in_distribution",
        "demand_shift",
        "lead_time_shift",
        "capacity_disruption",
        "demand_spike",
        "combined_shift"
    ]
    
    seeds = [0, 1, 2, 3, 4]
    
    print("Starting Phase 7 Evaluation...")
    
    # 1. OUT Baseline
    for scenario in scenarios:
        for seed in seeds:
            res_file = f"results/phase7/out_{scenario}_s{seed}.json"
            if os.path.exists(res_file):
                continue
            print(f"Evaluating OUT | {scenario} | seed {seed}")
            res = evaluate_scenario("OUT", None, "configs/phase4_ippo.yaml", scenario, seed)
            with open(res_file, "w") as f:
                json.dump(res, f, indent=2)
            
    # 2. Phase 6 Policies
    phase6_manifests = glob.glob("results/phase6/*_manifest.json")
    # exclude ci
    phase6_manifests = [m for m in phase6_manifests if "ci_test" not in m]
    
    for manifest_path in phase6_manifests:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
            
        run_id = manifest["experiment_id"]
        ckpt_path = manifest["checkpoint_path"]
        algo = manifest["algorithm"]
        condition = manifest["condition"]
        train_seed = manifest["seed"]
        
        # We evaluate each trained policy on all scenarios, but which seed?
        # The Master Prompt says: "Evaluate the trained policies across the six Phase 7 scenarios ... seeds {0,1,2,3,4}."
        # This implies we evaluate EACH trained policy on 5 evaluation seeds?
        # Or does the evaluation seed match the training seed?
        # Phase 8 prompt says: "The independent training seeds are: {0,1,2,3,4}... Do NOT treat every episode from the same trained policy as an independent training run... Episode-level data: Used for estimating evaluation variability and bootstrap CIs."
        # Wait, if we evaluate each trained policy on 5 evaluation seeds (which are the episode seeds), then each policy produces 5 episodes!
        # Let's evaluate each policy on all 5 evaluation seeds.
        
        for scenario in scenarios:
            for eval_seed in seeds:
                res_file = f"results/phase7/{run_id}_{scenario}_evals{eval_seed}.json"
                if os.path.exists(res_file):
                    continue
                print(f"Evaluating {run_id} | {scenario} | eval_seed {eval_seed}")
                
                # We need to construct a unique identifier for the algorithm string if it includes comm/variance
                full_algo_name = f"{algo}_{condition}"
                
                res = evaluate_scenario(full_algo_name, ckpt_path, "configs/phase4_ippo.yaml", scenario, eval_seed, manifest_path=manifest_path)
                with open(res_file, "w") as f:
                    json.dump(res, f, indent=2)

    print("Phase 7 Evaluation Complete.")

if __name__ == "__main__":
    main()
