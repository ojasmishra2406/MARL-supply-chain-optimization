import os
import json
from rl.phase7_evaluator import evaluate_scenario

def generate_s2_evals():
    scenarios = [
        "in_distribution", "demand_shift", "lead_time_shift", 
        "capacity_disruption", "demand_spike", "combined_shift"
    ]
    model_id = "phase6_mappo_comm_baseline_s2"
    config_path = "configs/phase4_ippo.yaml"
    ckpt_path = f"results/phase6/{model_id}.pt"
    
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    for scenario in scenarios:
        for eval_seed in range(5):
            print(f"Evaluating {model_id} - {scenario} - seed {eval_seed}...")
            res = evaluate_scenario(
                algorithm="mappo_comm",
                checkpoint_path=ckpt_path,
                config_path=config_path,
                scenario_id=scenario,
                seed=eval_seed,
                manifest_path=ckpt_path.replace(".pt", "_manifest.json")
            )
            res_file = f"results/phase7/{model_id}_{scenario}_evals{eval_seed}.json"
            with open(res_file, "w") as f:
                json.dump(res, f, indent=2)
                
if __name__ == "__main__":
    generate_s2_evals()
