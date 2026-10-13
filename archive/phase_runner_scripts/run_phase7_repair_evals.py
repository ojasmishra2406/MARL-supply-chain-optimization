import os
from rl.phase5_evaluator import evaluate_phase5_policy
from rl.phase7_evaluator import evaluate_scenario
from rl.checkpoint import load_checkpoint

def main():
    missing_models = ["phase6_mappo_comm_baseline_s1", "phase6_mappo_comm_baseline_s2"]
    scenarios = ["in_distribution", "demand_shift", "lead_time_shift", "capacity_disruption", "demand_spike", "combined_shift"]
    
    for model_id in missing_models:
        print(f"Evaluating {model_id}...")
        ckpt_path = f"results/phase6/{model_id}.pt"
        manifest_path = f"results/phase6/{model_id}_manifest.json"
        
        ablation = {"centralized_critic": True, "communication": True}
        
        for scenario in scenarios:
            try:
                evaluate_scenario(model_id, ckpt_path, manifest_path, "configs/phase4_ippo.yaml", ablation, scenario, "mappo_comm", "results/phase7")
            except Exception as e:
                print(f"Failed {model_id} - {scenario}: {e}")
                
if __name__ == "__main__":
    main()
