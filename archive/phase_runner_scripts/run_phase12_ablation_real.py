import os
import torch
import numpy as np
from rl.phase6_trainer import Phase6Trainer
from rl.phase7_evaluator import evaluate_scenario
from rl.checkpoint import save_checkpoint
import pandas as pd

def compute_cohens_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pool_sd = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / dof)
    return (np.mean(x) - np.mean(y)) / pool_sd

def main():
    print("Phase 12 Real Ablation: 2-Iteration (Undertrained) vs 200-Iteration (Fully Trained Baseline)")
    os.makedirs("results/phase12", exist_ok=True)
    
    print("Training undertrained MAPPO baseline for 2 iterations...")
    trainer = Phase6Trainer(config_path="configs/phase4_ippo.yaml", ablation_config={"algorithm": "mappo", "communication": True, "shared_params": False})
    for _ in range(2): trainer.train(total_updates=1)
        
    undertrained_path = "results/phase12/ablation_mappo_comm_2iters.pt"
    undertrained_manifest = "results/phase12/ablation_mappo_comm_2iters_manifest.json"
    state = {name: agent.state_dict() for name, agent in trainer.agents.items()}
    save_checkpoint(state, undertrained_path, undertrained_manifest)
    
    print("Evaluating Undertrained Model (5 seeds)...")
    costs_under = []
    for s in range(5):
        res = evaluate_scenario("mappo_comm", undertrained_path, "configs/phase4_ippo.yaml", "in_distribution", s, manifest_path=undertrained_manifest)
        costs_under.append(res["cost"])
    
    trained_path = "results/phase6/phase6_mappo_comm_baseline_s2.pt"
    trained_manifest = "results/phase6/phase6_mappo_comm_baseline_s2_manifest.json"
    if not os.path.exists(trained_path):
        print("Pre-trained baseline not found. Training for 200 iterations...")
        trainer_full = Phase6Trainer(config_path="configs/phase4_ippo.yaml", ablation_config={"algorithm": "mappo", "communication": True, "shared_params": False})
        for _ in range(200): trainer_full.train(total_updates=1)
        state_full = {name: agent.state_dict() for name, agent in trainer_full.agents.items()}
        save_checkpoint(state_full, trained_path, trained_manifest)
        
    print(f"Evaluating Fully Trained Model from {trained_path} (5 seeds)...")
    costs_full = []
    for s in range(5):
        res = evaluate_scenario("mappo_comm", trained_path, "configs/phase4_ippo.yaml", "in_distribution", s, manifest_path=trained_manifest)
        costs_full.append(res["cost"])
    
    perf_under = [-c for c in costs_under]
    perf_full = [-c for c in costs_full]
    d = compute_cohens_d(perf_full, perf_under)
    
    print("-" * 50)
    print("ABLATION RESULTS (Real)")
    print("-" * 50)
    print(f"Undertrained (2 iters) Mean Cost: {np.mean(costs_under):.2f}")
    print(f"Fully Trained (200 iters) Mean Cost: {np.mean(costs_full):.2f}")
    print(f"Real Cohen's d: {d:.4f} (Positive means Fully Trained is better)")

if __name__ == "__main__":
    main()
