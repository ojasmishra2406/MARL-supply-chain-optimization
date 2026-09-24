import os
import json
import torch
import multiprocessing
import hashlib
from rl.mappo_trainer import Phase5Trainer
from rl.phase5_evaluator import evaluate_phase5_policy
import numpy as np

def run_experiment(exp_name, ablation_config, seeds, updates=25):
    os.makedirs(f"results/phase5/{exp_name}", exist_ok=True)
    
    all_results = {}
    
    for seed in seeds:
        print(f"[{exp_name}] Starting Seed {seed}", flush=True)
        ablation_config["seed"] = seed
        
        trainer = Phase5Trainer("configs/phase4_ippo.yaml", ablation_config)
        
        learning_curve = {}
        for u in range(updates + 1):
            if u in [0, 5, 10, updates]:
                res = evaluate_phase5_policy(
                    trainer.agents, 
                    "configs/phase4_ippo.yaml", 
                    ablation_config, 
                    seeds=[0,1,2], 
                    num_episodes_per_seed=2, 
                    deterministic=True
                )
                learning_curve[u] = res["mean_cost"]
                print(f"[{exp_name}|S{seed}|U{u}] Cost: {res['mean_cost']:.2f}", flush=True)
                
            if u < updates:
                metrics = trainer.train(1, use_wandb=False)
                if u % 5 == 0:
                    print(f"  > P-Loss: {metrics.get('retailer_policy_loss', 0):.2f} | V-Loss: {metrics.get('retailer_value_loss', 0):.2f}", flush=True)
                    
        # Final rigorous eval
        final_res = evaluate_phase5_policy(
            trainer.agents, 
            "configs/phase4_ippo.yaml", 
            ablation_config, 
            seeds=[0,1,2,3,4], 
            num_episodes_per_seed=10, 
            deterministic=True
        )
        
        all_results[seed] = {
            "final_eval": final_res,
            "learning_curve": learning_curve,
            "final_metrics": metrics # from last update
        }
        
        # Save checkpoint
        ckpt_path = f"results/phase5/{exp_name}/seed{seed}.pt"
        # We save a dict of agent state dicts
        state_dicts = {a: trainer.agents[a].state_dict() for a in trainer.agents_names}
        torch.save(state_dicts, ckpt_path)
        
    with open(f"results/phase5/{exp_name}/results.json", "w") as f:
        json.dump(all_results, f, indent=2)

def worker(task):
    exp_name, ablation_config, seeds = task
    run_experiment(exp_name, ablation_config, seeds, updates=20)

if __name__ == "__main__":
    seeds = [0, 1, 2, 3, 4]
    
    experiments = {
        "core_mappo": {
            "centralized_critic": True,
            "parameter_sharing": False,
            "communication": False,
        },
        "ablation1_decentralized": {
            "centralized_critic": False,
            "parameter_sharing": False,
            "communication": False,
        },
        "ablation2_shared_params": {
            "centralized_critic": True,
            "parameter_sharing": True,
            "communication": False,
        },
        "ablation3_communication": {
            "centralized_critic": True,
            "parameter_sharing": False,
            "communication": True,
        },
        "ablation4_backlog_dominant": {
            "centralized_critic": True,
            "parameter_sharing": False,
            "communication": False,
            "reward_weights": {"holding": 1, "backlog": 5, "ordering": 0.5}
        },
        "ablation4_holding_dominant": {
            "centralized_critic": True,
            "parameter_sharing": False,
            "communication": False,
            "reward_weights": {"holding": 3, "backlog": 1, "ordering": 0.5}
        },
        "divergence": {
            "centralized_critic": True,
            "parameter_sharing": False,
            "communication": False,
            "learning_rate": 3e-3,
            "reward_scale": 1.0
        }
    }
    
    # We will run them sequentially in a single process to avoid memory issues,
    # or use multiprocessing pool with 2-3 workers.
    # Given the hardware, 2 workers is safe.
    tasks = [(name, cfg, seeds) for name, cfg in experiments.items()]
    
    with multiprocessing.Pool(3) as pool:
        pool.map(worker, tasks)
        
    print("All Phase 5 experiments completed!")
