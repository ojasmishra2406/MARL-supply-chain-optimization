import os
import sys
import json

# Optimize CPU thrashing for PyTorch
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def train_model(seed):
    print(f"Starting phase6_mappo_comm_baseline_s{seed}...")
    from rl.phase6_trainer import Phase6Trainer
    from rl.checkpoint import save_checkpoint
    
    run_id = f"phase6_mappo_comm_baseline_s{seed}"
    ablation = {"centralized_critic": True, "communication": True}
    trainer = Phase6Trainer("configs/phase4_ippo.yaml", ablation)
    
    for _ in range(200):
        trainer.train(1, use_wandb=False)
        
    ckpt_path = f"results/phase6/{run_id}.pt"
    manifest_path = f"results/phase6/{run_id}_manifest.json"
    
    state_dicts = {a: trainer.agents[a].state_dict() for a in trainer.agents_names}
    hash_val = save_checkpoint(state_dicts, ckpt_path, manifest_path, {"iteration": 200})
    print(f"Finished {run_id}.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        train_model(int(sys.argv[1]))
    else:
        # Run sequentially if no argument provided
        train_model(1)
        train_model(2)
