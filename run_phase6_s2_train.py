import os
import time
import torch
import numpy as np

# Single thread to prevent contention
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def train_s2():
    from rl.phase6_trainer import Phase6Trainer
    from rl.checkpoint import save_checkpoint
    
    run_id = "phase6_mappo_comm_baseline_s2"
    ablation = {"centralized_critic": True, "communication": True}
    
    print(f"Starting {run_id} to 200 iterations...")
    trainer = Phase6Trainer("configs/phase4_ippo.yaml", ablation)
    
    for i in range(200):
        trainer.train(1, use_wandb=False)
        if (i+1) % 10 == 0:
            print(f"[{run_id}] Iteration {i+1}/200")
            
    ckpt_path = f"results/phase6/{run_id}.pt"
    manifest_path = f"results/phase6/{run_id}_manifest.json"
    
    state_dicts = {a: trainer.agents[a].state_dict() for a in trainer.agents_names}
    hash_val = save_checkpoint(state_dicts, ckpt_path, manifest_path, {"iteration": 200})
    print(f"Finished {run_id}. Hash: {hash_val}")

if __name__ == "__main__":
    train_s2()
