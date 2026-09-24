import time
import torch
from rl.phase6_trainer import Phase6Trainer

def benchmark():
    ablation = {"centralized_critic": True, "communication": False}
    
    # --- CPU ---
    import unittest.mock as mock
    with mock.patch("torch.cuda.is_available", return_value=False):
        trainer_cpu = Phase6Trainer("configs/phase4_ippo.yaml", ablation)
        
        t0 = time.time()
        trainer_cpu.train(1, use_wandb=False)
        t1 = time.time()
        print(f"CPU Time for 1 iteration (2048 steps * 32 envs): {t1 - t0:.2f} s")
        
    # --- GPU ---
    trainer_gpu = Phase6Trainer("configs/phase4_ippo.yaml", ablation)
    t0 = time.time()
    trainer_gpu.train(1, use_wandb=False)
    t1 = time.time()
    print(f"GPU Time for 1 iteration (2048 steps * 32 envs): {t1 - t0:.2f} s")

if __name__ == "__main__":
    benchmark()
