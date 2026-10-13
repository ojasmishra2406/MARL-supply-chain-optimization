import os
import time
import torch
import numpy as np

# Apply basic execution optimizations
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def benchmark_phase11():
    from rl.phase11_trainer import Phase11Trainer
    
    print(f"Testing Phase 11 Initialization...")
    start_time = time.time()
    trainer = Phase11Trainer("configs/phase4_ippo.yaml", forecast_horizon=2, forecaster_type="xgboost")
    init_time = time.time() - start_time
    print(f"Init time: {init_time:.2f}s")
    
    print("Testing 3 iterations...")
    iter_times = []
    
    for i in range(3):
        iter_start = time.time()
        trainer.train(1)
        iter_time = time.time() - iter_start
        iter_times.append(iter_time)
        print(f"Iteration {i+1} time: {iter_time:.2f}s")
        
    print(f"Average time per iteration: {np.mean(iter_times):.2f}s")

if __name__ == "__main__":
    benchmark_phase11()
