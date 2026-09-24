import os
import sys
import time
import subprocess
import argparse

def run_worker(device_type):
    # This function will be executed by the subprocess
    if device_type == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        # Ensure it sees the GPU
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        
    import torch
    from rl.phase10_trainer import Phase10Trainer
    
    # Verify device
    actual_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{device_type.upper()} Worker] PyTorch is using device: {actual_device}")
    
    # We use a deterministic ablation config to make it a fair race
    ablation = {
        "centralized_critic": True,
        "communication": False,
        "parameter_sharing": False,
        "seed": 42
    }
    
    trainer = Phase10Trainer("configs/phase4_ippo.yaml", ablation)
    
    iterations = 20
    
    print(f"[{device_type.upper()} Worker] Starting {iterations} iterations...")
    start_time = time.time()
    
    for i in range(iterations):
        trainer.train(1, use_wandb=False)
        
    elapsed = time.time() - start_time
    print(f"[{device_type.upper()} Worker] Finished {iterations} iterations in {elapsed:.2f} seconds.")
    print(f"[{device_type.upper()} Worker] Average time per iteration: {elapsed / iterations:.2f} seconds.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=str, choices=["cpu", "gpu"], default=None)
    args = parser.parse_args()
    
    if args.worker:
        run_worker(args.worker)
        return

    print("==================================================")
    print("      MARL CPU vs GPU BENCHMARK (Phase 10)        ")
    print("==================================================")
    
    print("\nRunning GPU Benchmark...")
    start_gpu = time.time()
    subprocess.run([sys.executable, __file__, "--worker", "gpu"])
    gpu_total = time.time() - start_gpu
    
    print("\nRunning CPU Benchmark...")
    start_cpu = time.time()
    subprocess.run([sys.executable, __file__, "--worker", "cpu"])
    cpu_total = time.time() - start_cpu
    
    print("\n==================================================")
    print("                  FINAL RESULTS                   ")
    print("==================================================")
    print(f"GPU Total Time: {gpu_total:.2f} seconds")
    print(f"CPU Total Time: {cpu_total:.2f} seconds")
    
    if cpu_total < gpu_total:
        speedup = gpu_total / cpu_total
        print(f"\n=> CPU IS FASTER by {speedup:.2f}x!")
    else:
        speedup = cpu_total / gpu_total
        print(f"\n=> GPU IS FASTER by {speedup:.2f}x!")

if __name__ == "__main__":
    main()
