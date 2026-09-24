import subprocess
import sys
import os
import time

MODELS = [
    "phase10_gnn_mappo_baseline_s0",
    "phase10_gnn_mappo_baseline_s1",
    "phase10_gnn_mappo_baseline_s2",
    "phase10_gnn_mappo_baseline_s3",
    "phase10_gnn_mappo_baseline_s4",
    "phase10_gnn_mappo_high_variance_s0",
    "phase10_gnn_mappo_high_variance_s1",
    "phase10_gnn_mappo_high_variance_s2",
    "phase10_gnn_mappo_high_variance_s3",
    "phase10_gnn_mappo_high_variance_s4",
]

WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_phase10_eval_worker.py")

active_procs = []
task_queue = MODELS.copy()

print(f"\nQueueing evaluations for {len(MODELS)} Phase 10 models. Max 4 concurrently...")

while task_queue or active_procs:
    while len(active_procs) < 4 and task_queue:
        run_id = task_queue.pop(0)
        
        manifest_path = f"results/phase10/{run_id}_manifest.json"
        if not os.path.exists(manifest_path):
            print(f"Skipping evaluation for {run_id} (Manifest missing - training not complete?)")
            continue
            
        print(f"Launching eval for {run_id}...")
        log_file = open(f"results/phase10/eval_{run_id}.log", "w")
        proc = subprocess.Popen(
            [sys.executable, WORKER_SCRIPT, run_id],
            stdout=log_file,
            stderr=subprocess.STDOUT
        )
        active_procs.append((proc, log_file, run_id))
        
    for proc, log_file, run_id in active_procs.copy():
        ret = proc.poll()
        if ret is not None:
            active_procs.remove((proc, log_file, run_id))
            log_file.close()
            if ret == 0:
                print(f"Eval Success: {run_id}")
            else:
                print(f"Eval FAILED: {run_id}. Check results/phase10/eval_{run_id}.log")
                
    time.sleep(2)

print("Phase 10 Evaluation Matrix Loop Complete.")
