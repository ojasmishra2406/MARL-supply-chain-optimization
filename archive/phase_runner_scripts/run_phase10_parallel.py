import subprocess
import sys
import os
import time

REMAINING = [
    ("phase10_gnn_mappo_baseline_s0", "gnn_mappo", "baseline", "0"),
    ("phase10_gnn_mappo_baseline_s1", "gnn_mappo", "baseline", "1"),
    ("phase10_gnn_mappo_baseline_s2", "gnn_mappo", "baseline", "2"),
    ("phase10_gnn_mappo_baseline_s3", "gnn_mappo", "baseline", "3"),
    ("phase10_gnn_mappo_baseline_s4", "gnn_mappo", "baseline", "4"),
    ("phase10_gnn_mappo_high_variance_s0", "gnn_mappo", "high_variance", "0"),
    ("phase10_gnn_mappo_high_variance_s1", "gnn_mappo", "high_variance", "1"),
    ("phase10_gnn_mappo_high_variance_s2", "gnn_mappo", "high_variance", "2"),
    ("phase10_gnn_mappo_high_variance_s3", "gnn_mappo", "high_variance", "3"),
    ("phase10_gnn_mappo_high_variance_s4", "gnn_mappo", "high_variance", "4"),
]

WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_phase10_worker.py")

print(f"\nQueueing {len(REMAINING)} independent Phase 10 workers. Max 2 concurrently for robust execution...")

active_procs = []
task_queue = REMAINING.copy()
completed = 0

os.makedirs("results/phase10", exist_ok=True)

while task_queue or active_procs:
    # Fill queue up to 2
    while len(active_procs) < 2 and task_queue:
        run_id, algo, condition, seed = task_queue.pop(0)
        
        manifest_path = f"results/phase10/{run_id}_manifest.json"
        if os.path.exists(manifest_path):
            print(f"Skipping {run_id}, manifest already exists.")
            completed += 1
            continue
            
        print(f"Launching {run_id}...")
        log_file = open(f"results/phase10/{run_id}.log", "w")
        proc = subprocess.Popen(
            [sys.executable, WORKER_SCRIPT, run_id, algo, condition, seed],
            stdout=log_file,
            stderr=subprocess.STDOUT
        )
        active_procs.append((proc, log_file, run_id))
        
    # Monitor active processes
    for proc, log_file, run_id in active_procs.copy():
        ret = proc.poll()
        if ret is not None:
            active_procs.remove((proc, log_file, run_id))
            log_file.close()
            if ret == 0:
                print(f"Success: {run_id}")
                completed += 1
            else:
                print(f"FAILED: {run_id}. Check results/phase10/{run_id}.log")
                
    time.sleep(5)

print(f"\nPhase 10 Training Loop Complete. {completed}/{len(REMAINING)} tasks valid.")
