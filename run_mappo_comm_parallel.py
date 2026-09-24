import subprocess
import sys
import os
import time

REMAINING = [
    ("phase6_mappo_comm_baseline_s0", "mappo_comm", "baseline", "0"),
    ("phase6_mappo_comm_baseline_s3", "mappo_comm", "baseline", "3"),
    ("phase6_mappo_comm_baseline_s4", "mappo_comm", "baseline", "4"),
    ("phase6_mappo_comm_high_variance_s0", "mappo_comm", "high_variance", "0"),
    ("phase6_mappo_comm_high_variance_s1", "mappo_comm", "high_variance", "1"),
    ("phase6_mappo_comm_high_variance_s2", "mappo_comm", "high_variance", "2"),
    ("phase6_mappo_comm_high_variance_s3", "mappo_comm", "high_variance", "3"),
    ("phase6_mappo_comm_high_variance_s4", "mappo_comm", "high_variance", "4"),
]

WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_mappo_comm_worker.py")

print(f"\nQueueing {len(REMAINING)} independent workers. Max 3 concurrently to avoid OOM...")

active_procs = []
task_queue = REMAINING.copy()

while task_queue or active_procs:
    # Fill queue up to 3
    while len(active_procs) < 3 and task_queue:
        run_id, algo, condition, seed = task_queue.pop(0)
        
        manifest_path = f"results/phase6/{run_id}_manifest.json"
        if os.path.exists(manifest_path):
            print(f"[SKIP] {run_id} already exists.")
            continue

        log_path = f"results/phase6/{run_id}_train.log"
        log_file = open(log_path, "w")

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"

        cmd = [sys.executable, "-u", WORKER_SCRIPT, run_id, algo, condition, seed]
        print(f"[LAUNCH] {run_id}")
        p = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, env=env)
        active_procs.append((run_id, p, log_file))
        
    time.sleep(10)
    still_running = []
    for run_id, p, lf in active_procs:
        ret = p.poll()
        if ret is None:
            still_running.append((run_id, p, lf))
        else:
            lf.close()
            if ret == 0:
                print(f"[DONE] {run_id}")
            else:
                print(f"[FAIL] {run_id} exited with code {ret} - check results/phase6/{run_id}_train.log")
    active_procs = still_running

print("\nAll workers finished.")
