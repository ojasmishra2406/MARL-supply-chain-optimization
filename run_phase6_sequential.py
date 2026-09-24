import subprocess
import sys
import datetime
import os

seeds = [0, 1, 2, 3, 4]
algo = "mappo_comm"
condition = "high_variance"

log_file = open("sequential_phase6.log", "a")

def log(msg):
    full_msg = f"[{datetime.datetime.now().isoformat()}] {msg}\n"
    print(full_msg, end="", flush=True)
    log_file.write(full_msg)
    log_file.flush()

log(f"Starting sequential execution of {len(seeds)} seeds for Phase 6.")

for seed in seeds:
    run_id = f"phase6_{algo}_{condition}_s{seed}"
    log(f"--- Starting {run_id} ---")
    
    cmd = [
        sys.executable, "-u", "run_mappo_comm_worker.py",
        run_id, algo, condition, str(seed)
    ]
    
    # Run process synchronously so they happen one after the other
    result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    
    if result.returncode != 0:
        log(f"WARNING: {run_id} failed with exit code {result.returncode}")
    else:
        log(f"SUCCESS: {run_id} completed.")

log("Finished sequential execution of all Phase 6 seeds.")
log_file.close()
