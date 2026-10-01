import os
import sys
import subprocess
import time

def main():
    os.makedirs("results/phase11", exist_ok=True)
    seeds = [0, 1, 2, 3, 4]
    
    tasks = []
    for s in seeds:
        tasks.append(("xgboost", s))
        
    MAX_CONCURRENT = 2
    active_processes = []
    
    while tasks or active_processes:
        # Clean up finished
        for p, run_id in active_processes[:]:
            if p.poll() is not None:
                active_processes.remove((p, run_id))
                print(f"Finished {run_id}")
                
        # Start new
        while len(active_processes) < MAX_CONCURRENT and tasks:
            f, s = tasks.pop(0)
            run_id = f"phase11_gnn_mappo_forecast_genuine_{f}_s{s}"
            
            cmd = f".\\.venv\\Scripts\\python.exe run_phase11_worker.py --forecaster_type {f} --seed {s}"
            print(f"Starting {run_id}...")
            
            # Use specific environment variables to prevent locking
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            
            p = subprocess.Popen(cmd, shell=True, env=env)
            active_processes.append((p, run_id))
            
        time.sleep(2)
        
if __name__ == "__main__":
    main()
