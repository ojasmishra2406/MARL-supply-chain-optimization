import os
import subprocess
import time
import sys

def main():
    print("Queueing evaluations for 10 Phase 11 models. Max 4 concurrently...")
    
    forecasters = ["xgboost", "ma"]
    seeds = [0, 1, 2, 3, 4]
    
    tasks = []
    for f in forecasters:
        for s in seeds:
            tasks.append(f"phase11_gnn_mappo_forecast_{f}_s{s}")
            
    MAX_CONCURRENT = 4
    active_processes = []
    
    try:
        while tasks or active_processes:
            # Clean up finished processes
            for p, run_id in active_processes[:]:
                if p.poll() is not None:
                    active_processes.remove((p, run_id))
                    if p.returncode != 0:
                        print(f"Eval FAILED: {run_id}. Check results/phase11/eval_{run_id}.log")
                    else:
                        print(f"Eval Success: {run_id}")
            
            # Spawn new processes if we have capacity
            while len(active_processes) < MAX_CONCURRENT and tasks:
                run_id = tasks.pop(0)
                print(f"Launching eval for {run_id}...")
                log_file = open(f"results/phase11/eval_{run_id}.log", "w")
                
                cmd = [
                    sys.executable,
                    "run_phase11_eval_worker.py",
                    run_id
                ]
                
                p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
                active_processes.append((p, run_id))
                
                time.sleep(1)
                
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("Keyboard interrupt! Terminating active evals...")
        for p, _ in active_processes:
            p.terminate()
            
    print("Phase 11 Evaluation Matrix Loop Complete.")

if __name__ == "__main__":
    main()
