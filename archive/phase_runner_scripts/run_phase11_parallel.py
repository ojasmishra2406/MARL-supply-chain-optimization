import os
import subprocess
import time
import sys

def main():
    print("Initializing Phase 11 Matrix Orchestrator...")
    os.makedirs("results/phase11", exist_ok=True)
    
    forecasters = ["xgboost", "ma"]
    seeds = [0, 1, 2, 3, 4]
    
    # Generate tasks
    tasks = []
    for f in forecasters:
        for s in seeds:
            tasks.append((f, s))
            
    # Filter completed tasks
    pending_tasks = []
    for f, s in tasks:
        manifest_path = f"results/phase11/phase11_gnn_mappo_forecast_{f}_s{s}_manifest.json"
        if not os.path.exists(manifest_path):
            pending_tasks.append((f, s))
            
    print(f"Total tasks: {len(tasks)}. Pending tasks: {len(pending_tasks)}")
    
    MAX_CONCURRENT = 2
    active_processes = []
    
    try:
        while pending_tasks or active_processes:
            # Clean up finished processes
            for p, f, s in active_processes[:]:
                if p.poll() is not None:
                    active_processes.remove((p, f, s))
                    print(f"[{f}_s{s}] Process finished with code {p.returncode}.")
            
            # Spawn new processes if we have capacity
            while len(active_processes) < MAX_CONCURRENT and pending_tasks:
                f, s = pending_tasks.pop(0)
                print(f"Spawning worker for {f} seed {s}...")
                log_file = open(f"results/phase11/phase11_gnn_mappo_forecast_{f}_s{s}.log", "a")
                
                cmd = [
                    sys.executable,
                    "run_phase11_worker.py",
                    "--forecaster_type", f,
                    "--seed", str(s)
                ]
                
                p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
                active_processes.append((p, f, s))
                
                # Small delay to prevent initial memory spikes colliding
                time.sleep(5)
                
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("Keyboard interrupt! Terminating active workers...")
        for p, f, s in active_processes:
            p.terminate()
            
    print("Phase 11 Matrix Loop Complete.")

if __name__ == "__main__":
    main()
