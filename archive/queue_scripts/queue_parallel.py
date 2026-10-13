import subprocess
import time
import psutil

commands = [
    [".\\.venv\\Scripts\\python.exe", "run_phase11_worker.py", "--forecaster_type", "ma", "--seed", "1"],
    [".\\.venv\\Scripts\\python.exe", "run_phase11_worker.py", "--forecaster_type", "ma", "--seed", "2"],
    [".\\.venv\\Scripts\\python.exe", "run_phase11_worker.py", "--forecaster_type", "xgboost", "--seed", "100"],
    [".\\.venv\\Scripts\\python.exe", "run_phase11_worker.py", "--forecaster_type", "xgboost", "--seed", "101"],
]

max_workers = 2
active_procs = []

def is_pid_running(pid):
    try:
        return psutil.Process(pid).is_running()
    except psutil.NoSuchProcess:
        return False

print("Starting max-2 concurrent queue...")

while commands or active_procs:
    for p in active_procs[:]:
        if p.poll() is not None:
            active_procs.remove(p)
            
    # Check if the external XGBoost 20-iter process is still running
    external_running = is_pid_running(18456)
    
    # Calculate how many slots we have available
    # If external is running, it consumes 1 slot
    current_total = len(active_procs) + (1 if external_running else 0)
    
    while current_total < max_workers and commands:
        cmd = commands.pop(0)
        print(f"Starting: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd)
        active_procs.append(proc)
        current_total += 1
        
    time.sleep(5)

print("All runs finished.")
