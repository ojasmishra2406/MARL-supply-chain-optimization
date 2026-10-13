import os
import subprocess

commands = [
    # Fast 20 iter XGBoost run for variance
    [".\\.venv\\Scripts\\python.exe", "run_phase11_worker_20.py", "--forecaster_type", "xgboost", "--seed", "123"],
    # Full runs for final stats
    [".\\.venv\\Scripts\\python.exe", "run_phase11_worker.py", "--forecaster_type", "ma", "--seed", "1"],
    [".\\.venv\\Scripts\\python.exe", "run_phase11_worker.py", "--forecaster_type", "ma", "--seed", "2"],
    # Full XGBoost runs
    [".\\.venv\\Scripts\\python.exe", "run_phase11_worker.py", "--forecaster_type", "xgboost", "--seed", "100"],
    [".\\.venv\\Scripts\\python.exe", "run_phase11_worker.py", "--forecaster_type", "xgboost", "--seed", "101"],
]

for cmd in commands:
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)
    print("Done.\n")
