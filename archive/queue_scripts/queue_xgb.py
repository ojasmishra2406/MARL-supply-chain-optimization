import subprocess

commands = [
    [".\\.venv\\Scripts\\python.exe", "run_phase11_worker.py", "--forecaster_type", "xgboost", "--seed", "100"],
    [".\\.venv\\Scripts\\python.exe", "run_phase11_worker.py", "--forecaster_type", "xgboost", "--seed", "101"],
]

for cmd in commands:
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print("Done.")
