import os
import subprocess

def main():
    models = [f"phase11_gnn_mappo_forecast_genuine_xgboost_s{i}" for i in range(5)]
    
    for model_id in models:
        print(f"Queueing {model_id} for Phase 11 eval...")
        subprocess.run([".\\.venv\\Scripts\\python.exe", "run_phase11_eval_worker.py", model_id])

if __name__ == "__main__":
    main()
