import os
import subprocess
import time

def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    print("Starting full repair pipeline...")
    
    # 1. Run training repairs
    run_cmd(".\\.venv\\Scripts\\python.exe run_repairs.py")
    
    # 2. Run missing Phase 7 evaluations for the 2 missing Phase 6 models
    # There are 2 missing models: phase6_mappo_comm_baseline_s1 and s2
    # The eval script is rl/phase7_evaluator.py, let's just write a quick wrapper to invoke it
    run_cmd(".\\.venv\\Scripts\\python.exe run_phase7_repair_evals.py")
    
    # 3. Run missing Phase 11 evaluations for genuine XGBoost
    run_cmd(".\\.venv\\Scripts\\python.exe run_phase11_repair_evals.py")
    
    # 4. Run fixed Phase 12 statistics
    run_cmd(".\\.venv\\Scripts\\python.exe run_phase12_stats_fixed.py")
    
    # 5. Run full test suite to confirm everything
    run_cmd(".\\.venv\\Scripts\\python.exe -m pytest tests/ > pytest_final.log 2>&1")
    
    print("All repairs and verifications completed!")

if __name__ == "__main__":
    main()
