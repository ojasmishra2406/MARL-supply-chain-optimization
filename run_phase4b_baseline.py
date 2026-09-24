import json
import uuid
import yaml
import time
from datetime import datetime, timezone
import subprocess
import os
import sys

# Temporarily patch PilotTrainer for iteration
from rl.pilot import PilotTrainer, run_pilot

def main():
    run_id = str(uuid.uuid4())
    start_dt = datetime.now(timezone.utc).isoformat()
    start_time = time.time()
    
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except:
        commit_hash = "unknown"
        
    print(f"Running Step 0 Baseline. Commit: {commit_hash}")
    
    # We will just use the run_pilot function but pass our own output file.
    # run_pilot already loops over seeds [0, 1, 2].
    # We will pass updates=25.
    
    output_file = "phase_4b_baseline_results.json"
    run_pilot(updates=25, seeds=[0, 1, 2], output_file=output_file)
    
    end_time = time.time()
    
    manifest = {
        "run_id": run_id,
        "timestamp": start_dt,
        "config_path": "configs/phase4_ippo.yaml",
        "execution_time_seconds": end_time - start_time,
        "commit_hash": commit_hash,
        "change_justification": "Step 0 Baseline Re-run: Reverting reward scaling to /100.0. Ordinal Actor retained. Seeds 0,1,2. Hyperparams standard.",
        "tags": ["phase_4b_baseline_rerun"]
    }
    
    with open("results/manifest_phase_4b_baseline.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    print("Baseline complete. Wrote manifest.")

if __name__ == "__main__":
    main()
