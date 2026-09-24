import json
import uuid
import yaml
import time
from datetime import datetime, timezone
import subprocess
import os

from rl.pilot import run_pilot

def run_scale(scale_factor):
    run_id = str(uuid.uuid4())
    start_dt = datetime.now(timezone.utc).isoformat()
    start_time = time.time()
    
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except:
        commit_hash = "unknown"
        
    print(f"Running Step 3 Sweep for scale {scale_factor}. Commit: {commit_hash}")
    
    # Modify base config
    with open('configs/phase4_ippo.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    config['reward_scale'] = float(scale_factor)
    with open('configs/phase4_ippo.yaml', 'w') as f:
        yaml.dump(config, f)
        
    output_file = f"phase_4b_step3_scale_{scale_factor}.json"
    run_pilot(updates=25, seeds=[0, 1, 2], output_file=output_file)
    
    end_time = time.time()
    
    manifest = {
        "run_id": run_id,
        "timestamp": start_dt,
        "config_path": "configs/phase4_ippo.yaml",
        "execution_time_seconds": end_time - start_time,
        "commit_hash": commit_hash,
        "change_justification": f"Step 3 Sweep: Testing reward scaling of /{scale_factor} to inspect value-loss dominance.",
        "tags": ["phase_4b_step3_sweep", f"scale_{scale_factor}"]
    }
    
    with open(f"results/manifest_phase_4b_step3_{scale_factor}.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Sweep {scale_factor} complete.")
    
    # Reset config for safety
    del config['reward_scale']
    with open('configs/phase4_ippo.yaml', 'w') as f:
        yaml.dump(config, f)

if __name__ == "__main__":
    run_scale(1000)
    run_scale(10000)
