import os
import sys
import subprocess
import time
import datetime

PYTHON_EXE = ".\\.venv\\Scripts\\python.exe"
OUTPUT_DIR = "results/phase11_controlled"
TOTAL_UPDATES = 50
SEEDS = [1, 2, 3]
FORECASTERS = ["ma", "xgboost"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file_path = f"{OUTPUT_DIR}/controlled_study_orchestrator.log"

def log(msg):
    ts = datetime.datetime.now().isoformat()
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(log_file_path, "a") as f:
        f.write(line + "\n")

log("=" * 60)
log(f"STARTING PHASE 11 CONTROLLED MULTI-SEED STUDY (N={len(SEEDS)}, Updates={TOTAL_UPDATES})")
log("=" * 60)

for f in FORECASTERS:
    for s in SEEDS:
        run_id = f"phase11_controlled_{f}_s{s}"
        ckpt_path = f"{OUTPUT_DIR}/{run_id}.pt"
        manifest_path = f"{OUTPUT_DIR}/{run_id}_manifest.json"
        
        # 1. Training Step
        log(f">>> Launching Training: {run_id} ({f}, seed {s}, updates {TOTAL_UPDATES})")
        train_cmd = [
            PYTHON_EXE, "-u", "run_phase11_worker.py",
            "--forecaster_type", f,
            "--seed", str(s),
            "--total_updates", str(TOTAL_UPDATES),
            "--output_dir", OUTPUT_DIR,
            "--run_id", run_id
        ]
        
        t0 = time.time()
        res = subprocess.run(train_cmd, capture_output=False)
        dur = (time.time() - t0) / 60.0
        
        if res.returncode != 0:
            log(f"ERROR: Training failed for {run_id} with exit code {res.returncode}")
            continue
        log(f"Training completed for {run_id} in {dur:.2f} minutes.")
        
        # 2. Evaluation Step
        log(f">>> Launching 30-Episode OOD Evaluation: {run_id}")
        eval_cmd = [
            PYTHON_EXE, "-u", "run_phase11_eval_worker.py",
            run_id, OUTPUT_DIR
        ]
        t_eval = time.time()
        res_eval = subprocess.run(eval_cmd, capture_output=False)
        eval_dur = time.time() - t_eval
        
        if res_eval.returncode != 0:
            log(f"ERROR: Evaluation failed for {run_id} with exit code {res_eval.returncode}")
        else:
            log(f"Evaluation completed for {run_id} in {eval_dur:.1f} seconds.")

log("=" * 60)
log("CONTROLLED STUDY EXECUTION COMPLETED! RUNNING FINAL AGGREGATION...")
log("=" * 60)

# Run final aggregation and evaluation summary
agg_cmd = [PYTHON_EXE, "-u", "evaluate_phase11_controlled.py"]
subprocess.run(agg_cmd)
