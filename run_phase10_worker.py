"""
Worker script for a single gnn_mappo training run.
Usage: python run_phase10_worker.py <run_id> <algo> <condition> <seed>
"""
import os
import sys
import json
import datetime
import subprocess

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

run_id   = sys.argv[1]
algo     = sys.argv[2]
condition= sys.argv[3]
seed     = int(sys.argv[4])

manifest_path = f"results/phase10/{run_id}_manifest.json"
if os.path.exists(manifest_path):
    print(f"[{run_id}] Already exists, skipping.")
    sys.exit(0)

print(f"[{run_id}] Starting at {datetime.datetime.now().isoformat()}...", flush=True)

import torch
from rl.phase10_trainer import Phase10Trainer
from rl.checkpoint import _hash_state_dict

ablation = {
    "centralized_critic": True,
    "communication": False,
    "parameter_sharing": False,
    "seed": seed
}
if condition == "high_variance":
    ablation["reward_weights"] = {"holding": 2.0, "backlog": 4.0, "ordering": 0.5}

trainer = Phase10Trainer("configs/phase4_ippo.yaml", ablation)

ckpt_path = f"results/phase10/{run_id}.pt"
intermediate_ckpt = f"results/phase10/{run_id}_temp.pt"
start_iter = 0

# Try to load intermediate checkpoint to resume
if os.path.exists(intermediate_ckpt):
    print(f"[{run_id}] Resuming from intermediate checkpoint...", flush=True)
    state_dict = torch.load(intermediate_ckpt)
    for name in trainer.agents_names:
        trainer.agents[name].load_state_dict(state_dict[name])
    start_iter = state_dict.get("iteration", 0)
    print(f"[{run_id}] Resumed at iteration {start_iter}", flush=True)

for i in range(start_iter, 200):
    trainer.train(1, use_wandb=False)
    if (i + 1) % 10 == 0:
        print(f"[{run_id}] Iteration {i+1}/200", flush=True)
        # Save intermediate
        temp_state_dict = {name: trainer.agents[name].state_dict() for name in trainer.agents_names}
        temp_state_dict["iteration"] = i + 1
        torch.save(temp_state_dict, intermediate_ckpt)

# Save final checkpoint
os.makedirs("results/phase10", exist_ok=True)
final_state_dict = {name: trainer.agents[name].state_dict() for name in trainer.agents_names}
torch.save(final_state_dict, ckpt_path)
if os.path.exists(intermediate_ckpt):
    os.remove(intermediate_ckpt)
hash_val = _hash_state_dict(final_state_dict)

try:
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
except Exception:
    git_commit = "unknown"

manifest = {
    "experiment_id": run_id,
    "algorithm": algo,
    "condition": condition,
    "seed": seed,
    "config_path": "configs/phase4_ippo.yaml",
    "checkpoint_path": ckpt_path,
    "checkpoint_sha256": hash_val,
    "software_version": "1.0",
    "git_commit": git_commit,
    "timestamp": datetime.datetime.now().isoformat(),
}
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"[{run_id}] DONE.", flush=True)
