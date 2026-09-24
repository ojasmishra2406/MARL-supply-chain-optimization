"""
Worker script for a single mappo_comm training run.
Called by run_mappo_comm_parallel.py as an independent subprocess.
Usage: python run_mappo_comm_worker.py <run_id> <algo> <condition> <seed>
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

manifest_path = f"results/phase6/{run_id}_manifest.json"
if os.path.exists(manifest_path):
    print(f"[{run_id}] Already exists, skipping.")
    sys.exit(0)

print(f"[{run_id}] Starting at {datetime.datetime.now().isoformat()}...", flush=True)

import torch
from rl.phase6_trainer import Phase6Trainer
from rl.phase5_evaluator import evaluate_phase5_policy
from rl.checkpoint import _hash_state_dict

ablation = {
    "centralized_critic": True,
    "communication": True,
    "parameter_sharing": False,
}
if condition == "high_variance":
    ablation["reward_weights"] = {"holding": 2.0, "backlog": 4.0, "ordering": 0.5}

trainer = Phase6Trainer("configs/phase4_ippo.yaml", ablation)

for i in range(200):
    trainer.train(1, use_wandb=False)
    if (i + 1) % 50 == 0:
        print(f"[{run_id}] Iteration {i+1}/200", flush=True)

# Save checkpoint
os.makedirs("results/phase6", exist_ok=True)
ckpt_path = f"results/phase6/{run_id}.pt"
state_dict = {name: trainer.agents[name].state_dict() for name in trainer.agents_names}
torch.save(state_dict, ckpt_path)
hash_val = _hash_state_dict(state_dict)

# Quick evaluation
eval_res = evaluate_phase5_policy(
    trainer.agents, "configs/phase4_ippo.yaml", ablation,
    seeds=[0], num_episodes_per_seed=1, deterministic=True
)

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
    "config_hash": "TODO",
    "checkpoint_path": ckpt_path,
    "checkpoint_sha256": hash_val,
    "evaluation_result": eval_res,
    "software_version": "1.0",
    "git_commit": git_commit,
    "timestamp": datetime.datetime.now().isoformat(),
}
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"[{run_id}] DONE.", flush=True)
