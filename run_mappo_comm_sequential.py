"""
Standalone sequential runner for the 8 remaining mappo_comm Phase 6 runs.
Runs in the MAIN process with no multiprocessing to avoid Windows OOM kills.
"""
import os
import sys
import hashlib
import datetime
import json

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

REMAINING = [
    ("phase6_mappo_comm_baseline_s0", "mappo_comm", "baseline", 0),
    ("phase6_mappo_comm_baseline_s3", "mappo_comm", "baseline", 3),
    ("phase6_mappo_comm_baseline_s4", "mappo_comm", "baseline", 4),
    ("phase6_mappo_comm_high_variance_s0", "mappo_comm", "high_variance", 0),
    ("phase6_mappo_comm_high_variance_s1", "mappo_comm", "high_variance", 1),
    ("phase6_mappo_comm_high_variance_s2", "mappo_comm", "high_variance", 2),
    ("phase6_mappo_comm_high_variance_s3", "mappo_comm", "high_variance", 3),
    ("phase6_mappo_comm_high_variance_s4", "mappo_comm", "high_variance", 4),
]

def run_one(run_id, algo, condition, seed):
    manifest_path = f"results/phase6/{run_id}_manifest.json"
    if os.path.exists(manifest_path):
        print(f"[{run_id}] Already exists, skipping.")
        return

    print(f"[{run_id}] Starting at {datetime.datetime.now().isoformat()}...")
    sys.stdout.flush()

    from rl.phase6_trainer import Phase6Trainer
    from rl.phase5_evaluator import evaluate_phase5_policy
    from rl.checkpoint import _hash_state_dict
    import torch

    ablation = {
        "centralized_critic": True,
        "communication": True,
        "parameter_sharing": False,
    }
    if condition == "high_variance":
        ablation["reward_weights"] = {"holding": 2.0, "backlog": 4.0, "order": 0.5}

    trainer = Phase6Trainer("configs/phase4_ippo.yaml", ablation)

    for i in range(200):
        trainer.train(1, use_wandb=False)
        if (i + 1) % 50 == 0:
            print(f"[{run_id}] Iteration {i+1}/200")
            sys.stdout.flush()

    # Save checkpoint
    os.makedirs("results/phase6", exist_ok=True)
    ckpt_path = f"results/phase6/{run_id}.pt"

    # Build combined state dict keyed by agent name
    state_dict = {name: trainer.agents[name].state_dict() for name in trainer.agents_names}
    torch.save(state_dict, ckpt_path)
    hash_val = _hash_state_dict(state_dict)

    # Quick evaluation
    eval_res = evaluate_phase5_policy(
        trainer.agents, "configs/phase4_ippo.yaml", ablation,
        seeds=[0], num_episodes_per_seed=1, deterministic=True
    )

    # Write manifest
    import subprocess
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

    print(f"[{run_id}] DONE. Manifest saved.")
    sys.stdout.flush()


if __name__ == "__main__":
    print(f"Running {len(REMAINING)} mappo_comm jobs sequentially in main process.")
    for args in REMAINING:
        run_one(*args)
    print("All mappo_comm runs complete!")
