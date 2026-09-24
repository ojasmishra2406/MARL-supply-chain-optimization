import json
import os
import argparse
from rl.phase6_trainer import Phase6Trainer
from rl.phase5_evaluator import evaluate_phase5_policy
from rl.checkpoint import save_checkpoint
import datetime
import subprocess

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "unknown"

def generate_manifest(run_id, algo, condition, seed, config, ckpt_path, ckpt_hash, eval_res):
    manifest = {
        "experiment_id": run_id,
        "algorithm": algo,
        "condition": condition,
        "seed": seed,
        "config_path": "configs/phase4_ippo.yaml",
        "config_hash": "TODO",
        "checkpoint_path": ckpt_path,
        "checkpoint_sha256": ckpt_hash,
        "evaluation_result": eval_res,
        "software_version": "1.0",
        "git_commit": get_git_commit(),
        "timestamp": datetime.datetime.now().isoformat()
    }
    manifest_path = f"results/phase6/{run_id}_manifest.json"
    os.makedirs("results/phase6", exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Actually run the expensive matrix")
    args = parser.parse_args()

    with open("configs/phase6_matrix.json", "r") as f:
        matrix = json.load(f)

    print("Phase 6 Distribution Matrix Plan:")
    
    total_rl = len(matrix["algorithms"]) * len(matrix["conditions"]) * len(matrix["seeds"])
    total_out = matrix["out_runs"]
    print(f"RL Runs: {total_rl}")
    print(f"OUT Runs: {total_out}")
    print(f"Total Runs: {total_rl + total_out} (Resolved Accounting Inconsistency: 40 + 30 = 70, not 270)")

    print("Executing Phase 6 Matrix...")
    
    os.makedirs("results/phase6", exist_ok=True)
    
    tasks = []
    
    for algo in matrix["algorithms"]:
        for condition in matrix["conditions"]:
            for seed in matrix["seeds"]:
                run_id = f"phase6_{algo}_{condition}_s{seed}"
                
                # Check if already done
                if os.path.exists(f"results/phase6/{run_id}_manifest.json"):
                    print(f"Skipping {run_id}, already exists.")
                    continue
                    
                tasks.append((run_id, algo, condition, seed))
                
    # Optimize CPU thrashing for PyTorch multiprocessing
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    print(f"Queued {len(tasks)} RL tasks.")
    
    import multiprocessing
    
    # MAPPO comm tasks have huge centralized critics — run sequentially to avoid OOM
    mappo_comm_tasks = [t for t in tasks if "mappo_comm" in t[0]]
    other_tasks = [t for t in tasks if "mappo_comm" not in t[0]]
    
    if other_tasks:
        with multiprocessing.Pool(4) as pool:
            pool.map(run_single_task, other_tasks)
    
    # Run mappo_comm sequentially (pool size 1)
    for task in mappo_comm_tasks:
        run_single_task(task)
        
    print("All Phase 6 RL tasks completed.")
    
def run_single_task(args):
    run_id, algo, condition, seed = args
    print(f"[{run_id}] Starting...")
    
    ablation = {}
    if algo == "mappo":
        ablation["centralized_critic"] = True
    else:
        ablation["centralized_critic"] = False
        
    if "comm" in algo:
        ablation["communication"] = True
    else:
        ablation["communication"] = False
        
    trainer = Phase6Trainer("configs/phase4_ippo.yaml", ablation)
    
    # Run 200 iterations
    for _ in range(200):
        trainer.train(1, use_wandb=False)
        
    ckpt_path = f"results/phase6/{run_id}.pt"
    manifest_path = f"results/phase6/{run_id}_manifest.json"
    
    state_dicts = {a: trainer.agents[a].state_dict() for a in trainer.agents_names}
    hash_val = save_checkpoint(state_dicts, ckpt_path, manifest_path, {"iteration": 200})
    
    # We do a quick evaluation (1 seed) to populate the manifest
    eval_res = evaluate_phase5_policy(trainer.agents, "configs/phase4_ippo.yaml", ablation, seeds=[0], num_episodes_per_seed=1, deterministic=True)
    
    generate_manifest(run_id, algo, condition, seed, "configs/phase4_ippo.yaml", ckpt_path, hash_val, eval_res)
    
    print(f"[{run_id}] Completed.")

if __name__ == "__main__":
    main()
