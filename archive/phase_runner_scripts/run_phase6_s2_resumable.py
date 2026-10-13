"""
Phase 6 MAPPO_COMM Baseline Seed 2 — Resumable Training Script
- Saves intermediate checkpoint every 10 iterations.
- Automatically resumes from checkpoint if interrupted.
- No torch.set_num_threads() restriction (uses hardware defaults matching s1 run).
"""
import os
import sys
import json
import time
import datetime
import torch

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

SEED = 2
RUN_ID = f"phase6_mappo_comm_baseline_s{SEED}"
TEMP_CKPT = f"results/phase6/{RUN_ID}_temp.pt"
FINAL_CKPT = f"results/phase6/{RUN_ID}.pt"
MANIFEST  = f"results/phase6/{RUN_ID}_manifest.json"
LOG_FILE  = f"results/phase6/{RUN_ID}_progress.log"
TOTAL_ITERS = 200
SAVE_EVERY  = 10   # save intermediate every N iterations

def log(msg):
    ts = datetime.datetime.now().isoformat()
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def save_temp(trainer, iteration):
    state = {
        "iteration": iteration,
        "agents": {a: trainer.agents[a].state_dict() for a in trainer.agents_names},
        "timestamp": datetime.datetime.now().isoformat(),
    }
    torch.save(state, TEMP_CKPT)
    log(f"Intermediate checkpoint saved at iteration {iteration}.")

def main():
    from rl.phase6_trainer import Phase6Trainer
    from rl.checkpoint import save_checkpoint

    os.makedirs("results/phase6", exist_ok=True)

    ablation = {"centralized_critic": True, "communication": True}
    log(f"Initializing trainer for {RUN_ID}...")
    trainer = Phase6Trainer("configs/phase4_ippo.yaml", ablation)

    # --- Resume from intermediate checkpoint if available ---
    start_iter = 0
    if os.path.exists(TEMP_CKPT):
        log(f"Found intermediate checkpoint at {TEMP_CKPT}. Loading...")
        try:
            state = torch.load(TEMP_CKPT, map_location="cpu")
            for a in trainer.agents_names:
                trainer.agents[a].load_state_dict(state["agents"][a])
            start_iter = state["iteration"]
            log(f"Resumed from iteration {start_iter}.")
        except Exception as e:
            log(f"WARNING: Could not load temp checkpoint ({e}). Starting from 0.")
            start_iter = 0
    else:
        log("No intermediate checkpoint found. Starting from scratch.")

    # --- Training loop ---
    wall_start = time.time()
    iter_times = []

    for i in range(start_iter, TOTAL_ITERS):
        iter_start = time.time()
        trainer.train(1, use_wandb=False)
        elapsed = time.time() - iter_start
        iter_times.append(elapsed)

        current_iter = i + 1

        if current_iter % SAVE_EVERY == 0:
            save_temp(trainer, current_iter)

            # Estimate time remaining
            avg_iter = sum(iter_times[-20:]) / len(iter_times[-20:])  # rolling 20-iter avg
            remaining = (TOTAL_ITERS - current_iter) * avg_iter
            eta = datetime.datetime.now() + datetime.timedelta(seconds=remaining)
            log(f"Iteration {current_iter}/{TOTAL_ITERS} | "
                f"avg {avg_iter:.1f}s/iter | "
                f"ETA: {eta.strftime('%Y-%m-%d %H:%M:%S')}")

    # --- Final checkpoint ---
    log("Training complete! Saving final research checkpoint...")
    state_dicts = {a: trainer.agents[a].state_dict() for a in trainer.agents_names}
    hash_val = save_checkpoint(state_dicts, FINAL_CKPT, MANIFEST, {"iteration": TOTAL_ITERS})
    log(f"Final checkpoint saved: {FINAL_CKPT}")
    log(f"SHA-256: {hash_val}")

    # Clean up temp
    if os.path.exists(TEMP_CKPT):
        os.remove(TEMP_CKPT)
        log("Intermediate checkpoint removed.")

    total_wall = time.time() - wall_start
    log(f"DONE. Total wall time: {total_wall/3600:.2f} hours.")

if __name__ == "__main__":
    main()
