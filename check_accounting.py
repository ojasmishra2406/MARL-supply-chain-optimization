import json
import os

with open("configs/phase6_matrix.json", "r") as f:
    matrix = json.load(f)

tasks = []
for algo in matrix["algorithms"]:
    for condition in matrix["conditions"]:
        for seed in matrix["seeds"]:
            run_id = f"phase6_{algo}_{condition}_s{seed}"
            tasks.append((run_id, algo, condition, seed))

completed = []
pending = []

for t in tasks:
    run_id = t[0]
    if os.path.exists(f"results/phase6/{run_id}_manifest.json"):
        completed.append(t)
    else:
        pending.append(t)

# CPython multiprocessing chunking calculation
pool_size = 8
chunksize, extra = divmod(len(pending), pool_size * 4)
if extra:
    chunksize += 1

print(f"Required: {len(tasks)}")
print(f"Completed: {len(completed)}")
print(f"Pending: {len(pending)}")
print(f"Chunksize: {chunksize}")

running = []
queued = []

for chunk_idx in range(pool_size):
    # The first item in the chunk is currently running
    start_idx = chunk_idx * chunksize
    if start_idx < len(pending):
        running.append(pending[start_idx])
        # The rest of the chunk is queued locally in the worker
        for i in range(start_idx + 1, min(start_idx + chunksize, len(pending))):
            queued.append(pending[i])

# The remaining chunks are queued in the pool
for i in range(pool_size * chunksize, len(pending)):
    queued.append(pending[i])

print(f"Running: {len(running)}")
print(f"Queued: {len(queued)}")

print("\n--- COMPLETED ---")
for r in completed:
    print(f"{r[0]} (Seed {r[3]})")

print("\n--- RUNNING ---")
for r in running:
    print(f"{r[0]} (Seed {r[3]})")
