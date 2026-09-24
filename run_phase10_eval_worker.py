"""
Worker script for evaluating a Phase 10 checkpoint.
Usage: python run_phase10_eval_worker.py <run_id>
"""
import os
import sys
import json
import traceback

run_id = sys.argv[1]
manifest_path = f"results/phase10/{run_id}_manifest.json"

if not os.path.exists(manifest_path):
    print(f"Manifest not found for {run_id}")
    sys.exit(1)

with open(manifest_path, "r") as f:
    manifest = json.load(f)

algo = manifest["algorithm"]
condition = manifest["condition"]
seed = manifest["seed"]
ckpt_path = manifest["checkpoint_path"]

print(f"Evaluating {run_id} ({algo}, {condition}, seed {seed})")

from rl.phase7_evaluator import evaluate_scenario
from rl.phase10_agent import Phase10Agent
import torch
import yaml

ablation = {
    "centralized_critic": True,
    "communication": False,
    "parameter_sharing": False,
    "seed": seed
}

with open(manifest["config_path"], "r") as f:
    config = yaml.safe_load(f)

# Initialize agents
agents_names = ["retailer", "wholesaler", "distributor", "factory"]
agents = {name: Phase10Agent(5, 4, 28, agent_index=i, ablation=ablation) for i, name in enumerate(agents_names)}

# Load checkpoint
state_dict = torch.load(ckpt_path)
for name, agent in agents.items():
    agent.load_state_dict(state_dict[name])

scenarios = [
    "in_distribution",
    "demand_shift",
    "lead_time_shift",
    "capacity_disruption",
    "demand_spike",
    "combined_shift"
]

eval_seeds = [0, 1, 2, 3, 4]
completed = 0

for scenario in scenarios:
    for e_seed in eval_seeds:
        out_file = f"results/phase10/{run_id}_{scenario}_evals{e_seed}.json"
        if os.path.exists(out_file):
            completed += 1
            continue
            
        print(f"Evaluating {scenario} seed {e_seed}...")
        try:
            res = evaluate_scenario(agents, "configs/phase4_ippo.yaml", ablation, scenario, e_seed)
            # Annotate with phase 10 specific metadata
            res["experiment_id"] = f"eval_{scenario}_{run_id}_{e_seed}"
            res["algorithm"] = algo
            res["checkpoint"] = ckpt_path
            res["checkpoint_hash"] = manifest.get("checkpoint_sha256", "unknown")
            with open(out_file, "w") as f:
                json.dump(res, f, indent=2)
            completed += 1
        except Exception as e:
            print(f"Failed {scenario} {e_seed}: {e}")
            traceback.print_exc()

print(f"Evaluations completed for {run_id}: {completed}/30")
