import json
import datetime
import subprocess

try:
    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
except Exception:
    commit_hash = "unknown"

with open("results/step1_2_imitation_recreated.json", "r") as f:
    data = json.load(f)

manifest = {
    "phase": "4d",
    "condition": "phase_4b_step1_checkpoint_recreation",
    "seed": 42, # Training seed for data generation
    "git_commit": commit_hash,
    "config_identity": "run_phase4b_step1.py",
    "observation_specification": "aggregate (5 dims)",
    "reward_scale": "N/A (Supervised)",
    "actor_architecture": "ActorNetwork(obs_dim=5, action_dim=101, hidden_size=128)",
    "checkpoint_path": data["checkpoint_path"],
    "checkpoint_hash": data["sha256"],
    "provenance": "The original Phase 4B actor weights were not serialized. This checkpoint was regenerated using the documented Phase 4B Step 1 procedure and is therefore a protocol-matched recreation, not the literal original checkpoint.",
    "change_justification": "Recreate the Phase 4B Step 1 imitation actor under the same documented protocol and serialize its weights so that Phase 4D can be performed legitimately.",
    "validation_metric": data["val_nll"],
    "deployed_evaluation_result": data["mean_deployed_cost"],
    "creation_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
}

with open("results/manifest_phase_4b_step1_checkpoint_recreation.json", "w") as f:
    json.dump(manifest, f, indent=2)

print("Created manifest.")
