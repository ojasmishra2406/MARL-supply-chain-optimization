import json
import datetime
import subprocess

try:
    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
except Exception:
    commit_hash = "unknown"

for condition in ["frozen", "ppo_bootstrap"]:
    manifest = {
        "phase": "4d",
        "condition": f"phase_4d_{condition}",
        "seed": "[0, 1, 2]",
        "source_imitation_checkpoint": "results/phase4b_step1_imitation_actor_recreated.pt",
        "checkpoint_hash": "9695b2bd390553e954fcbd36fd07277780b589d963ceb7127eb9c638d541ca3f",
        "git_commit": commit_hash,
        "config_identity": "configs/phase4_ippo.yaml",
        "reward_scale": 100.0,
        "ppo_hyperparameters": "lr=3e-4, gamma=0.99, gae=0.95, clip=0.2, epochs=10, min_batch=64, rollout=2048, num_envs=8, hidden=128, layers=2, Tanh, entropy=0.01, value=0.5, grad_norm=0.5",
        "observation_specification": "aggregate",
        "update_count": 50,
        "change_justification": f"Test whether standard PPO updates preserve or destroy an already-good OUT-imitation policy ({condition} condition).",
        "creation_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
    with open(f"results/manifest_phase_4d_{condition}.json", "w") as f:
        json.dump(manifest, f, indent=2)

manifest_micro = {
    "phase": "4d",
    "condition": "phase_4d_one_update",
    "seed": "[0, 1, 2]",
    "source_imitation_checkpoint": "results/phase4b_step1_imitation_actor_recreated.pt",
    "checkpoint_hash": "9695b2bd390553e954fcbd36fd07277780b589d963ceb7127eb9c638d541ca3f",
    "git_commit": commit_hash,
    "config_identity": "configs/phase4_ippo.yaml",
    "reward_scale": 100.0,
    "ppo_hyperparameters": "lr=3e-4, gamma=0.99, gae=0.95, clip=0.2, epochs=10, min_batch=64, rollout=2048, num_envs=8, hidden=128, layers=2, Tanh, entropy=0.01, value=0.5, grad_norm=0.5",
    "observation_specification": "aggregate",
    "update_count": 1,
    "change_justification": "One-update micro-diagnostic to mechanically inspect PPO updates on an already-good OUT-imitation policy.",
    "creation_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
}
with open("results/manifest_phase_4d_one_update.json", "w") as f:
    json.dump(manifest_micro, f, indent=2)

print("Created all manifests.")
