import torch
import numpy as np
from rl.pilot import PilotTrainer
from evaluator import evaluate_policy
import hashlib
import json

def hash_tensor_dict(state_dict):
    m = hashlib.sha256()
    for k, v in sorted(state_dict.items()):
        m.update(k.encode())
        m.update(v.cpu().numpy().tobytes())
    return m.hexdigest()

def extract_action_stats(vec_env, actor):
    obs_list = vec_env.reset(seed=42)
    agent = "retailer"
    obs_tensor = torch.tensor(np.array([o[agent] for o in obs_list]), dtype=torch.float32)
    with torch.no_grad():
        dist = actor(obs_tensor)
        actions = dist.sample().cpu().numpy()
        entropy = dist.entropy().mean().item()
        
    return {
        "mean_action": float(np.mean(actions)),
        "std_action": float(np.std(actions)),
        "quantiles": [float(q) for q in np.percentile(actions, [0, 25, 50, 75, 100])],
        "entropy": float(entropy)
    }

def audit_micro():
    print("=== One-Update Micro-Diagnostic ===")
    config_path = "configs/phase4_ippo.yaml"
    checkpoint_path = "results/phase4b_step1_imitation_actor_recreated.pt"
    
    seed = 0
    trainer = PilotTrainer(config_path)
    trainer.seed = seed
    
    state_dict = torch.load(checkpoint_path, map_location=trainer.device)
    trainer.agents["retailer"].actor.load_state_dict(state_dict)
    
    # BEFORE
    actor_hash_before = hash_tensor_dict(trainer.agents["retailer"].actor.state_dict())
    critic_hash_before = hash_tensor_dict(trainer.agents["retailer"].critic.state_dict())
    
    eval_before = evaluate_policy(trainer.agents["retailer"].actor, seeds=[0,1,2], num_episodes_per_seed=10)
    act_stats_before = extract_action_stats(trainer.vec_env, trainer.agents["retailer"].actor)
    
    # Update
    metrics = trainer.train(1, use_wandb=False)
    
    # AFTER
    actor_hash_after = hash_tensor_dict(trainer.agents["retailer"].actor.state_dict())
    critic_hash_after = hash_tensor_dict(trainer.agents["retailer"].critic.state_dict())
    
    eval_after = evaluate_policy(trainer.agents["retailer"].actor, seeds=[0,1,2], num_episodes_per_seed=10)
    act_stats_after = extract_action_stats(trainer.vec_env, trainer.agents["retailer"].actor)
    
    report = {
        "before": {
            "actor_hash": actor_hash_before,
            "critic_hash": critic_hash_before,
            "cost": eval_before["mean_cost"],
            "fill_rate": eval_before["mean_fill_rate"],
            "bullwhip": eval_before["mean_bullwhip"],
            "action_stats": act_stats_before
        },
        "during": {
            "policy_loss": float(metrics.get("retailer_policy_loss", 0)),
            "value_loss": float(metrics.get("retailer_value_loss", 0)),
            "entropy": float(metrics.get("retailer_entropy", 0)),
            "explained_var": float(metrics.get("retailer_explained_var", 0)),
            "adv_var": float(metrics.get("retailer_adv_var", 0))
        },
        "after": {
            "actor_hash": actor_hash_after,
            "critic_hash": critic_hash_after,
            "cost": eval_after["mean_cost"],
            "fill_rate": eval_after["mean_fill_rate"],
            "bullwhip": eval_after["mean_bullwhip"],
            "action_stats": act_stats_after
        }
    }
    
    print(json.dumps(report, indent=2))
    
if __name__ == "__main__":
    audit_micro()
