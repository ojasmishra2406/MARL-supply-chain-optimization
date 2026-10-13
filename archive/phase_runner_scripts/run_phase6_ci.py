import supersuit as ss
from envs.supply_chain_env import SupplyChainParallelEnv
from rl.mappo_trainer import Phase5Trainer
from rl.checkpoint import save_checkpoint, load_checkpoint
import os

def run_ci():
    print("Starting Phase 6 CI Smoke Test...")
    
    from rl.phase6_trainer import Phase6Trainer
    ablation = {"centralized_critic": True, "communication": False}
    trainer = Phase6Trainer("configs/phase4_ippo.yaml", ablation)
    print(f"SuperSuit environment created with {trainer.env.num_envs} envs.")

    
    print("Training for exactly 50 iterations...")
    for u in range(50):
        trainer.train(1, use_wandb=False)
        print(f"Completed iteration {u+1}/50")
        
    # 3. Save, Hash
    ckpt_path = "results/phase6/ci_test_model.pt"
    manifest_path = "results/phase6/ci_test_manifest.json"
    
    state_dicts = {a: trainer.agents[a].state_dict() for a in trainer.agents_names}
    hash_val = save_checkpoint(state_dicts, ckpt_path, manifest_path, {"iteration": 50})
    print(f"Checkpoint saved with SHA-256: {hash_val}")
    
    # 4. Load, Verify
    loaded_state = load_checkpoint(ckpt_path, manifest_path)
    print("Checkpoint loaded and verified successfully.")
    
    # 5. Evaluate
    # We use evaluate_phase5_policy but we need to pass the loaded agents.
    from rl.phase5_evaluator import evaluate_phase5_policy
    for a in trainer.agents_names:
        trainer.agents[a].load_state_dict(loaded_state[a])
        
    eval_res = evaluate_phase5_policy(trainer.agents, "configs/phase4_ippo.yaml", ablation, seeds=[0], num_episodes_per_seed=1, deterministic=True)
    print(f"Evaluation Cost: {eval_res['mean_cost']:.2f}")
    
    print("Phase 6 CI Smoke Test PASSED.")

if __name__ == "__main__":
    run_ci()
