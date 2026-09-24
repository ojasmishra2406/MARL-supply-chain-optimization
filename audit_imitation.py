import torch
from rl.networks import ActorNetwork
from evaluator import evaluate_policy
import numpy as np
import scipy.stats as st

def audit_imitation():
    print("=== Recreated Imitation Checkpoint Evaluation ===")
    actor = ActorNetwork(obs_dim=5, action_dim=101, hidden_size=128)
    actor.load_state_dict(torch.load("results/phase4b_step1_imitation_actor_recreated.pt"))
    actor.eval()
    
    # Evaluate using the true episode evaluator
    res = evaluate_policy(actor, seeds=[0, 1, 2], num_episodes_per_seed=10)
    
    print(f"Mean Cost: {res['mean_cost']:.2f}")
    print(f"Std Cost: {res['std_cost']:.2f}")
    
    # 95% Bootstrap CI over the 30 episodes
    costs = np.array(res['all_costs'])
    bootstrap_means = [np.mean(np.random.choice(costs, size=len(costs), replace=True)) for _ in range(1000)]
    ci_lower = np.percentile(bootstrap_means, 2.5)
    ci_upper = np.percentile(bootstrap_means, 97.5)
    print(f"95% CI (across 30 episodes): [{ci_lower:.2f}, {ci_upper:.2f}]")
    print(f"Fill Rate: {res['mean_fill_rate']:.4f}")
    print(f"Bullwhip: {res['mean_bullwhip']:.4f}")
    
    for seed, data in res["seeds"].items():
        print(f"Seed {seed} Cost: {data['mean_cost']:.2f}")

if __name__ == "__main__":
    audit_imitation()
