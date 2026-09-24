import numpy as np
from baselines.out import OUTPolicy
from rl.pilot_env import PilotOneEchelonEnv

def eval_out():
    print("=== Evaluate OUT Baseline ===")
    policy = OUTPolicy(z_scores=[0.5], lead_times=[2], capacity=100)
    env = PilotOneEchelonEnv()
    
    seeds = [0, 1, 2]
    all_costs = []
    
    for seed in seeds:
        env.reset(seed=seed)
        episodes_completed = 0
        while episodes_completed < 10:
            current_ep_cost = 0.0
            done = False
            while not done:
                fake_state = env.get_fake_simulator_state()
                action = policy.get_actions(fake_state)[0]
                _, rews, terms, truncs, infos = env.step({"retailer": action})
                current_ep_cost += infos["retailer"]["cost"]
                done = terms["retailer"] or truncs["retailer"]
            all_costs.append(current_ep_cost)
            episodes_completed += 1
            env.reset()
            
    mean_cost = np.mean(all_costs)
    print(f"OUT Mean Cost: {mean_cost:.2f} ± {np.std(all_costs):.2f}")

if __name__ == "__main__":
    eval_out()
