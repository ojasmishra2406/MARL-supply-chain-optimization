import torch
import hashlib
from rl.networks import ActorNetwork
from rl.pilot_env import PilotOneEchelonEnv
import numpy as np

def verify_checkpoint():
    path = "results/phase4b_step1_imitation_actor_recreated.pt"
    
    with open(path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
        
    print(f"SHA256: {file_hash}")
    
    actor = ActorNetwork(obs_dim=5, action_dim=101, hidden_size=128)
    state_dict = torch.load(path)
    
    # Verify keys
    expected_keys = set(actor.state_dict().keys())
    actual_keys = set(state_dict.keys())
    assert expected_keys == actual_keys, "Keys do not match"
    print("Keys verified.")
    
    # Verify shapes and finitude
    for k, v in state_dict.items():
        assert actor.state_dict()[k].shape == v.shape, f"Shape mismatch for {k}"
        assert torch.isfinite(v).all(), f"Non-finite values in {k}"
    print("Shapes and finitude verified.")
    
    actor.load_state_dict(state_dict)
    actor.eval()
    
    # Evaluate
    env = PilotOneEchelonEnv()
    costs = []
    for seed in [0, 1, 2]:
        env.reset(seed=seed)
        done = False
        cost = 0.0
        while not done:
            obs = env._get_observations()["retailer"]
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                dist = actor(obs_t)
                action = dist.sample().item()
            _, rews, terms, truncs, _ = env.step({"retailer": float(action)})
            cost += -rews["retailer"]
            done = terms["retailer"] or truncs["retailer"]
        costs.append(cost)
        
    mean_cost = np.mean(costs)
    print(f"Loaded evaluation mean cost: {mean_cost:.2f}")

if __name__ == "__main__":
    verify_checkpoint()
