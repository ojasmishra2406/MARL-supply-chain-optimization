import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from rl.pilot_env import PilotOneEchelonEnv
from baselines.out import OUTPolicy
from rl.networks import ActorNetwork

def generate_dataset(num_transitions=10000):
    env = PilotOneEchelonEnv()
    policy = OUTPolicy(z_scores=[0.5], lead_times=[2], capacity=100)
    
    states = []
    actions = []
    
    env.reset(seed=42)
    for _ in range(num_transitions):
        fake_state = env.get_fake_simulator_state()
        action_list = policy.get_actions(fake_state)
        action = action_list[0]
        
        obs = env._get_observations()["retailer"]
        states.append(obs)
        actions.append(action)
        
        _, _, terms, truncs, _ = env.step({"retailer": action})
        if terms["retailer"] or truncs["retailer"]:
            env.reset()
            
    return np.array(states, dtype=np.float32), np.array(actions, dtype=np.float32)

def main():
    print("Generating dataset...")
    X, y = generate_dataset(10000)
    
    # Split
    split = int(0.8 * len(X))
    X_train, X_val = torch.tensor(X[:split]), torch.tensor(X[split:])
    y_train, y_val = torch.tensor(y[:split]), torch.tensor(y[split:])
    
    print("Training Supervised Imitation Policy...")
    actor = ActorNetwork(obs_dim=5, action_dim=101, hidden_size=128)
    optimizer = optim.Adam(actor.parameters(), lr=1e-3)
    
    epochs = 50
    batch_size = 64
    
    for epoch in range(epochs):
        permutation = torch.randperm(X_train.size()[0])
        epoch_loss = 0.0
        
        for i in range(0, X_train.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]
            
            # Actor network outputs a Categorical distribution.
            # We want to train its mean to match the target.
            # Actually, the user says "Train this network via plain supervised regression/imitation... negative log-likelihood".
            dist = actor(batch_x)
            # The action space is discrete, but our target is a float/int.
            # Negative log-likelihood of the rounded action:
            target_discrete = torch.clamp(torch.round(batch_y), 0, 100).long()
            loss = -dist.log_prob(target_discrete).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        with torch.no_grad():
            val_dist = actor(X_val)
            val_target = torch.clamp(torch.round(y_val), 0, 100).long()
            val_loss = -val_dist.log_prob(val_target).mean().item()
            # Calculate MSE of the mean just for reporting
            mean_pred = actor.net(X_val / 100.0) * 100.0
            val_mse = nn.MSELoss()(mean_pred.squeeze(), y_val).item()
            
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: Train NLL={epoch_loss:.4f}, Val NLL={val_loss:.4f}, Val MSE={val_mse:.4f}")
            
    print("Evaluating Imitation Policy in Environment...")
    env = PilotOneEchelonEnv()
    seeds = [0, 1, 2]
    costs = []
    
    actor.eval()
    for seed in seeds:
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
        print(f"Seed {seed} cost: {cost:.2f}")
        
    print(f"Mean cost: {np.mean(costs):.2f}")
    
    checkpoint_path = "results/phase4b_step1_imitation_actor_recreated.pt"
    torch.save(actor.state_dict(), checkpoint_path)
    print(f"Saved recreated checkpoint to {checkpoint_path}")
    
    import hashlib
    with open(checkpoint_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    print(f"SHA256: {file_hash}")
    
    with open("results/step1_2_imitation_recreated.json", "w") as f:
        json.dump({
            "val_nll": val_loss,
            "val_mse": val_mse,
            "deployed_costs": costs,
            "mean_deployed_cost": np.mean(costs),
            "checkpoint_path": checkpoint_path,
            "sha256": file_hash
        }, f, indent=2)

if __name__ == "__main__":
    main()
