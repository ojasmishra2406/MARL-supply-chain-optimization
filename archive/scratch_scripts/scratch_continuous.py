import os, sys, yaml, json, time, torch
from torch import nn
from torch.distributions.normal import Normal
import numpy as np

sys.path.insert(0, os.path.abspath(''))
from rl.pilot import PilotTrainer, SyncPilotVectorEnv
import rl.trainer

# Define continuous Actor
class ContinuousActorNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.log_std = nn.Parameter(torch.zeros(1))

    def forward(self, obs: torch.Tensor) -> Normal:
        mean = self.net(obs).squeeze(-1)
        # Scale mean to [0, 100] approximately? Wait, Tanh is better.
        # But let's just use raw linear mean for now.
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

# Override agent init
import rl.ppo
class ContinuousIPPOAgent(rl.ppo.IPPOAgent):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 128):
        super().__init__(obs_dim, action_dim, hidden_size)
        self.actor = ContinuousActorNetwork(obs_dim, hidden_size)

rl.ppo.IPPOAgent = ContinuousIPPOAgent

config = yaml.safe_load(open('configs/phase4_ippo.yaml'))
config['simulator_config'] = 'configs/phase1_simulator.yaml'
yaml.dump(config, open('configs/test2.yaml', 'w'))

trainer = PilotTrainer('configs/test2.yaml')

# Also scale obs/rewards as before
original_step = trainer.vec_env.step
def scaled_step(actions_list):
    # actions_list comes from action.cpu().numpy()
    # Continuous action is a float, we clip it and format it
    clipped_actions = []
    for a in actions_list:
        val = float(np.clip(a['retailer'], 0, 100))
        clipped_actions.append({'retailer': val})
        
    obs, rews, terms, truncs, infos = original_step(clipped_actions)
    for i in range(len(obs)):
        obs[i]['retailer'] = obs[i]['retailer'] / 100.0
        rews[i]['retailer'] = rews[i]['retailer'] / 100.0
        if 'final_observation' in infos[i]:
            infos[i]['final_observation']['retailer'] = infos[i]['final_observation']['retailer'] / 100.0
    return obs, rews, terms, truncs, infos

original_reset = trainer.vec_env.reset
def scaled_reset(seed=None):
    obs = original_reset(seed)
    for i in range(len(obs)):
        obs[i]['retailer'] = obs[i]['retailer'] / 100.0
    return obs

trainer.vec_env.step = scaled_step
trainer.vec_env.reset = scaled_reset
trainer.seed = 0

best_cost = float('inf')
for update in range(1, 51):
    metrics = trainer.train(1, use_wandb=False)
    cost = metrics['retailer_reward'] * -100.0
    best_cost = min(best_cost, cost)
    print(f"Update {update}: cost = {cost:.2f} (best: {best_cost:.2f})")
