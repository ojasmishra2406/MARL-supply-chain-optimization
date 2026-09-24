import os, sys, yaml, json, time, torch
sys.path.insert(0, os.path.abspath(''))
from rl.pilot import PilotTrainer, SyncPilotVectorEnv
import rl.trainer

config = yaml.safe_load(open('configs/phase4_ippo.yaml'))
config['simulator_config'] = 'configs/phase1_simulator.yaml'
yaml.dump(config, open('configs/test.yaml', 'w'))

trainer = PilotTrainer('configs/test.yaml')

original_step = trainer.vec_env.step
def scaled_step(actions_list):
    obs, rews, terms, truncs, infos = original_step(actions_list)
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
for update in range(1, 101):
    metrics = trainer.train(1, use_wandb=False)
    # the total_cost metric is based on rewards, so we must multiply by 100
    cost = metrics['retailer_reward'] * -100.0
    best_cost = min(best_cost, cost)
    print(f"Update {update}: cost = {cost:.2f} (best: {best_cost:.2f})")
