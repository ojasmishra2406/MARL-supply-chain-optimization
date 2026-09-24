import os, sys, yaml, json, time, torch
sys.path.insert(0, os.path.abspath(''))
from rl.pilot import PilotTrainer, SyncPilotVectorEnv

config = yaml.safe_load(open('configs/phase4_ippo.yaml'))
config['simulator_config'] = 'configs/phase1_simulator.yaml'
yaml.dump(config, open('configs/test.yaml', 'w'))

trainer = PilotTrainer('configs/test.yaml')
trainer.seed = 0

best_cost = float('inf')
for update in range(1, 16):
    metrics = trainer.train(1, use_wandb=False)
    cost = metrics['total_cost']
    best_cost = min(best_cost, cost)
    print(f"Update {update}: cost = {cost:.2f} (best: {best_cost:.2f})")
