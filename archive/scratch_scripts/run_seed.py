import sys
import yaml
from rl.pilot import PilotTrainer
import json

seed_str = sys.argv[1]
seed = int(seed_str)

config = yaml.safe_load(open('configs/phase4_ippo.yaml'))
config['simulator_config'] = 'configs/phase1_simulator.yaml'
config['seeds'] = [seed]
with open(f'configs/test_seed_{seed}.yaml', 'w') as f:
    yaml.dump(config, f)

trainer = PilotTrainer(f'configs/test_seed_{seed}.yaml')
metrics = trainer.train(updates=50)

with open(f'results/pilot_convergence_500_seed{seed}.json', 'w') as f:
    json.dump(metrics, f)
