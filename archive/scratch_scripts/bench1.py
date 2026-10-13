import time
import sys
import os
sys.path.insert(0, '.')
from simulator.core import SupplyChainSimulator

repo_root = os.path.abspath('.')
config_path = os.path.join(repo_root, 'configs', 'phase1_simulator.yaml')
sim = SupplyChainSimulator(config_path, seed=42)
num_steps = 10000
action = [10, 10, 10, 10]

start = time.perf_counter()
for _ in range(num_steps):
    sim.current_step = 0
    sim.step(action)
print(num_steps / (time.perf_counter() - start))
