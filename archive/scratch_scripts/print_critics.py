import json
import numpy as np

data_agg = json.load(open('results/phase_4c_aggregate.json'))
data_item = json.load(open('results/phase_4c_itemized.json'))

for mode, data in [("CONTROL (Agg)", data_agg), ("EXPERIMENT (Item)", data_item)]:
    print(f"--- {mode} ---")
    val_losses = []
    exp_vars = []
    adv_vars = []
    entropies = []
    
    for seed in ['0', '1', '2']:
        traj = data['seeds'][seed]['trajectory']
        # Take mean over last 5 updates
        last_5 = traj[-5:]
        val_losses.append(np.mean([t['value_loss'] for t in last_5]))
        entropies.append(np.mean([t['entropy'] for t in last_5]))
    
    print(f"Mean Final Value Loss: {np.mean(val_losses):.4f}")
    print(f"Mean Final Entropy: {np.mean(entropies):.4f}")
    print()
