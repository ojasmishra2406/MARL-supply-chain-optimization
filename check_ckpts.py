import torch
s2 = torch.load('results/phase6/phase6_mappo_comm_baseline_s2.pt', map_location='cpu')
s0 = torch.load('results/phase6/phase6_ippo_baseline_s0.pt', map_location='cpu')

def print_model_info(name, d):
    print(f'=== {name} ===')
    if 'model_state_dict' in d:
        print('Format: Resumable (new schema)')
        print(f'Iteration: {d.get("iteration")}')
        print(f'Train Seed: {d.get("train_seed")}')
        state = d['model_state_dict']
    else:
        print('Format: Direct state_dict (old schema)')
        state = d
    print(f'Keys: {list(state.keys())}')
    
    agent_key = list(state.keys())[0]
    substate = state[agent_key]
    print(f'{agent_key} Keys: {len(substate.keys())}')
    for k in list(substate.keys())[:3]:
        print(f'  {k}: {substate[k].shape}')
    print('VALID\n')

print_model_info('s2 (MAPPO_COMM)', s2)
print_model_info('s0 (IPPO)', s0)
