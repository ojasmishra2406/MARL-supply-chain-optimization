import json
data_agg = json.load(open('results/phase_4c_aggregate.json'))
data_item = json.load(open('results/phase_4c_itemized.json'))

print('--- AGGREGATE ---')
for seed in ['0', '1', '2']:
    d = data_agg['seeds'][seed]
    print(f"Seed {seed} | Initial: {d['initial_cost']:.0f} | Best: {d['best_cost']:.0f} | Final: {d['final_cost']:.0f} | OUT: {d['out_cost']:.0f}")

print('\n--- ITEMIZED ---')
for seed in ['0', '1', '2']:
    d = data_item['seeds'][seed]
    print(f"Seed {seed} | Initial: {d['initial_cost']:.0f} | Best: {d['best_cost']:.0f} | Final: {d['final_cost']:.0f} | OUT: {d['out_cost']:.0f}")
