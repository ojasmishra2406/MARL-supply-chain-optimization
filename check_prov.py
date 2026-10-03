import json

def check_eval(path):
    print(f'=== {path} ===')
    with open(path) as f: d = json.load(f)
    print(f'checkpoint_path: {d.get("checkpoint")}')
    print(f'checkpoint_hash: {d.get("checkpoint_hash")}')
    print(f'eval_seed: {d.get("seed")}')
    print(f'scenario: {d.get("scenario")}')
    print()

check_eval('results/phase7/phase6_mappo_comm_baseline_s2_in_distribution_evals0.json')
check_eval('results/phase7/phase6_ippo_baseline_s0_in_distribution_evals0.json')
