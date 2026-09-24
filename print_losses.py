import json
try:
    data = json.load(open('results/phase_4b_baseline_results.json'))
    for t in data['seeds']['0']['trajectory']:
        print(f"Update {t['update']}: Cost {t['cost']:.0f}, ValueLoss {t['value_loss']:.4f}, PolicyLoss {t['policy_loss']:.4f}, Entropy {t['entropy']:.4f}")
except Exception as e:
    print(e)
