"""Verify GNN disruption SD discrepancy: episode-level vs model-level."""
import os, glob, json, numpy as np

os.chdir(r"c:\Users\mishr\OneDrive\Documents\MARL")

# Load all GNN baseline disruption episodes
records = []
for f in glob.glob("results/phase10/phase10_gnn_mappo_baseline_s*_capacity_disruption_evals*.json"):
    with open(f) as fp:
        d = json.load(fp)
    bn = os.path.basename(f)
    seed = int(bn.split("_s")[1].split("_")[0])
    records.append({"seed": seed, "cost": float(d["total_cost"])})

print("=== GNN BASELINE DISRUPTION EPISODES ===")
print(f"Total episodes: {len(records)}")

# Episode-level SD (incorrect for model-level inference)
all_costs = [r["cost"] for r in records]
print(f"\nEpisode-level mean: {np.mean(all_costs):.2f}")
print(f"Episode-level SD (ddof=1): {np.std(all_costs, ddof=1):.2f}")

# Model-level aggregation (correct)
by_seed = {}
for r in records:
    by_seed.setdefault(r["seed"], []).append(r["cost"])

model_means = []
for s in sorted(by_seed):
    m = np.mean(by_seed[s])
    sd_within = np.std(by_seed[s], ddof=1) if len(by_seed[s]) > 1 else 0
    model_means.append(m)
    print(f"  Seed {s}: n={len(by_seed[s])} episodes, mean={m:.2f}, within-model SD={sd_within:.2f}")

print(f"\nModel-level (N={len(model_means)}):")
print(f"  Mean of model means: {np.mean(model_means):.2f}")
print(f"  SD across models (ddof=1): {np.std(model_means, ddof=1):.2f}")
print(f"  SE of model means: {np.std(model_means, ddof=1)/np.sqrt(len(model_means)):.2f}")

# The report says SD = 6701.37. Let's check if that's the within-seed SD for a specific seed
for s in sorted(by_seed):
    costs = by_seed[s]
    if len(costs) > 1:
        sd = np.std(costs, ddof=1)
        print(f"  Within-model SD for seed {s}: {sd:.2f}")

# Also check MAPPO baseline disruption
print("\n=== MAPPO BASELINE DISRUPTION EPISODES ===")
mappo_records = []
for f in glob.glob("results/phase7/phase6_mappo_baseline_s*_capacity_disruption_evals*.json"):
    with open(f) as fp:
        d = json.load(fp)
    bn = os.path.basename(f)
    seed = int(bn.split("_s")[1].split("_")[0])
    mappo_records.append({"seed": seed, "cost": float(d["cost"])})

print(f"Total episodes: {len(mappo_records)}")

mappo_by_seed = {}
for r in mappo_records:
    mappo_by_seed.setdefault(r["seed"], []).append(r["cost"])

mappo_model_means = []
for s in sorted(mappo_by_seed):
    m = np.mean(mappo_by_seed[s])
    sd_within = np.std(mappo_by_seed[s], ddof=1) if len(mappo_by_seed[s]) > 1 else 0
    mappo_model_means.append(m)
    print(f"  Seed {s}: n={len(mappo_by_seed[s])} episodes, mean={m:.2f}, within-model SD={sd_within:.2f}")

print(f"\nModel-level (N={len(mappo_model_means)}):")
print(f"  Mean of model means: {np.mean(mappo_model_means):.2f}")
print(f"  SD across models (ddof=1): {np.std(mappo_model_means, ddof=1):.2f}")

# Final: compute CORRECT Welch test at model level
from scipy import stats
t_res = stats.ttest_ind(model_means, mappo_model_means, equal_var=False)
print(f"\n=== CORRECT MODEL-LEVEL WELCH TEST ===")
print(f"  t = {t_res.statistic:.4f}")
print(f"  p = {t_res.pvalue:.6f}")

# Compare with report values
print(f"\n=== REPORT CLAIMS vs GROUND TRUTH ===")
print(f"  Report GNN SD:   6,701.37   Actual: {np.std(model_means, ddof=1):.2f}")
print(f"  Report MAPPO SD: 6,052.26   Actual: {np.std(mappo_model_means, ddof=1):.2f}")
print(f"  Report t:        0.380      Actual: {t_res.statistic:.4f}")
print(f"  Report p:        0.7099     Actual: {t_res.pvalue:.6f}")
