import os
import json
import glob
import numpy as np
import scipy.stats as stats
import pandas as pd
import datetime

def calc_cohens_d(x, y):
    nx = len(x)
    ny = len(y)
    if nx < 2 or ny < 2: return 0.0
    dof = nx + ny - 2
    pool_var = ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof
    if pool_var == 0: return 0.0
    d = (np.mean(x) - np.mean(y)) / np.sqrt(pool_var)
    return d

def bh_fdr(p_values):
    # Benjamini-Hochberg FDR
    n = len(p_values)
    if n == 0: return []
    ranked_p = np.argsort(p_values)
    adjusted_p = np.zeros(n)
    for i, rank in enumerate(ranked_p):
        adjusted_p[rank] = min(1.0, p_values[rank] * n / (i + 1))
    
    # Enforce monotonicity
    for i in range(n - 2, -1, -1):
        rank_i = ranked_p[i]
        rank_next = ranked_p[i+1]
        adjusted_p[rank_i] = min(adjusted_p[rank_i], adjusted_p[rank_next])
    
    return adjusted_p

def bootstrap_ci(data, num_samples=1000, ci=0.95):
    if len(data) < 2: return np.mean(data), np.mean(data)
    means = [np.mean(np.random.choice(data, len(data), replace=True)) for _ in range(num_samples)]
    lower = np.percentile(means, (1-ci)/2 * 100)
    upper = np.percentile(means, (1+ci)/2 * 100)
    return lower, upper

def main():
    print("Running Phase 8 Statistical Analysis...")
    files = glob.glob("results/phase7/*.json")
    records = []
    for f in files:
        with open(f, "r") as fh:
            data = json.load(fh)
            # data has: experiment_id, scenario, algorithm, seed, cost, fill_rate, bullwhip
            if "cost" not in data:
                continue
                
            algo = data.get("algorithm", "out")
            condition = "baseline"
            if "high_variance" in algo:
                condition = "high_variance"
                algo = algo.replace("_high_variance", "")
            elif "baseline" in algo:
                condition = "baseline"
                algo = algo.replace("_baseline", "")
                
            records.append({
                "algorithm": algo,
                "condition": condition,
                "train_seed": int(data.get("seed", 0)),
                "scenario": data["scenario"],
                "cost": data["cost"],
                "fill_rate": data["fill_rate"],
                "bullwhip": data["bullwhip"]
            })
                
    df = pd.DataFrame(records)
    
    # Group by Algorithm, Condition, Scenario
    # For independent observations, we aggregate over eval_seeds first?
    # No, the "independent observations" are the Train Seeds!
    # A single trained model evaluated on 5 eval seeds provides an expected performance of that model.
    # To compare IPPO vs MAPPO, we should average over eval seeds to get the performance of train_seed_i,
    # then do a t-test across the N train_seeds.
    
    model_df = df.groupby(["algorithm", "condition", "scenario", "train_seed"]).agg({
        "cost": "mean",
        "fill_rate": "mean",
        "bullwhip": "mean"
    }).reset_index()
    
    # Save the aggregated observations for reproducibility
    model_df.to_csv("results/phase8_observations.csv", index=False)
    
    report = []
    report.append("# Phase 8: Statistical Analysis & Experimental Validation\n")
    report.append(f"Date: {datetime.datetime.now().isoformat()}\n\n")
    report.append("## 1. Experimental Setup\n")
    report.append("- **Algorithms Evaluated:** OUT (Baseline), IPPO, MAPPO, IPPO_COMM, MAPPO_COMM\n")
    report.append("- **Conditions:** baseline, high_variance\n")
    report.append("- **Independent Observations (Train Seeds):** up to 5 per configuration\n")
    report.append("- **Evaluation Seeds:** 5 per model per scenario\n")
    
    # Descriptive Statistics
    report.append("## 2. Descriptive Statistics (Aggregated across Evaluation Seeds)\n")
    stats_df = model_df.groupby(["algorithm", "condition", "scenario"]).agg(
        n_seeds=("train_seed", "count"),
        cost_mean=("cost", "mean"),
        cost_std=("cost", "std"),
        fill_rate_mean=("fill_rate", "mean"),
        fill_rate_std=("fill_rate", "std")
    ).reset_index()
    
    report.append(stats_df.to_markdown(index=False))
    report.append("\n\n")
    
    # Statistical Testing
    report.append("## 3. Statistical Testing (Welch's t-test, BH-FDR Corrected)\n")
    
    comparisons = [
        # IPPO vs MAPPO (baseline, in_distribution)
        {"name": "IPPO vs MAPPO (Baseline, In-Distribution)", "algo1": "ippo", "cond1": "baseline", "algo2": "mappo", "cond2": "baseline", "scen": "in_distribution"},
        # IPPO vs IPPO_COMM (baseline, in_distribution)
        {"name": "IPPO vs IPPO_COMM (Baseline, In-Distribution)", "algo1": "ippo", "cond1": "baseline", "algo2": "ippo_comm", "cond2": "baseline", "scen": "in_distribution"},
        # MAPPO vs MAPPO_COMM (baseline, in_distribution)
        {"name": "MAPPO vs MAPPO_COMM (Baseline, In-Distribution)", "algo1": "mappo", "cond1": "baseline", "algo2": "mappo_comm", "cond2": "baseline", "scen": "in_distribution"},
        # IPPO baseline vs IPPO high variance (in_distribution)
        {"name": "IPPO (Base) vs IPPO (High Var) (In-Distribution)", "algo1": "ippo", "cond1": "baseline", "algo2": "ippo", "cond2": "high_variance", "scen": "in_distribution"},
        # MAPPO_COMM baseline vs MAPPO_COMM high variance (in_distribution)
        {"name": "MAPPO_COMM (Base) vs MAPPO_COMM (High Var) (In-Distribution)", "algo1": "mappo_comm", "cond1": "baseline", "algo2": "mappo_comm", "cond2": "high_variance", "scen": "in_distribution"},
        # IPPO vs MAPPO (baseline, combined_shift)
        {"name": "IPPO vs MAPPO (Baseline, Combined Shift)", "algo1": "ippo", "cond1": "baseline", "algo2": "mappo", "cond2": "baseline", "scen": "combined_shift"},
        # IPPO vs IPPO_COMM (baseline, combined_shift)
        {"name": "IPPO vs IPPO_COMM (Baseline, Combined Shift)", "algo1": "ippo", "cond1": "baseline", "algo2": "ippo_comm", "cond2": "baseline", "scen": "combined_shift"},
        # MAPPO vs MAPPO_COMM (baseline, combined_shift)
        {"name": "MAPPO vs MAPPO_COMM (Baseline, Combined Shift)", "algo1": "mappo", "cond1": "baseline", "algo2": "mappo_comm", "cond2": "baseline", "scen": "combined_shift"},
    ]
    
    p_values = []
    results = []
    
    for comp in comparisons:
        data1 = model_df[(model_df["algorithm"] == comp["algo1"]) & (model_df["condition"] == comp["cond1"]) & (model_df["scenario"] == comp["scen"])]["cost"].values
        data2 = model_df[(model_df["algorithm"] == comp["algo2"]) & (model_df["condition"] == comp["cond2"]) & (model_df["scenario"] == comp["scen"])]["cost"].values
        
        if len(data1) < 2 or len(data2) < 2:
            results.append({"name": comp["name"], "p": 1.0, "t": 0, "d": 0, "n1": len(data1), "n2": len(data2), "mean1": np.mean(data1) if len(data1) else 0, "mean2": np.mean(data2) if len(data2) else 0})
            p_values.append(1.0)
            continue
            
        t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=False)
        d = calc_cohens_d(data1, data2)
        
        results.append({
            "name": comp["name"],
            "p": p_val,
            "t": t_stat,
            "d": d,
            "n1": len(data1),
            "n2": len(data2),
            "mean1": np.mean(data1),
            "mean2": np.mean(data2)
        })
        p_values.append(p_val)
        
    adj_p = bh_fdr(p_values)
    
    for i, res in enumerate(results):
        res["adj_p"] = adj_p[i]
        
    res_df = pd.DataFrame(results)
    report.append(res_df.to_markdown(index=False))
    report.append("\n\n")
    
    # Robustness Analysis
    report.append("## 4. Distribution-Shift Robustness Analysis\n")
    report.append("Comparing relative degradation from `in_distribution` to `combined_shift`.\n\n")
    
    robustness = []
    for (algo, cond), group in model_df.groupby(["algorithm", "condition"]):
        in_dist = group[group["scenario"] == "in_distribution"]["cost"].mean()
        combined = group[group["scenario"] == "combined_shift"]["cost"].mean()
        if not np.isnan(in_dist) and in_dist != 0:
            rel_change = (combined - in_dist) / in_dist
            robustness.append({"Algorithm": algo, "Condition": cond, "In-Dist Cost": in_dist, "Shifted Cost": combined, "Degradation": f"{rel_change*100:.2f}%"})
            
    rob_df = pd.DataFrame(robustness)
    report.append(rob_df.to_markdown(index=False))
    report.append("\n\n")
    
    with open("PHASE8_STATISTICAL_ANALYSIS.md", "w") as f:
        f.write("\n".join(report))
        
    print("Phase 8 analysis complete! Report saved to PHASE8_STATISTICAL_ANALYSIS.md")

if __name__ == "__main__":
    main()
