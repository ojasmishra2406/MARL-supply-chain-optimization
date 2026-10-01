import os
import sys

def main():
    print("--- REPAIRING ISSUE 3: Phase 12 Pseudoreplication ---")
    
    import json
    import glob
    import numpy as np
    import pandas as pd
    from scipy import stats
    import statsmodels.stats.multitest as mt
    
    def load_results(phase_dir):
        files = glob.glob(os.path.join(phase_dir, "*_evals*.json")) + glob.glob(os.path.join(phase_dir, "eval*.json"))
        data = []
        for f in files:
            if "manifest" in f: continue
            with open(f, "r") as file:
                try:
                    res = json.load(file)
                    data.append(res)
                except:
                    pass
        return pd.DataFrame(data)
        
    df10 = load_results("results/phase10")
    df11 = load_results("results/phase11")
    
    def extract_group(exp_id):
        if "phase10_gnn_mappo_baseline" in exp_id: return "P10_Baseline"
        elif "phase11_gnn_mappo_forecast_genuine_xgboost" in exp_id: return "P11_Forecasting_GenuineXGB"
        return "Other"
        
    df10["group"] = df10["experiment_id"].apply(extract_group)
    df11["group"] = df11["experiment_id"].apply(extract_group)
    
    df = pd.concat([df10, df11], ignore_index=True)
    df_compare = df[df["group"].isin(["P10_Baseline", "P11_Forecasting_GenuineXGB"])].copy()
    
    # ISSUE 3 FIX: Aggregate evaluation seeds at the training run (seed) level
    # The experimental unit must be the training seed (N=5).
    agg_df = df_compare.groupby(["group", "scenario", "seed"]).agg({
        "total_cost": "mean",
        "service_level": "mean",
        "bullwhip_ratio": "mean"
    }).reset_index()
    
    metrics = ["total_cost", "service_level", "bullwhip_ratio"]
    scenarios = agg_df["scenario"].unique()
    
    p_values = []
    test_records = []
    
    for scenario in scenarios:
        df_s = agg_df[agg_df["scenario"] == scenario]
        group_base = df_s[df_s["group"] == "P10_Baseline"]
        group_fore = df_s[df_s["group"] == "P11_Forecasting_GenuineXGB"]
        
        for metric in metrics:
            val_base = group_base[metric].values
            val_fore = group_fore[metric].values
            
            # Anti-pseudoreplication check!
            if len(val_base) > 5 or len(val_fore) > 5:
                raise ValueError(f"Pseudoreplication detected! N={len(val_base)} but expected 5.")
                
            if len(val_base) < 2 or len(val_fore) < 2:
                continue
                
            t_stat, p_val = stats.ttest_ind(val_base, val_fore, equal_var=False)
            
            # Cohen's d
            nx = len(val_base)
            ny = len(val_fore)
            dof = nx + ny - 2
            pooled_std = np.sqrt(((nx-1)*np.std(val_base, ddof=1)**2 + (ny-1)*np.std(val_fore, ddof=1)**2) / dof)
            cohens_d = (np.mean(val_base) - np.mean(val_fore)) / pooled_std
            
            test_records.append({
                "scenario": scenario,
                "metric": metric,
                "baseline_mean": float(np.mean(val_base)),
                "forecast_mean": float(np.mean(val_fore)),
                "t_stat": float(t_stat),
                "raw_p_value": float(p_val),
                "cohens_d": float(cohens_d),
                "n_baseline": int(nx),
                "n_forecast": int(ny)
            })
            p_values.append(p_val)
            
    if p_values:
        reject, pvals_corrected, _, _ = mt.multipletests(p_values, alpha=0.05, method='fdr_bh')
        for i, record in enumerate(test_records):
            record["adj_p_value"] = float(pvals_corrected[i])
            record["significant"] = bool(reject[i])
            
    final_output = {"hypothesis_tests": test_records}
    with open("results/phase12/ablation_stats_fixed.json", "w") as f:
        json.dump(final_output, f, indent=4)
        
    print("Phase 12 Fixed Statistical Analysis Complete!")

if __name__ == "__main__":
    main()
