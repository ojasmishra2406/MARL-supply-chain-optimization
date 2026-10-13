import os
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
        # Ignore manifests
        if "manifest" in f:
            continue
            
        with open(f, "r") as file:
            try:
                res = json.load(file)
                data.append(res)
            except:
                pass
                
    return pd.DataFrame(data)

def main():
    print("Loading Phase 10 and Phase 11 results...")
    
    df10 = load_results("results/phase10")
    df11 = load_results("results/phase11")
    
    # Standardize experiment group names
    # For Phase 10, experiment_id looks like: eval_{scenario}_phase10_gnn_mappo_baseline_s0_{eval_seed}
    # For Phase 11, experiment_id looks like: eval_{scenario}_phase11_gnn_mappo_forecast_xgboost_s0_{eval_seed}
    
    def extract_group(exp_id):
        if "phase10_gnn_mappo_baseline" in exp_id:
            return "P10_Baseline"
        elif "phase10_gnn_mappo_high_variance" in exp_id:
            return "P10_HighVariance"
        elif "phase11_gnn_mappo_forecast_xgboost" in exp_id:
            return "P11_Forecasting_XGB"
        elif "phase11_gnn_mappo_forecast_ma" in exp_id:
            return "P11_Forecasting_MA"
        return "Unknown"
        
    df10["group"] = df10["experiment_id"].apply(extract_group)
    df11["group"] = df11["experiment_id"].apply(extract_group)
    
    df = pd.concat([df10, df11], ignore_index=True)
    
    print(f"Loaded {len(df)} total evaluation records.")
    
    # We want to compare P11_Forecasting_MA vs P10_Baseline
    # (Since XGBoost and MA both used MA under the hood, we can just aggregate them or use one)
    # Let's aggregate all P11 vs P10_Baseline
    
    df["is_forecasting"] = df["group"].apply(lambda x: "P11" in x)
    df["is_baseline"] = df["group"].apply(lambda x: x == "P10_Baseline")
    
    # Filter to only the comparison we care about
    df_compare = df[df["is_forecasting"] | df["is_baseline"]].copy()
    
    # Metrics to analyze
    metrics = ["total_cost", "service_level", "bullwhip_ratio"]
    
    results = []
    
    # We do a t-test for each scenario and metric
    scenarios = df_compare["scenario"].unique()
    
    p_values = []
    test_records = []
    
    for scenario in scenarios:
        df_s = df_compare[df_compare["scenario"] == scenario]
        
        group_base = df_s[df_s["is_baseline"]]
        group_fore = df_s[df_s["is_forecasting"]]
        
        for metric in metrics:
            val_base = group_base[metric].values
            val_fore = group_fore[metric].values
            
            if len(val_base) == 0 or len(val_fore) == 0:
                continue
                
            # Welch's t-test
            t_stat, p_val = stats.ttest_ind(val_base, val_fore, equal_var=False)
            
            mean_base = np.mean(val_base)
            mean_fore = np.mean(val_fore)
            
            # Improvement (cost/bullwhip lower is better, service level higher is better)
            if metric in ["total_cost", "bullwhip_ratio"]:
                imp = (mean_base - mean_fore) / mean_base * 100
            else:
                imp = (mean_fore - mean_base) / mean_base * 100
                
            p_values.append(p_val)
            test_records.append({
                "scenario": scenario,
                "metric": metric,
                "baseline_mean": float(mean_base),
                "forecast_mean": float(mean_fore),
                "improvement_pct": float(imp),
                "t_stat": float(t_stat),
                "raw_p_value": float(p_val)
            })
            
    # Benjamini-Hochberg FDR Correction
    if p_values:
        reject, pvals_corrected, _, _ = mt.multipletests(p_values, alpha=0.05, method='fdr_bh')
        
        for i, record in enumerate(test_records):
            record["adj_p_value"] = float(pvals_corrected[i])
            record["significant"] = bool(reject[i])
            
    # Aggregate Global Stats
    global_stats = {}
    for metric in metrics:
        val_base = df_compare[df_compare["is_baseline"]][metric].values
        val_fore = df_compare[df_compare["is_forecasting"]][metric].values
        
        mean_base = np.mean(val_base)
        mean_fore = np.mean(val_fore)
        
        if metric in ["total_cost", "bullwhip_ratio"]:
            imp = (mean_base - mean_fore) / mean_base * 100
        else:
            imp = (mean_fore - mean_base) / mean_base * 100
            
        global_stats[metric] = {
            "baseline_mean": float(mean_base),
            "forecast_mean": float(mean_fore),
            "improvement_pct": float(imp)
        }
        
    final_output = {
        "global_summary": global_stats,
        "hypothesis_tests": test_records
    }
    
    os.makedirs("results/phase12", exist_ok=True)
    with open("results/phase12/ablation_stats.json", "w") as f:
        json.dump(final_output, f, indent=4)
        
    print("Phase 12 Statistical Analysis Complete!")
    print("Results saved to results/phase12/ablation_stats.json")
    
if __name__ == "__main__":
    main()
