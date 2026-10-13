import json
import pandas as pd
import numpy as np
from scipy import stats

def check_scenario_stats():
    with open("results/phase12/ablation_stats_fixed.json", "r") as f:
        data = json.load(f)
    
    # We will pick the first scenario: 'in_distribution'
    scenario_records = [r for r in data["hypothesis_tests"] if r["scenario"] == "in_distribution"]
    
    # Let's manually compute baseline and forecast for in_distribution total_cost
    # Load raw data
    import glob
    df10 = []
    for f in glob.glob("results/phase10/eval*.json"):
        if "manifest" in f: continue
        with open(f, "r") as file:
            try:
                res = json.load(file)
                if "phase10_gnn_mappo_baseline" in res["experiment_id"] and res["scenario"] == "in_distribution":
                    df10.append(res)
            except Exception: pass
    
    df11 = []
    for f in glob.glob("results/phase11/eval*.json"):
        if "manifest" in f: continue
        with open(f, "r") as file:
            try:
                res = json.load(file)
                if "phase11_gnn_mappo_forecast_genuine_xgboost" in res["experiment_id"] and res["scenario"] == "in_distribution":
                    df11.append(res)
            except Exception: pass
                
    df10 = pd.DataFrame(df10)
    df11 = pd.DataFrame(df11)
    
    if len(df10) > 0 and len(df11) > 0:
        base_agg = df10.groupby("seed")["total_cost"].mean().values
        fore_agg = df11.groupby("seed")["total_cost"].mean().values
        
        print("Raw base N=", len(base_agg), "mean:", np.mean(base_agg))
        print("Raw fore N=", len(fore_agg), "mean:", np.mean(fore_agg))
        
        t_stat, p_val = stats.ttest_ind(base_agg, fore_agg, equal_var=False)
        print("T-stat:", t_stat, "P-val:", p_val)
        
        # Cohen's d
        nx, ny = len(base_agg), len(fore_agg)
        dof = nx + ny - 2
        pooled_std = np.sqrt(((nx-1)*np.std(base_agg, ddof=1)**2 + (ny-1)*np.std(fore_agg, ddof=1)**2) / dof)
        cohens_d = (np.mean(base_agg) - np.mean(fore_agg)) / pooled_std
        print("Cohen's d:", cohens_d)
        
        for r in scenario_records:
            if r["metric"] == "total_cost":
                print("Script T-stat:", r["t_stat"], "Script P-val:", r["raw_p_value"])
                print("Script Cohen's d:", r["cohens_d"])
                
if __name__ == "__main__":
    check_scenario_stats()
