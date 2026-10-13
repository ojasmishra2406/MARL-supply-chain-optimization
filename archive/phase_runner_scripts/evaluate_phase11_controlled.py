import os
import glob
import json
import numpy as np
import pandas as pd
from rl.checkpoint import load_checkpoint

RESULTS_DIR = "results/phase11_controlled"

def audit_and_recompute_run(run_id, ckpt_name):
    ckpt_path = os.path.join(RESULTS_DIR, f"{ckpt_name}.pt")
    manifest_path = os.path.join(RESULTS_DIR, f"{ckpt_name}_manifest.json")
    
    # 1. Canonical checkpoint verification
    assert os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}"
    assert os.path.exists(manifest_path), f"Manifest missing: {manifest_path}"
    
    try:
        load_checkpoint(ckpt_path, manifest_path)
        canonical_verified = True
    except Exception as e:
        canonical_verified = False
        load_err = str(e)
        
    # 2. Evaluate from raw episode JSONs
    pattern = os.path.join(RESULTS_DIR, f"{ckpt_name}_*eval*.json")
    files = sorted(glob.glob(pattern))
    assert len(files) == 30, f"Expected 30 evaluation files for {ckpt_name}, got {len(files)}"
    
    costs = []
    service_levels = []
    bullwhips = []
    fulfilled_units = []
    demands = []
    
    for f in files:
        with open(f, "r") as fh:
            d = json.load(fh)
        costs.append(d["total_cost"])
        service_levels.append(d["service_level"])
        bullwhips.append(d["bullwhip_ratio"])
        
    mean_cost = float(np.mean(costs))
    mean_sl = float(np.mean(service_levels))
    mean_bw = float(np.mean(bullwhips))
    
    # Telemetry check
    csv_path = os.path.join(RESULTS_DIR, f"training_metrics_{ckpt_name}.csv")
    csv_rows = len(pd.read_csv(csv_path)) if os.path.exists(csv_path) else 0
    
    return {
        "run_id": run_id,
        "checkpoint_path": ckpt_path,
        "manifest_path": manifest_path,
        "canonical_verified": canonical_verified,
        "episodes_evaluated": len(files),
        "telemetry_rows": csv_rows,
        "mean_cost": mean_cost,
        "mean_service_level": mean_sl,
        "mean_bullwhip": mean_bw
    }

if __name__ == "__main__":
    from scipy import stats
    
    seeds = [1, 2, 3]
    forecasters = ["ma", "xgboost"]
    
    print("=" * 60)
    print("PHASE 11 CONTROLLED RECOMPUTED EVALUATION REPORT")
    print("=" * 60)
    
    results = {"ma": {}, "xgboost": {}}
    
    for f in forecasters:
        for s in seeds:
            ckpt_name = f"phase11_controlled_{f}_s{s}"
            ckpt_path = os.path.join(RESULTS_DIR, f"{ckpt_name}.pt")
            if os.path.exists(ckpt_path):
                pattern = os.path.join(RESULTS_DIR, f"{ckpt_name}_*eval*.json")
                if len(glob.glob(pattern)) == 30:
                    res = audit_and_recompute_run(f"{f}_s{s}", ckpt_name)
                    results[f][s] = res
                    print(f"[{f.upper()} Seed {s}]")
                    print(f"  Cost: {res['mean_cost']:10.2f} | Service Level: {res['mean_service_level']*100:6.2f}% | Bullwhip: {res['mean_bullwhip']:6.2f}")
                else:
                    print(f"[{f.upper()} Seed {s}] Checkpoint exists but evaluations pending ({len(glob.glob(pattern))}/30)")
            else:
                print(f"[{f.upper()} Seed {s}] Not yet trained.")
                
    # If we have at least 2 seeds for each, compute Welch t-test
    xgb_runs = list(results["xgboost"].values())
    ma_runs = list(results["ma"].values())
    
    if len(xgb_runs) >= 2 and len(ma_runs) >= 2:
        xgb_costs = [r["mean_cost"] for r in xgb_runs]
        xgb_sls = [r["mean_service_level"] for r in xgb_runs]
        ma_costs = [r["mean_cost"] for r in ma_runs]
        ma_sls = [r["mean_service_level"] for r in ma_runs]
        
        print("\n--- AGGREGATE STATISTICAL COMPARISON ---")
        print(f"XGBoost Cost: {np.mean(xgb_costs):10.2f} +/- {np.std(xgb_costs, ddof=1):.2f}")
        print(f"XGBoost SL:   {np.mean(xgb_sls)*100:9.2f}% +/- {np.std(xgb_sls, ddof=1)*100:.2f}%")
        print(f"MA Cost:      {np.mean(ma_costs):10.2f} +/- {np.std(ma_costs, ddof=1):.2f}")
        print(f"MA SL:        {np.mean(ma_sls)*100:9.2f}% +/- {np.std(ma_sls, ddof=1)*100:.2f}%")
        
        t_cost, p_cost = stats.ttest_ind(xgb_costs, ma_costs, equal_var=False)
        t_sl, p_sl = stats.ttest_ind(xgb_sls, ma_sls, equal_var=False)
        print(f"\nCost Welch t: t = {t_cost:8.4f}, p = {p_cost:8.4f}")
        print(f"SL Welch t:   t = {t_sl:8.4f}, p = {p_sl:8.4f}")
    print("=" * 60)
