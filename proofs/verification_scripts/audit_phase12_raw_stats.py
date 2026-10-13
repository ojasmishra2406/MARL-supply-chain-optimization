import os
import glob
import json
import numpy as np
import pandas as pd
from scipy import stats

def run_audit():
    print("=== INGESTING RAW EVALUATION DATA ===")
    
    # 1. Ingest Phase 7
    p7_files = glob.glob("results/phase7/*_eval*.json")
    print(f"Found {len(p7_files)} Phase 7 evaluation JSONs")
    
    p7_records = []
    scenarios = ['in_distribution', 'demand_shift', 'lead_time_shift', 'capacity_disruption', 'demand_spike', 'combined_shift']
    
    for f in p7_files:
        with open(f, "r") as fp:
            d = json.load(fp)
        base = os.path.basename(f)
        sc = None
        for s in scenarios:
            if f"_{s}_" in base:
                sc = s
                model_name = base.split(f"_{s}_")[0]
                eval_part = base.split(f"_{s}_")[1].replace(".json", "")
                eval_seed = int(eval_part.replace("evals", "").replace("eval", ""))
                break
        if not sc:
            continue
            
        train_seed = int(model_name.split("_s")[-1])
        regime = "high_variance" if "high_variance" in model_name else "baseline"
        if "ippo_comm" in model_name:
            algo = "ippo_comm"
        elif "mappo_comm" in model_name:
            algo = "mappo_comm"
        elif "ippo" in model_name:
            algo = "ippo"
        elif "mappo" in model_name:
            algo = "mappo"
        else:
            algo = "other"
            
        p7_records.append({
            "phase": "phase7",
            "model_name": model_name,
            "algorithm": algo,
            "regime": regime,
            "training_seed": train_seed,
            "scenario": sc,
            "eval_seed": eval_seed,
            "cost": float(d["cost"]),
            "service_level": float(d["fill_rate"]),
            "bullwhip": float(d["bullwhip"]),
            "file": f
        })
    df_p7 = pd.DataFrame(p7_records)
    print(f"Loaded {len(df_p7)} Phase 7 records")
    
    # Check MAPPO vs IPPO baseline models
    mappo_base_episodes = df_p7[(df_p7["algorithm"] == "mappo") & (df_p7["regime"] == "baseline")]
    ippo_base_episodes = df_p7[(df_p7["algorithm"] == "ippo") & (df_p7["regime"] == "baseline")]
    
    print(f"MAPPO baseline episodes: {len(mappo_base_episodes)} (expected 5 seeds x 6 sc x 5 evals = 150)")
    print(f"IPPO baseline episodes: {len(ippo_base_episodes)} (expected 5 seeds x 6 sc x 5 evals = 150)")
    
    # 2. Episode-level summary (if someone erroneously treated episodes as independent)
    print("\n--- EPISODE-LEVEL CALCULATION (N=150 episodes per arm) ---")
    mb_bw = mappo_base_episodes["bullwhip"].values
    ib_bw = ippo_base_episodes["bullwhip"].values
    res_ep_bw = stats.ttest_ind(mb_bw, ib_bw, equal_var=False)
    print(f"Episode-level Bullwhip: MAPPO mean={np.mean(mb_bw):.4f}, sd={np.std(mb_bw, ddof=1):.4f} | IPPO mean={np.mean(ib_bw):.4f}, sd={np.std(ib_bw, ddof=1):.4f}")
    print(f"Episode-level Welch t={res_ep_bw.statistic:.4f}, p={res_ep_bw.pvalue:.4e}")
    
    mappo_base_episodes["cost_per_sl"] = mappo_base_episodes["cost"] / mappo_base_episodes["service_level"]
    ippo_base_episodes["cost_per_sl"] = ippo_base_episodes["cost"] / ippo_base_episodes["service_level"]
    mb_csl = mappo_base_episodes["cost_per_sl"].values
    ib_csl = ippo_base_episodes["cost_per_sl"].values
    res_ep_csl = stats.ttest_ind(mb_csl, ib_csl, equal_var=False)
    print(f"Episode-level Cost/SL: MAPPO mean={np.mean(mb_csl):.2f}, sd={np.std(mb_csl, ddof=1):.2f} | IPPO mean={np.mean(ib_csl):.2f}, sd={np.std(ib_csl, ddof=1):.2f}")
    print(f"Episode-level Welch t={res_ep_csl.statistic:.4f}, p={res_ep_csl.pvalue:.4e}")
    
    # 3. Model-level aggregation (Independent unit = training_seed, N=5 per arm)
    print("\n--- MODEL-LEVEL AGGREGATION (Independent unit = training_seed, N=5 per arm) ---")
    p7_model = df_p7.groupby(["algorithm", "regime", "training_seed"]).agg({
        "cost": "mean",
        "service_level": "mean",
        "bullwhip": "mean"
    }).reset_index()
    p7_model["cost_per_sl"] = p7_model["cost"] / p7_model["service_level"]
    
    mb_model = p7_model[(p7_model["algorithm"] == "mappo") & (p7_model["regime"] == "baseline")]
    ib_model = p7_model[(p7_model["algorithm"] == "ippo") & (p7_model["regime"] == "baseline")]
    
    print("\nMAPPO Baseline model seeds:")
    print(mb_model[["training_seed", "cost", "service_level", "bullwhip", "cost_per_sl"]])
    print("\nIPPO Baseline model seeds:")
    print(ib_model[["training_seed", "cost", "service_level", "bullwhip", "cost_per_sl"]])
    
    print("\n--- COMPUTING MODEL-LEVEL WELCH TESTS (N=5) ---")
    # Bullwhip
    x_bw = mb_model["bullwhip"].values
    y_bw = ib_model["bullwhip"].values
    res_bw = stats.ttest_ind(x_bw, y_bw, equal_var=False)
    
    nx, ny = len(x_bw), len(y_bw)
    mx, my = np.mean(x_bw), np.mean(y_bw)
    sx, sy = np.std(x_bw, ddof=1), np.std(y_bw, ddof=1)
    vx, vy = sx**2 / nx, sy**2 / ny
    df_bw = (vx + vy)**2 / ((vx**2)/(nx-1) + (vy**2)/(ny-1))
    
    s_pooled = np.sqrt(((nx - 1)*(sx**2) + (ny - 1)*(sy**2)) / (nx + ny - 2))
    d_bw = (mx - my) / s_pooled
    J = 1.0 - (3.0 / (4.0 * (nx + ny - 2) - 1.0))
    g_bw = d_bw * J
    mwu_bw = stats.mannwhitneyu(x_bw, y_bw, alternative="two-sided")
    
    print(f"Model-level Bullwhip:")
    print(f"  MAPPO (N=5): Mean={mx:.6f}, SD={sx:.6f}, SE={sx/np.sqrt(5):.6f}")
    print(f"  IPPO  (N=5): Mean={my:.6f}, SD={sy:.6f}, SE={sy/np.sqrt(5):.6f}")
    print(f"  Diff = {mx - my:.6f} ({(mx-my)/my * 100:.2f}%)")
    print(f"  Welch t = {res_bw.statistic:.6f}, df = {df_bw:.4f}, p = {res_bw.pvalue:.6f}")
    print(f"  Mann-Whitney U = {mwu_bw.statistic}, p = {mwu_bw.pvalue:.6f}")
    print(f"  Cohen's d = {d_bw:.4f}, Hedges' g = {g_bw:.4f}")
    
    # Cost/SL
    x_csl = mb_model["cost_per_sl"].values
    y_csl = ib_model["cost_per_sl"].values
    res_csl = stats.ttest_ind(x_csl, y_csl, equal_var=False)
    
    mx_csl, my_csl = np.mean(x_csl), np.mean(y_csl)
    sx_csl, sy_csl = np.std(x_csl, ddof=1), np.std(y_csl, ddof=1)
    vx_csl, vy_csl = sx_csl**2 / nx, sy_csl**2 / ny
    df_csl = (vx_csl + vy_csl)**2 / ((vx_csl**2)/(nx-1) + (vy_csl**2)/(ny-1))
    
    s_pooled_csl = np.sqrt(((nx - 1)*(sx_csl**2) + (ny - 1)*(sy_csl**2)) / (nx + ny - 2))
    d_csl = (mx_csl - my_csl) / s_pooled_csl
    g_csl = d_csl * J
    mwu_csl = stats.mannwhitneyu(x_csl, y_csl, alternative="two-sided")
    
    print(f"\nModel-level Cost / Service Level:")
    print(f"  MAPPO (N=5): Mean={mx_csl:.2f}, SD={sx_csl:.2f}, SE={sx_csl/np.sqrt(5):.2f}")
    print(f"  IPPO  (N=5): Mean={my_csl:.2f}, SD={sy_csl:.2f}, SE={sy_csl/np.sqrt(5):.2f}")
    print(f"  Diff = {mx_csl - my_csl:.2f} ({(mx_csl-my_csl)/my_csl * 100:.2f}%)")
    print(f"  Welch t = {res_csl.statistic:.6f}, df = {df_csl:.4f}, p = {res_csl.pvalue:.6f}")
    print(f"  Mann-Whitney U = {mwu_csl.statistic}, p = {mwu_csl.pvalue:.6f}")
    print(f"  Cohen's d = {d_csl:.4f}, Hedges' g = {g_csl:.4f}")

    # Also Cost and Service Level
    x_c = mb_model["cost"].values
    y_c = ib_model["cost"].values
    res_c = stats.ttest_ind(x_c, y_c, equal_var=False)
    print(f"\nModel-level Cost:")
    print(f"  MAPPO (N=5): Mean={np.mean(x_c):.2f}, SD={np.std(x_c, ddof=1):.2f}")
    print(f"  IPPO  (N=5): Mean={np.mean(y_c):.2f}, SD={np.std(y_c, ddof=1):.2f}")
    print(f"  Welch t = {res_c.statistic:.4f}, p = {res_c.pvalue:.4f}")
    
    x_sl = mb_model["service_level"].values
    y_sl = ib_model["service_level"].values
    res_sl = stats.ttest_ind(x_sl, y_sl, equal_var=False)
    print(f"\nModel-level Service Level:")
    print(f"  MAPPO (N=5): Mean={np.mean(x_sl):.4f}, SD={np.std(x_sl, ddof=1):.4f}")
    print(f"  IPPO  (N=5): Mean={np.mean(y_sl):.4f}, SD={np.std(y_sl, ddof=1):.4f}")
    print(f"  Welch t = {res_sl.statistic:.4f}, p = {res_sl.pvalue:.4f}")

if __name__ == "__main__":
    run_audit()
