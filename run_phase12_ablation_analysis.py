import os
import glob
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

os.makedirs("results/phase12/plots", exist_ok=True)

# ==============================================================================
# 1. DATA INGESTION FUNCTIONS
# ==============================================================================

SCENARIOS = ['in_distribution', 'demand_shift', 'lead_time_shift', 'capacity_disruption', 'demand_spike', 'combined_shift']

def extract_model_and_scenario(filepath):
    """Extracts canonical model_name, scenario, and eval_seed from filename."""
    base = os.path.basename(filepath)
    for sc in SCENARIOS:
        if f"_{sc}_" in base:
            model_name = base.split(f"_{sc}_")[0]
            eval_part = base.split(f"_{sc}_")[1].replace(".json", "")
            eval_seed = int(eval_part.replace("evals", "").replace("eval", ""))
            return model_name, sc, eval_seed
    raise ValueError(f"Could not parse scenario from {filepath}")

def load_phase7_runs():
    """Load Phase 7 evaluations (Phase 6 models: IPPO, MAPPO, IPPO_COMM, MAPPO_COMM)."""
    files = glob.glob("results/phase7/*_eval*.json")
    records = []
    for f in files:
        with open(f, "r") as fp:
            d = json.load(fp)
        model_name, scenario, eval_seed = extract_model_and_scenario(f)
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
            
        records.append({
            "phase": "phase6_7",
            "model_name": model_name,
            "algorithm": algo,
            "architecture": "mlp",
            "communication": ("comm" in algo),
            "forecasting": "none",
            "regime": regime,
            "training_seed": train_seed,
            "scenario": scenario,
            "eval_seed": eval_seed,
            "cost": float(d["cost"]),
            "service_level": float(d["fill_rate"]),
            "bullwhip": float(d["bullwhip"]),
            "checkpoint": d.get("checkpoint"),
            "checkpoint_hash": d.get("checkpoint_hash")
        })
    return pd.DataFrame(records)

def load_phase10_runs():
    """Load Phase 10 evaluations (GNN-MAPPO baseline and high_variance)."""
    files = glob.glob("results/phase10/*_eval*.json")
    records = []
    for f in files:
        with open(f, "r") as fp:
            d = json.load(fp)
        model_name, scenario, eval_seed = extract_model_and_scenario(f)
        train_seed = int(model_name.split("_s")[-1])
        regime = "high_variance" if "high_variance" in model_name else "baseline"
        
        records.append({
            "phase": "phase10",
            "model_name": model_name,
            "algorithm": "gnn_mappo",
            "architecture": "gnn",
            "communication": False,
            "forecasting": "none",
            "regime": regime,
            "training_seed": train_seed,
            "scenario": scenario,
            "eval_seed": eval_seed,
            "cost": float(d["total_cost"]),
            "service_level": float(d["service_level"]),
            "bullwhip": float(d["bullwhip_ratio"]),
            "checkpoint": d.get("checkpoint"),
            "checkpoint_hash": d.get("checkpoint_hash")
        })
    return pd.DataFrame(records)

def load_phase11_controlled_runs():
    """Load Phase 11 Controlled evaluations (GNN-MAPPO + Moving Average vs XGBoost)."""
    files = glob.glob("results/phase11_controlled/*_eval*.json")
    records = []
    for f in files:
        with open(f, "r") as fp:
            d = json.load(fp)
        model_name, scenario, eval_seed = extract_model_and_scenario(f)
        # Skip smoke runs for primary statistical testing
        if "smoke" in model_name:
            continue
        train_seed = int(model_name.split("_s")[-1])
        forecaster = d.get("forecaster", "ma")
        
        records.append({
            "phase": "phase11_controlled",
            "model_name": model_name,
            "algorithm": f"gnn_mappo_forecast_{forecaster}",
            "architecture": "gnn",
            "communication": False,
            "forecasting": forecaster,
            "regime": "baseline",
            "training_seed": train_seed,
            "scenario": scenario,
            "eval_seed": eval_seed,
            "cost": float(d["total_cost"]),
            "service_level": float(d["service_level"]),
            "bullwhip": float(d["bullwhip_ratio"]),
            "checkpoint": d.get("checkpoint"),
            "checkpoint_hash": d.get("checkpoint_hash")
        })
    return pd.DataFrame(records)

# ==============================================================================
# 2. STATISTICAL ENGINE & METRIC AGGREGATION
# ==============================================================================

def compute_welch_stats(x, y):
    """Computes Welch's t-test, Mann-Whitney U, Cohen's d, Hedges' g, and 95% CI."""
    nx, ny = len(x), len(y)
    mx, my = np.mean(x), np.mean(y)
    sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)
    diff = mx - my
    pct_diff = (diff / my) * 100 if my != 0 else np.nan
    
    # Welch t-test
    res = stats.ttest_ind(x, y, equal_var=False)
    t_stat = float(res.statistic)
    p_val = float(res.pvalue)
    
    # Welch-Satterthwaite df
    vx = (sx**2) / nx if nx > 0 else 0
    vy = (sy**2) / ny if ny > 0 else 0
    df = (vx + vy)**2 / ((vx**2)/(nx-1) + (vy**2)/(ny-1)) if (vx + vy) > 0 else np.nan
    
    # Pooled SD and Hedges g
    s_pooled = np.sqrt(((nx - 1)*(sx**2) + (ny - 1)*(sy**2)) / (nx + ny - 2)) if (nx + ny > 2) else np.nan
    d = diff / s_pooled if (s_pooled and s_pooled > 0) else np.nan
    J = 1.0 - (3.0 / (4.0 * (nx + ny - 2) - 1.0)) if (nx + ny > 2) else 1.0
    g = d * J if not np.isnan(d) else np.nan
    
    # Mann-Whitney U
    mwu = stats.mannwhitneyu(x, y, alternative="two-sided")
    
    # 95% CI
    se_diff = np.sqrt(vx + vy)
    crit = stats.t.ppf(0.975, df) if not np.isnan(df) else 1.96
    ci_lower = float(diff - crit * se_diff)
    ci_upper = float(diff + crit * se_diff)
    
    return {
        "nx": nx, "ny": ny,
        "mean_x": float(mx), "sd_x": float(sx),
        "mean_y": float(my), "sd_y": float(sy),
        "diff": float(diff), "pct_diff": float(pct_diff),
        "welch_t": t_stat, "df": float(df), "welch_p": p_val,
        "mwu_u": float(mwu.statistic), "mwu_p": float(mwu.pvalue),
        "cohens_d": float(d), "hedges_g": float(g),
        "ci_lower": ci_lower, "ci_upper": ci_upper
    }

def apply_bh_fdr(tests_list):
    """Applies Benjamini-Hochberg FDR correction across a list of test dictionaries."""
    n = len(tests_list)
    if n == 0:
        return tests_list
    
    p_vals = [t["welch_p"] for t in tests_list]
    ranked_indices = np.argsort(p_vals)
    
    adj_p = np.zeros(n)
    for i, rank in enumerate(ranked_indices):
        adj_p[rank] = min(1.0, p_vals[rank] * n / (i + 1))
        
    # Enforce monotonicity
    for i in range(n - 2, -1, -1):
        rank_curr = ranked_indices[i]
        rank_next = ranked_indices[i + 1]
        adj_p[rank_curr] = min(adj_p[rank_curr], adj_p[rank_next])
        
    for i, t in enumerate(tests_list):
        t["fdr_q"] = float(adj_p[i])
        t["significant_fdr"] = bool(adj_p[i] < 0.05)
        
    return tests_list

# ==============================================================================
# 3. MAIN EXECUTION PIPELINE
# ==============================================================================

def main():
    print("=" * 70)
    print("PHASE 12: COMPREHENSIVE ABLATIONS & STATISTICAL AUDIT ENGINE")
    print("=" * 70)
    
    # 1. Ingest datasets
    df_p7 = load_phase7_runs()
    df_p10 = load_phase10_runs()
    df_p11 = load_phase11_controlled_runs()
    
    df_all = pd.concat([df_p7, df_p10, df_p11], ignore_index=True)
    print(f"Total raw evaluation rows ingested: {len(df_all)}")
    print(f"  - Phase 7 (IPPO/MAPPO/COMM): {len(df_p7)} episodes")
    print(f"  - Phase 10 (GNN-MAPPO):      {len(df_p10)} episodes")
    print(f"  - Phase 11 (Forecasting):    {len(df_p11)} episodes")
    
    # 2. Aggregate by independent experimental unit (Training Seed)
    # Each model is evaluated on 30 episodes (6 scenarios x 5 eval seeds)
    run_df = df_all.groupby(["phase", "algorithm", "architecture", "communication", "forecasting", "regime", "training_seed"]).agg({
        "cost": "mean",
        "service_level": "mean",
        "bullwhip": "mean"
    }).reset_index()
    run_df["cost_per_sl"] = run_df["cost"] / run_df["service_level"]
    
    print(f"\nTotal independent trained models: {len(run_df)}")
    
    # Save run-level table
    run_df.to_csv("results/phase12/ablation_summary.csv", index=False)
    print("Saved run-level results: results/phase12/ablation_summary.csv")
    
    # 3. Scenario-level breakdown
    scenario_df = df_all.groupby(["algorithm", "regime", "scenario"]).agg({
        "cost": ["mean", "std"],
        "service_level": ["mean", "std"],
        "bullwhip": ["mean", "std"]
    }).reset_index()
    scenario_df.columns = ["_".join(col).strip("_") for col in scenario_df.columns.values]
    scenario_df.to_csv("results/phase12/ablation_scenario_breakdown.csv", index=False)
    print("Saved scenario breakdown: results/phase12/ablation_scenario_breakdown.csv")
    
    # 4. Execute Ablation Hypothesis Testing Families
    families = {}
    
    # --------------------------------------------------------------------------
    # FAMILY A: Graph Architecture (GNN-MAPPO vs MLP-MAPPO)
    # --------------------------------------------------------------------------
    fam_a = []
    # Baseline regime (N=5 each, 200 updates)
    gnn_base = run_df[(run_df["algorithm"] == "gnn_mappo") & (run_df["regime"] == "baseline")]
    mlp_base = run_df[(run_df["algorithm"] == "mappo") & (run_df["regime"] == "baseline")]
    
    for metric in ["cost", "service_level", "bullwhip", "cost_per_sl"]:
        res = compute_welch_stats(gnn_base[metric].values, mlp_base[metric].values)
        res.update({"family": "A_Topology", "comparison": "GNN_vs_MLP_Baseline", "metric": metric})
        fam_a.append(res)
        
    # High variance regime (N=5 each, 200 updates)
    gnn_hv = run_df[(run_df["algorithm"] == "gnn_mappo") & (run_df["regime"] == "high_variance")]
    mlp_hv = run_df[(run_df["algorithm"] == "mappo") & (run_df["regime"] == "high_variance")]
    for metric in ["cost", "service_level", "bullwhip", "cost_per_sl"]:
        res = compute_welch_stats(gnn_hv[metric].values, mlp_hv[metric].values)
        res.update({"family": "A_Topology", "comparison": "GNN_vs_MLP_HighVariance", "metric": metric})
        fam_a.append(res)
    families["Family_A_Topology"] = apply_bh_fdr(fam_a)
    
    # --------------------------------------------------------------------------
    # FAMILY B: Forecasting Conditioning in GNN-MAPPO (XGBoost vs Moving Average)
    # --------------------------------------------------------------------------
    fam_b = []
    xgb_runs = run_df[run_df["algorithm"] == "gnn_mappo_forecast_xgboost"]
    ma_runs = run_df[run_df["algorithm"] == "gnn_mappo_forecast_ma"]
    
    for metric in ["cost", "service_level", "bullwhip", "cost_per_sl"]:
        res = compute_welch_stats(xgb_runs[metric].values, ma_runs[metric].values)
        res.update({"family": "B_Forecasting_Conditioning", "comparison": "XGBoost_vs_MA_50updates", "metric": metric})
        fam_b.append(res)
    families["Family_B_Forecasting"] = apply_bh_fdr(fam_b)
    
    # --------------------------------------------------------------------------
    # FAMILY C: Communication (IPPO_COMM vs IPPO, MAPPO_COMM vs MAPPO)
    # --------------------------------------------------------------------------
    fam_c = []
    ippo_base = run_df[(run_df["algorithm"] == "ippo") & (run_df["regime"] == "baseline")]
    ippo_comm_base = run_df[(run_df["algorithm"] == "ippo_comm") & (run_df["regime"] == "baseline")]
    mappo_comm_base = run_df[(run_df["algorithm"] == "mappo_comm") & (run_df["regime"] == "baseline")]
    
    for metric in ["cost", "service_level", "bullwhip", "cost_per_sl"]:
        res = compute_welch_stats(ippo_comm_base[metric].values, ippo_base[metric].values)
        res.update({"family": "C_Communication", "comparison": "IPPO_COMM_vs_IPPO", "metric": metric})
        fam_c.append(res)
        
    for metric in ["cost", "service_level", "bullwhip", "cost_per_sl"]:
        res = compute_welch_stats(mappo_comm_base[metric].values, mlp_base[metric].values)
        res.update({"family": "C_Communication", "comparison": "MAPPO_COMM_vs_MAPPO", "metric": metric})
        fam_c.append(res)
    families["Family_C_Communication"] = apply_bh_fdr(fam_c)
    
    # --------------------------------------------------------------------------
    # FAMILY D: Critic Centralization (MAPPO vs IPPO)
    # --------------------------------------------------------------------------
    fam_d = []
    for metric in ["cost", "service_level", "bullwhip", "cost_per_sl"]:
        res = compute_welch_stats(mlp_base[metric].values, ippo_base[metric].values)
        res.update({"family": "D_Centralization", "comparison": "MAPPO_vs_IPPO_Baseline", "metric": metric})
        fam_d.append(res)
        
    ippo_hv = run_df[(run_df["algorithm"] == "ippo") & (run_df["regime"] == "high_variance")]
    for metric in ["cost", "service_level", "bullwhip", "cost_per_sl"]:
        res = compute_welch_stats(mlp_hv[metric].values, ippo_hv[metric].values)
        res.update({"family": "D_Centralization", "comparison": "MAPPO_vs_IPPO_HighVariance", "metric": metric})
        fam_d.append(res)
    families["Family_D_Centralization"] = apply_bh_fdr(fam_d)
    
    # Save statistical test results to JSON
    with open("results/phase12/ablation_statistical_tests.json", "w") as fp:
        json.dump(families, fp, indent=4)
    print("Saved statistical tests: results/phase12/ablation_statistical_tests.json")
    
    # 5. Generate Comprehensive Plots
    generate_ablation_plots(run_df, df_all)
    print("Generated all plots in results/phase12/plots/")
    
    print("=" * 70)
    print("PHASE 12 STATISTICAL AUDIT COMPLETE!")
    print("=" * 70)

def generate_ablation_plots(run_df, df_all):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # Plot 1: Key Architectures Cost vs Service Level
    fig, ax = plt.subplots(figsize=(10, 6))
    algos = ["ippo", "ippo_comm", "mappo", "mappo_comm", "gnn_mappo"]
    labels = ["IPPO", "IPPO + Comm", "MAPPO", "MAPPO + Comm", "GNN-MAPPO"]
    colors = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c"]
    
    for algo, label, color in zip(algos, labels, colors):
        sub = run_df[(run_df["algorithm"] == algo) & (run_df["regime"] == "baseline")]
        ax.scatter(sub["service_level"] * 100, sub["cost"], label=label, color=color, s=80, alpha=0.85, edgecolors="none")
        ax.errorbar(sub["service_level"].mean() * 100, sub["cost"].mean(),
                    xerr=sub["service_level"].std() * 100, yerr=sub["cost"].std(),
                    fmt="o", color=color, ecolor=color, elinewidth=2, capsize=4, markersize=10)
        
    ax.set_title("Architectural Ablation: Cost vs. Service Level (N=5 Independent Seeds, 200 Updates)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean Service Level (%) [Higher is Better]", fontsize=11)
    ax.set_ylabel("Mean Global Operational Cost [Lower is Better]", fontsize=11)
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig("results/phase12/plots/ablation_family_comparison.png", dpi=300)
    plt.close()
    
    # Plot 2: Bullwhip Effect Across Architectures
    fig, ax = plt.subplots(figsize=(9, 5))
    means = [run_df[(run_df["algorithm"] == a) & (run_df["regime"] == "baseline")]["bullwhip"].mean() for a in algos]
    stds = [run_df[(run_df["algorithm"] == a) & (run_df["regime"] == "baseline")]["bullwhip"].std() for a in algos]
    
    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors, alpha=0.85)
    ax.set_title("Bullwhip Effect Across Ablation Architectures (N=5, 200 Updates)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Bullwhip Ratio (Var(Orders) / Var(Demand)) [Lower is Better]", fontsize=11)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.2f}", ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig("results/phase12/plots/bullwhip_comparison.png", dpi=300)
    plt.close()
    
    # Plot 3: Scenario-Level Cost Profiles for Primary Baselines
    fig, ax = plt.subplots(figsize=(12, 6))
    scenarios = ["in_distribution", "demand_shift", "lead_time_shift", "capacity_disruption", "demand_spike", "combined_shift"]
    sc_clean = ["In-Dist", "Demand Shift", "Lead Time", "Cap Disruption", "Spike", "Combined"]
    
    x = np.arange(len(scenarios))
    width = 0.16
    
    for i, (algo, label, color) in enumerate(zip(algos, labels, colors)):
        vals = []
        for sc in scenarios:
            v = df_all[(df_all["algorithm"] == algo) & (df_all["regime"] == "baseline") & (df_all["scenario"] == sc)]["cost"].mean()
            vals.append(v)
        ax.bar(x + (i - 2) * width, vals, width, label=label, color=color, alpha=0.85)
        
    ax.set_xticks(x)
    ax.set_xticklabels(sc_clean, fontsize=10)
    ax.set_title("Scenario-Level Operational Cost Profiles Across Ablation Groups", fontsize=13, fontweight="bold")
    ax.set_ylabel("Operational Cost", fontsize=11)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig("results/phase12/plots/ablation_scenarios.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
