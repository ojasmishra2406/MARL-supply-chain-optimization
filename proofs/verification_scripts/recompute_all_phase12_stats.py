import os
import glob
import json
import numpy as np
import pandas as pd
from scipy import stats

SCENARIOS = ['in_distribution', 'demand_shift', 'lead_time_shift', 'capacity_disruption', 'demand_spike', 'combined_shift']

def parse_phase7_files():
    records = []
    for f in glob.glob("results/phase7/*_eval*.json"):
        with open(f, "r") as fp:
            d = json.load(fp)
        base = os.path.basename(f)
        for sc in SCENARIOS:
            if f"_{sc}_" in base:
                model_name = base.split(f"_{sc}_")[0]
                eval_part = base.split(f"_{sc}_")[1].replace(".json", "")
                eval_seed = int(eval_part.replace("evals", "").replace("eval", ""))
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
                    "scenario": sc,
                    "eval_seed": eval_seed,
                    "cost": float(d["cost"]),
                    "service_level": float(d["fill_rate"]),
                    "bullwhip": float(d["bullwhip"]),
                })
                break
    return records

def parse_phase10_files():
    records = []
    for f in glob.glob("results/phase10/*_eval*.json"):
        with open(f, "r") as fp:
            d = json.load(fp)
        base = os.path.basename(f)
        for sc in SCENARIOS:
            if f"_{sc}_" in base:
                model_name = base.split(f"_{sc}_")[0]
                eval_part = base.split(f"_{sc}_")[1].replace(".json", "")
                eval_seed = int(eval_part.replace("evals", "").replace("eval", ""))
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
                    "scenario": sc,
                    "eval_seed": eval_seed,
                    "cost": float(d["total_cost"]),
                    "service_level": float(d["service_level"]),
                    "bullwhip": float(d["bullwhip_ratio"]),
                })
                break
    return records

def parse_phase11_files():
    records = []
    for f in glob.glob("results/phase11_controlled/*_eval*.json"):
        with open(f, "r") as fp:
            d = json.load(fp)
        base = os.path.basename(f)
        if "smoke" in base:
            continue
        for sc in SCENARIOS:
            if f"_{sc}_" in base:
                model_name = base.split(f"_{sc}_")[0]
                eval_part = base.split(f"_{sc}_")[1].replace(".json", "")
                eval_seed = int(eval_part.replace("evals", "").replace("eval", ""))
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
                    "scenario": sc,
                    "eval_seed": eval_seed,
                    "cost": float(d["total_cost"]),
                    "service_level": float(d["service_level"]),
                    "bullwhip": float(d["bullwhip_ratio"]),
                })
                break
    return records

def compute_welch(x, y):
    nx, ny = len(x), len(y)
    mx, my = np.mean(x), np.mean(y)
    sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)
    diff = mx - my
    pct_diff = (diff / my) * 100 if my != 0 else np.nan
    
    res = stats.ttest_ind(x, y, equal_var=False)
    t_stat = float(res.statistic)
    p_val = float(res.pvalue)
    
    vx = (sx**2) / nx if nx > 0 else 0
    vy = (sy**2) / ny if ny > 0 else 0
    df = (vx + vy)**2 / ((vx**2)/(nx-1) + (vy**2)/(ny-1)) if (vx + vy) > 0 else np.nan
    
    s_pooled = np.sqrt(((nx - 1)*(sx**2) + (ny - 1)*(sy**2)) / (nx + ny - 2)) if (nx + ny > 2) else np.nan
    d = diff / s_pooled if (s_pooled and s_pooled > 0) else np.nan
    J = 1.0 - (3.0 / (4.0 * (nx + ny - 2) - 1.0)) if (nx + ny > 2) else 1.0
    g = d * J if not np.isnan(d) else np.nan
    
    mwu = stats.mannwhitneyu(x, y, alternative="two-sided")
    
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

def main():
    p7 = parse_phase7_files()
    p10 = parse_phase10_files()
    p11 = parse_phase11_files()
    
    print(f"Parsed {len(p7)} P7 rows, {len(p10)} P10 rows, {len(p11)} P11 rows. Total = {len(p7)+len(p10)+len(p11)}")
    
    df_all = pd.DataFrame(p7 + p10 + p11)
    
    # Check aggregation
    run_df = df_all.groupby(["phase", "algorithm", "architecture", "communication", "forecasting", "regime", "training_seed"]).agg({
        "cost": "mean",
        "service_level": "mean",
        "bullwhip": "mean"
    }).reset_index()
    run_df["cost_per_sl"] = run_df["cost"] / run_df["service_level"]
    
    print(f"Total independent runs: {len(run_df)}")
    
    # Compare against existing results/phase12/ablation_summary.csv
    existing_summary = pd.read_csv("results/phase12/ablation_summary.csv")
    print(f"Existing summary length: {len(existing_summary)}")
    
    # Check max absolute difference between recomputed and existing
    diff_cost = np.max(np.abs(run_df["cost"].values - existing_summary["cost"].values))
    diff_sl = np.max(np.abs(run_df["service_level"].values - existing_summary["service_level"].values))
    diff_bw = np.max(np.abs(run_df["bullwhip"].values - existing_summary["bullwhip"].values))
    print(f"Max diff vs existing ablation_summary.csv: cost={diff_cost:.10f}, sl={diff_sl:.10f}, bw={diff_bw:.10f}")
    
    # Recompute statistical tests
    with open("results/phase12/ablation_statistical_tests.json", "r") as fp:
        existing_tests = json.load(fp)
        
    print("\nVerifying recomputed tests against results/phase12/ablation_statistical_tests.json:")
    
    # Family D: MAPPO vs IPPO Baseline
    mlp_base = run_df[(run_df["algorithm"] == "mappo") & (run_df["regime"] == "baseline")]
    ippo_base = run_df[(run_df["algorithm"] == "ippo") & (run_df["regime"] == "baseline")]
    
    for metric in ["cost", "service_level", "bullwhip", "cost_per_sl"]:
        calc = compute_welch(mlp_base[metric].values, ippo_base[metric].values)
        # Find in existing_tests
        ext = [t for t in existing_tests["Family_D_Centralization"] if t["comparison"] == "MAPPO_vs_IPPO_Baseline" and t["metric"] == metric][0]
        t_diff = abs(calc["welch_t"] - ext["welch_t"])
        p_diff = abs(calc["welch_p"] - ext["welch_p"])
        print(f"Family D Baseline {metric}:")
        print(f"  Calc: MeanX={calc['mean_x']:.2f} (SD={calc['sd_x']:.2f}) vs MeanY={calc['mean_y']:.2f} (SD={calc['sd_y']:.2f}), t={calc['welch_t']:.4f}, p={calc['welch_p']:.6f}, g={calc['hedges_g']:.4f}")
        print(f"  Ext:  MeanX={ext['mean_x']:.2f} (SD={ext['sd_x']:.2f}) vs MeanY={ext['mean_y']:.2f} (SD={ext['sd_y']:.2f}), t={ext['welch_t']:.4f}, p={ext['welch_p']:.6f}, g={ext['hedges_g']:.4f}")
        print(f"  Delta t={t_diff:.10f}, Delta p={p_diff:.10f}")

if __name__ == "__main__":
    main()
