"""
Independent recomputation of Phase 12 statistics from raw evaluation JSONs.
Does NOT use the run_phase12_ablation_analysis.py script or its outputs.
Recomputes everything from scratch and compares against the saved JSON.
"""
import os, glob, json, sys
import numpy as np
from scipy import stats

SCENARIOS = ['in_distribution', 'demand_shift', 'lead_time_shift', 
             'capacity_disruption', 'demand_spike', 'combined_shift']

def extract_model_and_scenario(filepath):
    base = os.path.basename(filepath)
    for sc in SCENARIOS:
        if f"_{sc}_" in base:
            model_name = base.split(f"_{sc}_")[0]
            eval_part = base.split(f"_{sc}_")[1].replace(".json", "")
            eval_seed = int(eval_part.replace("evals", "").replace("eval", ""))
            return model_name, sc, eval_seed
    raise ValueError(f"Could not parse scenario from {filepath}")

def load_all_raw():
    """Load all raw evaluation JSONs independently."""
    records = []
    
    # Phase 7
    for f in glob.glob("results/phase7/*_eval*.json"):
        with open(f) as fp:
            d = json.load(fp)
        mn, sc, es = extract_model_and_scenario(f)
        ts = int(mn.split("_s")[-1])
        regime = "high_variance" if "high_variance" in mn else "baseline"
        if "ippo_comm" in mn: algo = "ippo_comm"
        elif "mappo_comm" in mn: algo = "mappo_comm"
        elif "ippo" in mn: algo = "ippo"
        elif "mappo" in mn: algo = "mappo"
        else: algo = "other"
        records.append({
            "phase": "phase7", "model_name": mn, "algorithm": algo,
            "regime": regime, "training_seed": ts, "scenario": sc,
            "eval_seed": es, "cost": float(d["cost"]),
            "service_level": float(d["fill_rate"]),
            "bullwhip": float(d["bullwhip"])
        })
    
    # Phase 10
    for f in glob.glob("results/phase10/*_eval*.json"):
        with open(f) as fp:
            d = json.load(fp)
        mn, sc, es = extract_model_and_scenario(f)
        ts = int(mn.split("_s")[-1])
        regime = "high_variance" if "high_variance" in mn else "baseline"
        records.append({
            "phase": "phase10", "model_name": mn, "algorithm": "gnn_mappo",
            "regime": regime, "training_seed": ts, "scenario": sc,
            "eval_seed": es, "cost": float(d["total_cost"]),
            "service_level": float(d["service_level"]),
            "bullwhip": float(d["bullwhip_ratio"])
        })
    
    # Phase 11 controlled
    for f in glob.glob("results/phase11_controlled/*_eval*.json"):
        with open(f) as fp:
            d = json.load(fp)
        mn, sc, es = extract_model_and_scenario(f)
        if "smoke" in mn:
            continue
        ts = int(mn.split("_s")[-1])
        forecaster = d.get("forecaster", "ma")
        records.append({
            "phase": "phase11", "model_name": mn,
            "algorithm": f"gnn_mappo_forecast_{forecaster}",
            "regime": "baseline", "training_seed": ts, "scenario": sc,
            "eval_seed": es, "cost": float(d["total_cost"]),
            "service_level": float(d["service_level"]),
            "bullwhip": float(d["bullwhip_ratio"])
        })
    
    return records

def compute_welch(x, y):
    nx, ny = len(x), len(y)
    mx, my = np.mean(x), np.mean(y)
    sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)
    res = stats.ttest_ind(x, y, equal_var=False)
    mwu = stats.mannwhitneyu(x, y, alternative="two-sided")
    s_pooled = np.sqrt(((nx-1)*sx**2 + (ny-1)*sy**2)/(nx+ny-2))
    d = (mx - my)/s_pooled if s_pooled > 0 else np.nan
    J = 1.0 - 3.0/(4*(nx+ny-2)-1) if nx+ny > 2 else 1.0
    return {
        "nx": nx, "ny": ny, "mean_x": mx, "sd_x": sx, "mean_y": my, "sd_y": sy,
        "welch_t": float(res.statistic), "welch_p": float(res.pvalue),
        "mwu_u": float(mwu.statistic), "mwu_p": float(mwu.pvalue),
        "cohens_d": d, "hedges_g": d*J
    }

def main():
    records = load_all_raw()
    
    # Count unique files per phase
    phase_counts = {}
    for r in records:
        phase_counts[r["phase"]] = phase_counts.get(r["phase"], 0) + 1
    
    print("=" * 70)
    print("INDEPENDENT RECOMPUTATION FROM RAW EVALUATION JSONs")
    print("=" * 70)
    print(f"\nTotal raw evaluation episodes loaded: {len(records)}")
    for p, c in sorted(phase_counts.items()):
        print(f"  {p}: {c} episodes")
    
    # Count unique trained models
    unique_models = set()
    for r in records:
        unique_models.add((r["algorithm"], r["regime"], r["training_seed"]))
    print(f"\nUnique trained model identities: {len(unique_models)}")
    for m in sorted(unique_models):
        count = sum(1 for r in records if r["algorithm"]==m[0] and r["regime"]==m[1] and r["training_seed"]==m[2])
        print(f"  {m[0]} / {m[1]} / seed={m[2]}: {count} episodes")
    
    # Aggregate by independent experimental unit (model = algo+regime+training_seed)
    model_metrics = {}
    for r in records:
        key = (r["algorithm"], r["regime"], r["training_seed"])
        if key not in model_metrics:
            model_metrics[key] = {"cost": [], "service_level": [], "bullwhip": []}
        model_metrics[key]["cost"].append(r["cost"])
        model_metrics[key]["service_level"].append(r["service_level"])
        model_metrics[key]["bullwhip"].append(r["bullwhip"])
    
    # Average across episodes for each model
    model_avgs = {}
    for key, vals in model_metrics.items():
        model_avgs[key] = {
            "cost": np.mean(vals["cost"]),
            "service_level": np.mean(vals["service_level"]),
            "bullwhip": np.mean(vals["bullwhip"]),
            "cost_per_sl": np.mean(vals["cost"]) / np.mean(vals["service_level"])
        }
    
    # Extract model-level values for key comparisons
    def get_values(algo, regime, metric):
        return np.array([model_avgs[k][metric] for k in sorted(model_avgs) 
                        if k[0]==algo and k[1]==regime])
    
    print("\n" + "=" * 70)
    print("KEY COMPARISONS (INDEPENDENT RECOMPUTATION)")
    print("=" * 70)
    
    # Family D: MAPPO vs IPPO Baseline
    comparisons = [
        ("Family_D: MAPPO vs IPPO Baseline", "mappo", "ippo", "baseline"),
        ("Family_D: MAPPO vs IPPO HighVariance", "mappo", "ippo", "high_variance"),
        ("Family_A: GNN vs MLP Baseline", "gnn_mappo", "mappo", "baseline"),
        ("Family_A: GNN vs MLP HighVariance", "gnn_mappo", "mappo", "high_variance"),
        ("Family_B: XGBoost vs MA 50updates", "gnn_mappo_forecast_xgboost", "gnn_mappo_forecast_ma", "baseline"),
    ]
    
    recomputed = {}
    for label, algo_x, algo_y, regime in comparisons:
        print(f"\n--- {label} ---")
        for metric in ["cost", "service_level", "bullwhip", "cost_per_sl"]:
            x = get_values(algo_x, regime, metric)
            y = get_values(algo_y, regime, metric)
            if len(x) == 0 or len(y) == 0:
                print(f"  {metric}: SKIPPED (no data: x={len(x)}, y={len(y)})")
                continue
            r = compute_welch(x, y)
            print(f"  {metric}: mean_x={r['mean_x']:.6f} sd_x={r['sd_x']:.6f} "
                  f"mean_y={r['mean_y']:.6f} sd_y={r['sd_y']:.6f} "
                  f"t={r['welch_t']:.4f} p={r['welch_p']:.6f} g={r['hedges_g']:.4f} "
                  f"U={r['mwu_u']:.1f} mwu_p={r['mwu_p']:.6f}")
            recomputed[(label, metric)] = r
    
    # Now compare with saved JSON
    print("\n" + "=" * 70)
    print("COMPARISON WITH SAVED ablation_statistical_tests.json")
    print("=" * 70)
    
    with open("results/phase12/ablation_statistical_tests.json") as fp:
        saved = json.load(fp)
    
    total_tests_in_json = sum(len(v) for v in saved.values())
    print(f"\nTotal tests in saved JSON: {total_tests_in_json}")
    for fam, tests in saved.items():
        print(f"  {fam}: {len(tests)} tests")
    
    # Spot-check: Family D MAPPO vs IPPO Baseline bullwhip
    fam_d = saved["Family_D_Centralization"]
    for t in fam_d:
        if t["comparison"] == "MAPPO_vs_IPPO_Baseline" and t["metric"] == "bullwhip":
            key = ("Family_D: MAPPO vs IPPO Baseline", "bullwhip")
            if key in recomputed:
                r = recomputed[key]
                print(f"\n  MAPPO vs IPPO bullwhip:")
                print(f"    Saved:      mean_x={t['mean_x']:.6f} sd_x={t['sd_x']:.6f} t={t['welch_t']:.4f} p={t['welch_p']:.6f}")
                print(f"    Recomputed: mean_x={r['mean_x']:.6f} sd_x={r['sd_x']:.6f} t={r['welch_t']:.4f} p={r['welch_p']:.6f}")
                delta_t = abs(t['welch_t'] - r['welch_t'])
                delta_p = abs(t['welch_p'] - r['welch_p'])
                print(f"    Δt={delta_t:.8f}  Δp={delta_p:.8f}")
                if delta_t < 1e-4 and delta_p < 1e-4:
                    print(f"    STATUS: MATCH")
                else:
                    print(f"    STATUS: MISMATCH!")
    
    # Spot-check: Family D MAPPO vs IPPO Baseline cost_per_sl
    for t in fam_d:
        if t["comparison"] == "MAPPO_vs_IPPO_Baseline" and t["metric"] == "cost_per_sl":
            key = ("Family_D: MAPPO vs IPPO Baseline", "cost_per_sl")
            if key in recomputed:
                r = recomputed[key]
                print(f"\n  MAPPO vs IPPO cost_per_sl:")
                print(f"    Saved:      mean_x={t['mean_x']:.2f} sd_x={t['sd_x']:.2f} t={t['welch_t']:.4f} p={t['welch_p']:.6f}")
                print(f"    Recomputed: mean_x={r['mean_x']:.2f} sd_x={r['sd_x']:.2f} t={r['welch_t']:.4f} p={r['welch_p']:.6f}")
                delta_t = abs(t['welch_t'] - r['welch_t'])
                delta_p = abs(t['welch_p'] - r['welch_p'])
                print(f"    Δt={delta_t:.8f}  Δp={delta_p:.8f}")
                if delta_t < 1e-4 and delta_p < 1e-4:
                    print(f"    STATUS: MATCH")
                else:
                    print(f"    STATUS: MISMATCH!")

    # Comprehensive spot-check: all 28 tests
    print("\n" + "=" * 70)
    print("FULL 28-TEST SPOT CHECK AGAINST SAVED JSON")
    print("=" * 70)
    
    comparison_map = {
        "GNN_vs_MLP_Baseline": ("gnn_mappo", "mappo", "baseline"),
        "GNN_vs_MLP_HighVariance": ("gnn_mappo", "mappo", "high_variance"),
        "XGBoost_vs_MA_50updates": ("gnn_mappo_forecast_xgboost", "gnn_mappo_forecast_ma", "baseline"),
        "IPPO_COMM_vs_IPPO": ("ippo_comm", "ippo", "baseline"),
        "MAPPO_COMM_vs_MAPPO": ("mappo_comm", "mappo", "baseline"),
        "MAPPO_vs_IPPO_Baseline": ("mappo", "ippo", "baseline"),
        "MAPPO_vs_IPPO_HighVariance": ("mappo", "ippo", "high_variance"),
    }
    
    matches = 0
    mismatches = 0
    for fam_name, tests in saved.items():
        for t in tests:
            comp = t["comparison"]
            metric = t["metric"]
            if comp not in comparison_map:
                print(f"  UNKNOWN comparison: {comp}")
                mismatches += 1
                continue
            ax, ay, reg = comparison_map[comp]
            x = get_values(ax, reg, metric)
            y = get_values(ay, reg, metric)
            if len(x) == 0 or len(y) == 0:
                print(f"  {fam_name}/{comp}/{metric}: NO DATA (x={len(x)}, y={len(y)})")
                mismatches += 1
                continue
            r = compute_welch(x, y)
            dt = abs(t["welch_t"] - r["welch_t"])
            dp = abs(t["welch_p"] - r["welch_p"])
            dm = abs(t["mean_x"] - r["mean_x"])
            if dt < 1e-3 and dp < 1e-3 and dm < 1.0:
                matches += 1
            else:
                mismatches += 1
                print(f"  MISMATCH: {fam_name}/{comp}/{metric}")
                print(f"    Saved:  mean_x={t['mean_x']:.4f} t={t['welch_t']:.4f} p={t['welch_p']:.6f}")
                print(f"    Recomp: mean_x={r['mean_x']:.4f} t={r['welch_t']:.4f} p={r['welch_p']:.6f}")
                print(f"    Δmean={dm:.4f} Δt={dt:.6f} Δp={dp:.6f}")
    
    print(f"\n  RESULT: {matches}/{matches+mismatches} tests MATCH, {mismatches} MISMATCHES")
    
    # Check for smoke test contamination in Phase 11
    print("\n" + "=" * 70)
    print("PHASE 11 SMOKE TEST CONTAMINATION CHECK")
    print("=" * 70)
    smoke_files = glob.glob("results/phase11_controlled/*smoke*")
    print(f"  Smoke test files found: {len(smoke_files)}")
    p11_non_smoke = [r for r in records if r["phase"] == "phase11"]
    p11_models = set((r["algorithm"], r["training_seed"]) for r in p11_non_smoke)
    print(f"  Non-smoke Phase 11 models: {len(p11_models)}")
    for m in sorted(p11_models):
        print(f"    {m[0]} seed={m[1]}")

    # GNN disruption-specific check
    print("\n" + "=" * 70)
    print("GNN DISRUPTION-SPECIFIC STATISTICS")
    print("=" * 70)
    gnn_disruption = [r for r in records if r["algorithm"]=="gnn_mappo" 
                      and r["regime"]=="baseline" and r["scenario"]=="capacity_disruption"]
    mappo_disruption = [r for r in records if r["algorithm"]=="mappo" 
                        and r["regime"]=="baseline" and r["scenario"]=="capacity_disruption"]
    
    if gnn_disruption:
        gnn_costs = [r["cost"] for r in gnn_disruption]
        print(f"  GNN baseline disruption episodes: {len(gnn_disruption)}")
        print(f"  GNN disruption cost: mean={np.mean(gnn_costs):.2f} sd={np.std(gnn_costs, ddof=1):.2f}")
        print(f"  GNN disruption costs per seed:")
        by_seed = {}
        for r in gnn_disruption:
            by_seed.setdefault(r["training_seed"], []).append(r["cost"])
        for s in sorted(by_seed):
            vals = by_seed[s]
            print(f"    seed {s}: episodes={len(vals)} mean={np.mean(vals):.2f}")
    
    if mappo_disruption:
        mappo_costs = [r["cost"] for r in mappo_disruption]
        print(f"  MAPPO baseline disruption episodes: {len(mappo_disruption)}")
        print(f"  MAPPO disruption cost: mean={np.mean(mappo_costs):.2f} sd={np.std(mappo_costs, ddof=1):.2f}")
    
    # Model-level aggregation for GNN vs MAPPO disruption
    if gnn_disruption and mappo_disruption:
        gnn_model_means = [np.mean(v) for v in by_seed.values()]
        mappo_by_seed = {}
        for r in mappo_disruption:
            mappo_by_seed.setdefault(r["training_seed"], []).append(r["cost"])
        mappo_model_means = [np.mean(v) for v in mappo_by_seed.values()]
        print(f"\n  Model-level GNN disruption means: {[f'{x:.2f}' for x in gnn_model_means]}")
        print(f"  Model-level MAPPO disruption means: {[f'{x:.2f}' for x in mappo_model_means]}")
        print(f"  GNN model-level: mean={np.mean(gnn_model_means):.2f} sd={np.std(gnn_model_means, ddof=1):.2f}")
        print(f"  MAPPO model-level: mean={np.mean(mappo_model_means):.2f} sd={np.std(mappo_model_means, ddof=1):.2f}")
        
        t_res = stats.ttest_ind(gnn_model_means, mappo_model_means, equal_var=False)
        print(f"  Welch t={t_res.statistic:.4f} p={t_res.pvalue:.6f}")

if __name__ == "__main__":
    os.chdir(r"c:\Users\mishr\OneDrive\Documents\MARL")
    main()
