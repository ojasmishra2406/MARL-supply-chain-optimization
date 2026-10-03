"""
benchmark_statistics.py — Independent reconstruction of Phase 8 statistical analysis.
"""
import os, glob, json, datetime, subprocess, sys, math
from collections import defaultdict
import numpy as np
from scipy import stats

ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PHASE7_DIR = os.path.join(ROOT, "results", "phase7")
OBS_CSV    = os.path.join(ROOT, "results", "phase8_observations.csv")

FAILURES = []
TOL = 1e-6

def git_commit():
    try:
        return subprocess.check_output(["git","rev-parse","HEAD"],
                                       cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"

def extract_train_seed(ckpt_path):
    if not ckpt_path:
        return None
    base = os.path.splitext(os.path.basename(ckpt_path))[0]
    parts = base.split("_")
    if parts and parts[-1].startswith("s") and parts[-1][1:].isdigit():
        return int(parts[-1][1:])
    return None

def cohens_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return None
    pooled_std = math.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    if pooled_std == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled_std

def bh_fdr(pvals):
    n = len(pvals)
    if n == 0:
        return []
    idx_sorted = np.argsort(pvals)
    ranked = np.empty(n, dtype=float)
    for rank, i in enumerate(idx_sorted):
        ranked[i] = rank + 1
    adj = np.array(pvals, dtype=float) * n / ranked
    result = adj.copy()
    for i in range(n-2, -1, -1):
        ri = idx_sorted[i]
        ri1 = idx_sorted[i+1]
        result[ri] = min(result[ri], result[ri1])
    return np.minimum(result, 1.0).tolist()

def run():
    ts  = datetime.datetime.utcnow().isoformat() + "Z"
    commit = git_commit()

    eval_files = sorted(glob.glob(os.path.join(PHASE7_DIR, "*.json")))
    raw_records = []
    for fpath in eval_files:
        with open(fpath) as f:
            d = json.load(f)
        if "cost" not in d:
            continue
        algo = d.get("algorithm", "out")
        cond = "baseline"
        if "high_variance" in algo:
            cond = "high_variance"
            algo = algo.replace("_high_variance", "")
        elif "baseline" in algo:
            algo = algo.replace("_baseline", "")

        ckpt = d.get("checkpoint")
        train_seed = extract_train_seed(ckpt)
        eval_seed  = d.get("seed")

        raw_records.append({
            "algorithm": algo,
            "condition": cond,
            "train_seed": train_seed,
            "eval_seed": eval_seed,
            "scenario": d.get("scenario"),
            "cost": d["cost"],
            "fill_rate": d.get("fill_rate"),
            "bullwhip": d.get("bullwhip"),
        })

    total_raw = len(raw_records)
    rl_records = [r for r in raw_records if r.get("train_seed") is not None]
    out_records = [r for r in raw_records if r.get("train_seed") is None]

    groups = defaultdict(list)
    for r in rl_records:
        key = (r["algorithm"], r["condition"], r["scenario"], r["train_seed"])
        groups[key].append(r["cost"])

    model_obs = defaultdict(list)
    train_seed_counts = defaultdict(set)

    for (algo, cond, scen, ts_val), costs in groups.items():
        agg_cost = float(np.mean(costs))
        eval_seed_count_this = len(costs)
        model_obs[(algo, cond, scen)].append({
            "train_seed": ts_val,
            "eval_seeds_averaged": eval_seed_count_this,
            "mean_cost": agg_cost,
        })
        train_seed_counts[(algo, cond, scen)].add(ts_val)

    n_report = []
    for (algo, cond, scen), obs in sorted(model_obs.items()):
        n = len(obs)
        eval_seeds_per = [o["eval_seeds_averaged"] for o in obs]
        n_report.append({
            "algorithm": algo,
            "condition": cond,
            "scenario": scen,
            "statistical_N": n,
            "eval_seeds_per_train_seed_min": min(eval_seeds_per),
            "eval_seeds_per_train_seed_max": max(eval_seeds_per),
            "source": "raw Phase 7 JSON aggregation",
        })

    MAX_TRAIN_SEEDS = 5
    for item in n_report:
        if item["statistical_N"] > MAX_TRAIN_SEEDS:
            FAILURES.append(f"PSEUDOREPLICATION DETECTED: {item['algorithm']}/{item['condition']}/{item['scenario']} "
                            f"has N={item['statistical_N']} > {MAX_TRAIN_SEEDS} max training seeds")

    def get_costs(algo, cond, scen):
        return [o["mean_cost"] for o in model_obs.get((algo, cond, scen), [])]

    comparisons = [
        {"name": "IPPO vs MAPPO (Baseline, In-Distribution)",
         "algo1": "ippo", "cond1": "baseline",
         "algo2": "mappo", "cond2": "baseline", "scen": "in_distribution"},
        {"name": "MAPPO vs MAPPO_COMM (Baseline, In-Distribution)",
         "algo1": "mappo", "cond1": "baseline",
         "algo2": "mappo_comm", "cond2": "baseline", "scen": "in_distribution"},
        {"name": "IPPO vs IPPO_COMM (Baseline, In-Distribution)",
         "algo1": "ippo", "cond1": "baseline",
         "algo2": "ippo_comm", "cond2": "baseline", "scen": "in_distribution"},
        {"name": "MAPPO vs MAPPO_COMM (Baseline, Combined Shift)",
         "algo1": "mappo", "cond1": "baseline",
         "algo2": "mappo_comm", "cond2": "baseline", "scen": "combined_shift"},
    ]

    p_values = []
    stats_results = []
    for comp in comparisons:
        d1 = get_costs(comp["algo1"], comp["cond1"], comp["scen"])
        d2 = get_costs(comp["algo2"], comp["cond2"], comp["scen"])
        rec = {"name": comp["name"], "N1": len(d1), "N2": len(d2),
               "mean1": float(np.mean(d1)) if d1 else None,
               "mean2": float(np.mean(d2)) if d2 else None}
        if len(d1) >= 2 and len(d2) >= 2:
            t, p = stats.ttest_ind(d1, d2, equal_var=False)
            d_val = cohens_d(d1, d2)
            rec.update({"t_statistic": round(float(t), 6),
                        "p_value_raw": round(float(p), 8),
                        "cohens_d": round(float(d_val), 6) if d_val is not None else None})
            p_values.append(float(p))
        else:
            rec["insufficient_data"] = True
            p_values.append(1.0)
        stats_results.append(rec)

    adj_p = bh_fdr(p_values)
    for i, rec in enumerate(stats_results):
        rec["bh_adjusted_p"] = round(adj_p[i], 8)

    stored_comparison = None
    if os.path.exists(OBS_CSV):
        import csv
        stored_rows = []
        with open(OBS_CSV) as f:
            reader = csv.DictReader(f)
            for row in reader:
                stored_rows.append(row)
        stored_n = len(stored_rows)
        stored_keys = set((r["algorithm"], r["condition"], r["scenario"], r["train_seed"])
                          for r in stored_rows)
        computed_keys = set(str(k) for k in groups.keys())
        stored_comparison = {
            "stored_observation_rows": stored_n,
            "independently_reconstructed_rows": len(groups),
            "match": stored_n == len(groups),
        }
        if stored_n != len(groups):
            FAILURES.append(f"STORED OBS COUNT {stored_n} != RECONSTRUCTED {len(groups)}")

    result = {
        "total_raw_eval_files": total_raw,
        "rl_eval_records": len(rl_records),
        "out_eval_records": len(out_records),
        "training_seed_groups": len(groups),
        "n_per_config": n_report,
        "statistical_comparisons": stats_results,
        "stored_vs_computed": stored_comparison,
        "pseudoreplication_check": "PASS" if not any("PSEUDOREPLICATION" in f for f in FAILURES) else "FAIL",
        "failures": FAILURES,
        "timestamp": ts,
        "git_commit": commit,
        "source": f"raw scan of {PHASE7_DIR}",
        "method": "groupby(algo,cond,scen,train_seed) → mean over eval_seeds → Welch t-test + BH-FDR",
    }
    return result

if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2, default=str))
