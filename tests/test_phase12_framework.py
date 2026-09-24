import os
import json
import numpy as np
import pandas as pd
from scipy import stats
import pytest

from run_phase12_ablation import get_experiment_id, get_combinations, is_duplicate, validate_run

# --- STATISTICAL ENGINE ---

def bh_fdr_grouped(p_values_dict):
    """
    Applies Benjamini-Hochberg FDR correction *within* defined families.
    p_values_dict: { "family_name": { "comparison_name": p_value } }
    Returns: { "family_name": { "comparison_name": adjusted_p_value } }
    """
    adjusted_dict = {}
    for family, comparisons in p_values_dict.items():
        if not comparisons:
            continue
            
        names = list(comparisons.keys())
        p_vals = list(comparisons.values())
        
        n = len(p_vals)
        ranked_indices = np.argsort(p_vals)
        
        adjusted_p = np.zeros(n)
        for i, rank in enumerate(ranked_indices):
            adjusted_p[rank] = min(1.0, p_vals[rank] * n / (i + 1))
            
        # Enforce monotonicity
        for i in range(n - 2, -1, -1):
            rank_curr = ranked_indices[i]
            rank_next = ranked_indices[i + 1]
            adjusted_p[rank_curr] = min(adjusted_p[rank_curr], adjusted_p[rank_next])
            
        adjusted_dict[family] = {names[i]: adjusted_p[i] for i in range(n)}
        
    return adjusted_dict

def aggregate_and_test(df_results, control_algo, treatment_algo, metric="cost"):
    """
    1. Averages over eval_seeds to yield 1 value per train_seed (N=5).
    2. Runs Welch's t-test.
    """
    # 1. Experimental unit aggregation
    agg_df = df_results.groupby(["algorithm", "train_seed"]).agg({metric: "mean"}).reset_index()
    
    # 2. Extract arrays
    control_data = agg_df[agg_df["algorithm"] == control_algo][metric].values
    treatment_data = agg_df[agg_df["algorithm"] == treatment_algo][metric].values
    
    if len(control_data) < 2 or len(treatment_data) < 2:
        return np.nan
        
    t_stat, p_val = stats.ttest_ind(control_data, treatment_data, equal_var=False)
    return p_val

# --- TESTS ---

def test_phase12_combinations():
    config = {
        "ablation_family": "A",
        "algorithm": ["ippo", "mappo"],
        "regime": "baseline",
        "seed": [0, 1]
    }
    combos = get_combinations(config)
    assert len(combos) == 4, f"Expected 4 combinations, got {len(combos)}"
    
    algos = [c["algorithm"] for c in combos]
    assert algos.count("ippo") == 2
    assert algos.count("mappo") == 2

def test_phase12_experiment_id():
    params1 = {"ablation_family": "A_Arch", "algorithm": "mappo", "seed": 42}
    params2 = {"ablation_family": "A_Arch", "algorithm": "mappo", "seed": 42}
    params3 = {"ablation_family": "A_Arch", "algorithm": "mappo", "seed": 0}
    
    id1 = get_experiment_id(params1)
    id2 = get_experiment_id(params2)
    id3 = get_experiment_id(params3)
    
    assert id1 == id2, "Deterministic ID failed"
    assert id1 != id3, "ID collision for different seeds"
    assert id1.startswith("p12_A_mappo")

def test_phase12_duplicate_detection(tmp_path):
    results_dir = str(tmp_path)
    exp_id = "p12_test_duplicate"
    
    assert not is_duplicate(exp_id, results_dir)
    
    # Create fake files
    with open(os.path.join(results_dir, f"{exp_id}.pt"), "w") as f: f.write("chkpt")
    with open(os.path.join(results_dir, f"{exp_id}_manifest.json"), "w") as f: f.write("{}")
    
    assert is_duplicate(exp_id, results_dir), "Duplicate detection failed"

def test_phase12_bh_fdr_grouping():
    raw_p_values = {
        "Family_A": {
            "GNN_vs_MLP": 0.01,
            "GNN_vs_MAPPO": 0.04
        },
        "Family_B": {
            "Forecast_vs_NoForecast": 0.03
        }
    }
    
    adj = bh_fdr_grouped(raw_p_values)
    
    # Family A: n=2
    # p=0.01 (rank 0) -> adj = 0.01 * 2 / 1 = 0.02
    # p=0.04 (rank 1) -> adj = 0.04 * 2 / 2 = 0.04
    assert np.isclose(adj["Family_A"]["GNN_vs_MLP"], 0.02)
    assert np.isclose(adj["Family_A"]["GNN_vs_MAPPO"], 0.04)
    
    # Family B: n=1
    # p=0.03 (rank 0) -> adj = 0.03 * 1 / 1 = 0.03
    assert np.isclose(adj["Family_B"]["Forecast_vs_NoForecast"], 0.03)

def test_phase12_experimental_unit_aggregation():
    # Mock data showing multiple eval_seeds per train_seed
    data = []
    for train_seed in range(5):
        for eval_seed in range(3):
            # Control has mean ~ 100
            data.append({"algorithm": "control", "train_seed": train_seed, "eval_seed": eval_seed, "cost": np.random.normal(100, 10)})
            # Treatment has mean ~ 80
            data.append({"algorithm": "treatment", "train_seed": train_seed, "eval_seed": eval_seed, "cost": np.random.normal(80, 10)})
            
    df = pd.DataFrame(data)
    
    # The function averages over eval_seeds (leaving 5 data points per algorithm) and runs Welch t-test
    p_val = aggregate_and_test(df, "control", "treatment")
    
    assert not np.isnan(p_val)
    assert 0 <= p_val <= 1.0
