import os
import json
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from run_phase12_ablation_analysis import (
    load_phase7_runs,
    load_phase10_runs,
    load_phase11_controlled_runs,
    compute_welch_stats,
    apply_bh_fdr
)
from rl.checkpoint import load_checkpoint

def test_data_ingestion_integrity():
    df_p7 = load_phase7_runs()
    df_p10 = load_phase10_runs()
    df_p11 = load_phase11_controlled_runs()
    
    assert len(df_p7) == 1200, f"Expected 1200 Phase 7 rows, got {len(df_p7)}"
    assert len(df_p10) == 300, f"Expected 300 Phase 10 rows, got {len(df_p10)}"
    assert len(df_p11) == 180, f"Expected 180 Phase 11 controlled rows, got {len(df_p11)}"
    
    df_all = pd.concat([df_p7, df_p10, df_p11], ignore_index=True)
    assert not df_all["cost"].isna().any(), "Found NaN in costs"
    assert not df_all["service_level"].isna().any(), "Found NaN in service levels"
    assert not df_all["bullwhip"].isna().any(), "Found NaN in bullwhip ratios"
    assert (df_all["service_level"] >= 0).all() and (df_all["service_level"] <= 1.05).all(), "Invalid service level range"

def test_checkpoint_provenance_verification():
    # Phase 6 checkpoint test
    p6_ckpt = "results/phase6/phase6_mappo_baseline_s0.pt"
    p6_manifest = "results/phase6/phase6_mappo_baseline_s0_manifest.json"
    p6_state = load_checkpoint(p6_ckpt, p6_manifest)
    assert "retailer" in p6_state
    
    # Phase 10 checkpoint test
    p10_ckpt = "results/phase10/phase10_gnn_mappo_baseline_s0.pt"
    p10_manifest = "results/phase10/phase10_gnn_mappo_baseline_s0_manifest.json"
    p10_state = load_checkpoint(p10_ckpt, p10_manifest)
    assert "retailer" in p10_state
    
    # Phase 11 controlled checkpoint test
    p11_ckpt = "results/phase11_controlled/phase11_controlled_ma_s1.pt"
    p11_manifest = "results/phase11_controlled/phase11_controlled_ma_s1_manifest.json"
    p11_state = load_checkpoint(p11_ckpt, p11_manifest)
    assert "retailer" in p11_state

def test_welch_statistical_calculations():
    np.random.seed(42)
    x = np.random.normal(100, 15, size=5)
    y = np.random.normal(85, 10, size=5)
    
    stats_dict = compute_welch_stats(x, y)
    
    # Compare against direct scipy
    scipy_res = stats.ttest_ind(x, y, equal_var=False)
    assert np.isclose(stats_dict["welch_t"], scipy_res.statistic)
    assert np.isclose(stats_dict["welch_p"], scipy_res.pvalue)
    assert stats_dict["nx"] == 5
    assert stats_dict["ny"] == 5
    assert stats_dict["ci_lower"] < stats_dict["diff"] < stats_dict["ci_upper"]

def test_bh_fdr_multiple_testing_correction():
    mock_tests = [
        {"welch_p": 0.001},
        {"welch_p": 0.04},
        {"welch_p": 0.08},
        {"welch_p": 0.70}
    ]
    fdr_results = apply_bh_fdr(mock_tests)
    
    assert len(fdr_results) == 4
    for r in fdr_results:
        assert 0 <= r["fdr_q"] <= 1.0
        assert "significant_fdr" in r
        
    # Check rank order adjustment
    # Lowest p should be adjusted by N / 1 = 4
    assert np.isclose(fdr_results[0]["fdr_q"], 0.001 * 4 / 1)

def test_experimental_unit_no_pseudoreplication():
    df_p7 = load_phase7_runs()
    ippo_base = df_p7[df_p7["algorithm"] == "ippo"]
    
    # Verify there are 5 unique training seeds
    unique_seeds = ippo_base["training_seed"].unique()
    assert len(unique_seeds) == 5, f"Expected 5 training seeds, got {len(unique_seeds)}"
    
    # Aggregate across 30 episodes per training seed
    run_df = ippo_base.groupby("training_seed").agg({"cost": "mean"}).reset_index()
    assert len(run_df) == 5, "Sample size must equal number of training seeds (N=5)"

def test_artifact_completeness():
    assert os.path.exists("results/phase12/ablation_summary.csv"), "Missing ablation_summary.csv"
    assert os.path.exists("results/phase12/ablation_scenario_breakdown.csv"), "Missing ablation_scenario_breakdown.csv"
    assert os.path.exists("results/phase12/ablation_statistical_tests.json"), "Missing ablation_statistical_tests.json"
    assert os.path.exists("results/phase12/plots/ablation_family_comparison.png"), "Missing ablation_family_comparison.png"
    assert os.path.exists("results/phase12/plots/ablation_scenarios.png"), "Missing ablation_scenarios.png"
    assert os.path.exists("results/phase12/plots/bullwhip_comparison.png"), "Missing bullwhip_comparison.png"
    
    with open("results/phase12/ablation_statistical_tests.json", "r") as fp:
        data = json.load(fp)
    assert "Family_A_Topology" in data
    assert "Family_B_Forecasting" in data
    assert "Family_C_Communication" in data
    assert "Family_D_Centralization" in data
