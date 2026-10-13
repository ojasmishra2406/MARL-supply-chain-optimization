"""
Phase 12 Independent Research Verification Test Suite
"""

import os
import json
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from results.phase12_verification.independent_statistical_recomputation import (
    load_all_raw_data,
    run_independent_welch,
    apply_fdr,
    compute_power_welch
)
from rl.checkpoint import load_checkpoint

def test_full_episode_dataset_integrity():
    df = load_all_raw_data()
    assert len(df) == 1680, f"Expected 1,680 episodes, got {len(df)}"
    
    # Check Phase breakdown
    p7_count = len(df[df["phase"] == "phase6_7"])
    p10_count = len(df[df["phase"] == "phase10"])
    p11_count = len(df[df["phase"] == "phase11_controlled"])
    
    assert p7_count == 1200, f"Expected 1,200 Phase 7 episodes, got {p7_count}"
    assert p10_count == 300, f"Expected 300 Phase 10 episodes, got {p10_count}"
    assert p11_count == 180, f"Expected 180 Phase 11 Controlled episodes, got {p11_count}"
    
    assert not df["cost"].isna().any()
    assert not df["service_level"].isna().any()
    assert not df["bullwhip"].isna().any()

def test_independent_model_aggregation():
    df = load_all_raw_data()
    run_df = df.groupby(["phase", "algorithm", "architecture", "communication", "forecasting", "regime", "training_seed"]).agg({
        "cost": "mean",
        "service_level": "mean",
        "bullwhip": "mean"
    }).reset_index()
    
    assert len(run_df) == 56, f"Expected 56 independent models, got {len(run_df)}"
    
    # Check model counts by phase
    assert len(run_df[run_df["phase"] == "phase6_7"]) == 40
    assert len(run_df[run_df["phase"] == "phase10"]) == 10
    assert len(run_df[run_df["phase"] == "phase11_controlled"]) == 6

def test_canonical_checkpoint_loader_sample():
    # Sample checkpoints from each phase
    samples = [
        ("results/phase6/phase6_ippo_baseline_s0.pt", "results/phase6/phase6_ippo_baseline_s0_manifest.json"),
        ("results/phase6/phase6_mappo_comm_high_variance_s3.pt", "results/phase6/phase6_mappo_comm_high_variance_s3_manifest.json"),
        ("results/phase10/phase10_gnn_mappo_baseline_s2.pt", "results/phase10/phase10_gnn_mappo_baseline_s2_manifest.json"),
        ("results/phase11_controlled/phase11_controlled_xgboost_s2.pt", "results/phase11_controlled/phase11_controlled_xgboost_s2_manifest.json")
    ]
    for ckpt, mf in samples:
        assert os.path.exists(ckpt), f"Missing {ckpt}"
        assert os.path.exists(mf), f"Missing {mf}"
        state = load_checkpoint(ckpt, mf)
        assert "retailer" in state

def test_statistical_calculations_match_json():
    # Load independently saved verification json and phase 12 json
    with open("results/phase12/ablation_statistical_tests.json", "r") as f:
        p12_data = json.load(f)
    with open("results/phase12_verification/independent_statistical_recomputation.json", "r") as f:
        ind_data = json.load(f)
        
    for fam in p12_data:
        p12_tests = p12_data[fam]
        ind_tests = ind_data[fam]
        assert len(p12_tests) == len(ind_tests)
        
        for t1, t2 in zip(p12_tests, ind_tests):
            assert t1["comparison"] == t2["comparison"]
            assert t1["metric"] == t2["metric"]
            assert np.isclose(t1["mean_x"], t2["mean_x"], rtol=1e-4)
            assert np.isclose(t1["mean_y"], t2["mean_y"], rtol=1e-4)
            assert np.isclose(t1["welch_t"], t2["welch_t"], rtol=1e-4)
            assert np.isclose(t1["welch_p"], t2["welch_p"], rtol=1e-4)
            assert np.isclose(t1["hedges_g"], t2["hedges_g"], rtol=1e-4)
            assert np.isclose(t1["fdr_q"], t2["fdr_q"], rtol=1e-4)

def test_power_calculation():
    # Power for very large effect size should be high
    pwr_high = compute_power_welch(5, 5, 1.0, 1.0, 3.0)
    assert pwr_high > 0.90
    
    # Power for tiny effect size should be low
    pwr_low = compute_power_welch(5, 5, 10.0, 10.0, 0.1)
    assert pwr_low < 0.10
