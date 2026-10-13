import os
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

def test_phase12_raw_data_count_and_parity():
    """Verify that all 1,680 raw evaluation records and 56 trained models are loaded."""
    summary_path = WORKSPACE_ROOT / "results" / "phase12" / "ablation_summary.csv"
    assert summary_path.exists(), "results/phase12/ablation_summary.csv missing"
    
    df_summary = pd.read_csv(summary_path)
    assert len(df_summary) == 56, f"Expected 56 models, got {len(df_summary)}"
    
    # Check algorithms present
    expected_algos = {"ippo", "ippo_comm", "mappo", "mappo_comm", "gnn_mappo", "gnn_mappo_forecast_ma", "gnn_mappo_forecast_xgboost"}
    assert set(df_summary["algorithm"].unique()) == expected_algos

def test_phase12_statistical_tests_json_consistency():
    """Verify exact numerical consistency in results/phase12/ablation_statistical_tests.json."""
    stats_path = WORKSPACE_ROOT / "results" / "phase12" / "ablation_statistical_tests.json"
    assert stats_path.exists(), "results/phase12/ablation_statistical_tests.json missing"
    
    with open(stats_path, "r") as f:
        data = json.load(f)
        
    assert "Family_A_Topology" in data
    assert "Family_B_Forecasting" in data
    assert "Family_C_Communication" in data
    assert "Family_D_Centralization" in data
    
    # Check Family D Baseline Bullwhip
    fam_d = data["Family_D_Centralization"]
    bw_test = next(t for t in fam_d if t["comparison"] == "MAPPO_vs_IPPO_Baseline" and t["metric"] == "bullwhip")
    
    assert bw_test["nx"] == 5
    assert bw_test["ny"] == 5
    assert pytest.approx(bw_test["mean_x"], abs=1e-3) == 6.333
    assert pytest.approx(bw_test["sd_x"], abs=1e-3) == 2.373
    assert pytest.approx(bw_test["mean_y"], abs=1e-3) == 1.476
    assert pytest.approx(bw_test["sd_y"], abs=1e-3) == 0.470
    assert pytest.approx(bw_test["welch_t"], abs=1e-3) == 4.490
    assert pytest.approx(bw_test["df"], abs=1e-2) == 4.31
    assert pytest.approx(bw_test["welch_p"], abs=1e-4) == 0.0092
    assert pytest.approx(bw_test["hedges_g"], abs=1e-3) == 2.565
    assert pytest.approx(bw_test["mwu_u"], abs=1e-1) == 25.0
    assert pytest.approx(bw_test["mwu_p"], abs=1e-4) == 0.0079
    # Multiple testing correction
    assert pytest.approx(bw_test["fdr_q"], abs=1e-3) == 0.073
    assert bw_test["significant_fdr"] is False

    # Check Family D Baseline Cost/Service Level
    csl_test = next(t for t in fam_d if t["comparison"] == "MAPPO_vs_IPPO_Baseline" and t["metric"] == "cost_per_sl")
    assert csl_test["nx"] == 5
    assert csl_test["ny"] == 5
    assert pytest.approx(csl_test["mean_x"], abs=1.0) == 106592.0
    assert pytest.approx(csl_test["sd_x"], abs=1.0) == 7798.0
    assert pytest.approx(csl_test["mean_y"], abs=1.0) == 95486.0
    assert pytest.approx(csl_test["sd_y"], abs=1.0) == 6081.0
    assert pytest.approx(csl_test["welch_t"], abs=1e-3) == 2.511
    assert pytest.approx(csl_test["df"], abs=1e-2) == 7.55
    assert pytest.approx(csl_test["welch_p"], abs=1e-4) == 0.0380
    assert pytest.approx(csl_test["hedges_g"], abs=1e-3) == 1.435
    assert pytest.approx(csl_test["fdr_q"], abs=1e-3) == 0.097
    assert csl_test["significant_fdr"] is False

def test_phase12_non_significant_comparisons():
    """Verify that non-significant comparisons (GNN topology, communication) have p > 0.05."""
    stats_path = WORKSPACE_ROOT / "results" / "phase12" / "ablation_statistical_tests.json"
    with open(stats_path, "r") as f:
        data = json.load(f)
        
    # GNN Baseline Cost
    gnn_cost = next(t for t in data["Family_A_Topology"] if t["comparison"] == "GNN_vs_MLP_Baseline" and t["metric"] == "cost")
    assert gnn_cost["welch_p"] > 0.30, f"Expected non-significant p > 0.30, got {gnn_cost['welch_p']}"
    
    # Comm Cost (IPPO)
    comm_cost = next(t for t in data["Family_C_Communication"] if t["comparison"] == "IPPO_COMM_vs_IPPO" and t["metric"] == "cost")
    assert comm_cost["welch_p"] > 0.30, f"Expected non-significant p > 0.30, got {comm_cost['welch_p']}"

def test_phase12_plots_exist():
    """Verify that Phase 12 plots exist and have non-zero size."""
    plot_dir = WORKSPACE_ROOT / "results" / "phase12" / "plots"
    for name in ["ablation_family_comparison.png", "ablation_scenarios.png", "bullwhip_comparison.png"]:
        p = plot_dir / name
        assert p.exists(), f"Plot {name} missing"
        assert p.stat().st_size > 50000, f"Plot {name} appears empty or truncated"
