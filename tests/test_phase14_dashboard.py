import os
import sys
import pandas as pd
import pytest
from pathlib import Path

WORKSPACE_ROOT = str(Path(__file__).resolve().parent.parent)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from dashboard.data_loader import (
    load_registry,
    load_registry_dataframe,
    load_evaluations,
    load_phase12_ablation_summary,
    load_phase12_scenario_breakdown,
    load_phase12_statistical_tests,
    load_forecasting_metrics,
    load_reproducibility_data
)

def test_load_registry_safely():
    # Test handling of missing file
    registry = load_registry("dummy_missing_path.json")
    assert isinstance(registry, dict)
    assert len(registry) == 0

def test_load_registry_dataframe_real():
    """Verify loading real registry as a structured DataFrame with orthogonal statuses."""
    df_reg = load_registry_dataframe.__wrapped__()
    assert isinstance(df_reg, pd.DataFrame)
    assert len(df_reg) == 84, f"Expected 84 models in registry, found {len(df_reg)}"
    
    # Assert orthogonal status columns
    assert "integrity_status" in df_reg.columns
    assert "evaluation_status" in df_reg.columns
    assert "research_status" in df_reg.columns
    assert "status" in df_reg.columns
    
    # Check Phase 4 is HISTORICAL_SUPERSEDED
    p4 = df_reg[df_reg["model_id"] == "phase4_final_random_init"]
    assert len(p4) == 1
    assert p4.iloc[0]["research_status"] == "HISTORICAL_SUPERSEDED"
    assert p4.iloc[0]["integrity_status"] == "ARTIFACT_VALID"

    # Check Phase 12 diagnostic is TRAINING_INCOMPLETE
    p12 = df_reg[df_reg["model_id"] == "phase12_ablation_mappo_comm_2iters"]
    assert len(p12) == 1
    assert p12.iloc[0]["research_status"] == "TRAINING_INCOMPLETE"

def test_load_evaluations_safely():
    # Test handling of missing directory using __wrapped__ to bypass streamlit context
    df = load_evaluations.__wrapped__()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_load_real_evaluations():
    # Test loading real data without crashing (graceful missing data handling)
    df = load_evaluations.__wrapped__()
    assert isinstance(df, pd.DataFrame)
    if not df.empty:
        required_cols = {"experiment_id", "scenario", "algorithm"}
        assert required_cols.issubset(df.columns), "Missing required columns in evaluation DataFrame"
        assert len(df) >= 1680, f"Expected at least 1680 evaluation episodes, got {len(df)}"

def test_load_phase12_statistical_tests():
    """Verify that dashboard loads authoritative Phase 12 stats matching ground truth."""
    df_stats, raw_json = load_phase12_statistical_tests.__wrapped__()
    assert isinstance(df_stats, pd.DataFrame)
    assert len(df_stats) == 28, f"Expected 28 statistical tests, got {len(df_stats)}"
    
    # Assert exact model-level findings
    bw = df_stats[(df_stats["Comparison"] == "MAPPO_vs_IPPO_Baseline") & (df_stats["Metric"] == "bullwhip")]
    assert len(bw) == 1
    assert pytest.approx(bw.iloc[0]["Raw p-value"], abs=1e-4) == 0.0092
    assert pytest.approx(bw.iloc[0]["Hedges g"], abs=1e-3) == 2.565
    assert pytest.approx(bw.iloc[0]["FDR q-value"], abs=1e-3) == 0.073
    assert bw.iloc[0]["Nominal Sig (p < 0.05)"] == True
    assert bw.iloc[0]["FDR Sig (q < 0.05)"] == False

    csl = df_stats[(df_stats["Comparison"] == "MAPPO_vs_IPPO_Baseline") & (df_stats["Metric"] == "cost_per_sl")]
    assert len(csl) == 1
    assert pytest.approx(csl.iloc[0]["Raw p-value"], abs=1e-4) == 0.0380
    assert pytest.approx(csl.iloc[0]["Hedges g"], abs=1e-3) == 1.435
    assert pytest.approx(csl.iloc[0]["FDR q-value"], abs=1e-3) == 0.097
    assert csl.iloc[0]["Nominal Sig (p < 0.05)"] == True
    assert csl.iloc[0]["FDR Sig (q < 0.05)"] == False

def test_load_forecasting_metrics():
    """Verify loading Phase 9 forecasting benchmark table."""
    df_fc = load_forecasting_metrics.__wrapped__()
    assert isinstance(df_fc, pd.DataFrame)
    assert not df_fc.empty
    assert "MAE" in df_fc.columns
    assert "Bias" in df_fc.columns
    assert "Model" in df_fc.columns
    assert set(df_fc["Model"].unique()) == {"Naive", "MovingAverage", "ExpSmoothing", "XGBoost"}

def test_load_reproducibility_data():
    """Verify loading Phase 13 reproducibility audit and integrity summaries."""
    rep_data, int_data = load_reproducibility_data.__wrapped__()
    assert isinstance(rep_data, dict)
    assert isinstance(int_data, dict)
    assert "evaluation_replay" in rep_data
    assert rep_data["evaluation_replay"]["cost_match"] is True
    assert rep_data["evaluation_replay"]["bullwhip_match"] is True
    assert "environment" in rep_data
    assert "registry_integrity" in rep_data
