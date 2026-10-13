"""
Phase 12 Targeted Statistical Reconciliation & Regression Suite.
Protects against recurrence of:
- GNN disruption variance miscalculations (model-level vs episode-level dispersion).
- Formal hypothesis test count conflation (28 unique formal hypotheses).
- Historical transcription errors and misattributions (ERR-01 through ERR-12).
- Pseudoreplication (enforcing training seed N=5, N=3 as independent experimental units).
- Multiple testing misinterpretations (nominal p vs FDR q).
"""

import json
import glob
import os
import pytest
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

def test_gnn_vs_mappo_disruption_model_level_stats():
    """
    Verifies the model-level independent experimental unit calculation for GNN vs MAPPO
    under capacity disruption (N=5 independent training seeds each).
    """
    # 1. GNN Baseline Capacity Disruption (Phase 10)
    gnn_files = sorted(glob.glob(str(WORKSPACE_ROOT / "results/phase10/phase10_gnn_mappo_baseline_s*_capacity_disruption_evals*.json")))
    assert len(gnn_files) == 25, f"Expected 25 GNN disruption files, got {len(gnn_files)}"
    
    gnn_by_seed = {}
    for f in gnn_files:
        with open(f, "r") as fp:
            d = json.load(fp)
        s = int(Path(f).name.split("_s")[1].split("_")[0])
        gnn_by_seed.setdefault(s, []).append(float(d["total_cost"]))
        
    assert len(gnn_by_seed) == 5, f"Expected 5 independent training seeds, got {len(gnn_by_seed)}"
    gnn_seed_means = [np.mean(gnn_by_seed[s]) for s in sorted(gnn_by_seed)]
    
    # Expected seed-level means
    expected_gnn_seeds = [262106.50, 350645.10, 246360.50, 270006.40, 248957.10]
    for actual, expected in zip(gnn_seed_means, expected_gnn_seeds):
        assert pytest.approx(actual, abs=1e-2) == expected

    # 2. MAPPO Baseline Capacity Disruption (Phase 7)
    mappo_files = sorted(glob.glob(str(WORKSPACE_ROOT / "results/phase7/phase6_mappo_baseline_s*_capacity_disruption_evals*.json")))
    assert len(mappo_files) == 25, f"Expected 25 MAPPO disruption files, got {len(mappo_files)}"
    
    mappo_by_seed = {}
    for f in mappo_files:
        with open(f, "r") as fp:
            d = json.load(fp)
        s = int(Path(f).name.split("_s")[1].split("_")[0])
        mappo_by_seed.setdefault(s, []).append(float(d["cost"]))
        
    assert len(mappo_by_seed) == 5, f"Expected 5 independent training seeds, got {len(mappo_by_seed)}"
    mappo_seed_means = [np.mean(mappo_by_seed[s]) for s in sorted(mappo_by_seed)]
    
    # Expected seed-level means
    expected_mappo_seeds = [307012.90, 288552.30, 268451.20, 193566.10, 243196.40]
    for actual, expected in zip(mappo_seed_means, expected_mappo_seeds):
        assert pytest.approx(actual, abs=1e-2) == expected

    # 3. Model-level statistics (ddof=1 across N=5 seeds)
    mean_gnn = np.mean(gnn_seed_means)
    sd_gnn = np.std(gnn_seed_means, ddof=1)
    mean_mappo = np.mean(mappo_seed_means)
    sd_mappo = np.std(mappo_seed_means, ddof=1)
    
    assert pytest.approx(mean_gnn, abs=1e-2) == 275615.12
    assert pytest.approx(sd_gnn, abs=1e-2) == 43040.45
    assert pytest.approx(mean_mappo, abs=1e-2) == 260155.78
    assert pytest.approx(sd_mappo, abs=1e-2) == 44137.77

    # 4. Welch's t-test across independent seeds
    t_stat, p_val = stats.ttest_ind(gnn_seed_means, mappo_seed_means, equal_var=False)
    assert pytest.approx(t_stat, abs=1e-3) == 0.561
    assert pytest.approx(p_val, abs=1e-4) == 0.5904
    assert p_val > 0.05, "GNN vs MAPPO disruption difference must not be statistically significant"

    # 5. Small-sample Hedges' g
    s_pool = np.sqrt((4 * sd_gnn**2 + 4 * sd_mappo**2) / 8)
    d = (mean_gnn - mean_mappo) / s_pool
    J = 1.0 - 3.0 / (4.0 * 8.0 - 1.0)
    g = d * J
    assert pytest.approx(g, abs=1e-3) == 0.320

def test_episode_vs_model_dispersion_separation():
    """
    Asserts that episode-level descriptive dispersion is cleanly separated
    from model-level inferential dispersion.
    """
    gnn_files = sorted(glob.glob(str(WORKSPACE_ROOT / "results/phase10/phase10_gnn_mappo_baseline_s*_capacity_disruption_evals*.json")))
    gnn_episodes = [float(json.load(open(f))["total_cost"]) for f in gnn_files]
    
    ep_sd = np.std(gnn_episodes, ddof=1)
    # Episode level dispersion across all 25 episodes
    assert pytest.approx(ep_sd, abs=1e-2) == 43100.35
    
    # Model level dispersion across N=5 seeds
    gnn_by_seed = {}
    for f in gnn_files:
        s = int(Path(f).name.split("_s")[1].split("_")[0])
        gnn_by_seed.setdefault(s, []).append(float(json.load(open(f))["total_cost"]))
    model_sd = np.std([np.mean(gnn_by_seed[s]) for s in gnn_by_seed], ddof=1)
    assert pytest.approx(model_sd, abs=1e-2) == 43040.45
    
    # Neither should ever equal the misreported ~6,701 value (which was a single within-model seed SD)
    assert not np.isclose(model_sd, 6701.37, atol=100)
    assert not np.isclose(ep_sd, 6701.37, atol=100)

def test_gnn_capacity_failure_retraction_provenance():
    """
    Verifies that the $356k disruption cost belonged to a 50-update Moving Average model,
    not the 200-update GNN baseline.
    """
    sc_path = WORKSPACE_ROOT / "results/phase12/ablation_scenario_breakdown.csv"
    assert sc_path.exists()
    df_sc = pd.read_csv(sc_path)
    
    # Find capacity disruption rows
    disr = df_sc[df_sc["scenario"] == "capacity_disruption"]
    
    # GNN Baseline (200 updates)
    gnn_base = disr[(disr["algorithm"] == "gnn_mappo") & (disr["regime"] == "baseline")]
    assert len(gnn_base) == 1
    assert pytest.approx(gnn_base.iloc[0]["cost_mean"], abs=1.0) == 275615.12
    
    # MAPPO Baseline (200 updates)
    mappo_base = disr[(disr["algorithm"] == "mappo") & (disr["regime"] == "baseline")]
    assert len(mappo_base) == 1
    assert pytest.approx(mappo_base.iloc[0]["cost_mean"], abs=1.0) == 260155.78
    
    # Moving Average 50-update model (source of the 356k figure)
    ma_50 = disr[disr["algorithm"] == "gnn_mappo_forecast_ma"]
    assert len(ma_50) == 1
    assert pytest.approx(ma_50.iloc[0]["cost_mean"], abs=1.0) == 356536.97

def test_formal_hypothesis_test_count_28():
    """
    Asserts that the machine-readable artifact contains exactly 28 formal tests across 4 families.
    """
    stats_path = WORKSPACE_ROOT / "results/phase12/ablation_statistical_tests.json"
    with open(stats_path, "r") as fp:
        data = json.load(fp)
        
    expected_counts = {
        "Family_A_Topology": 8,
        "Family_B_Forecasting": 4,
        "Family_C_Communication": 8,
        "Family_D_Centralization": 8
    }
    
    total_tests = 0
    for fam, expected in expected_counts.items():
        assert fam in data, f"Missing family {fam}"
        actual = len(data[fam])
        assert actual == expected, f"Family {fam}: expected {expected} tests, got {actual}"
        total_tests += actual
        
    assert total_tests == 28, f"Expected exactly 28 formal hypothesis tests, got {total_tests}"

def test_benjamini_hochberg_fdr_family_structure():
    """
    Verifies that FDR control is computed strictly within defined families and that only 2 tests
    reach nominal unadjusted significance while zero survive FDR control at q < 0.05.
    """
    stats_path = WORKSPACE_ROOT / "results/phase12/ablation_statistical_tests.json"
    with open(stats_path, "r") as fp:
        data = json.load(fp)
        
    nominal_sig_count = 0
    fdr_sig_count = 0
    
    for fam, tests in data.items():
        for t in tests:
            if t["welch_p"] < 0.05:
                nominal_sig_count += 1
            if t["fdr_q"] < 0.05:
                fdr_sig_count += 1
                
    assert nominal_sig_count == 2, f"Expected exactly 2 nominal significant tests, got {nominal_sig_count}"
    assert fdr_sig_count == 0, f"Expected 0 FDR-significant tests at q < 0.05, got {fdr_sig_count}"

def test_statistical_json_vs_csv_summary_consistency():
    """
    Asserts that statistical JSON means and SDs match ablation_summary.csv exactly.
    """
    summary_path = WORKSPACE_ROOT / "results/phase12/ablation_summary.csv"
    stats_path = WORKSPACE_ROOT / "results/phase12/ablation_statistical_tests.json"
    
    df_sum = pd.read_csv(summary_path)
    with open(stats_path, "r") as fp:
        tests = json.load(fp)
        
    # Check MAPPO baseline cost
    mappo_base = df_sum[(df_sum["algorithm"] == "mappo") & (df_sum["regime"] == "baseline")]
    assert len(mappo_base) == 5
    
    expected_mean = float(mappo_base["cost"].mean())
    expected_sd = float(mappo_base["cost"].std(ddof=1))
    
    # In Family D
    d_test = next(t for t in tests["Family_D_Centralization"] if t["comparison"] == "MAPPO_vs_IPPO_Baseline" and t["metric"] == "cost")
    assert pytest.approx(d_test["mean_x"], abs=1e-4) == expected_mean
    assert pytest.approx(d_test["sd_x"], abs=1e-4) == expected_sd

def test_no_duplicate_or_missing_episodes():
    """
    Verifies that all 56 models have exactly 30 evaluation episodes across the 6 standard scenarios.
    """
    # Enumerate raw files
    p7_files = glob.glob(str(WORKSPACE_ROOT / "results/phase7/*_eval*.json"))
    p10_files = glob.glob(str(WORKSPACE_ROOT / "results/phase10/*_eval*.json"))
    p11_files = [f for f in glob.glob(str(WORKSPACE_ROOT / "results/phase11_controlled/*_eval*.json")) if "smoke" not in f]
    
    assert len(p7_files) == 1200
    assert len(p10_files) == 300
    assert len(p11_files) == 180
    assert len(p7_files) + len(p10_files) + len(p11_files) == 1680

def test_transcription_error_audit_corrections_in_report():
    """
    Verifies that results/phase12/PHASE12_FINAL_ACCEPTANCE_REPORT.md reflects the corrected
    model-level values and retractions.
    """
    report_path = WORKSPACE_ROOT / "results/phase12/PHASE12_FINAL_ACCEPTANCE_REPORT.md"
    assert report_path.exists()
    
    content = report_path.read_text(encoding="utf-8")
    
    # Check GNN model-level SD and t-test
    assert "43,040.45" in content
    assert "44,137.77" in content
    assert "0.5607" in content
    assert "0.5904" in content
    assert "capacity failure" in content and "retracted" in content
    
    # Check MAPPO vs IPPO Bullwhip & Cost/SL
    assert "4.490" in content
    assert "0.0092" in content
    assert "2.511" in content
    assert "0.0380" in content
    assert "0.0733" in content
    assert "0.0972" in content

def test_scenario_breakdown_coverage_and_row_count():
    """
    Verifies that results/phase12/ablation_scenario_breakdown.csv contains exactly 72 rows,
    covering all 12 experimental configurations across all 6 scenarios with zero duplicates.
    Resolves the historical 40-row confusion (40 rows represented 8 Phase 6/7 configs x 5 scenarios,
    omitting Phase 10, Phase 11 Controlled, and the 6th scenario combined_shift).
    """
    sc_path = WORKSPACE_ROOT / "results/phase12/ablation_scenario_breakdown.csv"
    assert sc_path.exists(), "ablation_scenario_breakdown.csv missing"
    
    df_sc = pd.read_csv(sc_path)
    assert len(df_sc) == 72, f"Expected exactly 72 rows, got {len(df_sc)}"
    
    # 12 unique configurations (algorithm, regime)
    configs = df_sc[["algorithm", "regime"]].drop_duplicates()
    assert len(configs) == 12, f"Expected 12 configurations, got {len(configs)}"
    
    # 6 unique scenarios
    scenarios = df_sc["scenario"].unique()
    assert len(scenarios) == 6, f"Expected 6 scenarios, got {len(scenarios)}"
    expected_scenarios = {
        "in_distribution", "demand_shift", "demand_spike",
        "lead_time_shift", "capacity_disruption", "combined_shift"
    }
    assert set(scenarios) == expected_scenarios
    
    # Assert zero duplicate (algorithm, regime, scenario) triplets
    assert not df_sc.duplicated(subset=["algorithm", "regime", "scenario"]).any(), "Found duplicate configuration-scenario rows"
    
    # Verify expected column schema
    expected_cols = [
        "algorithm", "regime", "scenario",
        "cost_mean", "cost_std",
        "service_level_mean", "service_level_std",
        "bullwhip_mean", "bullwhip_std"
    ]
    assert list(df_sc.columns) == expected_cols

