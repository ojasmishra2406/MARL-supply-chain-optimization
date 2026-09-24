import pytest
import numpy as np

from analysis.statistics import check_normality, compare_groups
from analysis.effect_sizes import cohens_d, is_practically_significant
from analysis.corrections import benjamini_hochberg_correction
from analysis.bootstrap import bootstrap_ci
from analysis.report import extract_seed_level_data

def test_shapiro_wilk():
    # Normal data
    np.random.seed(42)
    data = np.random.normal(loc=0, scale=1, size=100)
    res = check_normality(data)
    assert res["is_normal"] == True
    assert res["n"] == 100
    
    # Non-normal data
    data2 = np.random.exponential(scale=1, size=100)
    res2 = check_normality(data2)
    assert res2["is_normal"] == False

def test_compare_groups():
    np.random.seed(42)
    # Both normal -> Welch's t-test
    g1 = np.random.normal(loc=0, scale=1, size=50)
    g2 = np.random.normal(loc=2, scale=1.5, size=50)
    res = compare_groups(g1, g2)
    assert res["test_name"] == "Welch's t-test"
    assert res["p_value"] < 0.05
    
    # Non-normal -> Mann-Whitney U
    g3 = np.random.exponential(scale=1, size=50)
    g4 = np.random.exponential(scale=2, size=50)
    res2 = compare_groups(g3, g4)
    assert res2["test_name"] == "Mann-Whitney U test"
    assert res2["p_value"] < 0.05

def test_effect_sizes():
    g1 = [1, 2, 3, 4, 5]
    g2 = [6, 7, 8, 9, 10]
    
    d = cohens_d(g1, g2)
    assert d is not None
    # mean1=3, mean2=8. d = -5 / std. std for [1,2,3,4,5] is ~1.58. 
    assert abs(d) >= 0.5
    assert is_practically_significant(d) == True

def test_fdr_correction():
    p_values = [0.01, 0.04, 0.03, 0.001]
    # Sorted: 0.001, 0.01, 0.03, 0.04
    # BH logic:
    # rank 4 (0.04): 0.04 * 4/4 = 0.04 -> min(1, 0.04) = 0.04
    # rank 3 (0.03): 0.03 * 4/3 = 0.04 -> min(0.04, 0.04) = 0.04
    # rank 2 (0.01): 0.01 * 4/2 = 0.02 -> min(0.04, 0.02) = 0.02
    # rank 1 (0.001): 0.001 * 4/1 = 0.004 -> min(0.02, 0.004) = 0.004
    adj = benjamini_hochberg_correction(p_values)
    assert len(adj) == 4
    # Original order: [0.01 (rank 2), 0.04 (rank 4), 0.03 (rank 3), 0.001 (rank 1)]
    # Expected adj: [0.02, 0.04, 0.04, 0.004]
    np.testing.assert_almost_equal(adj, [0.02, 0.04, 0.04, 0.004])

def test_bootstrap_ci():
    np.random.seed(42)
    data = np.random.normal(loc=10, scale=2, size=100)
    
    ci = bootstrap_ci(data, n_resamples=1000)
    assert ci["resamples"] == 1000
    assert ci["point_estimate"] is not None
    assert ci["lower_bound"] < ci["upper_bound"]
    assert 9 < ci["point_estimate"] < 11
    
    with pytest.raises(ValueError):
        bootstrap_ci(data, n_resamples=500)

def test_deterministic_bootstrap():
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    ci1 = bootstrap_ci(data, seed=123)
    ci2 = bootstrap_ci(data, seed=123)
    
    assert ci1["lower_bound"] == ci2["lower_bound"]
    assert ci1["upper_bound"] == ci2["upper_bound"]

def test_episode_vs_seed_distinction():
    """
    Test that extract_seed_level_data correctly aggregates multiple episodes per seed,
    ensuring we do not treat episodes as independent training runs.
    """
    # 2 seeds, 3 episodes each
    synthetic_results = [
        {"algorithm": "ippo", "scenario": "in_distribution", "seed": 0, "cost": 10},
        {"algorithm": "ippo", "scenario": "in_distribution", "seed": 0, "cost": 20},
        {"algorithm": "ippo", "scenario": "in_distribution", "seed": 0, "cost": 30},
        {"algorithm": "ippo", "scenario": "in_distribution", "seed": 1, "cost": 100},
        {"algorithm": "ippo", "scenario": "in_distribution", "seed": 1, "cost": 110},
        {"algorithm": "ippo", "scenario": "in_distribution", "seed": 1, "cost": 120},
    ]
    
    aggs = extract_seed_level_data(synthetic_results, "ippo", "in_distribution", "cost")
    
    # We should have exactly 2 independent observations, not 6.
    assert len(aggs) == 2
    assert aggs[0] == 20.0  # mean of 10, 20, 30
    assert aggs[1] == 110.0 # mean of 100, 110, 120
