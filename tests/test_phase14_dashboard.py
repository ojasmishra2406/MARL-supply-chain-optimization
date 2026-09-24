import os
import sys
import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dashboard.data_loader import load_registry, load_evaluations

def test_load_registry_safely():
    # Test handling of missing file
    registry = load_registry("dummy_missing_path.json")
    assert isinstance(registry, dict)
    assert len(registry) == 0

def test_load_evaluations_safely():
    # Test handling of missing directory
    df = load_evaluations("dummy_missing_dir")
    assert isinstance(df, pd.DataFrame)
    assert df.empty

def test_load_real_evaluations():
    # Test loading real data without crashing (graceful missing data handling)
    df = load_evaluations("results/phase7")
    assert isinstance(df, pd.DataFrame)
    # The dataframe might be empty if running in an environment without the artifacts,
    # but it must not crash.
    if not df.empty:
        required_cols = {"experiment_id", "model_id", "scenario", "algorithm", "seed", "cost"}
        assert required_cols.issubset(df.columns), "Missing required columns in evaluation DataFrame"
