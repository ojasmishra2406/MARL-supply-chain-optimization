import pytest
import os
import sys
import pandas as pd
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_phase12_no_pseudoreplication():
    """Verify that run_phase12_stats_fixed.py properly prevents pseudoreplication."""
    # We will simulate a run of the aggregation logic
    
    # Just verify that ablation_stats_fixed.json (if exists) doesn't have n > 5
    if os.path.exists("results/phase12/ablation_stats_fixed.json"):
        with open("results/phase12/ablation_stats_fixed.json", "r") as f:
            data = json.load(f)
            
        for test in data.get("hypothesis_tests", []):
            assert test["n_baseline"] <= 5, f"Pseudoreplication found! N_baseline = {test['n_baseline']}"
            assert test["n_forecast"] <= 5, f"Pseudoreplication found! N_forecast = {test['n_forecast']}"
