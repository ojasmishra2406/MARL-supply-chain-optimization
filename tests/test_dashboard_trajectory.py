import pytest
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashboard.data_loader import generate_trajectory

from unittest.mock import patch

@patch("streamlit.spinner")
@patch("streamlit.error")
def test_dashboard_no_mocked_random_trajectory(mock_error, mock_spinner):
    # Ensure generate_trajectory does not produce random mock values and handles loading gracefully.
    registry = {
        "fake_model": {
            "status": "VALID",
            "metadata": {"algorithm": "ippo", "config_path": "configs/phase1_simulator.yaml"},
            "checkpoint_path": "nonexistent.pt"
        }
    }
    
    # Because checkpoint nonexistent.pt doesn't exist, it should return an error dataframe, not a mocked trajectory
    df = generate_trajectory("fake_model", "in_distribution", 0, registry)
    
    assert "error" in df.columns, "generate_trajectory should fail gracefully and not mock data"
    assert df.iloc[0]["error"] == "Real trajectory data unavailable."
