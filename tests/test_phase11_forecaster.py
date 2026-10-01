import pytest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.phase11_trainer import Phase11Trainer
from forecasting.models import XGBoostForecaster, MovingAverageForecaster

def test_phase11_forecaster_instantiation():
    """Regression test ensuring explicit xgboost mapping works and fallback applies correctly."""
    
    # Test XGBoost explicit instantiation
    trainer_xgb = Phase11Trainer("configs/phase4_ippo.yaml", forecast_horizon=2, forecaster_type="xgboost")
    assert isinstance(trainer_xgb.forecaster, XGBoostForecaster), "Trainer did not instantiate XGBoostForecaster for 'xgboost'"
    
    # Test MA fallback
    trainer_ma = Phase11Trainer("configs/phase4_ippo.yaml", forecast_horizon=2, forecaster_type="moving_average")
    assert isinstance(trainer_ma.forecaster, MovingAverageForecaster), "Trainer did not instantiate MovingAverageForecaster for 'moving_average'"
