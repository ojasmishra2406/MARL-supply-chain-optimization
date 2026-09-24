# Phase 11: Forecasting + GNN-MAPPO

Date: 2026-09-23

## 1. Architecture Design

Combines the `DemandForecastWrapper` (Phase 9) and the `GNN-MAPPO` architecture (Phase 10).

### Observation Design
- **Original Node Dimension**: 5
- **Forecast Dimension**: 2 (Configurable horizon, tested on `ma` forecaster)
- **Enhanced Node Dimension**: 7
- **Global Critic Dimension**: 28 (4 agents * 7 features)

The forecast features are strictly additive, appended to the end of the original vector.

### Data Leakage Prevention
Explicit verification in `tests/test_phase11_combined.py` confirms that:
- The Forecaster strictly relies on `demand_history`.
- Future environment demand is never leaked into the observation payload at step `t`.
- No evaluation metrics depend on future scenario peeks.

## 2. Experimental Plan (Configured, GPU Pending)

- **Comparisons**: `MAPPO` vs `GNN-MAPPO` vs `Forecasting + GNN-MAPPO`
- **Goal**: Isolate performance benefit of GNN topology vs Time-series Forecasting explicitly.

## 3. Implementation Status
**IMPLEMENTED & TESTED**
- Wrapper dynamically intercepts observations and concats forecast output.
- All integration tests passed cleanly.

**EXPERIMENTAL VALIDATION**: `BLOCKED` (by required GPU execution time; reserved for future deployment)
