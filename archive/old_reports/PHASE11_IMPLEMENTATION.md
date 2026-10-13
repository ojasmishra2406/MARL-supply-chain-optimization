# Phase 11: Combined Demand Forecasting + GNN-MAPPO

## Overview
Phase 11 extends the project by combining the independent forecasting engine (Phase 9) and the graph-based MARL representation (Phase 10) into a unified, synergistic architecture. The implementation was carefully engineered through strict composition to avoid modifying or invalidating the existing Phase 0-10 baselines.

## Architecture

### 1. Data Flow & Composition
The combined model dynamically wraps the standard environment:
`Simulator -> DemandForecastWrapper -> GNN_MAPPO`

1. **Simulator**: Generates standard supply chain observations.
2. **Forecaster**: The `MovingAverageForecaster` (or `XGBoostForecaster`) intercepts the raw observations, extracts historical demand, and predicts a future horizon `H`.
3. **Wrapper**: The `DemandForecastWrapper` concatenates the `H`-step forecast onto the local 5-dimensional observation, scaling the `obs_dim` to `5 + H`.
4. **GNN-MAPPO**: The Phase 10 agent dynamically adapts its `node_features` to match the enhanced observation dimension, naturally propagating the forecast through both the local Actor and the global Centralized GNN Critic.

### 2. Observation Structure
By default, the local observation dimension expands from 5 to 7:
- `[0:5]`: `[inventory, backlog, pipeline_inventory, last_demand, last_order]` (Standard)
- `[5:7]`: `[forecast_t+1, forecast_t+2]` (Horizon = 2)

### 3. GNN Interface
The `SupplyChainGNNEncoder` processes the global state.
- **Node Features:** 7 dimensions per node
- **Graph Topology:** 4-node line graph with bidirectional informational/material flows.
- **Global Input Tensor:** Reshaped dynamically from `28` back to `(4 nodes, 7 features)` before GCN message passing.

### 4. MAPPO Interface (CTDE)
Centralized Training with Decentralized Execution (CTDE) is strictly preserved:
- **Actor:** A local MLP that only sees its own node's 7-dimensional forecast-enhanced vector. It operates decently.
- **Critic:** A graph-based value function (`GNN_MAPPO_Critic`) that processes the full 28-dimensional global state during centralized training to model exact supply chain cascades.

## Configuration & Integration
The integration is handled entirely by `Phase11Trainer` (`rl/phase11_trainer.py`). It dynamically extracts the wrapped environment's new observation dimension and initializes the GNN parameters to match.

## Testing
Comprehensive testing is implemented in `tests/test_phase11_combined.py`:
1. **Forecast Dimension:** Verifies dynamic space expansion.
2. **Graph Input:** Asserts correct node feature alignment.
3. **GNN Forward Pass:** Asserts successful forward passes with the new dimensions.
4. **Gradient Flow:** Verifies that optimizer gradients successfully flow backward through the GNN encoder from the MAPPO loss.
5. **No Data Leakage:** Ensures that the forecaster strictly uses historical data and does not accidentally peek at the simulator's future evaluation ground truth.
6. **Save/Load:** Verifies checkpoint integrity.
7. **Smoke Test:** `run_phase11_smoke.py` acts as an end-to-end CPU integration test.

## Status & Known Limitations
- **IMPLEMENTED:** Yes. The pipeline is fully composed.
- **TESTED:** Yes. Lightweight tests confirm shape matching, gradients, and leakage prevention.
- **EXECUTED / EXPERIMENTALLY VALIDATED:** **PENDING**. Multi-seed training and distribution shift evaluations are blocked waiting for the GPU to become available from Phase 6.

*No performance claims can be made until the full multi-seed evaluation matrix is executed.*
