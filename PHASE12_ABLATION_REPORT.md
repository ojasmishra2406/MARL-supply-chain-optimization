# Phase 12: Comprehensive Ablation & Robustness

Date: 2026-09-23

## 1. Experimental Design

Phase 12 controls for individual isolated variables rather than bulk architectural changes. 
The statistical baseline adheres strictly to Phase 8 methodology (BH-FDR, Welch's t-test over independent training runs).

### Comparison Matrix
1. **GNN Topology Benefit**: `MAPPO` vs `GNN-MAPPO`
2. **Forecast Benefit**: `MAPPO` vs `Forecast + MAPPO`
3. **Synergistic Benefit**: `GNN-MAPPO` vs `Forecast + GNN-MAPPO`
4. **Communication Benefit**: `IPPO` vs `IPPO_COMM` and `MAPPO` vs `MAPPO_COMM` (Already successfully completed in Phase 7/8).
5. **Forecasting Efficacy**: `Naive` vs `MA` vs `ES` vs `XGBoost` (Already successfully evaluated on Test sets in Phase 9).

## 2. Statistical Framework Validation

- Framework requires experimental unit = **Independent Training Run**.
- Duplicate detection script `tools/detect_duplicates.py` prevents overlapping evaluations.
- Missing configuration detections are supported dynamically.

## 3. Status

- Comparison Families 4 & 5 are **COMPLETE** and verified in Phase 8 and Phase 9.
- Comparison Families 1, 2, and 3 are **BLOCKED** pending the multi-day GPU execution of Phase 10 and 11 training matrices.
- The automation scripts required to resume safely (`run_phase12.py` equivalents in `rl/`) have been designed with deterministic seed isolation.
