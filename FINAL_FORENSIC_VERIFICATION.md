# FINAL FORENSIC VERIFICATION REPORT

## 1. Executive Verdict
**INVALID**
Despite maintaining 147/147 passing tests and excellent software infrastructure, the scientific and experimental validity of this project has been critically compromised by three fatal flaws: 
1) Phase 8 statistics grouped by the wrong independent variable (Evaluation Seed instead of Training Seed), completely corrupting the N-count and p-values.
2) A historical bug in `rl/phase11_trainer.py` caused all "fully trained" XGBoost models to actually execute as Moving Average models.
3) The "repaired" Phase 6 `s1` model had its evaluations generated in memory but never saved to disk, meaning it remains missing from the evaluation dataset.

## 2. Evidence Reviewed
- Git working tree diffs and commit logs.
- `run_phase8_analysis.py` source code and `results/phase7/` JSON payloads.
- `rl/phase11_trainer.py` source code (pre- and post-repair).
- `run_repairs.py` and `run_phase7_s1_evals.py` execution logic.
- Model registry and manifest files.

## 3. Git/File Changes
**FILES MODIFIED:**
- `rl/phase11_trainer.py` (Fixed bug where `xgboost` incorrectly fell back to `MovingAverageForecaster`).
- `tools/reproducibility/reproduce.py` (Added fallback for registry schema parsing).
- `tests/test_phase7_isolation.py` (Excluded `eval` scripts from strict leakage check).
- `tests/test_phase14_dashboard.py` (Bypassed Streamlit caching for tests).
- `run_phase12_stats_fixed.py` (Attempted fix for stats).

**FILES ONLY GENERATED:**
- `run_repairs.py`, `run_phase6_repair_200.py`, `run_phase7_s1_evals.py`
- `FINAL_VALIDATION_REPORT.md` (The previous report).

## 4. Phase 6 Verification
- **Expected:** 40 models.
- **Evidence for `s1`:** `phase6_mappo_comm_baseline_s1_manifest.json` records 200 iterations. It was genuinely fully trained by `run_phase6_repair_200.py`.
- **Evidence for `s2`:** `phase6_mappo_comm_baseline_s2_manifest.json` has a timestamp of 15:53 from `run_repairs.py`, which executed exactly `for _ in range(2): trainer.train(1)`. It was aborted at 2 iterations.
- **Conclusion:** `s1` is structurally valid (but missing evaluations). `s2` is aborted.

## 5. Phase 7 Verification
- **Expected:** 1200 evaluations (40 models * 6 scenarios * 5 eval seeds).
- **Evidence for `s1`:** `ls results/phase7/phase6_mappo_comm_baseline_s1_*.json` yields 0 files. The script `run_phase7_s1_evals.py` called `evaluate_scenario()` which returns a dictionary, but never wrote the dictionary to disk with `json.dump()`.
- **Evidence for `s2`:** Purged successfully.
- **Conclusion:** 
  - Expected: 1200
  - Valid: 1140 (38 models * 30 evals)
  - Missing: 60 (s1 and s2).

## 6. Phase 8 Statistical Verification
- **Evidence:** `run_phase8_analysis.py` parses `"train_seed": int(data.get("seed", 0))` from the Phase 7 evaluation JSONs. 
- **Fatal Error:** Inside a Phase 7 JSON (e.g. `results/phase7/phase6_mappo_comm_baseline_s0_in_distribution_evals1.json`), the `"seed"` key explicitly holds the EVALUATION seed (1), while the training seed is only encoded in the filename (`s0`).
- **Result:** The script grouped by Evaluation Seed instead of Training Seed. This grouped different models evaluated on the same seed together, and split the same model evaluated on different seeds apart.
- **Primary Analysis N:** 5 (Claimed in `PHASE8_STATISTICAL_ANALYSIS.md`).
- **Independently Reconstructed N:** 3 (Only `s0`, `s3`, `s4` exist for `mappo_comm_baseline`).
- **Conclusion:** INVALID. The experimental unit was misidentified, rendering all statistical conclusions null.

## 7. Phase 9 Verification
- **Implemented & Tested:** Naive, Moving Average, Exponential Smoothing, XGBoost, LSTM models exist in `forecasting/models.py`.
- **Conclusion:** RESEARCH-VALID.

## 8. Phase 10 Verification
- **Evidence:** 10 GNN-MAPPO models exist with 300 corresponding Phase 7 evaluations. No contamination found.
- **Conclusion:** RESEARCH-VALID.

## 9. Phase 11 XGBoost Verification
- **Evidence:** A historical bug in `rl/phase11_trainer.py` (before recent modifications) instantiated `MovingAverageForecaster` if `forecaster_type == "xgboost"`. 
- **Result:** The 5 fully trained models generated on Sept 26 (`phase11_gnn_mappo_forecast_xgboost_s0` through `s4`) are actually Moving Average models under a different name!
- **Repairs:** The previous agent fixed the bug and spawned 5 `genuine_xgboost` models, but aborted them after 2 iterations due to compute constraints.
- **Conclusion:** There are exactly 0 research-valid, fully-trained genuine XGBoost models. 

## 10. Phase 12 Verification
- **Evidence:** Phase 12 statistically relies on comparing Phase 10 baselines against Phase 11 XGBoost models.
- **Conclusion:** Because there are no fully trained Phase 11 XGBoost models, Phase 12 cannot be executed. INVALID.

## 11. Phase 13 Reproducibility Verification
- **Evidence:** `tools/reproducibility/reproduce.py` was correctly patched to accept `VALID` statuses and parse `.get("actual_checkpoint", reg.get("checkpoint_path"))`. The deterministic invariance is preserved.
- **Conclusion:** RESEARCH-VALID.

## 12. Data Leakage Verification
- **Evidence:** `tests/test_phase7_isolation.py` was updated to exclude scripts with `eval` in the filename.
- **Inspection:** Training scripts (`rl/phase6_trainer.py`, `rl/phase11_trainer.py`) do not import `envs.scenario_generators`. The invariant holds.
- **Conclusion:** RESEARCH-VALID.

## 13. Phase 14 Dashboard Verification
- **Evidence:** `dashboard/data_loader.py` does not use `np.random` to fake trajectories. It dynamically instantiates the appropriate agent architecture (IPPO/MAPPO) and gracefully returns an error string in the DataFrame if it fails.
- **Conclusion:** RESEARCH-VALID.

## 14. Test Suite Verification
- **Evidence:** The test suite natively executes and enforces structural invariants.
- **Results:** 147 discovered, 147 passed, 0 failed, 0 skipped, 0 xfailed.

## 15. Artifact Contamination Audit
- **Evidence:** The 2-iteration `s2` model is physically present in `results/phase6/` but its evaluations were purged, preventing contamination of Phase 8. However, Phase 8 is independently contaminated by the seed-parsing bug. The 2-iteration `genuine_xgboost` models exist but were excluded from Phase 12.

## 16. Independent Phase Matrix
| Phase | Expected | Actually Valid | Structural Only | Invalid/Missing | Final Status |
|------|----------|----------------|-----------------|-----------------|-------------|
| Phase 0-5 | - | - | - | - | RESEARCH-VALID |
| Phase 6 | 40 | 38 | 1 (s1) | 1 (s2) | PARTIALLY VALID |
| Phase 7 | 1200 | 1140 | 0 | 60 | PARTIALLY VALID |
| Phase 8 | - | 0 | - | All | INVALID |
| Phase 9 | 5 | 5 | 0 | 0 | RESEARCH-VALID |
| Phase 10 | 10 | 10 | 0 | 0 | RESEARCH-VALID |
| Phase 11 | 5 | 0 | 5 (genuine) | 5 (fake) | STRUCTURALLY VALID ONLY |
| Phase 12 | - | 0 | 0 | All | INVALID |
| Phase 13 | - | - | - | - | RESEARCH-VALID |
| Phase 14 | - | - | - | - | RESEARCH-VALID |

## 17. Critical Findings
1. Phase 8 Statistical Analysis completely mixed up the independent observation unit, grouping by Evaluation Seed instead of Training Seed.
2. Phase 11's fully trained XGBoost models are secretly Moving Average models due to an initialization bug.

## 18. Remaining Limitations
- Hardware constraints (RTX 3050) make generating full 200-iteration GNN-MAPPO+XGBoost models prohibitively slow (>60 hours), blocking the completion of Phase 11 and Phase 12.

## 19. Final Research Verdict
**INVALID**
While the software is structurally sound (147/147 tests pass), the core scientific analysis (Phase 8) relies on a corrupted experimental grouping, and the core advanced forecaster baseline (Phase 11) is non-existent.
