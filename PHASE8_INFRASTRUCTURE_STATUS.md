# Phase 8A Infrastructure Status

## Statistical Workflow
The Phase 8 Statistical Analysis Pipeline provides a reproducible, end-to-end framework to answer predefined research questions. The workflow enforces:
1. **Shapiro-Wilk Normality Test** to determine distributional properties.
2. **Comparison Selection**: Uses Welch's t-test if parametric assumptions hold; otherwise, falls back to the Mann-Whitney U test.
3. **Effect Size**: Computes Cohen's *d* and enforces a threshold of $|d| \ge 0.5$ for practical significance.
4. **Multiple Comparison Correction**: Applies Benjamini-Hochberg FDR correction exclusively within distinct research families.
5. **Bootstrap Confidence Intervals**: Generates 95% CIs via empirical bootstrapping with 1000 resamples.
6. **Headline Claim Contract**: Generates string representations detailing the test, unadjusted and adjusted p-values, effect size, and sample sizes.

## Data Validation Rules
Before any statistics are computed, `load_and_validate_results` guarantees data integrity:
* Required keys (e.g., `experiment_id`, `cost`, `fill_rate`, `scenario_hash`).
* Values must be within physical bounds (e.g., cost > 0, fill_rate between 0 and 1).
* Duplicate rows are detected and rejected.
* Missing files or invalid JSON formats are gracefully caught and reported in `exclusions.json`.

## Research-Question Families
Configured in `analysis/config.py`:
1. `algorithm_comparison`: IPPO vs MAPPO, MAPPO vs OUT.
2. `scenario_robustness`: MAPPO against each of the 5 shifted distributions.
3. `ablations`: Comparisons against parameter-sharing, communication, centralized critic, and reward weighting configurations.

## Inference Levels
**CRITICAL:** The analysis strictly partitions episode-level stochasticity from training-level robustness. `extract_seed_level_data` aggregates metrics per seed before performing inferential statistics across seeds. This prevents artificially inflating the sample size (e.g., analyzing 50 episodes as $N=50$ when the true training sample size is $N=5$).

## Reproducibility Mechanism
The root script `run_phase8_analysis.py` produces an `analysis_manifest.json` documenting the fixed statistical seed, Python environment version, Git commit hash, timestamp, and row counts.

## Test Coverage
`tests/test_phase8_statistics.py` tests all statistical units (Shapiro-Wilk logic, Welch/Mann-Whitney dispatch, Cohen's *d*, Benjamini-Hochberg ranking logic, 1000-resample bootstrapping, and seed-level aggregation) using fully synthetic data fixtures. 

## Final State
**No actual Phase 7 results were analyzed or fabricated.** The pipeline correctly identifies the missing execution artifacts and gracefully halts.

**STATUS: PHASE 8A — INFRASTRUCTURE READY**
