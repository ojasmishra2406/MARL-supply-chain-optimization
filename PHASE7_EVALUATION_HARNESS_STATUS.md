# Phase 7 Evaluation Harness Status

## Architecture
The Phase 7 Evaluation Harness (`rl/phase7_evaluator.py`) is designed as a standalone evaluation pipeline that runs policies (RL checkpoints or the analytic OUT baseline) against the specific, mathematically isolated evaluation distributions defined in Phase 7A. The primary entry point is `evaluate_scenario`, which correctly delegates execution based on the chosen algorithm without polluting training modules.

## Supported Algorithms
1. **OUT Baseline** (`out`): Loads the `OUTPolicy` mapping inventory position to target parameters.
2. **IPPO** (`ippo`): Fully decentralized actors evaluated in determinist mode.
3. **MAPPO** (`mappo`): Centralized critic (ignored during eval), evaluated symmetrically via shared or distinct parameters.
4. **Phase 6 Implementations**: All communication-enabled architectures evaluate natively through the same code path by parsing the underlying `config` keys.

## Supported Scenarios
* `in_distribution`
* `demand_shift`
* `lead_time_shift`
* `capacity_disruption`
* `demand_spike`
* `combined_shift`

## Evaluator Contract
* Each evaluation consists of precisely 52 steps representing exactly one uninterrupted episode.
* No data leaks occur: Scenario loading happens inside the evaluation harness. 
* Metrics calculated directly align with previous phases:
    1. **Total Cost**
    2. **Fill Rate**
    3. **Bullwhip Ratio**
* No optimization routines run during evaluation (`torch.no_grad()` is strictly enforced).
* Network weights are locked into `.eval()` mode.

## Checkpoint Handling
All RL checkpoints are routed through `rl.checkpoint.load_checkpoint`, which requires a verified `.json` manifest containing the SHA-256 hash. Invalid or corrupted checkpoints automatically fail (proven by `test_invalid_checkpoint`). 

## Reproducibility
* `seed` argument deterministically dictates episode trajectory.
* Evaluator emits a machine-readable JSON-like dictionary containing:
    * Configuration identifiers (Algorithm, Scenario, Seed).
    * `checkpoint_hash`: The verified SHA-256 string for the model weights.
    * `scenario_hash`: The SHA-256 string for the scenario yaml, guaranteeing no silent drift in the evaluation bounds.
    * `reproducibility/git_commit`: Automatic Git SHAs.

## Unresolved Scenario Parameters
As noted in Phase 7A, the evaluation harness successfully loads these schemas but acknowledges several `UNRESOLVED` parameter definitions that require finalized values before actual Phase 7 execution:
1. `demand_spike.yaml`: `spike.magnitude` and `spike.step`. (Currently filled with placeholders `150` and `26`).
2. `capacity_disruption.yaml`: The exact `capacity.units_per_step_per_echelon`. (Currently filled with placeholder `25`).
3. `combined_shift.yaml`: The precise interaction/combinations of these vectors.

## Testing Status
`tests/test_phase7_evaluation.py` proves:
* All 6 scenarios evaluate correctly under the OUT baseline.
* IPPO checkpoints evaluate successfully yielding all metrics and hashes.
* Corrupted metadata rejects the checkpoint completely.
* Trajectories (cost and fill rate) are perfectly reproducible across multiple runs using identical seeds.

**STATUS: Phase 7B Evaluation Harness - Complete.**
*(Awaiting Phase 5 closure and final matrix execution).*
