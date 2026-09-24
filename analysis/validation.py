import os
import json
import logging

REQUIRED_KEYS = {
    "experiment_id", "scenario", "algorithm", "seed",
    "checkpoint", "checkpoint_hash", "scenario_hash",
    "episode_count", "cost", "fill_rate", "bullwhip",
    "runtime", "reproducibility"
}

VALID_SCENARIOS = {
    "in_distribution", "demand_shift", "lead_time_shift",
    "capacity_disruption", "demand_spike", "combined_shift"
}

def load_and_validate_results(results_dir="results/phase7"):
    """
    Loads and validates Phase 7 results.
    Rejects malformed datasets with explicit errors.
    Returns valid results and any exclusion records.
    """
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Phase 7 results unavailable: Directory '{results_dir}' does not exist.")
        
    valid_results = []
    exclusions = []
    seen_experiments = set()
    
    files = [f for f in os.listdir(results_dir) if f.endswith(".json")]
    if not files:
        raise FileNotFoundError(f"Phase 7 results unavailable: No JSON files found in '{results_dir}'.")
        
    for filename in files:
        filepath = os.path.join(results_dir, filename)
        with open(filepath, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                exclusions.append({"file": filename, "reason": "Invalid JSON format"})
                continue
                
        # Handle if data is a single dict or a list of dicts
        if isinstance(data, dict):
            records = [data]
        elif isinstance(data, list):
            records = data
        else:
            exclusions.append({"file": filename, "reason": "Root element is not a dict or list"})
            continue
            
        for idx, rec in enumerate(records):
            reason = _validate_record(rec, seen_experiments)
            if reason:
                exclusions.append({"file": filename, "index": idx, "reason": reason})
            else:
                seen_experiments.add(rec["experiment_id"])
                valid_results.append(rec)
                
    if not valid_results:
        raise ValueError("No valid results found after validation.")
        
    return valid_results, exclusions

def _validate_record(rec, seen_experiments):
    if not isinstance(rec, dict):
        return "Record is not a dictionary"
        
    missing_keys = REQUIRED_KEYS - set(rec.keys())
    if missing_keys:
        return f"Missing keys: {missing_keys}"
        
    if rec["scenario"] not in VALID_SCENARIOS:
        return f"Invalid scenario: {rec['scenario']}"
        
    if not isinstance(rec["seed"], int) or rec["seed"] not in {0, 1, 2, 3, 4}:
        return f"Invalid seed: {rec['seed']}"
        
    if rec["experiment_id"] in seen_experiments:
        return f"Duplicate experiment_id: {rec['experiment_id']}"
        
    # Check numeric bounds
    try:
        cost = float(rec["cost"])
        fill_rate = float(rec["fill_rate"])
        bullwhip = float(rec["bullwhip"])
    except (ValueError, TypeError):
        return "Metrics must be numeric"
        
    if cost < 0:
        return f"Impossible cost: {cost}"
    if fill_rate < 0.0 or fill_rate > 1.0:
        return f"Impossible fill_rate: {fill_rate}"
    if bullwhip < 0.0:
        return f"Impossible bullwhip: {bullwhip}"
        
    return None
