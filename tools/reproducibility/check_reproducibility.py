"""
Phase 13 Automated Reproducibility & Integrity Checker
Lightweight, robust CLI utility for CI/CD and pre-flight research verification.

Audits:
1. Python environment and core dependency versions
2. Registry schema and checkpoint/manifest disk integrity (dual-hash verification)
3. Seed propagation and environment reset determinism
4. Deterministic evaluation replay against recorded ground-truth evaluation JSON
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Ensure MARL root is in path
WORKSPACE_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import numpy as np
import torch

from tools.registry import ArtifactRegistry, REGISTRY_DEFAULT_PATH
from rl.checkpoint import load_checkpoint, compute_file_sha256, _hash_state_dict
from envs.scenario_generators import create_scenario_env
from rl.phase7_evaluator import evaluate_scenario

def check_environment() -> Tuple[bool, Dict[str, Any]]:
    """Checks Python version, PyTorch, and core scientific stack."""
    results = {}
    passed = True
    
    # Python
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_pass = sys.version_info.major == 3 and sys.version_info.minor >= 10
    results["python"] = {"version": py_ver, "passed": py_pass}
    if not py_pass:
        passed = False
        
    # PyTorch
    torch_ver = torch.__version__
    torch_pass = int(torch_ver.split(".")[0]) >= 2
    results["torch"] = {
        "version": torch_ver,
        "cuda_available": torch.cuda.is_available(),
        "passed": torch_pass
    }
    if not torch_pass:
        passed = False
        
    # Packages
    packages = ["numpy", "pandas", "scipy", "statsmodels", "xgboost", "yaml"]
    for pkg in packages:
        try:
            mod = __import__(pkg)
            v = getattr(mod, "__version__", "installed")
            results[pkg] = {"version": v, "passed": True}
        except ImportError:
            results[pkg] = {"version": "missing", "passed": False}
            passed = False
            
    return passed, results

def check_registry_integrity(registry_path: str = REGISTRY_DEFAULT_PATH,
                             sample_limit: Optional[int] = None) -> Tuple[bool, Dict[str, Any]]:
    """Audits registry schema, manifests, and checkpoint hashes on disk."""
    if not os.path.exists(registry_path):
        return False, {"error": f"Registry file not found at {registry_path}"}
        
    registry = ArtifactRegistry(registry_path)
    summary = registry.verify_all()
    duplicates = registry.detect_duplicates()
    missing_evals = registry.find_missing_evaluations()
    stale_paths = registry.report_stale_paths()
    
    has_critical_failure = (
        summary["corrupted"] > 0 or 
        len(duplicates["duplicate_checkpoints"]) > 0 or
        len(duplicates["identical_state_dict_hashes"]) > 0 or
        len(stale_paths) > 0
    )
    
    details = {
        "total_models": summary["total_registered"],
        "valid_canonical": summary["valid_canonical"],
        "valid_historical_physical": summary["valid_historical_physical"],
        "valid_offline": summary["valid_offline"],
        "corrupted": summary["corrupted"],
        "missing_paths": summary["missing"],
        "missing_evals_count": len(missing_evals),
        "stale_paths_count": len(stale_paths),
        "duplicate_checkpoints_count": len(duplicates["duplicate_checkpoints"]),
        "passed": not has_critical_failure
    }
    return not has_critical_failure, details

def check_seed_propagation() -> Tuple[bool, Dict[str, Any]]:
    """Verifies environment seed isolation and determinism."""
    try:
        env = create_scenario_env("in_distribution", canonical_config_path="configs/phase1_simulator.yaml", comm_enabled=False)
        actions = {a: 10 for a in env.possible_agents}
        
        # 1. Reset with seed 42 (run A) and step 3 steps
        env.reset(seed=42)
        for _ in range(3):
            env.step(actions)
        demands_42_a = list(env.simulator.state.echelons[0].demand_history)
        cost_42_a = float(env.simulator.state.total_cost)
        
        # 2. Reset with seed 42 (run B) and step 3 steps
        env.reset(seed=42)
        for _ in range(3):
            env.step(actions)
        demands_42_b = list(env.simulator.state.echelons[0].demand_history)
        cost_42_b = float(env.simulator.state.total_cost)
        
        # 3. Reset with seed 99 (run C) and step 3 steps
        env.reset(seed=99)
        for _ in range(3):
            env.step(actions)
        demands_99 = list(env.simulator.state.echelons[0].demand_history)
        cost_99 = float(env.simulator.state.total_cost)
        
        deterministic = bool(demands_42_a == demands_42_b and np.isclose(cost_42_a, cost_42_b))
        independent = bool(demands_42_a != demands_99 and not np.isclose(cost_42_a, cost_99))
        
        passed = bool(deterministic and independent)
        return passed, {
            "deterministic_reproducibility": deterministic,
            "seed_entropy_independence": independent,
            "demands_seed42": [float(x) for x in demands_42_a],
            "demands_seed99": [float(x) for x in demands_99],
            "passed": passed
        }
    except Exception as e:
        return False, {"error": str(e), "passed": False}

def replay_deterministic_evaluation(model_id: str = "phase6_ippo_baseline_s0",
                                     scenario: str = "in_distribution",
                                     seed: int = 0) -> Tuple[bool, Dict[str, Any]]:
    """
    Executes a 1-episode evaluation replay of a canonical model and verifies
    exact numeric match against the recorded evaluation JSON.
    """
    ckpt_path = f"results/phase6/{model_id}.pt"
    manifest_path = f"results/phase6/{model_id}_manifest.json"
    record_path = f"results/phase7/{model_id}_{scenario}_evals{seed}.json"
    
    if not os.path.exists(ckpt_path) or not os.path.exists(record_path):
        return False, {"error": f"Missing artifacts for replay: {ckpt_path} or {record_path}", "passed": False}
        
    with open(record_path, "r") as f:
        recorded_data = json.load(f)
    expected_cost = float(recorded_data["cost"])
    expected_fill_rate = float(recorded_data["fill_rate"])
    expected_bullwhip = float(recorded_data["bullwhip"])
    
    # Replay evaluation via canonical loader
    replay_res = evaluate_scenario(
        algorithm="ippo_baseline",
        checkpoint_path=ckpt_path,
        config_path="configs/phase4_ippo.yaml",
        scenario_id=scenario,
        seed=seed,
        manifest_path=manifest_path
    )
    
    replayed_cost = float(replay_res["cost"])
    replayed_fill_rate = float(replay_res["fill_rate"])
    replayed_bullwhip = float(replay_res["bullwhip"])
    
    cost_match = bool(np.isclose(replayed_cost, expected_cost, atol=1e-3))
    fill_rate_match = bool(np.isclose(replayed_fill_rate, expected_fill_rate, atol=1e-4))
    bullwhip_match = bool(np.isclose(replayed_bullwhip, expected_bullwhip, atol=1e-4))
    
    passed = bool(cost_match and fill_rate_match and bullwhip_match)
    return passed, {
        "model_id": model_id,
        "scenario": scenario,
        "seed": seed,
        "expected_cost": expected_cost,
        "replayed_cost": replayed_cost,
        "cost_match": bool(cost_match),
        "expected_fill_rate": expected_fill_rate,
        "replayed_fill_rate": replayed_fill_rate,
        "fill_rate_match": bool(fill_rate_match),
        "expected_bullwhip": expected_bullwhip,
        "replayed_bullwhip": replayed_bullwhip,
        "bullwhip_match": bool(bullwhip_match),
        "passed": passed
    }

def run_all_checks(registry_path: str = REGISTRY_DEFAULT_PATH,
                   skip_replay: bool = False) -> Tuple[bool, Dict[str, Any]]:
    """Runs the full battery of Phase 13 reproducibility checks."""
    print("=" * 70)
    print("  PHASE 13: AUTOMATED REPRODUCIBILITY & INTEGRITY AUDIT")
    print("=" * 70)
    
    report = {}
    all_passed = True
    
    # 1. Environment
    print("\n[1/4] Auditing Python runtime and dependencies...")
    env_pass, env_data = check_environment()
    report["environment"] = env_data
    if env_pass:
        print(f"      PASS: Python {env_data['python']['version']}, PyTorch {env_data['torch']['version']}")
    else:
        print("      FAIL: Incompatible runtime or missing packages.")
        all_passed = False
        
    # 2. Registry Integrity
    print("\n[2/4] Auditing registry schema & dual-hash checkpoint integrity...")
    reg_pass, reg_data = check_registry_integrity(registry_path)
    report["registry_integrity"] = reg_data
    if reg_pass:
        print(f"      PASS: {reg_data['total_models']} models audited:")
        print(f"            - Valid Canonical State-Dict: {reg_data['valid_canonical']}")
        print(f"            - Valid Historical Physical:  {reg_data['valid_historical_physical']}")
        print(f"            - Valid Offline Benchmarks:   {reg_data['valid_offline']}")
        print(f"            - Corrupted Checkpoints:      {reg_data['corrupted']}")
    else:
        print(f"      FAIL: Integrity issues detected: {reg_data}")
        all_passed = False
        
    # 3. Seed Propagation & Entropy Isolation
    print("\n[3/4] Auditing seed propagation and RNG entropy isolation...")
    seed_pass, seed_data = check_seed_propagation()
    report["seed_propagation"] = seed_data
    if seed_pass:
        print("      PASS: Deterministic reset verified; independent RNG seeds verified.")
    else:
        print(f"      FAIL: Seed failure: {seed_data}")
        all_passed = False
        
    # 4. Evaluation Replay Test
    if not skip_replay:
        print("\n[4/4] Executing deterministic evaluation replay on representative model...")
        replay_pass, replay_data = replay_deterministic_evaluation()
        report["evaluation_replay"] = replay_data
        if replay_pass:
            print(f"      PASS: Replayed {replay_data['model_id']} ({replay_data['scenario']}, seed {replay_data['seed']}):")
            print(f"            Recorded Cost: {replay_data['expected_cost']:.1f} | Replayed: {replay_data['replayed_cost']:.1f} (MATCH)")
            print(f"            Fill Rate:     {replay_data['expected_fill_rate']:.4f} | Replayed: {replay_data['replayed_fill_rate']:.4f} (MATCH)")
            print(f"            Bullwhip:      {replay_data['expected_bullwhip']:.4f} | Replayed: {replay_data['replayed_bullwhip']:.4f} (MATCH)")
        else:
            print(f"      FAIL: Replay deviation detected: {replay_data}")
            all_passed = False
    else:
        print("\n[4/4] Evaluation replay skipped by flag.")
        report["evaluation_replay"] = {"skipped": True}
        
    print("\n" + "=" * 70)
    if all_passed:
        print("  REPRODUCIBILITY AUDIT PASSED: ALL INTEGRITY CONTRACTS MET.")
    else:
        print("  REPRODUCIBILITY AUDIT FAILED: ONE OR MORE CHECKS FAILED.")
    print("=" * 70)
    
    return all_passed, report

def make_serializable(val: Any) -> Any:
    """Recursively converts NumPy data types to standard Python primitives."""
    if isinstance(val, np.bool_):
        return bool(val)
    if isinstance(val, (np.integer, np.int64, np.int32)):
        return int(val)
    if isinstance(val, (np.floating, np.float64, np.float32)):
        return float(val)
    if isinstance(val, dict):
        return {k: make_serializable(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [make_serializable(x) for x in val]
    return val

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 13 Reproducibility & Integrity Checker")
    parser.add_argument("--registry", default=REGISTRY_DEFAULT_PATH, help="Path to registry.json")
    parser.add_argument("--skip-replay", action="store_true", help="Skip the simulation evaluation replay check")
    parser.add_argument("--json", type=str, help="Save report to JSON file")
    args = parser.parse_args()
    
    passed, report = run_all_checks(registry_path=args.registry, skip_replay=args.skip_replay)
    
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        serializable_report = make_serializable(report)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(serializable_report, f, indent=2)
        print(f"\nReport written to {args.json}")
        
    sys.exit(0 if passed else 1)

if __name__ == "__main__":
    main()
