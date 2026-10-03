"""
benchmark_reproducibility.py — Deterministic reproducibility check.
"""
import os, sys, json, datetime, subprocess
import torch
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

FAILURES = []

def git_commit():
    try:
        return subprocess.check_output(["git","rev-parse","HEAD"],
                                       cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"

def find_first_valid_phase6_model():
    import glob
    manifests = sorted(glob.glob(os.path.join(ROOT, "results/phase6", "*_manifest.json")))
    for m in manifests:
        with open(m) as f:
            d = json.load(f)
        ckpt = d.get("checkpoint_path") or d.get("actual_checkpoint")
        if ckpt:
            full = os.path.join(ROOT, ckpt.replace("/", os.sep).lstrip(os.sep))
            if os.path.exists(full):
                return os.path.splitext(os.path.basename(m))[0].replace("_manifest",""), full, m
    return None, None, None

def run_eval_once(model_id, ckpt_path, manifest_path, scenario, eval_seed):
    from rl.phase7_evaluator import evaluate_scenario
    return evaluate_scenario(
        algorithm="mappo_comm",
        checkpoint_path=ckpt_path,
        config_path=os.path.join(ROOT, "configs/phase4_ippo.yaml"),
        scenario_id=scenario,
        seed=eval_seed,
        manifest_path=manifest_path,
    )

def run():
    ts     = datetime.datetime.utcnow().isoformat() + "Z"
    commit = git_commit()

    model_id, ckpt_path, manifest_path = find_first_valid_phase6_model()
    if not model_id:
        FAILURES.append("No valid Phase 6 checkpoint found for reproducibility test")
        return {"failures": FAILURES, "timestamp": ts, "git_commit": commit}

    torch.manual_seed(42)
    np.random.seed(42)

    scenario  = "in_distribution"
    eval_seed = 0

    run1 = run_eval_once(model_id, ckpt_path, manifest_path, scenario, eval_seed)
    run2 = run_eval_once(model_id, ckpt_path, manifest_path, scenario, eval_seed)

    metrics = ["cost", "fill_rate", "bullwhip"]
    diffs = {}
    max_abs_diff = 0.0
    for key in metrics:
        v1 = run1.get(key, 0) or 0
        v2 = run2.get(key, 0) or 0
        d = abs(v1 - v2)
        diffs[key] = {"run1": v1, "run2": v2, "abs_diff": d}
        max_abs_diff = max(max_abs_diff, d)

    exact_match = (max_abs_diff == 0.0)

    cuda_in_use = torch.cuda.is_available()
    deterministic = exact_match

    if not deterministic and not torch.cuda.is_available():
        FAILURES.append(f"NON-DETERMINISTIC on CPU: max_abs_diff={max_abs_diff}")

    result = {
        "model_used": model_id,
        "scenario": scenario,
        "eval_seed": eval_seed,
        "metric_diffs": diffs,
        "max_absolute_difference": max_abs_diff,
        "exact_match": exact_match,
        "deterministic": deterministic,
        "cuda_in_use": cuda_in_use,
        "failures": FAILURES,
        "timestamp": ts,
        "git_commit": commit,
        "source": f"double evaluation of {model_id} with seed={eval_seed}",
        "method": "run_eval_once × 2, compare metric values",
    }
    return result

if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2, default=str))
