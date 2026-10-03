"""
benchmark_evaluations.py — Phase 7 evaluation file inventory and provenance audit.
"""
import os, glob, json, datetime, subprocess, sys, hashlib
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PHASE7_DIR  = os.path.join(ROOT, "results", "phase7")
PHASE6_DIR  = os.path.join(ROOT, "results", "phase6")

EXPECTED_SCENARIOS  = 6
EXPECTED_EVAL_SEEDS = 5

FAILURES = []

def git_commit():
    try:
        return subprocess.check_output(["git","rev-parse","HEAD"],
                                       cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def extract_train_seed_from_ckpt_path(ckpt_path):
    if not ckpt_path:
        return None
    base = os.path.splitext(os.path.basename(ckpt_path))[0]
    parts = base.split("_")
    if parts and parts[-1].startswith("s") and parts[-1][1:].isdigit():
        return int(parts[-1][1:])
    return None

def run():
    ts = datetime.datetime.utcnow().isoformat() + "Z"
    commit = git_commit()

    eval_files = sorted(glob.glob(os.path.join(PHASE7_DIR, "*.json")))

    records = []
    models_seen = set()
    train_seeds_seen = defaultdict(set)
    eval_seeds_seen  = defaultdict(set)
    scenarios_seen   = defaultdict(set)
    hash_mismatches  = 0
    missing_ckpts    = 0
    unprovable_seeds = 0

    for fpath in eval_files:
        with open(fpath) as f:
            d = json.load(f)

        ckpt_path_raw = d.get("checkpoint")
        ckpt_hash_in_eval = d.get("checkpoint_hash")
        scenario = d.get("scenario")
        eval_seed = d.get("seed")
        algo = d.get("algorithm", "out")

        ckpt_path = None
        if ckpt_path_raw:
            ckpt_path = os.path.join(ROOT, ckpt_path_raw.replace("/", os.sep).lstrip(os.sep))
            if not os.path.exists(ckpt_path):
                ckpt_path = ckpt_path_raw

        train_seed = extract_train_seed_from_ckpt_path(ckpt_path_raw)
        ckpt_exists = bool(ckpt_path and os.path.exists(ckpt_path))
        hash_match = None

        if ckpt_exists and ckpt_hash_in_eval:
            actual_hash = sha256_file(ckpt_path)
            hash_match = (actual_hash == ckpt_hash_in_eval)
            if not hash_match:
                hash_mismatches += 1
                FAILURES.append(f"HASH MISMATCH in {os.path.basename(fpath)}: "
                                 f"eval claims {ckpt_hash_in_eval[:16]}... "
                                 f"file is {actual_hash[:16]}...")
        elif ckpt_path_raw and not ckpt_exists:
            missing_ckpts += 1
            FAILURES.append(f"MISSING CHECKPOINT referenced by {os.path.basename(fpath)}: {ckpt_path_raw}")

        if train_seed is None and algo.lower() not in ("out",):
            unprovable_seeds += 1
            FAILURES.append(f"CANNOT ESTABLISH train_seed from {os.path.basename(fpath)}")

        model_id = os.path.splitext(os.path.basename(fpath))[0]
        for scen in ["in_distribution","demand_shift","lead_time_shift",
                     "capacity_disruption","demand_spike","combined_shift"]:
            if f"_{scen}_" in model_id:
                model_prefix = model_id[:model_id.index(f"_{scen}_")]
                break
        else:
            model_prefix = model_id

        if ckpt_path_raw:
            models_seen.add(ckpt_path_raw)
        if train_seed is not None:
            train_seeds_seen[model_prefix].add(train_seed)
        if eval_seed is not None:
            eval_seeds_seen[model_prefix].add(eval_seed)
        if scenario:
            scenarios_seen[model_prefix].add(scenario)

        records.append({
            "file": os.path.basename(fpath),
            "algorithm": algo,
            "scenario": scenario,
            "eval_seed": eval_seed,
            "train_seed": train_seed,
            "checkpoint_exists": ckpt_exists,
            "hash_match": hash_match,
        })

    rl_records = [r for r in records if r["algorithm"].lower() not in ("out",)]
    out_records = [r for r in records if r["algorithm"].lower() in ("out",)]

    unique_rl_models = set(r["file"].split("_in_distribution_")[0]
                            .split("_demand_shift_")[0]
                            .split("_lead_time_shift_")[0]
                            .split("_capacity_disruption_")[0]
                            .split("_demand_spike_")[0]
                            .split("_combined_shift_")[0]
                           for r in rl_records)

    expected_rl = 40 * EXPECTED_SCENARIOS * EXPECTED_EVAL_SEEDS  # 1200
    actual_rl   = len(rl_records)
    if actual_rl != expected_rl:
        FAILURES.append(f"RL EVAL COUNT {actual_rl} != EXPECTED {expected_rl}")

    result = {
        "total_files": len(eval_files),
        "rl_evaluations": actual_rl,
        "out_evaluations": len(out_records),
        "expected_rl_evaluations": expected_rl,
        "missing_vs_expected": expected_rl - actual_rl,
        "unique_rl_models": len(unique_rl_models),
        "unique_checkpoints_referenced": len(models_seen),
        "unique_scenarios": len(set(r["scenario"] for r in records if r["scenario"])),
        "scenarios": sorted(set(r["scenario"] for r in records if r["scenario"])),
        "hash_mismatches": hash_mismatches,
        "missing_checkpoints": missing_ckpts,
        "unprovable_train_seeds": unprovable_seeds,
        "failures": FAILURES,
        "timestamp": ts,
        "git_commit": commit,
        "source": f"filesystem scan of {PHASE7_DIR}",
        "method": "json parse + sha256 provenance check",
    }
    return result

if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2, default=str))
