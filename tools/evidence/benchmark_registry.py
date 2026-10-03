"""
benchmark_registry.py — Registry forensics.
"""
import os, json, glob, hashlib, datetime, subprocess, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REGISTRY_PATH = os.path.join(ROOT, "models", "registry.json")

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

def run():
    ts = datetime.datetime.utcnow().isoformat() + "Z"
    commit = git_commit()

    with open(REGISTRY_PATH) as f:
        registry = json.load(f)

    all_ckpts_on_disk = set()
    for phase_dir in ["results/phase6", "results/phase10", "results/phase11"]:
        full = os.path.join(ROOT, phase_dir)
        if os.path.exists(full):
            for fname in os.listdir(full):
                if fname.endswith(".pt") and "_temp" not in fname and "ci_test" not in fname:
                    all_ckpts_on_disk.add(os.path.join(full, fname))

    model_ids = list(registry.keys())
    duplicate_ids = len(model_ids) - len(set(model_ids))

    entries = []
    missing_ckpts    = 0
    hash_mismatches  = 0
    invalid_statuses = 0

    for model_id, reg in registry.items():
        status = reg.get("status", "UNKNOWN")
        ckpt_path_raw = reg.get("checkpoint_path") or reg.get("actual_checkpoint")

        rec = {
            "model_id": model_id,
            "status": status,
            "checkpoint_path": ckpt_path_raw,
        }

        if ckpt_path_raw:
            ckpt_full = os.path.join(ROOT, ckpt_path_raw.replace("/", os.sep).lstrip(os.sep))
            rec["checkpoint_exists"] = os.path.exists(ckpt_full)

            if not rec["checkpoint_exists"]:
                missing_ckpts += 1
                if status in ("VALID", "TRAINED", "EVALUATED"):
                    FAILURES.append(f"REGISTRY: VALID entry '{model_id}' missing checkpoint: {ckpt_path_raw}")
                rec["hash_match"] = None
            else:
                all_ckpts_on_disk.discard(ckpt_full)
                stored_hash = reg.get("checkpoint_sha256")
                if stored_hash:
                    actual_hash = sha256_file(ckpt_full)
                    rec["hash_match"] = (actual_hash == stored_hash)
                    rec["actual_hash_prefix"] = actual_hash[:16]
                    rec["stored_hash_prefix"] = stored_hash[:16]
                    if not rec["hash_match"]:
                        hash_mismatches += 1
                        FAILURES.append(f"REGISTRY HASH MISMATCH '{model_id}': "
                                        f"stored={stored_hash[:16]}... actual={actual_hash[:16]}...")
                else:
                    rec["hash_match"] = "no_hash_stored"
        else:
            rec["checkpoint_exists"] = None
            rec["hash_match"] = None

        if status not in ("VALID", "TRAINED", "EVALUATED", "STRUCTURAL_ONLY", "INVALID",
                           "PENDING", "ABORTED"):
            invalid_statuses += 1
            FAILURES.append(f"REGISTRY: Unknown status '{status}' for '{model_id}'")

        entries.append(rec)

    orphans = [p for p in all_ckpts_on_disk
               if "_temp" not in p and "ci_test" not in p]

    result = {
        "registry_path": REGISTRY_PATH,
        "registered_models": len(model_ids),
        "duplicate_model_ids": duplicate_ids,
        "missing_checkpoints": missing_ckpts,
        "hash_mismatches": hash_mismatches,
        "invalid_status_entries": invalid_statuses,
        "orphan_checkpoints": len(orphans),
        "orphan_paths": orphans,
        "entries": entries,
        "failures": FAILURES,
        "timestamp": ts,
        "git_commit": commit,
        "source": REGISTRY_PATH,
        "method": "json parse + sha256 cross-reference against filesystem",
    }
    return result

if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2, default=str))
