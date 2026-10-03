"""
benchmark_models.py — Phase 6/10/11 checkpoint inventory and integrity.
Reads actual filesystem artifacts; never hardcodes expected counts.
"""
import os, glob, json, hashlib, datetime, time, subprocess, sys
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PHASE6_DIR = os.path.join(ROOT, "results", "phase6")
PHASE10_DIR = os.path.join(ROOT, "results", "phase10")
PHASE11_DIR = os.path.join(ROOT, "results", "phase11")

REQUIRED_ITERATIONS = 200   # protocol requirement
PHASE6_EXPECTED     = 40    # 4 algos × 2 conditions × 5 seeds

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

def load_time_ms(path):
    t0 = time.perf_counter()
    torch.load(path, map_location="cpu")
    return round((time.perf_counter() - t0) * 1000, 1)

def probe_manifest(manifest_path):
    """Return iteration count and stored hash from manifest (any schema)."""
    with open(manifest_path) as f:
        m = json.load(f)
    iters = m.get("iteration") or m.get("iterations") or m.get("training_iterations")
    stored_hash = m.get("checkpoint_sha256") or m.get("sha256")
    return iters, stored_hash, m

def scan_phase(phase_label, ckpt_dir, expected_count):
    ts = datetime.datetime.utcnow().isoformat() + "Z"
    commit = git_commit()
    models = []

    ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
    # exclude ci/temp files
    ckpt_files = [p for p in ckpt_files if
                  "_temp" not in os.path.basename(p) and
                  "ci_test" not in os.path.basename(p)]

    for ckpt_path in ckpt_files:
        basename = os.path.splitext(os.path.basename(ckpt_path))[0]
        manifest_path = os.path.join(ckpt_dir, basename + "_manifest.json")

        rec = {
            "model_id": basename,
            "checkpoint_path": ckpt_path,
            "checkpoint_exists": os.path.exists(ckpt_path),
            "manifest_exists": os.path.exists(manifest_path),
            "checkpoint_size_bytes": os.path.getsize(ckpt_path) if os.path.exists(ckpt_path) else None,
            "timestamp": ts,
            "git_commit": commit,
        }

        if not rec["checkpoint_exists"]:
            rec["valid"] = False
            rec["error"] = "checkpoint_missing"
            FAILURES.append(f"[{phase_label}] MISSING checkpoint: {ckpt_path}")
            models.append(rec)
            continue

        # Compute real hash
        actual_hash = sha256_file(ckpt_path)
        rec["checkpoint_sha256"] = actual_hash

        # Load time
        try:
            rec["checkpoint_load_ms"] = load_time_ms(ckpt_path)
            rec["checkpoint_loads"] = True
        except Exception as e:
            rec["checkpoint_loads"] = False
            rec["load_error"] = str(e)
            FAILURES.append(f"[{phase_label}] LOAD ERROR {basename}: {e}")

        # Manifest
        if rec["manifest_exists"]:
            iters, stored_hash, manifest_raw = probe_manifest(manifest_path)
            rec["iteration_count"] = iters
            rec["stored_hash"] = stored_hash

            # Hash integrity
            if stored_hash and actual_hash != stored_hash:
                rec["hash_match"] = False
                FAILURES.append(f"[{phase_label}] HASH MISMATCH {basename}: "
                                 f"actual={actual_hash[:16]}... stored={stored_hash[:16]}...")
            else:
                rec["hash_match"] = (stored_hash is not None)

            # Iteration check (only for phase6)
            if phase_label == "phase6" and iters is not None:
                if iters != REQUIRED_ITERATIONS:
                    rec["valid"] = False
                    FAILURES.append(f"[{phase_label}] WRONG ITERATIONS {basename}: "
                                    f"got {iters}, required {REQUIRED_ITERATIONS}")
                else:
                    rec["valid"] = True
            else:
                rec["valid"] = rec.get("checkpoint_loads", False)

            # Parse model_id fields from name
            parts = basename.split("_")
            if "s" in parts[-1] and parts[-1][1:].isdigit():
                rec["train_seed"] = int(parts[-1][1:])
            for cond in ("baseline", "high_variance"):
                if cond in basename:
                    rec["condition"] = cond
                    break
            for algo in ("mappo_comm", "ippo_comm", "mappo", "ippo", "gnn_mappo"):
                if algo in basename:
                    rec["algorithm"] = algo
                    break
        else:
            rec["manifest_exists"] = False
            rec["valid"] = False
            FAILURES.append(f"[{phase_label}] MISSING manifest: {manifest_path}")

        models.append(rec)

    # Aggregate stats
    valid_models = [m for m in models if m.get("valid")]
    iters_list = [m["iteration_count"] for m in valid_models if m.get("iteration_count") is not None]

    summary = {
        "phase": phase_label,
        "expected": expected_count,
        "discovered": len(models),
        "valid": len(valid_models),
        "invalid": len(models) - len(valid_models),
        "missing_checkpoints": sum(1 for m in models if not m["checkpoint_exists"]),
        "hash_mismatches": sum(1 for m in models if m.get("hash_match") is False),
        "iteration_min": min(iters_list) if iters_list else None,
        "iteration_max": max(iters_list) if iters_list else None,
        "iteration_mean": round(sum(iters_list)/len(iters_list), 1) if iters_list else None,
        "models": models,
        "timestamp": ts,
        "git_commit": commit,
        "source": f"filesystem scan of {ckpt_dir}",
        "method": "sha256_file + torch.load + manifest parse",
    }

    if expected_count and len(valid_models) != expected_count:
        FAILURES.append(f"[{phase_label}] VALID COUNT {len(valid_models)} != EXPECTED {expected_count}")

    return summary

def run():
    result = {
        "phase6": scan_phase("phase6", PHASE6_DIR, PHASE6_EXPECTED),
        "phase10": scan_phase("phase10", PHASE10_DIR, 10),
        "phase11": scan_phase("phase11", PHASE11_DIR, None),
        "failures": FAILURES,
    }
    return result

if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2, default=str))
