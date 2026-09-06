import hashlib
import json
import subprocess
import sys
from datetime import datetime
from uuid import UUID

from manifests.hardware import get_hardware_info
from manifests.schema import ManifestSchema


def get_git_commit() -> str:
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        return commit
    except subprocess.CalledProcessError:
        raise ValueError("No git commit found")
    except FileNotFoundError:
        raise ValueError("git command not found")


def get_config_hash(config_path: str) -> str:
    hasher = hashlib.sha256()
    with open(config_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def get_library_versions() -> dict[str, str]:
    try:
        output = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"]
        ).decode("utf-8")
        versions = {}
        for line in output.strip().split("\n"):
            if "==" in line:
                pkg, ver = line.split("==", 1)
                versions[pkg] = ver
        return versions
    except Exception:
        return {}


def write_manifest(
    run_id: UUID,
    config_path: str,
    seed: int,
    start_time: datetime,
    end_time: datetime,
    out_path: str,
) -> None:
    manifest = ManifestSchema(
        run_id=run_id,
        git_commit_hash=get_git_commit(),
        config_hash=get_config_hash(config_path),
        seed=seed,
        library_versions=get_library_versions(),
        hardware=get_hardware_info(),
        start_time=start_time,
        end_time=end_time,
    )
    with open(out_path, "w") as f:
        f.write(manifest.model_dump_json(indent=4))


def verify_manifest(manifest_path: str, config_path: str) -> bool:
    with open(manifest_path, "r") as f:
        data = json.load(f)

    # Will raise ValidationError if schema is invalid
    manifest = ManifestSchema(**data)

    current_hash = get_config_hash(config_path)
    if current_hash != manifest.config_hash:
        raise ValueError(
            f"Configuration drift detected: {current_hash} != {manifest.config_hash}"
        )

    return True
