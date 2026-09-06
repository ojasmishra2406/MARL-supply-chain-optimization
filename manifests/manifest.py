import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from uuid import UUID

import yaml

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


def get_repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get_relative_config_path(config_path: str) -> str:
    abs_path = os.path.abspath(config_path)
    repo_root = get_repo_root()
    if not abs_path.startswith(repo_root):
        raise ValueError("Configuration path must be within the repository")
    rel_path = os.path.relpath(abs_path, repo_root)
    if rel_path.startswith(".."):
        raise ValueError("Configuration path must be within the repository")
    return rel_path.replace("\\", "/")


def write_manifest(
    run_id: UUID,
    config_path: str,
    start_time: datetime,
    end_time: datetime,
    out_path: str,
) -> None:
    # 1. Ensure config_path is repository-relative
    rel_config_path = get_relative_config_path(config_path)

    # 2. Extract seed from the YAML configuration
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
    if not config_data or "seed" not in config_data:
        raise ValueError("Configuration must contain a 'seed'")
    seed = config_data["seed"]

    # 3. Create manifest
    manifest = ManifestSchema(
        run_id=run_id,
        git_commit_hash=get_git_commit(),
        config_hash=get_config_hash(config_path),
        config_path=rel_config_path,
        seed=seed,
        library_versions=get_library_versions(),
        hardware=get_hardware_info(),
        start_time=start_time,
        end_time=end_time,
    )
    with open(out_path, "w") as f:
        f.write(manifest.model_dump_json(indent=4))


def verify_manifest(manifest_path: str) -> bool:
    with open(manifest_path, "r") as f:
        data = json.load(f)

    # Will raise ValidationError if schema is invalid
    manifest = ManifestSchema(**data)

    repo_root = get_repo_root()
    # Ensure proper joining on the current OS
    config_path = os.path.normpath(os.path.join(repo_root, manifest.config_path))

    current_hash = get_config_hash(config_path)
    if current_hash != manifest.config_hash:
        raise ValueError(
            f"Configuration drift detected: {current_hash} != {manifest.config_hash}"
        )

    return True
