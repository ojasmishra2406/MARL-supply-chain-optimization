import hashlib
import json
import os
import subprocess
from datetime import datetime
from unittest import mock
from uuid import uuid4

import pytest

from manifests.hardware import get_hardware_info
from manifests.manifest import (
    get_config_hash,
    get_git_commit,
    get_library_versions,
    get_repo_root,
    verify_manifest,
    write_manifest,
)
from manifests.schema import ManifestSchema
from tests.conftest import compute_deterministic_hash


def test_manifest_schema_and_uuid():
    run_id = uuid4()
    manifest = ManifestSchema(
        run_id=run_id,
        git_commit_hash="fake_hash",
        config_hash="fake_config_hash",
        config_path="configs/test.yaml",
        seed=42,
        library_versions={"torch": "2.3.1"},
        hardware={"system": "Linux"},
        start_time=datetime.now(),
        end_time=datetime.now(),
    )
    assert manifest.run_id == run_id
    assert manifest.seed == 42


def test_exact_yaml_hash(temp_config_path):
    with open(temp_config_path, "rb") as f:
        expected_hash = hashlib.sha256(f.read()).hexdigest()
    actual_hash = get_config_hash(temp_config_path)
    assert actual_hash == expected_hash


def test_configuration_drift(temp_config_path, temp_manifest_path):
    run_id = uuid4()
    write_manifest(
        run_id=run_id,
        config_path=temp_config_path,
        start_time=datetime.now(),
        end_time=datetime.now(),
        out_path=temp_manifest_path,
    )
    assert verify_manifest(temp_manifest_path)

    with open(temp_config_path, "a") as f:
        f.write("modified: true\n")

    with pytest.raises(ValueError, match="Configuration drift detected"):
        verify_manifest(temp_manifest_path)


def test_hardware_detector():
    info = get_hardware_info()
    assert "system" in info
    assert "processor" in info
    assert "logical_cpus" in info


def test_manifest_round_trip(temp_config_path, temp_manifest_path):
    run_id = uuid4()
    write_manifest(
        run_id=run_id,
        config_path=temp_config_path,
        start_time=datetime.now(),
        end_time=datetime.now(),
        out_path=temp_manifest_path,
    )

    with open(temp_manifest_path, "r") as f:
        data = json.load(f)
    manifest = ManifestSchema(**data)
    assert manifest.run_id == run_id
    assert manifest.seed == 42
    assert manifest.config_hash == get_config_hash(temp_config_path)
    assert manifest.config_path == "configs/test_config.yaml"


def test_deterministic_fixture(deterministic_fixture_input):
    hash_1 = compute_deterministic_hash(deterministic_fixture_input)
    hash_2 = compute_deterministic_hash(deterministic_fixture_input)
    assert hash_1 == hash_2


def test_docker_reproducibility():
    if os.path.exists("host_hash.txt") and os.path.exists("container_hash.txt"):
        with open("host_hash.txt") as f:
            host_hash = f.read().strip()
        with open("container_hash.txt") as f:
            container_hash = f.read().strip()
        assert host_hash == container_hash


@mock.patch("subprocess.check_output", side_effect=FileNotFoundError)
def test_git_commit_file_not_found(mock_subp):
    with pytest.raises(ValueError, match="git command not found"):
        get_git_commit()


@mock.patch(
    "subprocess.check_output", side_effect=subprocess.CalledProcessError(128, [])
)
def test_git_commit_called_process_error(mock_subp):
    with pytest.raises(ValueError, match="No git commit found"):
        get_git_commit()


@mock.patch("subprocess.check_output", side_effect=Exception)
def test_get_library_versions_error(mock_subp):
    assert get_library_versions() == {}


@mock.patch.dict("sys.modules", {"torch": None})
def test_hardware_detector_no_torch():
    info = get_hardware_info()
    assert "gpu" not in info


# --- NEW FIX TESTS ---


def test_1_configuration_path_is_stored(temp_config_path, temp_manifest_path):
    run_id = uuid4()
    write_manifest(
        run_id=run_id,
        config_path=temp_config_path,
        start_time=datetime.now(),
        end_time=datetime.now(),
        out_path=temp_manifest_path,
    )
    with open(temp_manifest_path, "r") as f:
        data = json.load(f)

    # Path is stored, repository-relative (not absolute), using forward slashes
    assert data["config_path"] == "configs/test_config.yaml"


def test_2_seed_comes_from_yaml(temp_config_path, temp_manifest_path):
    run_id = uuid4()
    # Write manifest API does not take a seed parameter
    write_manifest(
        run_id=run_id,
        config_path=temp_config_path,
        start_time=datetime.now(),
        end_time=datetime.now(),
        out_path=temp_manifest_path,
    )
    with open(temp_manifest_path, "r") as f:
        data = json.load(f)
    # The seed in temp_config_path is 42
    assert data["seed"] == 42


def test_3_no_competing_seed_source(temp_manifest_path):
    repo_root = get_repo_root()
    test_yaml_path = os.path.join(repo_root, "configs", "test3.yaml")
    with open(test_yaml_path, "w") as f:
        f.write("seed: 123\n")

    try:
        run_id = uuid4()
        write_manifest(
            run_id=run_id,
            config_path=test_yaml_path,
            start_time=datetime.now(),
            end_time=datetime.now(),
            out_path=temp_manifest_path,
        )
        with open(temp_manifest_path, "r") as f:
            data = json.load(f)
        assert data["seed"] == 123

        # Modify the YAML seed
        with open(test_yaml_path, "w") as f:
            f.write("seed: 456\n")

        # Verification fails because the hash changed
        with pytest.raises(ValueError, match="Configuration drift detected"):
            verify_manifest(temp_manifest_path)
    finally:
        if os.path.exists(test_yaml_path):
            os.remove(test_yaml_path)


def test_5_same_basename_ambiguity(temp_manifest_path):
    repo_root = get_repo_root()
    dir_a = os.path.join(repo_root, "configs", "train")
    dir_b = os.path.join(repo_root, "configs", "eval")
    os.makedirs(dir_a, exist_ok=True)
    os.makedirs(dir_b, exist_ok=True)

    path_a = os.path.join(dir_a, "config.yaml")
    path_b = os.path.join(dir_b, "config.yaml")

    try:
        with open(path_a, "w") as f:
            f.write("type: train\nseed: 100\n")
        with open(path_b, "w") as f:
            f.write("type: eval\nseed: 200\n")

        manifest_a_path = temp_manifest_path + "_a"
        manifest_b_path = temp_manifest_path + "_b"

        write_manifest(uuid4(), path_a, datetime.now(), datetime.now(), manifest_a_path)
        write_manifest(uuid4(), path_b, datetime.now(), datetime.now(), manifest_b_path)

        with open(manifest_a_path, "r") as f:
            data_a = json.load(f)
        with open(manifest_b_path, "r") as f:
            data_b = json.load(f)

        assert data_a["config_path"] != data_b["config_path"]
        assert data_a["config_hash"] != data_b["config_hash"]

        assert verify_manifest(manifest_a_path)
        assert verify_manifest(manifest_b_path)
    finally:
        for p in [path_a, path_b]:
            if os.path.exists(p):
                os.remove(p)


def test_6_same_content_different_path(temp_manifest_path):
    repo_root = get_repo_root()
    dir_a = os.path.join(repo_root, "configs", "temp_a")
    dir_b = os.path.join(repo_root, "configs", "temp_b")
    os.makedirs(dir_a, exist_ok=True)
    os.makedirs(dir_b, exist_ok=True)

    path_a = os.path.join(dir_a, "config.yaml")
    path_b = os.path.join(dir_b, "config.yaml")

    try:
        content = "seed: 999\nsomething: else\n"
        with open(path_a, "w") as f:
            f.write(content)
        with open(path_b, "w") as f:
            f.write(content)

        manifest_a_path = temp_manifest_path + "_a"
        manifest_b_path = temp_manifest_path + "_b"

        write_manifest(uuid4(), path_a, datetime.now(), datetime.now(), manifest_a_path)
        write_manifest(uuid4(), path_b, datetime.now(), datetime.now(), manifest_b_path)

        with open(manifest_a_path, "r") as f:
            data_a = json.load(f)
        with open(manifest_b_path, "r") as f:
            data_b = json.load(f)

        assert data_a["config_hash"] == data_b["config_hash"]
        assert data_a["config_path"] != data_b["config_path"]
    finally:
        for p in [path_a, path_b]:
            if os.path.exists(p):
                os.remove(p)


def test_7_verification_uses_stored_path(temp_config_path, temp_manifest_path):
    # Verification API called without config_path argument
    write_manifest(
        run_id=uuid4(),
        config_path=temp_config_path,
        start_time=datetime.now(),
        end_time=datetime.now(),
        out_path=temp_manifest_path,
    )
    assert verify_manifest(temp_manifest_path)

    with open(temp_config_path, "a") as f:
        f.write("modified: true\n")

    with pytest.raises(ValueError, match="Configuration drift detected"):
        verify_manifest(temp_manifest_path)


def test_8_absolute_path_is_rejected_or_normalized(
    temp_config_path, temp_manifest_path
):
    abs_path = os.path.abspath(temp_config_path)
    write_manifest(
        run_id=uuid4(),
        config_path=abs_path,
        start_time=datetime.now(),
        end_time=datetime.now(),
        out_path=temp_manifest_path,
    )
    with open(temp_manifest_path, "r") as f:
        data = json.load(f)
    assert not os.path.isabs(data["config_path"])
    assert data["config_path"] == "configs/test_config.yaml"

    # Must reject paths outside repo
    with pytest.raises(
        ValueError, match="Configuration path must be within the repository"
    ):
        write_manifest(
            run_id=uuid4(),
            config_path=(
                "/tmp/outside_repo.yaml" if os.name != "nt" else "C:\\outside_repo.yaml"
            ),
            start_time=datetime.now(),
            end_time=datetime.now(),
            out_path=temp_manifest_path,
        )
