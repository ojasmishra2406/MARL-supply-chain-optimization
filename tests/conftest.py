import hashlib
import json

import pytest


@pytest.fixture
def temp_config_path():
    import os

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_file = os.path.join(repo_root, "configs", "test_config.yaml")
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    with open(config_file, "w") as f:
        f.write("learning_rate: 0.001\nbatch_size: 32\nseed: 42\n")
    yield config_file
    if os.path.exists(config_file):
        os.remove(config_file)


@pytest.fixture
def temp_manifest_path(tmp_path):
    return str(tmp_path / "manifest.json")


@pytest.fixture
def deterministic_fixture_input():
    return {"learning_rate": 0.001, "batch_size": 32, "seed": 42}


def compute_deterministic_hash(data: dict) -> str:
    serialized = json.dumps(data, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
