import hashlib
import json

import pytest


@pytest.fixture
def temp_config_path(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("learning_rate: 0.001\nbatch_size: 32\n")
    return str(config_file)


@pytest.fixture
def temp_manifest_path(tmp_path):
    return str(tmp_path / "manifest.json")


@pytest.fixture
def deterministic_fixture_input():
    return {"learning_rate": 0.001, "batch_size": 32, "seed": 42}


def compute_deterministic_hash(data: dict) -> str:
    serialized = json.dumps(data, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
