import pytest
import os
import hashlib
from uuid import uuid4
from datetime import datetime
from manifests.schema import ManifestSchema
from manifests.manifest import write_manifest, verify_manifest, get_config_hash
from manifests.hardware import get_hardware_info
from tests.conftest import compute_deterministic_hash

def test_manifest_schema_and_uuid():
    run_id = uuid4()
    manifest = ManifestSchema(
        run_id=run_id,
        git_commit_hash="fake_hash",
        config_hash="fake_config_hash",
        seed=42,
        library_versions={"torch": "2.3.1"},
        hardware={"system": "Linux"},
        start_time=datetime.now(),
        end_time=datetime.now()
    )
    assert manifest.run_id == run_id
    assert manifest.seed == 42

def test_exact_yaml_hash(temp_config_path):
    # Calculate hash independently
    with open(temp_config_path, "rb") as f:
        expected_hash = hashlib.sha256(f.read()).hexdigest()
    
    actual_hash = get_config_hash(temp_config_path)
    assert actual_hash == expected_hash

def test_configuration_drift(temp_config_path, temp_manifest_path):
    # Setup initial manifest
    run_id = uuid4()
    write_manifest(
        run_id=run_id,
        config_path=temp_config_path,
        seed=42,
        start_time=datetime.now(),
        end_time=datetime.now(),
        out_path=temp_manifest_path
    )
    
    # Verification should succeed
    assert verify_manifest(temp_manifest_path, temp_config_path)
    
    # Store original hash
    original_hash = get_config_hash(temp_config_path)
    
    # Modify the config silently
    with open(temp_config_path, "a") as f:
        f.write("modified: true\n")
        
    modified_hash = get_config_hash(temp_config_path)
    assert original_hash != modified_hash
    
    # Verification should fail
    with pytest.raises(ValueError, match="Configuration drift detected"):
        verify_manifest(temp_manifest_path, temp_config_path)

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
        seed=123,
        start_time=datetime.now(),
        end_time=datetime.now(),
        out_path=temp_manifest_path
    )
    
    # Read back and validate using pydantic
    import json
    with open(temp_manifest_path, "r") as f:
        data = json.load(f)
        
    manifest = ManifestSchema(**data)
    assert manifest.run_id == run_id
    assert manifest.seed == 123
    assert manifest.config_hash == get_config_hash(temp_config_path)

def test_deterministic_fixture(deterministic_fixture_input):
    hash_1 = compute_deterministic_hash(deterministic_fixture_input)
    hash_2 = compute_deterministic_hash(deterministic_fixture_input)
    assert hash_1 == hash_2

def test_docker_reproducibility():
    # If container_hash.txt and host_hash.txt are produced by the test script, verify they match
    if os.path.exists("host_hash.txt") and os.path.exists("container_hash.txt"):
        with open("host_hash.txt") as f:
            host_hash = f.read().strip()
        with open("container_hash.txt") as f:
            container_hash = f.read().strip()
            
        assert host_hash == container_hash
