import hashlib
import json


def compute_deterministic_hash(data: dict) -> str:
    serialized = json.dumps(data, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


if __name__ == "__main__":
    print(
        compute_deterministic_hash(
            {"learning_rate": 0.001, "batch_size": 32, "seed": 42}
        )
    )
