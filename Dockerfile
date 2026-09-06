FROM python:3.11.9-slim

WORKDIR /app

# Ensure we're running CPU-first
ENV PIP_NO_CACHE_DIR=1

COPY requirements.txt .

# Install from locked requirements only
RUN pip install -r requirements.txt

COPY . .

# Generate a deterministic hash based on deterministic fixture
# We'll use tests.conftest's deterministic_fixture_input
CMD ["python", "-c", "from tests.conftest import compute_deterministic_hash, deterministic_fixture_input; import pytest; print(compute_deterministic_hash(deterministic_fixture_input()))"]
