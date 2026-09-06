FROM python:3.11.9-slim

WORKDIR /app

# Ensure we're running CPU-first
ENV PIP_NO_CACHE_DIR=1

COPY requirements.txt .

# Install from locked requirements only
RUN pip install -r requirements.txt

COPY . .

# Generate a deterministic hash based on deterministic fixture
CMD ["python", "smoke_test.py"]
