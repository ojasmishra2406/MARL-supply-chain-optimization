# MARL Supply-Chain Optimization

## Purpose
A portfolio-quality Multi-Agent Reinforcement Learning (MARL) project for optimizing supply-chain Key Performance Indicators (KPIs).

## Repository Structure
- `simulator/`: Supply-chain environment simulator
- `envs/`: PettingZoo wrappers for the simulator
- `agents/`: Reinforcement learning agents (PPO, MAPPO, etc.)
- `baselines/`: Heuristic and standard baselines (e.g., OUT, forecasting)
- `experiments/`: Scripts for running experiments and hyperparameter tuning
- `analysis/`: Notebooks and scripts for statistical analysis
- `dashboard/`: Streamlit dashboard for visualizing results
- `tests/`: Automated tests
- `configs/`: Configuration files (Note: Evaluation scenario data will live in a separate namespace and will not be reachable from training/tuning code)
- `manifests/`: Run reproducibility manifests

## Environment
- **Python**: 3.11.9
- **Dependency Management**: pip-tools with exact version locking and hashes.
- **Hardware**: CPU-first environment by default.

### Local Development Workflow
1. Initialize Python 3.11.9 virtual environment: `python -m venv .venv`
2. Install pip-tools: `pip install pip-tools`
3. Install dependencies from locked requirements: `pip install -r requirements.txt -r requirements-dev.txt`
   *(Optional)* To compile requirements yourself: `pip-compile --generate-hashes requirements.in`

## Reproducibility
The project adopts a two-tier reproducibility model:

### Tier 1 - Exact/Hash-Verified
Applies to the simulator, deterministic data pipeline, statistical analysis, and other non-neural deterministic computation.
**Guarantee**: True bit-identical and hash-verified reproduction when inputs, seed, and locked environment are equivalent. 

### Tier 2 - Statistical Bounds
Applies to PyTorch neural-network training and GPU/cuDNN computations.
**Guarantee**: Deterministic within a fixed hardware class and deterministic PyTorch settings (`torch.use_deterministic_algorithms(True)`).
*Note: Universal bit-identical GPU reproduction across arbitrary GPU architectures is not claimed. Tier 2 reproduction is evaluated against originally reported 95% confidence-interval bounds.*

## Phase 1: Deterministic Supply-Chain Simulator
The core simulator is a standalone, deterministic 4-echelon supply-chain domain (Retailer -> Wholesaler -> Distributor -> Manufacturer).

- **Canonical Configuration**: Uses a YAML config (`configs/phase1_simulator.yaml`) specifying lead times (2, 2, 3, 4), normal demand (mu=20, sigma=5) at the retailer, and fixed 100 units/step capacity.
- **State Model**: Echelon state is explicitly modeled with `inventory`, `backlog`, `pipeline_inventory`, `demand_history`, and `last_order`.
- **RNG & Reproducibility**: Uses a single NumPy `Generator` seeded from the manifest. Repeated runs with the same seed are byte-identical.
- **Test Result**: All functional, property-based (Hypothesis), and regression tests passed.
- **Coverage**: 100.0% coverage on simulator core.
- **Performance**: Benchmark measured at ~23,500 steps/sec (single-threaded).

### Manifest System
Every run generates a reproducible manifest tracking:
- Exact YAML SHA-256 configuration hash (computed over the exact YAML file bytes)
- Repository-relative path of the exact configuration YAML used for the run
- Git commit hash
- Seed (derived from that same configuration)
- Installed library versions
- Framework-independent hardware metadata

Verification resolves the stored configuration path and re-computes the SHA256 to detect configuration drift.

## Docker
**Docker is a reproducibility artifact and is not required for day-to-day development.**
- CPU-first `Dockerfile` is provided for running the reproducible environment.
- CUDA 12.1 is an optional variant (`Dockerfile.cuda`). It is not the default and is not tested in default CI.

## Weights & Biases (W&B)
- **Project Name**: `marl-supply-chain`
- **Authentication**: W&B credentials must be provided via the `WANDB_API_KEY` environment variable only. Never place credentials in source, YAML, README, GitHub Actions, manifests, committed `.env`, or Dockerfiles.

## Testing & CI
- Run tests: `pytest -q`
- Coverage requirement (>=90% for Phase 0 new code): `pytest --cov --cov-report=term-missing`
- Default CI (GitHub Actions) runs on a CPU-only environment.
