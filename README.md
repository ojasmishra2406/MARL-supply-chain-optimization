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

### Manifest System
Every run generates a reproducible manifest tracking:
- Exact YAML SHA-256 configuration hash (configuration drift detection)
- Git commit hash
- Seed
- Installed library versions
- Framework-independent hardware metadata

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
