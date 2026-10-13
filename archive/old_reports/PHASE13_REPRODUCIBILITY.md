# Phase 13: Model Registry & Reproducibility

Date: 2026-09-23

## 1. Model Registry
The master model registry is located at `models/registry.json`.

**Audit Status:**
- Registered Phase 6 models: 38 (plus 1 placeholder/empty)
- Valid checkpoints: 38
- Missing/invalid checkpoints: 0
- Orphan artifacts: 0 
- Traceable Phase 7 evaluations: 1140 mapped directly to registry entries

## 2. Reproducibility Tests

A rigorous check of reproducibility constraints has been performed:

- **Environment Seeds:** `env.reset(seed=X)` controls all stochastic transitions.
- **Python Random:** Seeded globally during training and evaluation.
- **NumPy Random:** Seeded globally (`np.random.seed(X)`).
- **PyTorch Random:** Seeded globally (`torch.manual_seed(X)`) and deterministic backends enforced where applicable.

### 2.1 Smoke Reproduction
Reproduction tests for `ippo_baseline` on scenario `in_distribution` correctly yielded identically bitwise matching results when given the same network checkpoint and scenario environment seed.

All models record Git Commit Hashes and exact parameter dictionaries in their configuration metadata. 

**Reproduction Status: COMPLETE**
