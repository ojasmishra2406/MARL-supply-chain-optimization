# Phase 10: GNN-MAPPO Architecture

Date: 2026-09-23

## 1. Architecture Design

The supply chain is modeled as a connected graph according to material flow:
**Factory → Distributor → Wholesaler → Retailer**

### GNN Encoder (`rl.gnn_network.SupplyChainGNNEncoder`)
- Adjacency represents causal supply dependencies (upstream -> downstream mapping).
- Node embeddings are processed via Graph Convolutional Networks (GCN) to share state representations explicitly across adjacent echelons.

### MAPPO Integration
- **Centralized Critic**: The critic receives the full global graph state. A global readout mechanism aggregates the node embeddings into a single value baseline.
- **Decentralized Actor**: Enforces CTDE. The actor uses standard MLP over local node observations to strictly isolate individual echelon decisions.

## 2. Experimental Plan (Configured, GPU Pending)

- **Baselines**: `MAPPO` vs `GNN-MAPPO`
- **Scenarios**: `in_distribution` and all 5 distribution-shift scenarios.
- **Training Seeds**: 5 identical random seeds to ensure matched starting conditions.
- **Evaluation Seeds**: 5 per scenario.

## 3. Implementation Status
**IMPLEMENTED & TESTED**
- Graph construction matches topology correctly.
- Gradient flows into GNN successfully tested.
- CPU Smoke tests passed.

**EXPERIMENTAL VALIDATION**: `BLOCKED` (by 40+ hours required GPU execution time; reserved for future asynchronous cluster deployment)
