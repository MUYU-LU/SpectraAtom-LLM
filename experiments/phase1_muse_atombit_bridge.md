# Phase 1: MUSE-AtomBit Bridge

## Hypothesis

Frozen MUSE spectrum embeddings and frozen AtomBit atom/environment embeddings can be aligned through a lightweight bridge.

## Current Best Run

Path on current cluster:

```text
CLUSTER_PROJECT_ROOT/runs/fg26_muse_atombit_bridge_node_local_ddp64_h0h1h2norm_ep20_nw0_norc_20260616
```

Configuration:

```text
records: 29,250
train/val: 27,788 / 1,462
spec_dim: 768
atom_dim: 384
atom_feature_mode: h0_h1h2_norm
proj_dim: 256
num_spec_queries: 8
num_mol_queries: 8
trainable_params: 305,923
world_size: 64
effective_batch_size: 2,048
epochs: 20
optimizer_steps: 2,160
lr: 2e-4
```

Result:

```text
best spectrum-to-molecule Recall@1: 0.9925
```

## Interpretation

This result supports the Phase 1 alignment hypothesis on the current local validation split.

## Required Follow-Up

- Evaluate same-formula hard negatives.
- Evaluate scaffold-similar candidates.
- Add Z-shuffle and pair-shuffle ablations.
- Save a standardized retrieval report with all metrics and dataset split hashes.
