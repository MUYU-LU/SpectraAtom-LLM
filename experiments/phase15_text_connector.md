# Phase 1.5: Text Connector

## Hypothesis

Phase 1 bridge tokens can be aligned to short text views using MatterChat-style ITC/ITM/LM training.

## Current FG26 Checkpoint

Path on current cluster:

```text
CLUSTER_PROJECT_ROOT/runs/phase15_fg26_muse_v1_ddp48_ep20_balanced_preload_20260622_1120/best/phase15_v2_qformer.pt
```

Known issue:

```text
Checkpoint exists, but standardized metrics/history files are missing.
```

## Older Reference Result

Path on compute11:

```text
COMPUTE11_PROJECT_ROOT/runs/phase15_v3_matterchat_ddp8_ep6_from_v2best_20260609_084119
```

Metrics:

```text
loss: 0.2133
ITC: 0.0519
ITM: 0.0429
LM: 0.3949
ITM acc: 0.9873
Z->Text R@1: 0.9792
Text->Z R@1: 0.9816
```

## Required Follow-Up

- Re-run or re-evaluate the FG26 connector with saved metrics.
- Separate unique identity captions from non-unique captions.
- Do not treat formula-only text as strong all-batch ITC.
- Keep text views short and one-property-per-view.
