# Fixed Decode Evaluation

This repository keeps fixed Phase 2 generation-evaluation samples so checkpoints can be compared with the same inputs.

## Eval Sets

Location:

```text
manifests/eval_sets/phase2_fg26_non3d_fixed_v1/
```

Files:

```text
val_3_per_task.jsonl
val_20_per_task.jsonl
```

`val_3_per_task.jsonl` is for quick checkpoint triage. `val_20_per_task.jsonl` is the standard fixed decode set.

These files were sampled from the non-3D Phase 2 validation JSONL with seed `20260623`.

## Decode Policy

Use deterministic greedy decoding:

```text
do_sample = false
temperature/top_p/top_k = unused
forced JSON prefix = false for the main metric
```

The evaluator extracts the first JSON object from the raw model output. JSON parse rate is reported separately from task correctness, because formatting failures and semantic failures are different issues.

## Main Metrics

- Formula and SMILES tasks: exact match, with optional canonical SMILES when RDKit is available.
- Peak tasks: count accuracy, sorted peak MAE/RMSE, and tolerance hit rate.
- Candidate verification: status accuracy.
- Candidate ranking: selected candidate accuracy and ranked-id exact match.
- Relative energy: selected candidate accuracy.
- Scalar tasks: per-task MAE/RMSE; do not use the aggregate scalar error across mixed units as a headline metric.

Generated predictions and summaries are checkpoint-specific and should be stored under `runs/fixed_decode_eval/`, not committed as source.
