# Phase 2: LLM SFT

## Hypothesis

Qwen can use soft prefixes from `Z_spec` and `Z_mol` to answer chemistry instructions.

## Current Smoke Run

Path on current cluster:

```text
CLUSTER_PROJECT_ROOT/runs/phase2_fg26_v21_qwen25_7b_ddp48_smoke_20260623_1125
```

Configuration:

```text
model: Qwen2.5-7B-Instruct
Qwen trainable params: 0
connector trainable params: 102,224,640
world_size: 48
max_train_samples: 48,000
max_val_samples: 4,800
max_steps: 1,000
max_text_len: 1,024
learning_rate: 1e-4
warmup_ratio: 0.005
```

Eval loss:

```text
step 200: 0.7224
step 400: 0.6820
step 600: 0.6660
step 800: 0.6606
step 1000: 0.6605
```

Generation sanity:

```text
n: 50
json_parse_rate: 0.94
schema_valid_rate: 0.94
semantic_exact: 0.02
```

## Interpretation

The smoke run proves that the data path, connector loading, Qwen forward pass, and generation path are functional.

It does not prove semantic grounding.

Observed issues:

- SMILES generation is unreliable.
- Formula generation is weak.
- Scalar values show rough tendencies but are not reliable.
- Peak positions are poor.
- 3D generation fails due to both undertraining and truncation.

## Length Policy

Non-3D tasks:

```text
max_text_len = 512
```

3D tasks:

```text
max_text_len = 2304
```

Reason:

- Non-3D max length is about 150 tokens in sampled data.
- 3D max length is about 2,133 tokens.
- `max_text_len=1024` truncates about 27% of 3D examples.

## Next Run Plan

Run non-3D first:

```text
phase2_non3d_sft
max_text_len: 512
max_steps: 10k-20k
eval: formula/SMILES/scalar/peak/candidate metrics
required ablation: shuffle or drop Z tokens
```

Then run 3D separately:

```text
phase2_3d_sft
max_text_len: 2304
output format: compact XYZ JSON
metrics: parse rate, atom-count accuracy, element-count accuracy, RMSD after valid parse
```
