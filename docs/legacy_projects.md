# Legacy Projects

This document defines how the old compute11 project and the current 910-cluster project should be treated while this repository becomes the clean source of truth.

## Principle

Do not move, rename, or delete old remote project directories during cleanup.

Old directories are experiment archives. They contain working scripts, local paths, checkpoints, logs, and partially failed attempts. They are useful for audit and reproduction, but they are not clean source code.

The new repository should contain:

- cleaned source code,
- configs,
- data manifests,
- experiment notes,
- small examples,
- migration notes.

The new repository should not contain:

- raw FG26 datasets,
- hidden-state caches,
- AtomBit caches,
- Qwen/MUSE/BERT checkpoints,
- large training checkpoints,
- remote credentials.

## Legacy Roots

## Private Path Aliases

Real hostnames, IPs, usernames, and credentials are kept outside git.

```text
COMPUTE11_PROJECT_ROOT=<compute11 project root>
CLUSTER_PROJECT_ROOT=<910 cluster project root>
```

### compute11

```text
COMPUTE11_PROJECT_ROOT
```

Role:

- old AtomBit cache scripts,
- early bridge experiments,
- Phase 1.5 and Phase 2 prototypes,
- old evaluation scripts and generation samples.

Approximate inventory from the last audit:

```text
scripts: 81
runs:    64
```

Important old data directories:

```text
data/FG26/
data/alignment_cache/
data/alignment_cache_merged/
data/bridge_outputs/
data/muse_fg26_ir_raman_uv_hidden_full_fp16_v1/
data/phase15_matterchat_v2/
data/phase15_matterchat_v3/
data/phase15_matterchat_v4/
data/phase2_matterchat_v1/
data/phase2_matterchat_v2/
data/phase2_matterchat_v3/
data/phase2_matterchat_v3_1/
data/phase2_matterchat_v3_2/
data/phase2_matterchat_v4/
data/phase2_sft/
```

Representative old runs:

```text
runs/v21_bridge_A4b_hardneg_h0_logh1h2norm_d512_RK16_from_A3_ddp8_20260607_0215
runs/v21_bridge_normft_epoch3_A4b_ft10_from_A4b_ddp8_20260612_172200
runs/phase15_v3_matterchat_ddp8_ep6_from_v2best_20260609_084119
runs/phase2_qwen25_7b_ddp8_v2clean_full3ep_20260610_0315
runs/phase2_qwen25_7b_ddp8_v3_1_ep2_20260610_130718
```

### 910 Cluster

```text
CLUSTER_PROJECT_ROOT
```

Role:

- current FG26/MUSE work,
- current AtomBit norm-ft cache,
- current Phase 1 bridge,
- current Phase 1.5 connector,
- current Phase 2 Qwen SFT smoke tests.

Approximate inventory from the last audit:

```text
scripts: 98
runs:    28
```

Important current data directories:

```text
data/muse_fg26_ir_raman_uv_hidden_full_fp16_v1/
data/alignment_cache/fg26_atombit_normft_epoch3_omol25_raw_l012_v1/
data/bridge_outputs/fg26_muse_phase1_tokens_best_h0h1h2norm_fp16_v1/
data/phase15_matterchat_v4/
data/phase2_fg26_with_peaks_v2_1_clean/
```

Representative current runs:

```text
runs/fg26_muse_atombit_bridge_node_local_ddp64_h0h1h2norm_ep20_nw0_norc_20260616
runs/phase15_fg26_muse_v1_ddp48_ep20_balanced_preload_20260622_1120
runs/phase2_fg26_v21_qwen25_7b_ddp48_smoke_20260623_1125
```

## Sorting Policy

Use this policy to decide what to migrate.

### Keep As Archive

Keep old directories in place when they are:

- historical failed attempts,
- old non-current spectrum experiments,
- one-off debug scripts,
- temporary check scripts,
- large cache generation outputs,
- raw training runs with non-standard names.

These should only be referenced in docs.

### Migrate Into This Repo

Migrate scripts when they are part of the current reproducible pipeline:

- MUSE/FG26 hidden export,
- AtomBit h0/h1/h2 cache export,
- Phase 1 bridge training/evaluation/export,
- Phase 1.5 text connector data/training/evaluation,
- Phase 2 instruction data/training/evaluation,
- generation and metric evaluation tools.

Migration requirements:

- Rename script to a descriptive name.
- Replace hard-coded absolute paths with CLI args or YAML config.
- Remove cluster-specific defaults.
- Add a config example.
- Add output schema and required artifact paths.
- Add a short experiment note.

### Deprecate

Deprecate scripts when they:

- depend on non-current spectrum prompt assumptions,
- generate ambiguous or leaked instruction labels,
- use old Phase 2 data with `qm9s_index` in user-visible text,
- hard-code unsuitable prompt templates,
- require manual edits between runs.

Keep deprecated scripts out of the clean repo unless they are needed for audit.

## Migration Waves

### Wave 1: Documentation And Manifests

Goal: make the current state understandable without moving large files.

Deliverables:

- `docs/project_overview.md`
- `docs/experiment_timeline.md`
- `docs/system_inventory.md`
- `docs/legacy_projects.md`
- `legacy/script_mapping.md`
- `manifests/artifact_registry.md`

### Wave 2: Phase 1 Source

Goal: reproduce the current MUSE/AtomBit bridge.

Target scripts:

```text
scripts/export_muse_fg26_hidden.py
scripts/build_fg26_atombit_cache.py
scripts/train_phase1_muse_atombit_bridge.py
scripts/eval_phase1_retrieval.py
scripts/export_phase1_bridge_tokens.py
```

### Wave 3: Phase 1.5 Source

Goal: reproduce MatterChat-style connector pretraining.

Target scripts:

```text
scripts/build_phase15_text_pairs.py
scripts/train_phase15_text_connector.py
scripts/eval_phase15_connector.py
```

### Wave 4: Phase 2 Source

Goal: reproduce Qwen soft-prefix SFT and evaluation.

Target scripts:

```text
scripts/build_phase2_instruction_data.py
scripts/train_phase2_llm_sft.py
scripts/eval_phase2_generation.py
```

### Wave 5: Cleanup

Goal: make each experiment one hypothesis, one config, one output directory.

Deliverables:

- config templates,
- launch templates,
- standardized `history.jsonl`,
- standardized `metrics.json`,
- standardized generation reports.
