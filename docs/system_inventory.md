# System Inventory

This file records where the current working artifacts live. Large artifacts should not be committed to git.

## Private Path Aliases

Real hostnames, IPs, usernames, and credentials are not committed to this public project documentation. Keep them in private operations notes.

Aliases used below:

```text
COMPUTE11_PROJECT_ROOT=<compute11 project root>
CLUSTER_PROJECT_ROOT=<910 cluster project root>
```

## compute11

Access:

```text
host: <compute11 host>
port: <compute11 ssh port>
user: <compute11 user>
```

Main project:

```text
COMPUTE11_PROJECT_ROOT
```

Important directories:

```text
third_party/MUSE/
third_party/matterchat_code/
models/Qwen2.5-7B-Instruct/
models/bert-base-uncased/
models/muse/
data/FG26/
data/alignment_cache/
data/bridge_outputs/
data/muse_fg26_ir_raman_uv_hidden_full_fp16_v1/
data/phase15_matterchat_v4/
data/phase2_matterchat_v4/
runs/
eval_reports/
generation_samples/
```

Representative old runs:

```text
runs/v21_bridge_A4b_hardneg_h0_logh1h2norm_d512_RK16_from_A3_ddp8_20260607_0215
runs/v21_bridge_normft_epoch3_A4b_ft10_from_A4b_ddp8_20260612_172200
runs/phase15_v3_matterchat_ddp8_ep6_from_v2best_20260609_084119
runs/phase2_qwen25_7b_ddp8_v2clean_full3ep_20260610_0315
runs/phase2_qwen25_7b_ddp8_v3_1_ep2_20260610_130718
```

Use compute11 mainly as:

- historical record,
- older MUSE/FG26 prototypes and supporting scripts,
- old Phase 1/1.5/2 prototypes,
- code source for earlier scripts.

## Current 910 Cluster

Access used during development:

```text
master: <cluster master host>
main project: CLUSTER_PROJECT_ROOT
```

Important directories:

```text
data/muse_fg26_ir_raman_uv_hidden_full_fp16_v1/
data/alignment_cache/fg26_atombit_normft_epoch3_omol25_raw_l012_v1/
data/bridge_outputs/fg26_muse_phase1_tokens_best_h0h1h2norm_fp16_v1/
data/phase2_fg26_with_peaks_v2_1_clean/
models/Qwen2.5-7B-Instruct/
models/bert-base-uncased/
third_party/matterchat_code/
scripts/
runs/
```

Key current runs:

```text
runs/fg26_muse_atombit_bridge_node_local_ddp64_h0h1h2norm_ep20_nw0_norc_20260616
runs/phase15_fg26_muse_v1_ddp48_ep20_balanced_preload_20260622_1120
runs/phase2_fg26_v21_qwen25_7b_ddp48_smoke_20260623_1125
```

Key current checkpoints:

```text
CLUSTER_PROJECT_ROOT/runs/fg26_muse_atombit_bridge_node_local_ddp64_h0h1h2norm_ep20_nw0_norc_20260616/best.pt
CLUSTER_PROJECT_ROOT/runs/phase15_fg26_muse_v1_ddp48_ep20_balanced_preload_20260622_1120/best/phase15_v2_qformer.pt
CLUSTER_PROJECT_ROOT/runs/phase2_fg26_v21_qwen25_7b_ddp48_smoke_20260623_1125/best/connector.pt
```

Use the current cluster mainly for:

- FG26/MUSE hidden cache,
- AtomBit norm-ft cache,
- Phase 1 bridge training,
- Phase 1.5 connector training,
- Phase 2 Qwen SFT smoke and future medium runs.

## Code Migration Policy

Before copying old scripts into this repo:

1. Rename the script to describe its function.
2. Remove absolute cluster-specific paths from defaults.
3. Add a config file or CLI arguments.
4. Keep large data/checkpoint paths in manifests.
5. Add a short experiment doc that states hypothesis, inputs, outputs, and metrics.

Preferred script names:

```text
scripts/export_muse_fg26_hidden.py
scripts/build_fg26_atombit_cache.py
scripts/train_phase1_muse_atombit_bridge.py
scripts/export_phase1_bridge_tokens.py
scripts/build_phase15_text_pairs.py
scripts/train_phase15_text_connector.py
scripts/build_phase2_instruction_data.py
scripts/train_phase2_llm_sft.py
scripts/eval_phase2_generation.py
```

Avoid script names such as:

```text
train_v2.py
new_train.py
test_final.py
run2.py
```
