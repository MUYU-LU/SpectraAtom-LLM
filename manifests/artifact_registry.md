# Artifact Registry

This file records the current artifact names and locations. Keep this file updated when a run becomes a baseline.

## Datasets And Caches

| Artifact | Location | Notes |
| --- | --- | --- |
| FG26 raw data | `CLUSTER_PROJECT_ROOT/data/FG26` | Source data from the MUSE-related paper. |
| MUSE FG26 hidden cache | `CLUSTER_PROJECT_ROOT/data/muse_fg26_ir_raman_uv_hidden_full_fp16_v1` | Spectrum hidden states for IR/Raman/UV. |
| AtomBit norm-ft cache | `CLUSTER_PROJECT_ROOT/data/alignment_cache/fg26_atombit_normft_epoch3_omol25_raw_l012_v1` | h0/h1/h2 raw cache from `Norm_ft_from_epoch3.ckpt`; h1/h2 norms can be derived. |
| Phase 1 bridge tokens | `CLUSTER_PROJECT_ROOT/data/bridge_outputs/fg26_muse_phase1_tokens_best_h0h1h2norm_fp16_v1` | Exported Z tokens for downstream connector/LLM. |
| Phase 1.5 text pairs | `CLUSTER_PROJECT_ROOT/data/phase15_matterchat_v4` | Current connector training data family. |
| Phase 2 instruction data | `CLUSTER_PROJECT_ROOT/data/phase2_fg26_with_peaks_v2_1_clean` | Cleaned Phase 2 data with peak tasks. |

## Baseline Runs

| Stage | Run | Status | Key metric |
| --- | --- | --- | --- |
| Phase 1 | `CLUSTER_PROJECT_ROOT/runs/fg26_muse_atombit_bridge_node_local_ddp64_h0h1h2norm_ep20_nw0_norc_20260616` | current baseline | Spectrum-to-molecule Recall@1 = 0.992476 on local validation. |
| Phase 1.5 | `CLUSTER_PROJECT_ROOT/runs/phase15_fg26_muse_v1_ddp48_ep20_balanced_preload_20260622_1120` | current checkpoint, metrics need standardization | Checkpoint exists at `best/phase15_v2_qformer.pt`. |
| Phase 2 | `CLUSTER_PROJECT_ROOT/runs/phase2_fg26_v21_qwen25_7b_ddp48_smoke_20260623_1125` | smoke only | Eval loss 0.660457 at step 1000; generation quality still weak. |

## Model Checkpoints

| Model | Location | Notes |
| --- | --- | --- |
| Qwen2.5-7B-Instruct | `CLUSTER_PROJECT_ROOT/models/Qwen2.5-7B-Instruct` | Current Phase 2 LLM. |
| BERT base uncased | `CLUSTER_PROJECT_ROOT/models/bert-base-uncased` | Used for Q-Former/BERT-style connector initialization. |
| MUSE model | `CLUSTER_PROJECT_ROOT/models/muse` | Spectrum encoder source for FG26. |
| AtomBit norm-ft checkpoint | `COMPUTE11_ATOMBIT_FT_ROOT/checkpoints/Norm_ft_from_epoch3.ckpt` | Used for latest AtomBit h0/h1/h2 cache. |

## Open Issues

- Phase 1.5 latest FG26 run needs standardized `best/metrics.json`.
- Phase 2 needs task-level generation metrics before any formal claim.
- 3D generation requires a separate length policy and stricter geometric validation.
- Old non-current spectrum-prompt results should be kept out of the current MUSE/FG26 baseline.
