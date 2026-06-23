# Experiment Timeline

This document separates the older `compute11` work from the newer multi-node cluster work.

## Locations

### compute11

- Host: `<compute11 ssh endpoint>`
- Main project: `COMPUTE11_PROJECT_ROOT`

### Current cluster

- Master access used during development: `<cluster master ssh endpoint>`
- Main project: `CLUSTER_PROJECT_ROOT`

## Stage 0: Archived Non-Current Direction

Early work used a different spectrum pipeline. It is kept as private historical infrastructure only and is not part of the current GitHub migration.

Decision:

- Keep archived work outside this repo.
- Use FG26/MUSE as the current spectrum encoder direction.

Caution:

- Historical losses are not treated as final scientific evidence.

## Stage 1: FG26/MUSE Spectrum Encoder

Why changed:

- Spectral peak semantics are central to the project.
- We needed a spectrum representation where IR/Raman/UV positions are consistent with the data format.

Data/cache direction:

- FG26 data under project `data/FG26`.
- MUSE hidden cache:
  - `CLUSTER_PROJECT_ROOT/data/muse_fg26_ir_raman_uv_hidden_full_fp16_v1`

MUSE hidden precision:

- Hidden states were stored in fp16 for storage and training throughput.
- This is acceptable for bridge training, but exact spectral peak extraction should use the original spectrum arrays/metadata, not hidden states.

## Stage 2: Phase 1 FG26 MUSE-AtomBit Bridge

Current representative cluster run:

- Run: `runs/fg26_muse_atombit_bridge_node_local_ddp64_h0h1h2norm_ep20_nw0_norc_20260616`
- Records: 29,250
- Train/val split: 27,788 / 1,462
- Spectrum dimension: 768
- Atom feature dimension: 384
- Atom feature mode: `h0_h1h2_norm`
- Projection dimension: 256
- Spectrum queries: 8
- Molecule queries: 8
- Trainable parameters: 305,923
- World size: 64
- Effective batch size: 2,048
- Epochs: 20
- Optimizer steps: 2,160
- Learning rate: 2e-4
- Best spectrum-to-molecule Recall@1: 0.9925
- Elapsed time: 20,793 s

Interpretation:

- Phase 1 is currently the strongest part of the pipeline.
- The local retrieval objective is learnable with frozen MUSE and frozen AtomBit caches.

Open concern:

- The retrieval split is random and local to available FG26 records.
- Same-formula and hard-negative protocols should be formalized before claiming chemical identification performance.

## Stage 3: Phase 1.5 FG26 Text Connector

Current cluster checkpoint:

- Run directory: `runs/phase15_fg26_muse_v1_ddp48_ep20_balanced_preload_20260622_1120`
- Checkpoint: `best/phase15_v2_qformer.pt`
- Size: 781 MB

Known issue:

- The latest FG26 Phase 1.5 directory currently preserves the checkpoint but not standardized metrics JSON/history files.
- Future runs must always save `history.jsonl`, `best/metrics.json`, and `final/metrics.json`.

Intended Phase 1.5 data types:

- `mol_smiles`
- `spec_paired_smiles`
- formula and atom-count text views
- IR/Raman/UV major peak short views
- short property views where labels are reliable

Current design principle:

- Phase 1.5 should align `Z` space with text space.
- It should not become a large instruction-task mixture; Phase 2 is where text instructions drive task behavior.

## Stage 4: Phase 2 FG26 Qwen SFT Smoke

Current cluster smoke run:

- Run: `runs/phase2_fg26_v21_qwen25_7b_ddp48_smoke_20260623_1125`
- Model: Qwen2.5-7B-Instruct
- Qwen trainable parameters: 0
- Connector trainable parameters: 102,224,640
- World size: 48
- Train samples: 48,000
- Val samples: 4,800
- Global batch size: 48
- Max steps: 1,000
- Max text length: 1,024
- Learning rate: 1e-4 with cosine schedule ending at 1,000 steps

Eval loss:

- step 200: 0.7224
- step 400: 0.6820
- step 600: 0.6660
- step 800: 0.6606
- step 1000: 0.6605

Generation few-sample eval:

- Samples: 50
- JSON parse rate: 0.94
- Schema valid rate: 0.94
- Semantic exact: 0.02
- Formula accuracy over sampled formula fields: 0.1667
- SMILES exact: 0.0

Interpretation:

- The smoke run proves pipeline execution and schema learning.
- It does not prove semantic grounding in soft tokens.
- The model is not ready for molecule identity, peaks, scalar properties, or 3D generation.

## Length Findings

Non-3D tasks:

- Median total text length: about 89 tokens
- p95: about 126-127 tokens
- max: about 150-154 tokens
- `max_text_len=512` is safe.

3D conformer tasks:

- Median total length: about 850 tokens
- p95: about 1,299 tokens
- observed max: about 2,133 tokens
- with `max_text_len=1024`, about 27% of 3D examples are truncated.

Recommendation:

- Train non-3D tasks separately with `max_text_len=512`.
- Train 3D tasks separately with `max_text_len=2304`, or use a compact coordinate format.
