# Script Mapping

This file maps old script names to clean target names. The target names are the ones that should exist in this repository after migration.

## Phase 1: Spectrum Atom Bridge

| Old script or family | Source root | Clean target | Status |
| --- | --- | --- | --- |
| `export_muse_fg26_hidden.py` | cluster | `scripts/export_muse_fg26_hidden.py` | migrate |
| `build_fg26_atombit_cache_ft_raw.py` | compute11/cluster | `scripts/build_fg26_atombit_cache.py` | migrate |
| `train_fg26_muse_atombit_bridge.py` | cluster | `scripts/train_phase1_muse_atombit_bridge.py` | migrate |
| phase1 eval helpers | cluster | `scripts/eval_phase1_retrieval.py` | consolidate |
| phase1 bridge export helpers | cluster | `scripts/export_phase1_bridge_tokens.py` | consolidate |

Required cleanup:

- expose spectrum cache path, AtomBit cache path, split file, output dir, and run name as CLI args,
- remove hard-coded node-local paths,
- write `train_config.json`,
- write `history.jsonl`,
- write `best/metrics.json`.

## Phase 1.5: Text Connector

| Old script or family | Source root | Clean target | Status |
| --- | --- | --- | --- |
| `phase15_qformer.py` | compute11/cluster | `src/spectra_atom_llm/models/phase15_connector.py` | migrate |
| `train_phase15_v2_matterchat.py` | compute11 | `scripts/train_phase15_text_connector.py` | superseded |
| `train_phase15_v41_matterchat.py` | cluster | `scripts/train_phase15_text_connector.py` | migrate current logic |
| phase15 data builders | cluster | `scripts/build_phase15_text_pairs.py` | consolidate |
| phase15 eval helpers | cluster | `scripts/eval_phase15_connector.py` | consolidate |

Required cleanup:

- separate text-type registry from code,
- keep ITC/ITM/LM eligibility explicit per text type,
- keep examples short and non-concatenated,
- remove obsolete non-current spectrum templates from current FG26 path.

## Phase 2: LLM SFT

| Old script or family | Source root | Clean target | Status |
| --- | --- | --- | --- |
| `train_phase2_qformer_sft.py` | compute11 | `scripts/train_phase2_llm_sft.py` | migrate cautiously |
| `train_phase2_qformer_sft_ddp.py` | compute11/cluster | `scripts/train_phase2_llm_sft.py` | migrate current DDP logic |
| `build_phase2_fg26_with_peaks_v1_data.py` | cluster | `scripts/build_phase2_instruction_data.py` | superseded |
| `build_phase2_fg26_with_peaks_v2_data.py` | cluster | `scripts/build_phase2_instruction_data.py` | migrate cleaned logic |
| `convert_phase2_fg26_v1_clean_to_v2.py` | cluster | fold into builder | remove as standalone |
| `convert_phase2_fg26_v2_to_v21.py` | cluster | fold into builder | remove as standalone |
| phase2 generation eval helpers | cluster | `scripts/eval_phase2_generation.py` | consolidate |

Required cleanup:

- remove system prompts that mention soft tokens,
- keep user prompts natural,
- keep internal paths and sample IDs out of answer supervision,
- split non-3D and 3D length policies,
- add task-level metrics.

## Deprecated Or Archive Only

| Old script family | Reason |
| --- | --- |
| Old non-current spectrum prompt generation | Axis/prompt assumptions were not reliable enough for current peak-level work. |
| Old Phase 2 data with user-visible IDs | Risk of leakage and trivial learning. |
| One-off `check_*`, `tmp_*`, `debug_*` scripts | Useful for audit only, not clean source. |
| Early mean-pooling bridge scripts | Superseded by multi-query bridge. |
