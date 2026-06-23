# Project Overview

## Goal

Build an agent that can reason over molecular spectra and molecular structures by combining:

- a spectrum encoder,
- AtomBit/MOLE atom or environment embeddings,
- a trainable multimodal bridge,
- and an LLM interface for instruction-style tasks.

The intended input/output modes include:

- spectrum to molecule identity,
- molecule structure to properties,
- spectrum plus candidate structures to verification/ranking,
- spectrum or molecule to peak/property summaries,
- and eventually spectrum or molecule to 3D conformer text output.

## Current Architecture

### Phase 1: Spectrum to AtomBit Bridge

Frozen inputs:

- Spectrum side: MUSE/FG26 IR + Raman + UV hidden states.
- Atom side: AtomBit `h0`, with invariant norms from `h1/h2`.

Trainable bridge:

- spectrum query tokens,
- molecule query tokens,
- projection heads,
- global and fine-grained spectrum-to-atom scoring.

Primary objective:

- contrastive alignment and ranking over true spectrum-molecule pairs.

### Phase 1.5: Text Connector

Frozen/loaded inputs:

- Phase 1 bridge tokens `Z_spec` and `Z_mol`.

Trainable connector:

- Q-Former-style connector initialized from BERT/MatterChat-style code.

Objectives:

- ITC: contrastive alignment for unique text views.
- ITM: binary match/mismatch alignment.
- LM: teacher-forcing generation of short captions.

Purpose:

- map continuous bridge tokens into a text-aligned latent space before LLM SFT.

### Phase 2: LLM Instruction SFT

Inputs:

- soft prefix from `Z_spec`, `Z_mol`, or both,
- user prompt text,
- assistant JSON answer.

Model:

- Qwen2.5-7B-Instruct frozen in the latest smoke run.
- Phase 1.5 Q-Former connector remains trainable.
- `llm_proj`, `out_norm`, and prefix type embeddings are trainable.

Training loss:

- teacher-forcing next-token cross entropy on assistant answer tokens.

Evaluation:

- JSON parse/schema validity,
- exact formula/SMILES where appropriate,
- scalar MAE/RMSE,
- peak MAE/RMSE,
- candidate selection/ranking accuracy,
- 3D parse rate, atom-count accuracy, element-count accuracy, and RMSD only after valid parse.

## Key Design Lessons

- Free-form 3D JSON is too long and should not be trained together with short tasks at the same context length.
- Non-3D tasks fit safely in `max_text_len=512`.
- Current 3D tasks need approximately `max_text_len=2304` or a compact output format.
- Ranking tasks with candidates are weak evidence unless paired with shuffle/drop-Z ablations and identity-field checks.
- Phase 2 smoke loss alone is not enough; generation and ablation tests are required.
