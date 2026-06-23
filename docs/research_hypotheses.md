# Research Hypotheses

The project should be run as a sequence of falsifiable hypotheses rather than a collection of ad hoc training jobs.

## H1: Spectrum Embeddings Can Be Aligned With AtomBit Atom/Environment Embeddings

Claim:

- Frozen MUSE spectrum representations and frozen AtomBit atom/environment embeddings contain enough shared chemical information for spectrum-molecule retrieval.

Experiment:

- Train `phase1_muse_atombit_bridge` with contrastive/ranking loss.
- Use MUSE IR/Raman/UV hidden states and AtomBit `h0 + norm(h1/h2)` features.

Primary metrics:

- Spectrum-to-molecule Recall@1/5/10.
- Molecule-to-spectrum Recall@1/5/10.
- MRR.
- Same-formula hard-negative ranking.
- Shuffle-Z or wrong-pair ablation.

Current evidence:

- FG26 run with 29,250 records reached local spectrum-to-molecule Recall@1 = 0.9925.

Remaining risk:

- Random split and local retrieval may overestimate true chemical identification.
- Need same-formula and scaffold-similar candidate pools.

## H2: Phase 1 Bridge Tokens Can Be Mapped Into Text Space

Claim:

- `Z_spec` and `Z_mol` can be connected to short text descriptions through MatterChat-style connector training.

Experiment:

- Train Phase 1.5 Q-Former connector with ITC/ITM/LM.
- Use short, mostly unique text views.

Suitable ITC tasks:

- molecule canonical SMILES,
- spectrum paired canonical SMILES with lower weight or separate audit,
- identity-level descriptions that are relatively unique.

Suitable ITM tasks:

- molecule SMILES match/mismatch,
- spectrum paired SMILES match/mismatch,
- short peak-region descriptions,
- formula/count views as auxiliary, not as the strongest contrastive target.

Suitable LM tasks:

- all short views, but each view should stay short and focused.

Primary metrics:

- Z-to-text Recall@1/5.
- Text-to-Z Recall@1/5.
- ITM accuracy.
- LM teacher-forcing loss and free-running generation sanity.

Current evidence:

- Older compute11 Phase 1.5 v3 achieved Z-to-text R@1 = 0.9792 and text-to-Z R@1 = 0.9816.
- Latest FG26 Phase 1.5 checkpoint exists, but standardized metrics are missing and must be regenerated.

Remaining risk:

- Formula and atom-count captions are not unique and can create false negatives in ITC.
- Peak captions may not uniquely identify a molecule.

## H3: LLM SFT Can Use Soft Prefix Tokens To Answer Chemistry Instructions

Claim:

- Qwen can use soft prefixes derived from `Z_spec`/`Z_mol` to answer structured chemical instructions.

Experiment:

- Phase 2 SFT with Qwen2.5-7B-Instruct.
- Frozen Qwen first; train connector and projection modules.
- Later compare against LoRA-enabled Qwen.

Primary metrics:

- JSON parse rate.
- Schema validity.
- Formula exact accuracy.
- SMILES exact/canonical accuracy.
- Scalar MAE/RMSE.
- Peak MAE/RMSE.
- Candidate selection accuracy under same-formula/hard negatives.
- Shuffle/drop-Z ablation.

Current evidence:

- 1,000-step smoke learned output schema but not reliable semantics.
- JSON parse rate: 0.94.
- Semantic exact: 0.02.

Interpretation:

- Current smoke is only an execution test.
- It is not a meaningful final Phase 2 model.

## H4: 3D Coordinate Generation Needs Separate Treatment

Claim:

- Direct free-form 3D JSON generation is possible only if length, format, and evaluation are handled separately.

Experiment:

- Split 3D tasks from short tasks.
- Use `max_text_len=2304` or compact coordinate output.
- Evaluate parse validity before RMSD.

Primary metrics:

- JSON parse rate.
- Atom-count accuracy.
- Element-count accuracy.
- Coordinate completeness.
- RMSD after valid parse and atom matching.

Current evidence:

- With `max_text_len=1024`, about 27% of 3D training examples are truncated.
- Few-sample generation produced invalid JSON or collapsed near-zero coordinates.

Recommendation:

- Do not mix 3D generation into the same run as short tasks with `max_text_len=512`.
- Use a dedicated 3D run.

## Next Experimental Sequence

1. Re-run FG26 Phase 1.5 with standardized metrics saved.
2. Train Phase 2 non-3D with `max_text_len=512`.
3. Add shuffle/drop-Z ablation to prove the LLM uses soft tokens.
4. Run Phase 2 3D separately with compact coordinate format and `max_text_len=2304`.
5. Add same-formula candidate protocols for ranking and verification tasks.
