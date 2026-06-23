# Naming Conventions

Names should be understandable without reading the whole codebase.

## Phase Names

Use these canonical names:

- `phase1_muse_atombit_bridge`
- `phase15_text_connector`
- `phase2_llm_sft`
- `phase2_non3d_sft`
- `phase2_3d_sft`

Avoid ambiguous names:

- `v4`
- `v4_1`
- `new`
- `final`
- `test2`

If version labels are unavoidable, attach a semantic suffix:

- `phase15_text_connector_v4_short_views`
- `phase2_non3d_sft_v21_clean_json`
- `phase2_3d_sft_compact_xyz_v1`

## Run Directory Format

Recommended format:

```text
runs/YYYYMMDD_<phase>_<data>_<model>_<key_config>
```

Examples:

```text
runs/20260616_phase1_muse_atombit_fg26_rk8_d256_h0h1h2norm
runs/20260622_phase15_text_connector_fg26_muse_balanced_shortviews
runs/20260623_phase2_non3d_sft_qwen25_7b_len512_connector
runs/20260623_phase2_3d_sft_qwen25_7b_len2304_compactxyz
```

## Required Files Per Run

Every run should save:

```text
train_config.json
history.jsonl
best/metrics.json
best/checkpoint.pt
final/metrics.json
eval_summary.json
```

If generation is evaluated:

```text
generation_eval/
  predictions.jsonl
  summary.json
  examples_good.jsonl
  examples_bad.jsonl
```

## Data Directory Names

Use names that encode source, representation, and cleaning state:

```text
data/fg26_muse_hidden_ir_raman_uv_fp16_v1
data/fg26_atombit_cache_normft_omol25_h0h1h2_raw_v1
data/fg26_phase15_text_pairs_short_views_v1
data/fg26_phase2_non3d_instructions_clean_v1
data/fg26_phase2_3d_instructions_compact_xyz_v1
```

Do not name core datasets only as:

```text
data/v1
data/v2_clean
data/new
```

## Task Type Names

Use task names that describe input and output:

```text
spectrum_to_paired_formula
spectrum_to_paired_smiles
molecule_to_formula
molecule_to_smiles
molecule_scalar_homo_energy
spectrum_to_ir_peaks
spec_mol_verify_match
spec_mol_rank_k4
molecule_to_3d_conformer
spectrum_to_3d_conformer
```

For future relative tasks:

```text
compare_two_molecules_lower_energy_same_formula
rank_candidates_by_spectrum_consistency
verify_candidate_against_spectrum
```

## Metric Names

Use explicit metric names:

```text
json_parse_rate
schema_valid_rate
formula_exact
smiles_exact
smiles_canonical_exact
scalar_mae
scalar_rmse
peak_position_mae
peak_count_accuracy
selected_id_accuracy
same_formula_rank_accuracy
z_shuffle_drop
```

Avoid generic metrics without task context:

```text
acc
score
good
result
```
