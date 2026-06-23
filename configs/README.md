# Configs

Training and data-generation configs should live here.

Recommended names:

```text
phase1_muse_atombit_bridge.yaml
phase15_text_connector.yaml
phase2_non3d_sft.yaml
phase2_3d_sft.yaml
```

Every config should include:

- input artifact manifests,
- output run directory,
- model dimensions,
- optimizer and schedule,
- batch size and world size assumptions,
- evaluation frequency,
- random seed.

