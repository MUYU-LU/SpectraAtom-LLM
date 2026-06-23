# SpectraAtom-LLM

SpectraAtom-LLM is a research prototype for connecting molecular spectra, atom-level MLIP representations, and an instruction-following LLM.

The current system is organized as three stages:

1. `phase1_muse_atombit_bridge`: align spectrum embeddings with AtomBit atom/environment embeddings.
2. `phase15_text_connector`: align continuous bridge tokens with short text descriptions using MatterChat-style ITC/ITM/LM objectives.
3. `phase2_llm_sft`: use bridge tokens as soft prefixes for Qwen instruction tuning.

Current status:

- Phase 1 on FG26/MUSE + AtomBit is strong on the local retrieval objective.
- Phase 1.5 has a usable checkpoint, but the latest FG26 run is missing standardized metrics files.
- Phase 2 is only smoke-tested. It learns JSON/schema formatting but not reliable molecule identity, spectra peaks, scalar values, or 3D geometry yet.

Start here:

- [Project overview](docs/project_overview.md)
- [Experiment timeline](docs/experiment_timeline.md)
- [Research hypotheses](docs/research_hypotheses.md)
- [System inventory](docs/system_inventory.md)
- [Legacy projects](docs/legacy_projects.md)
- [Naming conventions](docs/naming_conventions.md)
- [Artifact registry](manifests/artifact_registry.md)

Important: this repository should contain source code, configs, experiment manifests, and small examples. Large datasets, hidden caches, checkpoints, and cluster-local paths stay outside git and should be referenced through manifests.
