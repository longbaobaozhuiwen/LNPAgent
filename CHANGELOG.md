# Changelog

## v0.25.0 - 2026-08-15

### Added

- Agent-callable adapters for local COMET efficacy/stability inference and
  LaMGen dual/triple protein-conditioned molecular generation.
- `lnp-agent --external-tools-status` for checking optional upstream runtime
  configuration before an execution request.
- `lnp-agent --comet-predict` and `lnp-agent --lamgen-generate` commands that
  write structured local manifests beside external-tool outputs.
- Packaging metadata versioned as `0.25.0` and integration documentation.

### Boundaries

- This release does not bundle COMET/LaMGen source, weights, checkpoints,
  protein embeddings, generated molecules, or private LNP data.
- COMET and LaMGen retain their own upstream terms and scientific limitations.
