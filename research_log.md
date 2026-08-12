# Research log

## v0.8.0 — public-data adapter (Exploration)

- **Hypothesis:** a public example becomes more useful when users can inspect
  its coverage through the package, while preserving assay semantics.
- **Change:** added `lnp-agent --public-summary`, which reports counts and
  value ranges for the public LNPDB schema without renaming its measurements
  to LNPAgent-native endpoints.
- **Result:** the bundled 100-row example now has a reproducible, lightweight
  inspection path; no private data or model weights are required.
- **Scientific limitation:** this is not a benchmark and does not establish
  transfection, immune, or clinical performance. Native-model training still
  requires a native-schema dataset.
- **Decision:** retain this adapter and next test a one-round lightweight demo
  before changing the native data contract.
