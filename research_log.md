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

## v0.9.0 — explicit synthetic measurement provenance (Exploration)

- **Hypothesis:** lab-like output must be impossible to confuse with measured data.
- **Change:** synthetic wet-lab outputs now carry explicit type and provenance columns.
- **Result:** downstream reports can distinguish oracle simulation from physical measurements.
- **Decision:** retain the simulator as a test fixture, never as evidence.
