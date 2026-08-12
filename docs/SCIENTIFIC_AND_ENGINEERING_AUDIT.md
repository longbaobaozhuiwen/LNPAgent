# Scientific and engineering audit

This public audit summarizes what LNPAgent currently supports, the main issues
identified during the open-source release review, and how the v0.8.0-v0.16.0
iterations resolved them without including private LNP data.

## Current scientific scope

- Closed-loop LNP design workflow: virtual library generation, prediction,
  exploitation/exploration selection, reporting, simulated measurement, and
  retraining states.
- Multi-objective candidate review: lower inflammation-related endpoints and
  higher transfection proxy can be ranked with Pareto-front diagnostics.
- Molecular and formulation features: formulation ratios, component identities,
  and Morgan-fingerprint feature paths are available for deployable baselines.
- Cross-validation and uncertainty primitives: leave-template-out splits,
  group-aware inner model selection, candidate-level ensemble dispersion,
  bootstrap intervals, and paired-delta utilities are present.
- Public-data workflow: the bundled LNPDB subset can be validated, summarized,
  and benchmarked for schema/coverage reproducibility.

## Scientific limitations

- The bundled LNPDB example is a public schema and assay-coverage example, not
  a native endpoint benchmark and not a formulation recommendation dataset.
- Synthetic wet-lab results come from `WetLabOracle`; they are software fixtures,
  not physical measurements or biological evidence.
- Candidate uncertainty is an ensemble-dispersion proxy and still needs
  calibration before probabilistic claims.
- Generalization to new chemistry, assay platforms, species, tissues, or cargo
  types remains an empirical question for future public datasets.

## Engineering issues resolved in this release cycle

- v0.8.0: added public LNPDB summary without relabeling assays as native endpoints.
- v0.9.0: labeled synthetic wet-lab outputs with explicit measurement provenance.
- v0.10.0: replaced global uncertainty proxy with candidate-level ensemble dispersion.
- v0.11.0: excluded non-finite objectives from Pareto-front membership.
- v0.12.0: removed fixed private-grid cardinality assertions and validated the
  transfection `log1p` domain.
- v0.13.0: used template-group-aware inner CV for tuned model configurations.
- v0.14.0: added reproducible public benchmark CLI output.
- v0.15.0: added provenance metadata to public benchmark artifacts.
- v0.16.0: published this audit and finalized the versioned research log cadence.

## Data boundary

The repository intentionally versions only the documented public LNPDB example
under `data/`. Full datasets, checkpoints, raw experimental outputs, private
measurements, and generated artifacts remain outside Git by default.
