# Research log

## v0.20.0 - explicit endpoint objective weighting (Exploration)

- **Hypothesis:** the acquisition policy should express the biological design
  priority explicitly; equal averaging of delivery and immune objectives can
  hide whether a candidate is useful for the intended decision.
- **Change:** experiment-value scoring now accepts endpoint objective weights.
  The public demo records a default multi-objective policy of
  `tx_log1p=0.50`, `immune_signal_a=0.25`, and `immune_signal_b=0.25`.
- **Result:** regression tests verify that changing endpoint weights changes the
  exploitation score direction, while the existing exploration, diversity, and
  batch-complementarity layers remain available.
- **Scientific limitation:** these weights encode a decision preference, not a
  learned biological utility function; they require domain review and
  retrospective/prospective validation.
- **Decision:** retain explicit weights as the public policy interface and next
  compare policy sensitivity across held-out rounds.

## v0.19.0 - batch-aware acquisition policy (Exploration)

- **Hypothesis:** an experiment round should be selected as a batch, not as
  independent top-ranked rows, because redundant candidates can waste wet-lab
  capacity even when each row has a high individual experiment-value score.
- **Change:** `select_experiment_value_batch` now uses greedy batch construction:
  the first candidate follows experiment-value score, and later candidates are
  penalized for similarity to already selected candidates across formulation
  ratios and design labels.
- **Result:** release smoke tests now verify that a slightly lower-scoring but
  complementary candidate can displace a near-duplicate candidate in a two-item
  batch. The public demo artifact also records the acquisition policy name,
  batch-level term, candidate batch scores, and mean selected complementarity.
- **Scientific limitation:** this is still a heuristic diversity penalty, not a
  calibrated expected information gain or D-optimal design objective.
- **Decision:** keep the batch-aware policy as the next public algorithm layer
  and use a held-out or redistribution-safe dataset to compare it against
  pure top-N and uncertainty-only acquisition.

## v0.18.0 - public acquisition diagnostics (Exploration)

- **Hypothesis:** an acquisition policy should produce reviewable diagnostics,
  not only a selected batch, so users can inspect whether a round was driven by
  exploitation, uncertainty, or design-space coverage.
- **Change:** the public one-round demo now includes retrospective mechanics
  diagnostics: score/public-assay Spearman, selected-vs-pool public assay mean
  delta, selected lipid diversity, and rationale counts.
- **Result:** `lnp-agent --demo-public` writes an artifact that exposes the
  policy's behavior while keeping `measurement_type=synthetic_public_demo` and
  `private_data_included=false`.
- **Scientific limitation:** these diagnostics are not performance validation;
  the synthetic demo score is derived from public table structure and must be
  replaced by held-out experimental rounds before making information-gain
  claims.
- **Decision:** keep this diagnostic layer and next add calibration against a
  native-schema held-out round or another redistribution-safe public dataset.

## v0.17.0 - experiment-value acquisition policy (Exploration)

- **Hypothesis:** the agent should rank candidates by experiment value rather
  than by a single predicted endpoint order, so batch selection can deliberately
  trade off exploitation, uncertainty-driven exploration, and design-space
  coverage.
- **Change:** added `compute_experiment_value_scores` and
  `select_experiment_value_batch` to `lnp_core.candidate_ranking`. Candidate
  outputs now include exploitation, exploration, diversity, combined
  `experiment_value_score`, and a short selection rationale.
- **Result:** the release smoke test now verifies that changing policy weights
  can select an uncertain, under-sampled candidate over a purely exploitative
  predicted winner.
- **Scientific limitation:** this is still a heuristic acquisition policy, not
  calibrated expected information gain. It needs comparison against retrospective
  held-out rounds and then prospective experimental rounds before becoming a
  central scientific claim.
- **Decision:** retain this as the first explicit algorithm-policy layer and
  next connect it to a public-safe one-round demo plus calibration diagnostics.

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

## v0.10.0 — candidate-level ensemble dispersion (Exploration)

- **Hypothesis:** model disagreement across candidate predictions is more useful
  for exploration than assigning every row the same fold-level metric spread.
- **Change:** candidate ranking now records per-row dispersion from a small OOF
  ensemble, with the benchmark spread retained only as fallback.
- **Decision:** keep this as an uncertainty proxy and calibrate it against held-out
  measurements before making probabilistic claims.

## v0.11.0 — invalid-objective exclusion (Consolidation)

- **Change:** Pareto dominance now excludes rows with non-finite objectives.
- **Reason:** missing predictions cannot support a scientific trade-off claim.

## v0.12.0 — native data contract loosening (Consolidation)

- **Hypothesis:** public users need native-schema data loading to work for
  datasets other than the historical private 100-row grid.
- **Change:** fixed row/template/design-cell counts are now logged as dataset
  profile metadata instead of enforced assertions.
- **Change:** transfection_efficiency <= -1 fails before log1p, so invalid
  values cannot silently become non-finite model inputs.
- **Decision:** keep structural keys and ratio diagnostics, but let dataset
  scale be data-derived.

## v0.13.0 — group-aware inner model selection (Consolidation)

- **Hypothesis:** template-aware outer folds need template-aware inner tuning to
  avoid selecting hyperparameters with row-level template leakage.
- **Change:** tuned model configs now use `GroupKFold` over `template_key` when
  templates are available, falling back to shuffled KFold only when grouping is
  impossible.
- **Decision:** keep this protocol for both benchmark metrics and OOF candidate
  predictions.

## v0.14.0 — public benchmark CLI (Consolidation)

- **Hypothesis:** open-source users need one command that produces a stable
  benchmark artifact without private data or model weights.
- **Change:** added `lnp-agent --benchmark-public`, which writes a JSON artifact
  summarizing public LNPDB schema coverage, assay counts, seed metadata, and
  scientific-use limits.
- **Decision:** keep this as a software/data-access benchmark only; it is not a
  predictive performance or biological efficacy claim.

## v0.15.0 — artifact provenance (Consolidation)

- **Hypothesis:** public artifacts should carry enough provenance for reviewers
  to identify the generator, package version, dataset source, license, and
  private-data boundary.
- **Change:** public benchmark JSON now includes a `provenance` block with the
  generator command, package metadata, artifact schema, public dataset/license,
  and `private_data_included=false`.
- **Decision:** require provenance fields on public-facing benchmark artifacts
  before users treat them as reproducibility evidence.

## v0.16.0 — public audit and release cadence (PaperReady)

- **Hypothesis:** a public release needs a durable audit trail that links
  scientific scope, limitations, GitHub issues, code changes, tests, and tags.
- **Change:** added `docs/SCIENTIFIC_AND_ENGINEERING_AUDIT.md` and completed
  the versioned research log through the v0.8.0-v0.16.0 release cycle.
- **Result:** the project now has public issue history, release tags, research
  rationale, and a current audit that can be reviewed without private LNP data.
- **Decision:** future versions should continue adding a short research-log
  entry before release, especially when scientific claims, data coverage,
  benchmark protocol, or artifact semantics change.
