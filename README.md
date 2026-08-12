# LNPAgent

> An open-source, uncertainty-aware research agent for lipid nanoparticle formulation design.

LNPAgent turns LNP formulation design into an auditable active-learning loop. It generates candidate libraries, builds molecular and formulation features, predicts multiple assay-oriented readouts, ranks candidates under uncertainty, records why each candidate was selected, and separates synthetic software fixtures from real experimental evidence.

The project is intentionally not a black-box recipe generator. Its core goal is to help scientists decide which experiment is worth running next, why that experiment is informative, and how new measurements should update the model.

![LNPAgent evidence loop](assets/evidence-loop.svg)

## Why This Exists

Most formulation ML pipelines stop after prediction: enter a candidate, get a score. LNPAgent treats prediction as one step inside a larger scientific decision process.

The research question is:

> How can an agent choose the next LNP experiment that is maximally informative while respecting biological trade-offs, uncertainty, assay provenance, and data-use boundaries?

That framing leads to a different software design:

- Candidate generation is tied to configurable formulation building blocks.
- Ranking combines predicted objectives, Pareto diagnostics, and candidate-level uncertainty proxies.
- Exploration and exploitation are separated so the agent can learn, not only chase current winners.
- Synthetic wet-lab outputs are explicitly labelled as simulation, never experimental evidence.
- Public examples and benchmark artifacts carry provenance and release-safety metadata.
- Private measurements, raw experimental outputs, checkpoints, and credentials are excluded from the public repository.

![LNPAgent core innovation](assets/core-innovation.svg)

## Latest Public-Release Progress

The current release cycle moved the project from a private research prototype toward a public, reviewable foundation. The most important updates are:

- **Public data adapter:** `lnp-agent --public-summary` inspects the bundled LNPDB example without relabelling public assay fields as native LNPAgent endpoints.
- **Synthetic measurement provenance:** simulated wet-lab outputs now carry explicit measurement type and provenance fields.
- **Candidate-specific uncertainty:** ranking outputs include per-candidate ensemble dispersion instead of relying only on one global spread metric.
- **Robust Pareto filtering:** rows with missing or non-finite objectives are excluded from Pareto-front claims.
- **Looser native data contract:** native-schema loading no longer assumes a fixed private-grid shape.
- **Group-aware tuning:** tuned models use template-aware inner cross-validation when template groups are available.
- **Public benchmark artifact:** `lnp-agent --benchmark-public` writes reproducible schema and assay-coverage metadata for the public LNPDB example.
- **Artifact provenance:** public benchmark JSON records generator command, package metadata, dataset source, license, schema, and `private_data_included=false`.
- **Release audit:** `docs/SCIENTIFIC_AND_ENGINEERING_AUDIT.md` documents scientific scope, limitations, resolved issues, and the public data boundary.

See [research_log.md](research_log.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), and [docs/SCIENTIFIC_AND_ENGINEERING_AUDIT.md](docs/SCIENTIFIC_AND_ENGINEERING_AUDIT.md) for the full audit trail.

## Public Data Boundary

This repository is designed to be public. That only works if the data boundary is strict.

![Public release boundary](assets/public-release-boundary.svg)

Versioned public data is limited to:

- `data/lnpdb_public_example.csv`
- `data/README.md`
- `data/LNPDB_NOTICE.md`

The bundled CSV is a 100-row example from the public, MIT-licensed [LNPDB](https://github.com/evancollins1/LNPDB) project. It is useful for package validation, schema inspection, and examples. It is not a biological efficacy benchmark, not a formulation recommendation dataset, and not clinical evidence.

Do not commit confidential LNP measurements, unpublished assay records, proprietary formulations, local checkpoints, model weights, generated artifacts, API keys, or credentials. The release audit script enforces this policy against tracked files:

```bash
python scripts/audit_public_repository.py
```

## How The Agent Works

![LNPAgent autonomous workflow](assets/agent-workflow.svg)

At a high level, LNPAgent coordinates five steps:

1. **Plan:** define the design objective and evidence boundary.
2. **Generate:** build a candidate library from formulation components and ratio rules.
3. **Rank:** predict assay-oriented readouts and score candidates under uncertainty.
4. **Review:** report trade-offs, Pareto status, and provenance for human inspection.
5. **Learn:** ingest supplied measurements or synthetic fixtures and update the model state.

The current public release provides the infrastructure for this loop. The next scientific frontier is stronger experiment-value acquisition: selecting batches by expected information gain under multi-objective biological constraints, not simply selecting the top predicted score.

![Algorithmic frontier](assets/algorithmic-frontier.svg)

## Roadmap

Near-term work should make the public project easier to run and scientifically sharper without weakening the private-data boundary:

- **Public demo loop:** expose a one-round CLI workflow that runs candidate generation, ranking, reporting, synthetic measurement, and retraining summary using public-safe fixtures.
- **Native/public schema bridge:** add a careful adapter from public LNPDB-style records to software examples while preserving assay semantics and avoiding exaggerated performance claims.
- **Calibrated uncertainty:** compare ensemble dispersion, applicability-domain checks, and conformal intervals against held-out measurements before presenting probabilistic confidence.
- **Mechanism-guided exploration:** prefer interpretable chemistry and ratio perturbations that teach design rules, not only numerical optimization.
- **Evidence memory:** summarize what each round learned, which hypotheses failed, and which design regions became more or less promising.

## Quick Start

LNPAgent requires Python 3.10 or newer. RDKit is easiest to install through conda-forge; the remaining package dependencies can then be installed with pip.

```bash
git clone https://github.com/longbaobaozhuiwen/LNPAgent.git
cd LNPAgent
conda create -n lnpagent -c conda-forge python=3.11 rdkit
conda activate lnpagent
python -m pip install -e '.[dev]'
lnp-agent --check-data
pytest
```

Optional GPU, AGILE, and local LLM integrations are supported by configuration but are not bundled with weights, checkpoints, or credentials:

```bash
python -m pip install -e '.[gpu,llm,agile]'
export LNP_AGENT_AGILE_CHECKPOINT=/path/to/agile_model.pth
export LNP_AGENT_GEMMA_MODEL=/path/to/gemma
```

## Public Data Commands

```bash
lnp-agent --check-data        # validate the configured CSV
lnp-agent --public-summary    # summarize the public LNPDB example
lnp-agent --benchmark-public  # write artifacts/benchmark_public_lnpdb.json
```

The public benchmark command produces a reproducibility artifact for schema and assay coverage. It is not a model-performance claim and not a candidate recommendation.

To use your own local data without editing source files:

```bash
export LNP_AGENT_DATA=/path/to/your/formulations.csv
export LNP_AGENT_ARTIFACTS=/path/to/lnpagent-artifacts
lnp-agent --check-data
```

Keep full datasets outside the repository unless redistribution rights, provenance, and audit allowlisting are all explicit.

## Repository Map

```text
src/lnp_agent/       agent engine, data managers, tools, and CLI
src/lnp_core/        feature engineering, model evaluation, ranking, uncertainty
data/                approved public LNPDB example and data-use policy
assets/              README diagrams and project illustrations
docs/                architecture, release policy, and scientific audit
tests/               release-level smoke and regression tests
scripts/             data extraction and public-release audit utilities
```

## Scientific Scope

Current capabilities include:

- formulation-library generation from configurable building blocks;
- molecular and formulation feature construction for deployable baselines;
- multi-objective candidate ranking with Pareto diagnostics;
- candidate-level uncertainty proxy from ensemble dispersion;
- bootstrap and paired-delta evaluation utilities;
- public LNPDB schema validation and coverage summaries;
- synthetic wet-lab simulation for software exercises.

Important limitations:

- The bundled public data is an example, not a native endpoint benchmark.
- Synthetic wet-lab outputs are software fixtures, not physical measurements.
- Candidate uncertainty is currently a proxy and still needs calibration before probabilistic claims.
- Generalization across chemistry, cargo, assay platform, species, tissue, or disease context remains an empirical question.

## Safety And Contribution Policy

LNPAgent is research software. Validate all candidate formulations experimentally. Do not use it for clinical decision-making.

Before opening a pull request, run:

```bash
python scripts/audit_public_repository.py
pytest
```

Contributions are governed by [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and [docs/PUBLIC_RELEASE_POLICY.md](docs/PUBLIC_RELEASE_POLICY.md). Please keep private data, restricted third-party data, generated artifacts, weights, checkpoints, and secrets out of Git.

## License

LNPAgent is released under the [MIT License](LICENSE). Dependencies and external datasets retain their own licenses and terms.
