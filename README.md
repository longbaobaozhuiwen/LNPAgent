# LNPAgent

> An agentic active-learning system for designing lipid nanoparticles under uncertainty.

LNPAgent is a research software project for turning LNP formulation design into an auditable closed loop: propose candidates, predict multiple biological readouts, choose what to test next, absorb new measurements, and update the model. The goal is not to ship a black-box formula recommender. The goal is to build an agent that can reason about exploration, evidence, uncertainty, and experimental next steps.

![LNPAgent core innovation](assets/core-innovation.svg)

## Core Idea

Most LNP modeling tools stop at prediction: given a formulation, estimate an endpoint. LNPAgent treats prediction as only one step inside a larger scientific decision process.

The central research question is:

> How can an agent make each new experiment maximally informative while respecting assay uncertainty, multi-objective trade-offs, data provenance, and the boundary between simulation and real measurements?

LNPAgent currently implements the infrastructure for this loop:

- Generates formulation libraries from configurable building blocks.
- Builds molecular and formulation features for deployable baselines.
- Predicts delivery and inflammation-related endpoints.
- Ranks candidates with Pareto diagnostics and candidate-level uncertainty proxies.
- Separates exploitation from exploration so the agent can improve rather than only chase current winners.
- Records public benchmark provenance and keeps private data outside the repository.
- Supports synthetic wet-lab simulation as a software fixture, explicitly labelled as non-experimental evidence.

## What Makes This Different

LNPAgent is moving toward an algorithmic research agent, not just a pipeline. The interesting part is the decision policy: deciding which formulation should be tested next, why that experiment is valuable, and how its result changes the model.

![Algorithmic frontier](assets/algorithmic-frontier.svg)

Near-term innovation targets:

- **Experiment-value acquisition:** choose batches by expected information gain under multi-objective biological constraints, not only top predicted score.
- **Uncertainty-aware formulation search:** combine ensemble disagreement, applicability-domain checks, and conformal intervals to avoid confident extrapolation.
- **Mechanism-guided exploration:** use chemistry-aware structure changes and lipid-ratio perturbations so exploration teaches the model interpretable design rules.
- **Closed-loop evidence memory:** summarize what each round learned, which hypotheses failed, and which design regions became more or less promising.
- **Data expansion as algorithm design:** actively seek public datasets or assays that reveal cross-platform, cargo, tissue, or species generalization limits.

These are the directions that should drive future versions. Documentation, validation, and provenance matter because they make the scientific loop reviewable, but the main research thrust is better active-learning policy for LNP design.

## Current Public Release

The public release is deliberately narrow and safe. It includes a small public LNPDB example for software validation and reproducible benchmark metadata. It does not include private LNP measurements, raw experimental outputs, model weights, AGILE checkpoints, or credentials.

Recent release work established the public foundation:

- Public LNPDB validation, summary, and reproducible benchmark CLI.
- Explicit synthetic-measurement provenance for the wet-lab oracle.
- Candidate-level ensemble uncertainty proxy in ranking outputs.
- Robust Pareto filtering for missing predictions.
- Native-schema loading without fixed private-grid assumptions.
- Group-aware inner cross-validation for tuned models.
- Provenance fields in public benchmark artifacts.
- A public scientific and engineering audit.

See [research_log.md](research_log.md) and [docs/SCIENTIFIC_AND_ENGINEERING_AUDIT.md](docs/SCIENTIFIC_AND_ENGINEERING_AUDIT.md) for the detailed audit trail.

## Quick Start

LNPAgent requires Python 3.10 or newer. RDKit is easiest to install through conda-forge; other dependencies can then be installed with pip.

```bash
git clone https://github.com/longbaobaozhuiwen/LNPAgent.git
cd LNPAgent
conda create -n lnpagent -c conda-forge python=3.11 rdkit
conda activate lnpagent
python -m pip install -e '.[dev]'
lnp-agent --check-data
pytest
```

Optional GPU, AGILE, and local LLM integrations are not bundled with weights or credentials:

```bash
python -m pip install -e '.[gpu,llm,agile]'
export LNP_AGENT_AGILE_CHECKPOINT=/path/to/agile_model.pth
export LNP_AGENT_GEMMA_MODEL=/path/to/gemma
```

## Public Data And Benchmark Commands

The versioned `data/lnpdb_public_example.csv` is a 100-row subset of the public, MIT-licensed [LNPDB](https://github.com/evancollins1/LNPDB) dataset. Its provenance and license notes are recorded in [data/LNPDB_NOTICE.md](data/LNPDB_NOTICE.md).

```bash
lnp-agent --check-data        # validate configured CSV
lnp-agent --public-summary    # inspect public LNPDB example without endpoint relabelling
lnp-agent --benchmark-public  # write artifacts/benchmark_public_lnpdb.json
```

The public benchmark is a schema and assay-coverage benchmark only. It is not a predictive performance result, not biological efficacy evidence, and not a candidate recommendation.

To use your own data without editing source:

```bash
export LNP_AGENT_DATA=/path/to/your/formulations.csv
export LNP_AGENT_ARTIFACTS=/path/to/lnpagent-artifacts
lnp-agent --check-data
```

## Repository Map

```text
src/lnp_agent/       agent engine, data managers, tools, and CLI
src/lnp_core/        feature engineering, model evaluation, ranking, uncertainty
data/                public LNPDB example and data-use policy
assets/              README diagrams and workflow illustrations
docs/                architecture and public scientific/engineering audit
tests/               release-level smoke and regression tests
```

## Reproducibility And Safety

LNPAgent is research software. Its behavior depends on data selection, simulator choice, feature construction, random seeds, model configuration, and hardware. Validate all candidate formulations experimentally. Do not use LNPAgent for clinical decision-making.

The bundled wet-lab tool is a deterministic, noisy synthetic oracle for software exercises. Its outputs are labelled `measurement_type=synthetic_oracle` and are not experimental evidence.

Contributions are governed by [CONTRIBUTING.md](CONTRIBUTING.md). Please do not commit confidential data, raw experimental outputs, redistribution-restricted third-party datasets, model weights, checkpoints, API keys, or other secrets.

## License

LNPAgent is released under the [MIT License](LICENSE). Dependencies and external datasets retain their own licenses and terms.
