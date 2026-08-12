# Architecture

LNPAgent implements a stateful, closed-loop experimental design workflow.

```text
library generation -> batch prediction -> exploitation / exploration selection
       ^                                                        |
       |                                                        v
model retraining <- simulated or supplied wet-lab measurements <- reporting
```

`lnp_agent.engine_v5` coordinates the state machine and tool permissions.
`lnp_agent.data_manager_v4` builds molecular and formulation features, trains
endpoint models, and estimates uncertainty. The `tools` package provides the
library, selection, visualization, and wet-lab integration boundaries.
`lnp_core` contains the bundled, lightweight feature engineering and
evaluation primitives required by the data manager.

Generated files are always written under `artifacts/` by default. Set
`LNP_AGENT_ARTIFACTS` to relocate them. This keeps source and run outputs
separate and makes repeated experiments safe to inspect.
