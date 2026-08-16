# Optional COMET And LaMGen Integrations

LNPAgent v0.25.0 can dispatch two optional local research runtimes. The
package contains adapters only: it does not redistribute COMET or LaMGen code,
datasets, checkpoints, protein embeddings, generated molecules, or LNP
measurements.

## Readiness Check

Set checkout locations directly or through environment variables, then inspect
the integration surface:

```bash
export LNP_AGENT_COMET_ROOT=/opt/COMET
export LNP_AGENT_LAMGEN_ROOT=/opt/LaMGen
lnp-agent --external-tools-status
```

The status response describes whether the expected upstream source files are
present. It does not imply that a compatible checkpoint or GPU environment is
available.

When `--comet-python` or `--lamgen-python` points to a separate conda
environment, LNPAgent executes a small packaged runner file with that Python
interpreter. The external environment therefore needs the upstream tool's
dependencies, not a second editable checkout of LNPAgent.

## COMET

[COMET](https://github.com/alvinchangw/COMET) is an upstream Composite Material
Transformer implementation for LNP tasks. Its documented runtime requires its
own CUDA/Uni-Core environment, a COMET-preprocessed LMDB input, and a compatible
local checkpoint. The LNPAgent adapter runs COMET's `unimol/infer_np.py` with a
selected upstream schema:

```bash
lnp-agent --comet-predict \
  --comet-root /opt/COMET \
  --comet-python /opt/conda/envs/comet/bin/python \
  --comet-input-lmdb /work/prepared_lnp.lmdb \
  --comet-checkpoint /work/checkpoint_best.pt \
  --comet-task stability \
  --comet-output /work/lnpagent-artifacts/comet
```

`--comet-task` accepts `lipid`, `pbae`, and `stability`. The stability option
selects COMET's lyophilized-LNP task schema; the interpretation of a prediction
is defined by the specific checkpoint and input data supplied by the operator.
LNPAgent writes `comet_inference_manifest.json` beside COMET's local result
files. The manifest records the invoked command and result paths, but does not
copy user data into this repository.

## LaMGen

[LaMGen](https://github.com/cholin01/LaMGen) is an upstream multi-target 3D
molecular-generation project. Its published generation interface conditions on
ESM-C protein embedding arrays, so LNPAgent accepts target identifiers paired
with one local `.npy` embedding per target. Those embeddings may be generated
from amino-acid sequences through LaMGen's upstream ESM-C workflow.

```bash
lnp-agent --lamgen-generate \
  --lamgen-root /opt/LaMGen \
  --lamgen-python /opt/conda/envs/lamgen/bin/python \
  --lamgen-mode dual \
  --lamgen-target TARGET_A=/work/esmc/TARGET_A.npy \
  --lamgen-target TARGET_B=/work/esmc/TARGET_B.npy \
  --lamgen-model /work/checkpoints/dual_target_ckpt \
  --lamgen-output /work/lnpagent-artifacts/lamgen_dual.csv \
  --lamgen-samples 100
```

For triple-target generation, pass `--lamgen-mode triple` and three
`--lamgen-target TARGET=EMBEDDING.npy` values with a compatible triple checkpoint.
The output is an upstream molecular-token CSV and a neighboring JSON manifest.
It is a generation artifact, not an LNP formulation, a binding claim, a
toxicity assessment, or an experimental recommendation.

## Agent Tools

`create_all_tools(...)` now registers the following callables for the LNPAgent
tool registry:

- `run_comet_inference`
- `generate_lamgen_molecules`

Both return a structured manifest suitable for recording in an experiment plan.
The agent keeps quantitative outputs separate from language-model explanation.

## Upstream Terms And Data Boundary

COMET and LaMGen remain separate upstream projects. COMET's GitHub repository
does not declare a machine-readable SPDX license at the time of this release;
review its repository terms before use or redistribution. LaMGen declares MIT.
LNPAgent is AGPL-3.0-only for its own code. The public LNPAgent source release
does not include external model weights, checkpoints, ESM-C embeddings, or
private/experimental LNP records.
