# v0.67 research execution report

Updated: 2026-09-26. Scope: audited retrospective research execution, not an
installable v0.67 package release. The public package version remains 0.25.0.

## Bottom line

Qwen performed real tool-mediated model selection in part of this experiment.
However, **v0.67 does not establish an LLM prediction benefit**. The central
hypothesis-controller and context-ablation comparison was confounded by
implementation and execution defects. Fixed aggregation was the strongest
observed default in this particular replay.

This is neither evidence that LLM-assisted research cannot work nor evidence
that biological context improved the model. Those questions require a valid
comparison.

## Task and shared numerical methods

The local experiment used COMET-associated Caco-2 retrospective prediction
tasks with B16-F10 and DC2.4 source information. Eight task settings combined
two folds, two seeds and target-label budgets of 32 or 64. The label budgets
included validation labels. These settings reuse one dataset and are **not
eight independent biological studies** or a fresh external validation set.

The candidate menu comprised a target mean, target-feature ridge regression,
calibrated source predictors and equal-weight target/source combinations.
`U_BD` is 0.5 times the target prediction plus 0.5 times the jointly calibrated
two-source prediction. `F_U` always selects it; `F_all` selects the minimum
validation-error candidate after probing both sources.

The primary numerical metric below is test group-MAE: average absolute row
error within each composition group, then average equally across groups.
Lower is better; values are on the existing task's response scale, not a
common physical transfection unit.

## Observed results and attribution

The means below reproduce the saved selected-output metrics across all eight
task settings, **including executor fallbacks**. They are not autonomous-only
performance estimates.

| Method | Mean test group-MAE | What actually happened |
|---|---:|---|
| F_U | 0.121280 | Fixed U_BD in all eight settings |
| F_all | 0.128761 | Fixed full-menu validation selection |
| F_card | 0.127585 | No source probes: incorrect context lookup invalidated the intended rule |
| Q_plan | 0.125575 | Seven valid plans, all identical; one fallback |
| Q_generic | 0.131376 | Seven explicit model submissions; one fallback |
| Q_hypothesis | 0.121280 | Zero explicit model choices; all outputs resolved to U_BD through defaults/fallbacks |
| Q_no_context | 0.121280 | No executed model-choice actions; fallback outputs |

Three Q_hypothesis records were labeled `origin=qwen`, but their original
final actions omitted both `spec` and `artifact`. The executor silently filled
in U_BD. The apparent improvement over F_all therefore belongs to the fixed
default, not to a demonstrated Qwen hypothesis-selection mechanism.

The generic controller did explicitly specify models in seven original final
actions. This establishes a limited execution capability; its aggregate score
does not show improved prediction over the fixed default.

## Why the main comparison is not conclusive

1. **Incomplete visible state.** U_BD existed in the numerical store but was
   absent from the initial table shown to Qwen. Its probe comparison was also
   missing. Fixed numerical routines and Qwen did not receive equivalent
   usable baseline information.
2. **Silent final-selection defaults.** Missing model identifiers were filled
   by the executor and sometimes labeled as Qwen selections.
3. **Invalid context ablation.** All 28 saved no-context responses failed the
   action parser; this mode also had different output instructions. Its
   failure cannot be interpreted as a biological-knowledge effect.
4. **Broken fixed-card baseline.** Source cards were nested under a different
   key than the rule inspected, so the intended probes never ran.
5. **Incorrect diagnostics.** A fit-correlation calculation used validation
   labels/groups, and some paired diagnostics used row-MAE rather than
   group-MAE. Qwen rationales referenced the incorrect correlation values.
6. **Budget starvation.** The ledger recorded 160 controller calls plus six
   card-extraction calls, at least 166 model calls in total. Stage caps did
   not reserve pilot capacity. The last task had zero calls in all four Qwen
   modes; this is not a normal completed trial of those methods.

The available pilot-root outputs contain four CSV tables, not the seven
claimed in an earlier local summary. Complete cost and hypothesis-update
tables were missing. Timing also omitted initialization and other shared
costs. No complete cost-efficiency advantage can be claimed.

Some proposed biological hypotheses concerned dose or normalization
compatibility, whereas the tools returned regression diagnostics. A favorable
regression error cannot establish those experimental facts or a delivery
mechanism.

## What follows from this audit

The appropriate v0.67 assessment is **partial execution capability, with the
central scientific comparison blocked by protocol/runtime defects**.

The bounded follow-up design calls for a complete candidate-state view, one
explicit action contract, no implicit final model selection, corrected
group-level diagnostics, identical non-context interfaces, separate phase
budgets and scoring of frozen prediction artifacts. Numerical hypotheses
must be computable by the actual tools. Capability checks and prediction
benefit must be reported separately; retaining U_BD is a valid decision.

Local v0.68 execution artifacts dated September 25 now exist. This report
does not audit or certify their outcome and should not be read as saying that
v0.68 is still unexecuted. The v0.67 findings remain historical evidence, not
a verdict on an unreviewed later version.

## Evidence and public-release boundary

This note is a manually prepared, aggregate research report based on local
selected-metric CSVs, original model responses, selection records, the call
ledger and the v0.67 implementation. The numerical means, explicit-final
counts and ledger totals were rechecked on September 26 against the existing
artifacts; no model or new test scoring was run for this update. The local
September 23 audit supplies the detailed implementation findings.

Raw research snapshots and execution artifacts are not part of this public
commit. Consequently this note alone is **not a publicly reproducible
benchmark package**. It contains no formulation-level measurements, model
weights, checkpoints, raw prompts, local access details or dataset files.
The repository's [public release policy](../PUBLIC_RELEASE_POLICY.md) remains
unchanged. This documentation update does not advertise the local research
controller as a new installable public runtime.
