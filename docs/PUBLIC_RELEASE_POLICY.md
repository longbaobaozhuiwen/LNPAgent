# Public release policy

This repository must not contain internal LNP measurements, unpublished assay
records, participant-level information, proprietary formulations, local model
weights, checkpoints, or generated run artifacts.

The only versioned data paths are the LNPDB public example and its two
documentation files under `data/`. The CI workflow runs
`scripts/audit_public_repository.py`, which rejects tracked `Data/`, historical
development snapshots, output directories, model artifact suffixes, and the
retired local-example filenames.

Before adding any new dataset, verify redistribution rights, record upstream
license and provenance, add it to the explicit allowlist in the audit script,
and add a regression test. When sensitive content is found, remove it from the
current tree and reachable Git history before continuing public development.
