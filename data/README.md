# Data policy

`lnpdb_public_example.csv` is a 100-row example from the publicly available,
MIT-licensed [LNPDB](https://github.com/evancollins1/LNPDB) project. It is a
compact, source-local subset of public in-vitro luminescence records; full
provenance and attribution are in [LNPDB_NOTICE.md](LNPDB_NOTICE.md).

The file is for package validation and software examples. It is not a general
benchmark across assays, experimental evidence for a new formulation, or a
clinical decision aid. To recreate the exact subset from an official LNPDB CSV:

```bash
python scripts/extract_lnpdb_example.py /path/to/LNPDB.csv
```

Do not commit confidential, participant-level, proprietary, licensed, or
third-party redistribution-restricted data. Store full datasets outside the
repository and point the package to a CSV with `LNP_AGENT_DATA=/path/to/data.csv`.
`lnp-agent --check-data` supports the public LNPDB schema and the LNPAgent-native
research schema, and reports which it found.
