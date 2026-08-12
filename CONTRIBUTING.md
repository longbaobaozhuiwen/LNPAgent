# Contributing

Issues and pull requests are welcome. Please open an issue before proposing a
large API or modelling change so the intended experimental contract is clear.

For a local development environment:

```bash
python -m pip install -e '.[dev]'
pytest
```

Keep changes focused, add tests for changed behavior, and do not commit model
weights, generated checkpoints, API keys, or source data that cannot be
redistributed. Claims about experimental performance should identify the data
split, endpoint, and evaluation procedure used.
