# Contributing to simtooreal-probe

Thanks for helping improve the open-core policy debugging layer.

## Development setup

```bash
git clone https://github.com/sim-too-real/simtooreal-probe.git
cd simtooreal-probe
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Optional full extras (Parquet + DuckDB + mcap):

```bash
pip install -e ".[full,dev]"
```

## Project layout

| Path | Role |
|------|------|
| `src/simtooreal_probe/` | Library code |
| `tests/` | Pytest suite |
| `docs/` | Extra documentation |

## Guidelines

1. **Local-first first.** Core paths must work with no account and no network.
2. **Keep core deps lean.** Runtime hard dependency is `numpy` only; heavier tools go in extras.
3. **Never raise into the training loop from sinks.** Sink failures degrade; they do not abort training.
4. **Preserve the `PolicyTraceEvent` envelope.** Bump `SCHEMA_VERSION` only for breaking shape changes.
5. **Tests required** for new public APIs and for onset / capture logic.

## Pull requests

- Open against `main`.
- Keep PRs focused; include tests and a short CHANGELOG note under `[Unreleased]` if needed.
- Run `pytest` and `python -m build` before requesting review.

## Code of conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Security

Do not open public issues for vulnerabilities. See [SECURITY.md](SECURITY.md).
