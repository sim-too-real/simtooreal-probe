# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Public stack narrative (GitHub-readable, no captcha): [docs/sincere-stack.md](docs/sincere-stack.md).

### Changed

- Public contact and company links: [simtooreal.com](https://simtooreal.com), [robosynx.com](https://robosynx.com), [vardhan@simtooreal.com](mailto:vardhan@simtooreal.com).

## [0.1.0] — 2026-08-07

### Added

- Open-core policy debugging package: `PolicyTraceEvent` at iteration / episode / step grains.
- Local-first store under `~/.probe/<run_id>/` (JSONL or Parquet when `pyarrow` is installed).
- Process API: `init`, `emit`, `close`, `watch`, `rsl_rl_capture`, `gym_capture`.
- Notebook API: `ProbeRun`, `LocalRunSummary`, CUSUM `first_abnormal` onset detector.
- Optional `HttpSink` (stdlib) with pluggable `.post(path, payload)` clients.
- Real-robot adapters in `adapt` (`emit_records`, `mcap_to_traces`).
- Standalone failure replay HTML via `ProbeRun.to_html` / `export_failure_html`.
- Optional extras: `query`, `plot`, `mcap`, `full`, `dev`.

### Notes

- PyPI publish may lag the GitHub tag; install from git until `pip install simtooreal-probe` is live.
