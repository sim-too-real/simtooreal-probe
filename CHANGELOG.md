# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-09-11

### Added
- `diagnose` / `ProbeRun.diagnose()` — run-level failure taxonomy: entropy collapse, KL spike, zero action-std (VLA silent killer), reward hacking, idle policy, instruction/skill decoupling, torque saturation, early termination, sim-vs-real covariate shift.
- `health_score` — 0–100 pre-certificate signal from findings + failure rate.
- `compare_sources` / `ProbeRun.compare()` — sim vs real distribution shift → `transfer_risk`.
- `lerobot_to_traces` — import on-disk LeRobot v2/v3 datasets (parquet or jsonl) into the same schema as RSL-RL rollouts, including language `instruction` tags.
- `LocalRunSummary.summary()` now includes `findings` and `health`.

### Fixed
- `probe.watch()` and `rsl_rl_capture` / `gym_capture` now force `capture=full` so per-step traces are not silently dropped (the previous default `episodes` discarded the moat data).
- STEP drops under `capture=episodes` emit a one-time warning.
- RSL-RL capture slices the env axis **on-device** before the D2H copy, aligns T across obs/action (T+1 vs T), reads `time_outs`, and tags episodes with `success` / `termination` so `failures()` works on the headline path.
- `infer_dim_registry` names Isaac Lab action terms.
- HTML replay and `DimRegistry.resolve` no longer crash on sanitized NaN/None (the interesting failure episodes).
- Gym capture ignores non-finite rewards when accumulating episode return.

### Notes
- Schema stays at version 1. New fields travel as optional tags (`instruction`, `primitive_success`) so existing consumers keep working.

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
