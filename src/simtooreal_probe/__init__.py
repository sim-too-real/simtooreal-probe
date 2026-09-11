"""
simtooreal-probe — open-core Policy Debugging Layer for any RL training run.

One import, one canonical trace contract (`PolicyTraceEvent`), framework-agnostic
capture, and a local-first store any researcher can read in a notebook — with no
account and no network. Pointing it at a remote backend is one optional sink.

Quickstart (local-first, zero account)
--------------------------------------
    import simtooreal_probe as probe

    probe.init(run_id="go2-flat-v3")            # writes ~/.probe/go2-flat-v3/
    cap = probe.rsl_rl_capture(source="sim:physx")
    # ... inside the training loop, once per iteration after rollout collection:
    cap.capture_rollout(runner, iteration=it)
    probe.close()

    # later, in a notebook:
    from simtooreal_probe import ProbeRun
    run = ProbeRun("go2-flat-v3")
    fail = run.failures()[0]
    print(run.first_abnormal(fail["episode_id"]))

Capture level (FPS escape hatch): env `PROBE_CAPTURE=off|metrics|episodes|full`
(default `episodes`). `full` adds the per-step policy trace.

Optional HTTP sink: set ``PROBE_HTTP_URL`` (or compatible platform env aliases)
and ``http=True`` / auto-detect to also ship batches to a remote ingest endpoint.
"""
from __future__ import annotations

import os
from typing import List, Optional

try:
    from importlib.metadata import PackageNotFoundError, version as _pkg_version

    try:
        __version__ = _pkg_version("simtooreal-probe")
    except PackageNotFoundError:
        __version__ = "0.2.0"
except Exception:  # pragma: no cover
    __version__ = "0.2.0"

from .trace import (
    DimRegistry,
    Grain,
    PolicyTraceEvent,
    SCHEMA_VERSION,
    SOURCE_MJWARP,
    SOURCE_MJX,
    SOURCE_MUJOCO,
    SOURCE_NEWTON,
    SOURCE_PHYSX,
    real_source,
)
from .sinks import HttpSink, LocalSink, Sink
from .writer import TraceWriter, resolve_capture
from .capture import GymStepCapture, RslRlStorageCapture, infer_dim_registry, policy_state_registry
from .query import ProbeRun
from . import adapt  # real-robot / external-log → trace adapters
from . import viz  # standalone HTML failure replay
from .analytics import LocalRunSummary
from .viz import export_failure_html, render_failure_replay_html
from .diagnose import Finding, diagnose, diagnose_run, health_score, CATALOG as FAILURE_CATALOG
from .compare import compare_run, compare_sources
from .adapt_lerobot import lerobot_to_traces

__all__ = [
    "__version__",
    "init",
    "close",
    "emit",
    "writer",
    "rsl_rl_capture",
    "gym_capture",
    "register_dims",
    "watch",
    "PolicyTraceEvent",
    "DimRegistry",
    "Grain",
    "ProbeRun",
    "LocalRunSummary",
    "LocalSink",
    "HttpSink",
    "Sink",
    "TraceWriter",
    "resolve_capture",
    "GymStepCapture",
    "RslRlStorageCapture",
    "infer_dim_registry",
    "policy_state_registry",
    "adapt",
    "viz",
    "export_failure_html",
    "render_failure_replay_html",
    "SCHEMA_VERSION",
    "SOURCE_PHYSX",
    "SOURCE_NEWTON",
    "SOURCE_MJX",
    "SOURCE_MUJOCO",
    "SOURCE_MJWARP",
    "real_source",
    "Finding",
    "diagnose",
    "diagnose_run",
    "health_score",
    "FAILURE_CATALOG",
    "compare_run",
    "compare_sources",
    "lerobot_to_traces",
]

# Single process-wide writer (one training run per process).
_WRITER: Optional[TraceWriter] = None
_RUN_ID: str = ""


def init(
    run_id: Optional[str] = None,
    *,
    root: Optional[str] = None,
    http: Optional[bool] = None,
    capture: Optional[str] = None,
    sinks: Optional[List[Sink]] = None,
    meta: Optional[dict] = None,
) -> TraceWriter:
    """Initialise the process-wide trace writer. Idempotent within a process.

    run_id  : defaults to env SIMTOOREAL_RUN_ID, else "local-run".
    http    : also attach an HttpSink (defaults to True iff SIMTOOREAL_URL or
              SIMTOOREAL_INGEST_TOKEN is set — i.e. running on the platform).
    capture : off|metrics|episodes|full (env PROBE_CAPTURE overrides None).
    """
    global _WRITER, _RUN_ID
    if _WRITER is not None:
        want_rid = run_id or os.environ.get("SIMTOOREAL_RUN_ID") or "local-run"
        if (run_id and want_rid != _WRITER.run_id) or (capture and resolve_capture(capture) != _WRITER.capture):
            import warnings
            warnings.warn(
                f"probe.init() is already active (run_id={_WRITER.run_id!r}, capture={_WRITER.capture!r}); "
                f"new args ignored. Call probe.close() first to reconfigure.",
                stacklevel=2,
            )
        return _WRITER

    rid = (
        run_id
        or os.environ.get("PROBE_RUN_ID")
        or os.environ.get("SIMTOOREAL_RUN_ID")
        or "local-run"
    )
    _RUN_ID = rid

    if sinks is None:
        sink_list: List[Sink] = [LocalSink(rid, root=root, meta=meta)]
        if http is None:
            http = bool(
                os.environ.get("PROBE_HTTP_URL")
                or os.environ.get("SIMTOOREAL_URL")
                or os.environ.get("SIMTOOREAL_INGEST_TOKEN")
            )
        if http:
            sink_list.append(HttpSink(rid))
    else:
        sink_list = list(sinks)

    _WRITER = TraceWriter(rid, sink_list, capture=capture)
    return _WRITER


def writer() -> TraceWriter:
    """Return the active writer, initialising a local-first one if needed."""
    return _WRITER if _WRITER is not None else init()


def emit(event: PolicyTraceEvent) -> None:
    writer().emit(event)


def register_dims(registry: DimRegistry) -> None:
    writer().register_dims(registry)


def rsl_rl_capture(*, source: str = SOURCE_PHYSX, traced_envs: int = 8) -> RslRlStorageCapture:
    """Capture per-step policy trace from an RSL-RL rollout buffer (the moat path)."""
    w = writer()
    if getattr(w, "capture", "episodes") != "full":
        w.set_capture("full")
    return RslRlStorageCapture(w, w.run_id, source=source, traced_envs=traced_envs)


def gym_capture(*, source: str = SOURCE_PHYSX) -> GymStepCapture:
    """Per-step capture for raw gym / SB3 / real-robot replay loops."""
    w = writer()
    if getattr(w, "capture", "episodes") != "full":
        w.set_capture("full")
    return GymStepCapture(w, w.run_id, source=source)


def watch(runner: object, *, env: object = None, source: str = SOURCE_PHYSX, traced_envs: int = 8, capture: str = "full") -> RslRlStorageCapture:
    """Convenience: init (if needed), register named dims from the env, and return
    an RSL-RL capture bound to `runner`. Call `.capture_rollout(runner, it)` each
    iteration. Capture defaults to ``full`` so the per-step policy trace actually
    lands — the previous default (``episodes``) silently dropped STEP events.
    """
    init(capture=capture)
    cap = rsl_rl_capture(source=source, traced_envs=traced_envs)
    target_env = env if env is not None else getattr(runner, "env", None)
    if target_env is not None:
        try:
            register_dims(infer_dim_registry(target_env, _RUN_ID))
        except Exception:
            register_dims(policy_state_registry(_RUN_ID))
    else:
        register_dims(policy_state_registry(_RUN_ID))
    return cap


def close(timeout: float = 10.0) -> None:
    """Flush and stop the writer. Safe to call multiple times."""
    global _WRITER
    if _WRITER is not None:
        _WRITER.close(timeout=timeout)
        _WRITER = None
