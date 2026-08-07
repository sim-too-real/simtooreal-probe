"""Smoke tests for the open-core simtooreal-probe package."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_source_constants() -> None:
    import simtooreal_probe as probe

    assert probe.SOURCE_PHYSX == "sim:physx"
    assert probe.SOURCE_NEWTON == "sim:newton"
    assert probe.SOURCE_MJX == "sim:mjx"
    assert probe.SOURCE_MUJOCO == "sim:mujoco"
    assert probe.SOURCE_MJWARP == "sim:mjwarp"
    assert probe.real_source("go2_01") == "real:go2_01"


def test_public_exports_include_viz() -> None:
    import simtooreal_probe as probe

    assert hasattr(probe, "viz")
    assert hasattr(probe, "export_failure_html")
    assert hasattr(probe.ProbeRun, "to_html")
    assert "to_html" in dir(probe.ProbeRun)


def test_local_roundtrip_and_summary(tmp_path: Path) -> None:
    import simtooreal_probe as probe
    from simtooreal_probe import LocalRunSummary, PolicyTraceEvent, ProbeRun
    from simtooreal_probe.sinks import LocalSink
    from simtooreal_probe.writer import TraceWriter

    run_id = "smoke-run"
    sink = LocalSink(run_id, root=str(tmp_path), prefer_parquet=False)
    writer = TraceWriter(run_id, [sink], capture="full")
    writer.emit(
        PolicyTraceEvent.iteration_event(
            run_id, 1, {"mean_reward": 1.5, "kl_divergence": 0.02}, source="sim:mujoco"
        )
    )
    writer.emit(
        PolicyTraceEvent.episode_event(
            run_id,
            "ep-0",
            source="sim:mujoco",
            ep_return=3.0,
            length=10,
            success=False,
            termination="base_contact",
        )
    )
    writer.emit(
        PolicyTraceEvent.step_event(
            run_id,
            "ep-0",
            step_idx=0,
            source="sim:mujoco",
            obs=[0.1, 0.2],
            action=[0.0],
            reward=0.5,
        )
    )
    writer.close()

    pr = ProbeRun(run_id, root=str(tmp_path))
    assert "sim:mujoco" in pr.sources()
    fails = pr.failures()
    assert len(fails) == 1
    summary = LocalRunSummary(run_id, root=str(tmp_path)).summary()
    assert summary["iteration_count"] >= 1
    assert summary["failure_count"] == 1
    csv_path = LocalRunSummary(run_id, root=str(tmp_path)).export_csv(tmp_path / "out.csv")
    assert csv_path.exists()


def test_http_sink_without_url_disables() -> None:
    from simtooreal_probe.sinks import HttpSink
    from simtooreal_probe.trace import PolicyTraceEvent

    sink = HttpSink("r1", base_url="")
    sink.write(
        [
            PolicyTraceEvent.iteration_event(
                "r1", 0, {"mean_reward": 1.0}, source="sim:physx"
            )
        ]
    )
    assert sink._enabled is False


def test_http_sink_accepts_custom_client() -> None:
    from simtooreal_probe.sinks import HttpSink
    from simtooreal_probe.trace import PolicyTraceEvent

    posted = []

    class FakeClient:
        def post(self, path, payload):
            posted.append((path, payload))
            return {"success": True}

    sink = HttpSink("r1", client=FakeClient())
    sink.write(
        [
            PolicyTraceEvent.iteration_event(
                "r1", 1, {"mean_reward": 2.0}, source="sim:physx"
            )
        ]
    )
    assert len(posted) == 1
    assert posted[0][0] == "/api/v2/trace"
    assert posted[0][1]["run_id"] == "r1"


def test_rsl_rl_namespace_capture(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    from simtooreal_probe.capture import RslRlStorageCapture
    from simtooreal_probe.sinks import LocalSink
    from simtooreal_probe.writer import TraceWriter

    run_id = "mjx-capture"
    sink = LocalSink(run_id, root=str(tmp_path), prefer_parquet=False)
    writer = TraceWriter(run_id, [sink], capture="full")
    cap = RslRlStorageCapture(writer, run_id, source="sim:mjx", traced_envs=2)

    T, N, OBS, ACT = 4, 3, 5, 2
    dones = np.zeros((T, N), dtype="float32")
    dones[-1, 0] = 1.0
    ns = SimpleNamespace(
        observations=np.random.rand(T, N, OBS).astype("float32"),
        actions=np.random.rand(T, N, ACT).astype("float32"),
        rewards=np.ones((T, N), dtype="float32"),
        actions_log_prob=np.zeros((T, N), dtype="float32"),
        values=np.zeros((T, N), dtype="float32"),
        returns=np.zeros((T, N), dtype="float32"),
        dones=dones,
    )
    emitted = cap.capture_rollout(ns, iteration=7)
    writer.close()
    assert emitted == T * 2
    path = tmp_path / run_id / "sim%3Amjx" / "step.jsonl"
    steps = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert len(steps) == T * 2
