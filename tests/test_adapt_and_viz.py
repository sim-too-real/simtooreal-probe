"""Adapters (emit_records) and HTML failure replay."""
from __future__ import annotations

from pathlib import Path

from simtooreal_probe.adapt import emit_records
from simtooreal_probe import ProbeRun, export_failure_html, close
from simtooreal_probe.sinks import LocalSink
from simtooreal_probe.trace import PolicyTraceEvent
from simtooreal_probe.writer import TraceWriter


def test_emit_records_roundtrip(tmp_path: Path) -> None:
    close()  # ensure clean process-wide writer
    run_id = "adapt-run"
    records = []
    for i in range(5):
        records.append(
            {
                "obs": [float(i), 0.1],
                "action": [0.2],
                "reward": 0.5,
                "done": i == 4,
                "success": False,
                "termination": "timeout" if i == 4 else None,
            }
        )
    out = emit_records(
        records,
        run_id=run_id,
        source="real:go2_01",
        http=False,
        root=str(tmp_path),
        obs_names=["j0", "j1"],
        action_names=["a0"],
    )
    assert out["steps"] == 5
    assert out["episodes"] == 1

    pr = ProbeRun(run_id, root=str(tmp_path))
    assert "real:go2_01" in pr.sources()
    steps = pr.steps(source="real:go2_01")
    # list or DataFrame
    n = len(steps) if not hasattr(steps, "shape") else int(steps.shape[0])
    assert n == 5


def test_to_html_and_export_failure_html(tmp_path: Path) -> None:
    run_id = "html-run"
    sink = LocalSink(run_id, root=str(tmp_path), prefer_parquet=False)
    writer = TraceWriter(run_id, [sink], capture="full")
    ep = "ep-0"
    for i in range(6):
        writer.emit(
            PolicyTraceEvent.step_event(
                run_id,
                ep,
                step_idx=i,
                source="sim:physx",
                obs=[0.1 * i, 0.2],
                action=[0.0, 0.1],
                reward=float(i),
                policy_state=[0.0, 1.0, 0.5],
            )
        )
    writer.emit(
        PolicyTraceEvent.episode_event(
            run_id,
            ep,
            source="sim:physx",
            ep_return=15.0,
            length=6,
            success=False,
            termination="base_contact",
        )
    )
    writer.close()

    pr = ProbeRun(run_id, root=str(tmp_path))
    path = pr.to_html(ep, source="sim:physx")
    p = Path(path)
    assert p.exists()
    text = p.read_text(encoding="utf-8")
    assert "Failure Replay" in text
    assert ep in text
    assert run_id in text
    assert "<!doctype html>" in text.lower() or "<!DOCTYPE html>" in text

    out2 = tmp_path / "custom_replay.html"
    path2 = export_failure_html(run_id, ep, root=str(tmp_path), source="sim:physx", out=str(out2))
    assert Path(path2) == out2
    assert out2.exists()


def test_to_html_empty_episode_still_writes(tmp_path: Path) -> None:
    run_id = "empty-html"
    sink = LocalSink(run_id, root=str(tmp_path), prefer_parquet=False)
    writer = TraceWriter(run_id, [sink], capture="full")
    writer.emit(
        PolicyTraceEvent.episode_event(
            run_id,
            "missing-steps",
            source="sim:physx",
            ep_return=0.0,
            length=0,
            success=False,
            termination="unknown",
        )
    )
    writer.close()
    pr = ProbeRun(run_id, root=str(tmp_path))
    path = pr.to_html("missing-steps")
    assert Path(path).exists()
    assert "missing-steps" in Path(path).read_text(encoding="utf-8")
