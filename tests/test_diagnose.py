"""Failure taxonomy detectors — entropy, KL, VLA action-std, reward hacking."""
from __future__ import annotations

from pathlib import Path

from simtooreal_probe.diagnose import diagnose, health_score
from simtooreal_probe.sinks import LocalSink
from simtooreal_probe.trace import PolicyTraceEvent
from simtooreal_probe.writer import TraceWriter
from simtooreal_probe import ProbeRun, diagnose_run, lerobot_to_traces


def test_entropy_collapse_fires() -> None:
    iters = [{"iteration": i, "entropy": 1.2 if i < 20 else 0.005} for i in range(40)]
    findings = diagnose(iterations=iters)
    ids = {f.id for f in findings}
    assert "entropy_collapse" in ids
    hit = next(f for f in findings if f.id == "entropy_collapse")
    assert hit.severity in ("critical", "high")
    assert hit.family == "rl"


def test_kl_spike_fires() -> None:
    iters = [{"iteration": i, "kl_divergence": 0.02 if i < 10 else 0.8} for i in range(20)]
    findings = diagnose(iterations=iters)
    assert any(f.id == "kl_spike" for f in findings)


def test_action_std_zero_is_critical() -> None:
    steps = []
    for i in range(24):
        steps.append(
            {
                "step_idx": i,
                "vectors": {"action": [0.1 * (i % 3), 0.0, 0.2]},  # dim 1 frozen
            }
        )
    findings = diagnose(steps=steps)
    hit = next(f for f in findings if f.id == "action_std_zero")
    assert hit.severity == "critical"
    assert 1 in hit.evidence["zero_dims"]


def test_reward_hacking_proxy_vs_success() -> None:
    iters = [{"iteration": i, "mean_reward": 1.0 + i * 0.1} for i in range(20)]
    episodes = [
        {"iteration": i, "tags": {"success": i < 8}, "scalars": {"return": float(i)}}
        for i in range(16)
    ]
    findings = diagnose(iterations=iters, episodes=episodes)
    assert any(f.id == "reward_hacking" for f in findings)


def test_idle_policy() -> None:
    steps = [{"vectors": {"action": [0.001, -0.002]}} for _ in range(20)]
    findings = diagnose(steps=steps)
    assert any(f.id == "idle_policy" for f in findings)


def test_instruction_decoupling() -> None:
    episodes = [
        {"tags": {"success": False, "primitive_success": True}} for _ in range(12)
    ]
    findings = diagnose(episodes=episodes)
    hit = next(f for f in findings if f.id == "instruction_decoupling")
    assert hit.family == "vla"


def test_health_score_penalizes_critical() -> None:
    findings = diagnose(
        iterations=[{"iteration": i, "entropy": 0.001} for i in range(20)]
    )
    h = health_score(findings, failure_rate=0.6)
    assert h["score"] < 80
    assert h["band"] in ("low", "moderate")


def test_probe_run_diagnose_roundtrip(tmp_path: Path) -> None:
    run_id = "diag-run"
    sink = LocalSink(run_id, root=str(tmp_path), prefer_parquet=False)
    writer = TraceWriter(run_id, [sink], capture="full")
    for i in range(24):
        writer.emit(
            PolicyTraceEvent.iteration_event(
                run_id,
                i,
                {"mean_reward": 1.0, "entropy": 1.1 if i < 10 else 0.004, "kl_divergence": 0.01},
                source="sim:physx",
            )
        )
        writer.emit(
            PolicyTraceEvent.episode_event(
                run_id,
                f"ep-{i}",
                source="sim:physx",
                ep_return=1.0,
                length=40 if i < 10 else 8,
                success=False,
                termination="base_contact",
            )
        )
    writer.close()
    pr = ProbeRun(run_id, root=str(tmp_path))
    found = diagnose_run(pr)
    ids = {f.id for f in found}
    assert "entropy_collapse" in ids
    summary = pr.findings()
    assert isinstance(summary, list)
    assert any(f["id"] == "entropy_collapse" for f in summary)


def test_lerobot_jsonl_import(tmp_path: Path) -> None:
    ds = tmp_path / "lerobot-ds"
    ds.mkdir()
    (ds / "meta").mkdir()
    (ds / "meta" / "info.json").write_text(
        '{"robot_type": "so101", "task": "pick the cube"}', encoding="utf-8"
    )
    frames = ds / "frames.jsonl"
    lines = []
    for i in range(6):
        rec = {
            "episode_index": 0 if i < 3 else 1,
            "frame_index": i,
            "observation.state": [0.1, 0.2],
            "action": [0.01, 0.02],
            "next.reward": 0.1,
            "next.done": i in (2, 5),
            "task": "pick the cube",
        }
        import json

        lines.append(json.dumps(rec))
    frames.write_text("\n".join(lines), encoding="utf-8")
    stats = lerobot_to_traces(str(ds), run_id="lerobot-import", root=str(tmp_path / "probe"))
    assert stats["steps"] == 6
    assert stats["episodes"] >= 2
    pr = ProbeRun("lerobot-import", root=str(tmp_path / "probe"))
    assert any(s.startswith("real:") for s in pr.sources())


def test_nan_html_export(tmp_path: Path) -> None:
    run_id = "nan-run"
    sink = LocalSink(run_id, root=str(tmp_path), prefer_parquet=False)
    writer = TraceWriter(run_id, [sink], capture="full")
    writer.emit(
        PolicyTraceEvent.step_event(
            run_id, "ep-0", 0, source="sim:mujoco", obs=[float("nan"), 1.0], action=[None], reward=float("nan")
        )
    )
    writer.emit(
        PolicyTraceEvent.episode_event(
            run_id, "ep-0", source="sim:mujoco", success=False, termination="base_contact", length=1
        )
    )
    writer.close()
    pr = ProbeRun(run_id, root=str(tmp_path))
    path = pr.to_html("ep-0")
    assert Path(path).exists()
    html = Path(path).read_text(encoding="utf-8")
    assert "ep-0" in html
