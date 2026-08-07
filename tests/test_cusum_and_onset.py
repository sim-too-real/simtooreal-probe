"""CUSUM change-point and ProbeRun.first_abnormal tests."""
from __future__ import annotations

from pathlib import Path

from simtooreal_probe.query import _cusum_changepoint
from simtooreal_probe.sinks import LocalSink
from simtooreal_probe.trace import PolicyTraceEvent
from simtooreal_probe.writer import TraceWriter
from simtooreal_probe import ProbeRun


def test_cusum_no_false_alarm_on_noise() -> None:
    # Stable series around 1.0 — must not fire.
    values = [1.0 + 0.01 * ((i % 5) - 2) for i in range(40)]
    assert _cusum_changepoint(values, low=True) is None
    assert _cusum_changepoint(values, low=False) is None


def test_cusum_detects_sustained_reward_collapse() -> None:
    # Warmup ~1.0 then sustained collapse to -10 (tens of σ).
    values = [1.0] * 15 + [-10.0] * 20
    res = _cusum_changepoint(values, low=True)
    assert res is not None
    onset, shift = res
    assert onset >= 12  # after warmup
    assert shift >= 5.0


def test_cusum_detects_action_explosion() -> None:
    values = [0.1] * 15 + [50.0] * 20
    res = _cusum_changepoint(values, low=False)
    assert res is not None
    onset, shift = res
    assert onset >= 12
    assert shift >= 5.0


def test_cusum_too_short_returns_none() -> None:
    assert _cusum_changepoint([1.0] * 10, low=True) is None


def test_first_abnormal_on_collapse_episode(tmp_path: Path) -> None:
    run_id = "onset-run"
    sink = LocalSink(run_id, root=str(tmp_path), prefer_parquet=False)
    writer = TraceWriter(run_id, [sink], capture="full")
    ep = "ep-collapse"
    n = 40
    for i in range(n):
        reward = 1.0 if i < 18 else -8.0
        value = 0.5 if i < 18 else -5.0
        writer.emit(
            PolicyTraceEvent.step_event(
                run_id,
                ep,
                step_idx=i,
                source="sim:mujoco",
                obs=[0.0, 0.0],
                action=[0.1, 0.1],
                reward=reward,
                policy_state=[0.0, value, 0.0],
            )
        )
    writer.emit(
        PolicyTraceEvent.episode_event(
            run_id,
            ep,
            source="sim:mujoco",
            ep_return=-50.0,
            length=n,
            success=False,
            termination="base_contact",
        )
    )
    writer.close()

    pr = ProbeRun(run_id, root=str(tmp_path))
    onset = pr.first_abnormal(ep, source="sim:mujoco")
    assert onset is not None
    assert onset["episode_id"] == ep
    assert onset["channel"] in ("reward_collapse", "value_collapse", "action_explosion")
    assert onset["detector"] == "cusum_v1"
    assert onset["onset_step"] >= 12
