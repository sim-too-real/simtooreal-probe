"""
probe.adapt — ingest EXTERNAL logs (real-robot bags, recorded datasets) into the
same probe trace store as sim rollouts, tagged with a `real:<serial>` source.

This is what makes "sim AND real in one schema" a demonstrated capability, not a
claim: once a real episode lands as `source=real:<serial>`, the SAME onset
detector (probe.query.first_abnormal / GET /runs/:id/onset) and the SAME
onset-compare endpoint work on it — so you can diff a real rollout against the
sim baseline.

Two layers:
  * `emit_records(...)` — the format-agnostic core. Feed it already-decoded
    per-step records (dicts with obs/action/reward[/done/success/termination])
    from ANY source; it emits probe traces via the SDK. Fully tested.
  * `mcap_to_traces(...)` — the MCAP/ROS2 front-end. Decodes an MCAP file into
    records, then calls `emit_records`. Needs `pip install mcap` (+
    `mcap-ros2-support` for ROS2 CDR topics); import-guarded with a clear error.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .trace import DimRegistry, V_ACTION, V_OBS, V_REWARD_TERMS, real_source


def emit_records(
    records: Iterable[Dict[str, Any]],
    *,
    run_id: str,
    source: str,
    http: Optional[bool] = None,
    obs_names: Optional[Sequence[str]] = None,
    action_names: Optional[Sequence[str]] = None,
    reward_term_names: Optional[Sequence[str]] = None,
    root: Optional[str] = None,
) -> Dict[str, int]:
    """Emit already-decoded per-step records as probe traces under a real-robot source.

    Each record is a dict that may carry: ``obs``, ``action``, ``reward``,
    ``reward_terms``, ``policy_state`` (all optional, any shape — flattened by the
    trace contract), plus episode-boundary hints ``done`` (bool), ``success``
    (bool), ``termination`` (str). One episode runs until a ``done`` record;
    a trailing partial episode is closed on exhaustion.

    Returns ``{"steps": n, "episodes": n}``. Flushes + closes the writer (this is
    an offline conversion, not a live loop), so the run is immediately queryable.

    Always uses ``capture="full"`` so per-step records are persisted (the process
    default is ``episodes``, which would drop STEP grains).
    """
    from . import close as _close, gym_capture, init, register_dims

    # Offline adapters must close any prior writer so capture=full is applied.
    _close()
    init(run_id=run_id, http=http, root=root, capture="full")
    if obs_names or action_names or reward_term_names:
        reg = DimRegistry(run_id=run_id)
        reg.register("policy_state", ["log_prob", "value", "return"])
        if obs_names:
            reg.register(V_OBS, list(obs_names))
        if action_names:
            reg.register(V_ACTION, list(action_names))
        if reward_term_names:
            reg.register(V_REWARD_TERMS, list(reward_term_names))
        register_dims(reg)

    cap = gym_capture(source=source)
    n_steps = 0
    n_eps = 0
    pending = False
    for rec in records:
        cap.record_step(
            obs=rec.get("obs"),
            action=rec.get("action"),
            reward=rec.get("reward"),
            reward_terms=rec.get("reward_terms"),
            policy_state=rec.get("policy_state"),
        )
        n_steps += 1
        pending = True
        if rec.get("done"):
            cap.end_episode(success=rec.get("success"), termination=rec.get("termination"))
            n_eps += 1
            pending = False
    if pending:
        cap.end_episode()
        n_eps += 1

    _close()
    return {"steps": n_steps, "episodes": n_eps}


def emit_episodes(
    episodes: Iterable[Sequence[Dict[str, Any]]],
    *,
    run_id: str,
    source: str,
    **kwargs: Any,
) -> Dict[str, int]:
    """Like `emit_records` but takes an explicit list of episodes (each a record
    sequence). Inserts a synthetic `done` after each so boundaries are exact."""
    def _flatten() -> Iterable[Dict[str, Any]]:
        for ep in episodes:
            ep = list(ep)
            for i, rec in enumerate(ep):
                r = dict(rec)
                if i == len(ep) - 1:
                    r.setdefault("done", True)
                yield r

    return emit_records(_flatten(), run_id=run_id, source=source, **kwargs)


# ── MCAP / ROS2 front-end ─────────────────────────────────────────────────────


def mcap_to_traces(
    path: str,
    *,
    run_id: str,
    serial: str,
    obs_topics: Sequence[str],
    action_topic: str,
    reward_topic: Optional[str] = None,
    done_topic: Optional[str] = None,
    field: str = "position",
    http: Optional[bool] = None,
    root: Optional[str] = None,
) -> Dict[str, int]:
    """Decode an MCAP (rosbag2) into probe traces with ``source=real:<serial>``.

    obs_topics  : topics whose numeric `field` arrays concatenate into the obs vector
                  (e.g. ["/joint_states", "/imu"]).
    action_topic: topic carrying the commanded action (`field` array).
    reward_topic/done_topic: optional scalar/bool topics.
    field       : message attribute to read as a float array (ROS2 JointState -> "position").

    Messages are bucketed onto the action topic's timeline (one record per action
    message; obs/reward use the most recent value per topic — standard zero-order
    hold). Requires `pip install mcap`; ROS2 CDR topics also need
    `mcap-ros2-support`. Foxglove-`json` encoded topics work with `mcap` alone.
    """
    records = _read_mcap_records(
        path, obs_topics, action_topic, reward_topic, done_topic, field
    )
    return emit_records(
        records, run_id=run_id, source=real_source(serial), http=http, root=root
    )


def _read_mcap_records(
    path: str,
    obs_topics: Sequence[str],
    action_topic: str,
    reward_topic: Optional[str],
    done_topic: Optional[str],
    field: str,
) -> List[Dict[str, Any]]:
    try:
        from mcap.reader import make_reader  # type: ignore
    except Exception as e:  # pragma: no cover - exercised only without the lib
        raise ImportError(
            "probe.adapt.mcap_to_traces requires the 'mcap' package "
            "(`pip install mcap`; add `mcap-ros2-support` for ROS2 CDR topics)."
        ) from e

    # Optional ROS2 CDR decoder; if absent we fall back to JSON-encoded messages.
    try:
        from mcap_ros2.decoder import DecoderFactory as _Ros2Decoder  # type: ignore
        decoders = [_Ros2Decoder()]
    except Exception:  # pragma: no cover
        decoders = []

    def _to_floats(msg: Any) -> List[float]:
        val = getattr(msg, field, None)
        if val is None and isinstance(msg, dict):
            val = msg.get(field)
        if val is None:
            return []
        try:
            return [float(x) for x in val]
        except TypeError:
            try:
                return [float(val)]
            except Exception:
                return []

    # Latest value per non-action topic (zero-order hold), flushed on each action msg.
    latest_obs: Dict[str, List[float]] = {t: [] for t in obs_topics}
    latest_reward: Optional[float] = None
    latest_done: bool = False
    records: List[Dict[str, Any]] = []

    with open(path, "rb") as f:
        reader = make_reader(f, decoder_factories=decoders) if decoders else make_reader(f)
        iterate = getattr(reader, "iter_decoded_messages", None) or reader.iter_messages
        for item in iterate():
            # iter_decoded_messages -> (schema, channel, message, decoded);
            # iter_messages -> (schema, channel, message)
            if len(item) == 4:
                _schema, channel, _message, decoded = item
            else:
                _schema, channel, decoded = item
            topic = getattr(channel, "topic", None)
            if topic in latest_obs:
                latest_obs[topic] = _to_floats(decoded)
            elif reward_topic and topic == reward_topic:
                fs = _to_floats(decoded)
                latest_reward = fs[0] if fs else None
            elif done_topic and topic == done_topic:
                fs = _to_floats(decoded)
                latest_done = bool(fs[0]) if fs else True
            elif topic == action_topic:
                obs_vec: List[float] = []
                for t in obs_topics:
                    obs_vec.extend(latest_obs.get(t, []))
                rec: Dict[str, Any] = {"obs": obs_vec, "action": _to_floats(decoded)}
                if latest_reward is not None:
                    rec["reward"] = latest_reward
                if latest_done:
                    rec["done"] = True
                    latest_done = False
                records.append(rec)
    return records
