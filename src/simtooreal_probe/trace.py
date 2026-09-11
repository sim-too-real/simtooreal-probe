"""
probe.trace — the canonical PolicyTraceEvent contract.

This is the SPINE of the Policy Debugging Layer. Every emitter (RSL-RL, SB3,
raw gym, LeRobot, real-robot log adapters) produces these events; every
consumer (Policy CI, Failure Reconstruction, the Debugging Layer) reads them.
Get this envelope right and the three pillars are *views* over one substrate.

Design principles
-----------------
* **Three grains, one schema.** Everything in RL policy observability lives at
  exactly one of three grains. They are discriminated by `grain`, not by table:
    - ITERATION : per training iteration   (mean reward, KL, entropy, grad-norm …)
    - EPISODE   : per episode rollup       (return, length, success, termination)
    - STEP      : per environment-frame    (obs, action, reward-terms, policy-state)
* **Orthogonal source axis.** `source` ("sim:physx" | "sim:newton" | "sim:mjx" |
  "real:<serial>") tags WHICH engine/robot produced the frame. The *same* schema
  holds a Newton rollout, a PhysX rollout, and a real Go2 episode — so the
  sim-to-real gap is a `WHERE source IN (...)` query, not a bespoke pipeline.
* **Named vectors.** `vectors` ("obs", "action", "reward_terms", "policy_state")
  carry raw floats; a per-run `DimRegistry` resolves component names once
  (reward_terms -> {"track_lin_vel": 1.2, "action_rate": -0.01}), so attribution
  is term-level by name, never an anonymous float index.

Zero hard dependencies beyond numpy — works inside the Isaac container.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence

# Bump when the envelope shape changes. Consumers branch on this; the v2 ingest
# endpoint up-converts older payloads. Never reuse a number.
SCHEMA_VERSION = 1


class Grain(str, Enum):
    """The granularity of a trace event. `str` mixin -> JSON-serialises to the value."""

    STEP = "step"
    EPISODE = "episode"
    ITERATION = "iteration"


# ── Source axis ───────────────────────────────────────────────────────────────
# Canonical engine identifiers. Free-form `real:<serial>` is allowed for hardware.

SOURCE_PHYSX = "sim:physx"
SOURCE_NEWTON = "sim:newton"
SOURCE_MJX = "sim:mjx"
SOURCE_MUJOCO = "sim:mujoco"
SOURCE_MJWARP = "sim:mjwarp"   # mjlab / MuJoCo-Warp (Isaac-Lab manager API on MuJoCo)


def real_source(serial: str) -> str:
    """Build a real-robot source tag, e.g. real_source('go2-0xA31F') -> 'real:go2-0xA31F'."""
    serial = (serial or "unknown").strip().replace(" ", "-")
    return f"real:{serial}"


def is_sim(source: str) -> bool:
    return source.startswith("sim:")


def is_real(source: str) -> bool:
    return source.startswith("real:")


# ── Vector field names (the well-known keys inside `vectors`) ─────────────────
V_OBS = "obs"
V_ACTION = "action"
V_REWARD_TERMS = "reward_terms"
V_POLICY_STATE = "policy_state"  # log_prob, value, action_mean, action_logstd, …

# Well-known scalar keys (free-form is allowed; these are the ones consumers know).
S_REWARD = "reward"
S_RETURN = "return"
S_VALUE = "value"
S_LOG_PROB = "log_prob"
S_KL = "kl_divergence"
S_ENTROPY = "entropy"
S_GRAD_NORM = "gradient_norm"
S_CLIP_FRACTION = "clip_fraction"
S_EXPLAINED_VAR = "explained_variance"
S_MEAN_REWARD = "mean_reward"
S_EPISODE_LENGTH = "episode_length"


def _fin(v: Any) -> Optional[float]:
    """Coerce to a finite float, or None for NaN/Inf/uncoercible.

    Non-finite values are common in the exact failure rollouts this tool captures
    (diverged value fn, exploded action). `None` -> JSON `null` keeps the trace
    valid for serde_json / DuckDB / any strict reader, while still marking the
    field as "went bad here" rather than silently coercing to 0.
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _to_floats(x: Any) -> List[Optional[float]]:
    """Best-effort coerce scalar / list / nested / numpy / torch -> flat list of floats.

    ALWAYS flattens to 1-D (image obs, [E,D] slices, nested lists all survive) and
    NEVER raises — the contract is that capture must not crash a training loop.
    Non-finite values become None. numpy is a core dep; torch is duck-typed
    (`.detach`/`.cpu`) so this module never imports torch.
    """
    if x is None:
        return []
    if isinstance(x, Mapping):  # e.g. reward_terms passed as a dict -> ordered values
        x = list(x.values())
    # Move torch/ProxyArray tensors to host without importing torch.
    if hasattr(x, "detach"):
        try:
            x = x.detach()
        except Exception:
            pass
    if hasattr(x, "cpu"):
        try:
            x = x.cpu()
        except Exception:
            pass
    # Primary path: numpy flattens any shape (incl. >=2-D) in one shot.
    try:
        import numpy as np
        arr = np.asarray(x, dtype="float64").reshape(-1)
        return [v if math.isfinite(v) else None for v in arr.tolist()]
    except Exception:
        pass
    # Fallback for objects numpy can't ingest: tolist + manual recursive flatten.
    if hasattr(x, "tolist"):
        try:
            x = x.tolist()
        except Exception:
            pass
    out: List[Optional[float]] = []
    _flatten_into(x, out)
    return out


def _flatten_into(x: Any, out: List[Optional[float]]) -> None:
    if isinstance(x, (int, float)):
        out.append(_fin(x))
        return
    if isinstance(x, (str, bytes)):
        out.append(None)
        return
    try:
        for v in x:
            _flatten_into(v, out)
    except TypeError:
        out.append(_fin(x))


def _san_scalars(d: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    return {k: _fin(v) for k, v in d.items()}


def _san_vectors(d: Mapping[str, Any]) -> Dict[str, List[Optional[float]]]:
    return {k: [(_fin(x)) for x in vec] for k, vec in d.items()}


@dataclass(slots=True)
class PolicyTraceEvent:
    """One self-describing observability event at a single grain.

    Construct via the `step()`, `episode()`, `iteration()` classmethods rather
    than the raw initialiser — they enforce the per-grain field contract.
    """

    grain: Grain
    run_id: str
    source: str = SOURCE_PHYSX
    schema_version: int = SCHEMA_VERSION

    # Coordinates (which subset is set depends on grain).
    iteration: Optional[int] = None
    episode_id: Optional[str] = None
    step_idx: Optional[int] = None
    env_idx: Optional[int] = None

    t_wall: float = field(default_factory=time.time)

    scalars: Dict[str, float] = field(default_factory=dict)
    vectors: Dict[str, List[float]] = field(default_factory=dict)

    # Free-form, low-cardinality tags (task, robot, terminated reason, …).
    tags: Dict[str, Any] = field(default_factory=dict)

    # ── Grain constructors (the public, contract-enforcing API) ───────────────

    @classmethod
    def iteration_event(
        cls,
        run_id: str,
        iteration: int,
        scalars: Mapping[str, float],
        *,
        source: str = SOURCE_PHYSX,
        tags: Optional[Mapping[str, Any]] = None,
    ) -> "PolicyTraceEvent":
        return cls(
            grain=Grain.ITERATION,
            run_id=run_id,
            source=source,
            iteration=int(iteration),
            scalars={k: f for k, v in scalars.items() if (f := _fin(v)) is not None},
            tags=dict(tags or {}),
        )

    @classmethod
    def episode_event(
        cls,
        run_id: str,
        episode_id: str,
        *,
        source: str = SOURCE_PHYSX,
        iteration: Optional[int] = None,
        env_idx: Optional[int] = None,
        ep_return: Optional[float] = None,
        length: Optional[int] = None,
        success: Optional[bool] = None,
        termination: Optional[str] = None,
        scalars: Optional[Mapping[str, float]] = None,
        reward_terms: Optional[Mapping[str, float]] = None,
        tags: Optional[Mapping[str, Any]] = None,
    ) -> "PolicyTraceEvent":
        sc: Dict[str, float] = {k: float(v) for k, v in (scalars or {}).items() if v is not None}
        if ep_return is not None:
            sc[S_RETURN] = float(ep_return)
        if length is not None:
            sc[S_EPISODE_LENGTH] = float(length)
        vec: Dict[str, List[float]] = {}
        if reward_terms:
            vec[V_REWARD_TERMS] = [float(v) for v in reward_terms.values()]
        tg = dict(tags or {})
        if success is not None:
            tg["success"] = bool(success)
        if termination is not None:
            tg["termination"] = str(termination)
        return cls(
            grain=Grain.EPISODE,
            run_id=run_id,
            source=source,
            iteration=iteration,
            episode_id=str(episode_id),
            env_idx=env_idx,
            scalars=sc,
            vectors=vec,
            tags=tg,
        )

    @classmethod
    def step_event(
        cls,
        run_id: str,
        episode_id: str,
        step_idx: int,
        *,
        source: str = SOURCE_PHYSX,
        iteration: Optional[int] = None,
        env_idx: Optional[int] = None,
        obs: Any = None,
        action: Any = None,
        reward: Optional[float] = None,
        reward_terms: Any = None,
        policy_state: Any = None,
        scalars: Optional[Mapping[str, float]] = None,
        tags: Optional[Mapping[str, Any]] = None,
    ) -> "PolicyTraceEvent":
        vec: Dict[str, List[float]] = {}
        if obs is not None:
            vec[V_OBS] = _to_floats(obs)
        if action is not None:
            vec[V_ACTION] = _to_floats(action)
        if reward_terms is not None:
            vec[V_REWARD_TERMS] = _to_floats(reward_terms)
        if policy_state is not None:
            vec[V_POLICY_STATE] = _to_floats(policy_state)
        sc: Dict[str, float] = {k: float(v) for k, v in (scalars or {}).items() if v is not None}
        if reward is not None:
            sc[S_REWARD] = float(reward)
        return cls(
            grain=Grain.STEP,
            run_id=run_id,
            source=source,
            iteration=iteration,
            episode_id=str(episode_id),
            step_idx=int(step_idx),
            env_idx=env_idx,
            scalars=sc,
            vectors=vec,
            tags=dict(tags or {}),
        )

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["grain"] = self.grain.value  # Enum -> str
        # Sanitize NaN/Inf -> null so every downstream reader gets valid JSON.
        d["scalars"] = _san_scalars(d.get("scalars") or {})
        d["vectors"] = _san_vectors(d.get("vectors") or {})
        # Drop None coordinates to keep payloads compact (95k+ events/run).
        for k in ("iteration", "episode_id", "step_idx", "env_idx"):
            if d.get(k) is None:
                d.pop(k, None)
        if not d["scalars"]:
            d.pop("scalars")
        if not d["vectors"]:
            d.pop("vectors")
        if not d["tags"]:
            d.pop("tags")
        return d

    def to_json(self) -> str:
        # allow_nan=False is safe because to_dict() already sanitized non-finite
        # values; it acts as a hard guard that we never emit bare NaN/Infinity.
        return json.dumps(self.to_dict(), separators=(",", ":"), allow_nan=False)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PolicyTraceEvent":
        try:
            grain = Grain(d.get("grain", "step"))
        except ValueError:
            grain = Grain.STEP  # tolerate forward-compat unknown grains
        return cls(
            grain=grain,
            run_id=d.get("run_id", ""),
            source=d.get("source", SOURCE_PHYSX),
            schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
            iteration=d.get("iteration"),
            episode_id=d.get("episode_id"),
            step_idx=d.get("step_idx"),
            env_idx=d.get("env_idx"),
            t_wall=float(d.get("t_wall", time.time())),
            scalars=dict(d.get("scalars", {}) or {}),
            vectors={k: list(v) for k, v in (d.get("vectors", {}) or {}).items()},
            tags=dict(d.get("tags", {}) or {}),
        )


# ── DimRegistry — resolves named components for vector fields, once per run ───


@dataclass(slots=True)
class DimRegistry:
    """Maps each vector field -> ordered component names.

    Registered once per run (cheap), shipped as a `_dims.json` sidecar so that
    `reward_terms[3]` can always be resolved back to e.g. "flat_orientation".
    Missing entries are tolerated: consumers fall back to positional `field[i]`.
    """

    run_id: str
    dims: Dict[str, List[str]] = field(default_factory=dict)

    def register(self, field_name: str, names: Sequence[str]) -> None:
        self.dims[field_name] = [str(n) for n in names]

    def names(self, field_name: str) -> List[str]:
        return self.dims.get(field_name, [])

    def resolve(self, field_name: str, values: Sequence[float]) -> Dict[str, float]:
        """Zip a raw vector back into {component_name: value}.

        Falls back to f"{field}_{i}" for any component without a registered name.
        """
        names = self.dims.get(field_name)
        out: Dict[str, float] = {}
        for i, v in enumerate(values):
            key = names[i] if names and i < len(names) else f"{field_name}_{i}"
            out[key] = float(v) if v is not None and _fin(v) is not None else float("nan")
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {"run_id": self.run_id, "schema_version": SCHEMA_VERSION, "dims": self.dims}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "DimRegistry":
        return cls(run_id=d.get("run_id", ""), dims={k: list(v) for k, v in (d.get("dims", {}) or {}).items()})
