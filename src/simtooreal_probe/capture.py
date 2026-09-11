"""
probe.capture — turn a training loop / rollout into PolicyTraceEvents.

The headline capability is `RslRlStorageCapture`: it reads RSL-RL's rollout
storage **once per iteration** and emits the per-step policy trace
(obs / action / reward / log_prob / value / return). Those tensors are ALREADY
buffered by the algorithm during rollout collection, so capturing them costs one
device->host copy per iteration — NOT a per-step Python callback. That is the
whole reason per-step capture can be cheap enough to leave on.

`GymStepCapture` covers the framework-agnostic path (raw gym/torch loops, SB3,
real-robot replay) where there is no rollout buffer — it emits one STEP event
per `env.step()` and rolls episodes up on `done`.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Sequence

from .trace import (
    DimRegistry,
    PolicyTraceEvent,
    SOURCE_PHYSX,
    V_ACTION,
    V_OBS,
    V_POLICY_STATE,
    V_REWARD_TERMS,
)
from .writer import TraceWriter


def _np():
    import numpy as np  # core dep
    return np


# ── RSL-RL rollout storage capture (the moat primitive) ───────────────────────

# Field name on rsl_rl RolloutStorage -> our policy_state component name.
# These are the tensors RSL-RL keeps for every (step, env) in the rollout buffer.
_STORAGE_FIELDS = (
    "observations",
    "actions",
    "rewards",
    "actions_log_prob",
    "values",
    "returns",
    "dones",
    "time_outs",
)


class RslRlStorageCapture:
    """Emit per-step policy trace by reading the RSL-RL rollout buffer.

    Usage (inside the monitored runner, after each rollout collection):

        cap = RslRlStorageCapture(writer, run_id, source="sim:physx", traced_envs=8)
        ...
        cap.capture_rollout(runner, iteration=it)

    `traced_envs` bounds STEP volume: we fully trace a fixed slice of parallel
    envs (e.g. 8 of 4096) at every step — enough to reconstruct failures while
    keeping per-iteration event counts in the thousands, not the 100-thousands.
    Aggregate signals still come from the trainer's existing per-iteration path.
    """

    def __init__(
        self,
        writer: TraceWriter,
        run_id: str,
        *,
        source: str = SOURCE_PHYSX,
        traced_envs: int = 8,
    ) -> None:
        self.writer = writer
        self.run_id = run_id
        self.source = source
        self.traced_envs = max(1, int(os.environ.get("PROBE_TRACED_ENVS", traced_envs)))
        # Episode-id bookkeeping per traced env (monotonic across iterations).
        self._ep_counter: Dict[int, int] = {}
        self._ep_len: Dict[int, int] = {}
        self._ep_rew: Dict[int, float] = {}   # running episodic reward SUM per env

    @staticmethod
    def _get_storage(runner_or_storage: Any) -> Any:
        stor = getattr(runner_or_storage, "storage", None)
        if stor is not None:
            return stor
        alg = getattr(runner_or_storage, "alg", None)
        if alg is not None:
            return getattr(alg, "storage", None)
        return runner_or_storage  # already a storage

    def _extract(self, storage: Any) -> Optional[Dict[str, Any]]:
        """Pull the buffered rollout tensors to numpy [T, E, D] arrays, once."""
        np = _np()
        out: Dict[str, Any] = {}
        for name in _STORAGE_FIELDS:
            t = getattr(storage, name, None)
            if t is None:
                continue
            # rsl_rl 5.x stores `observations` as a TensorDict / dict of obs groups.
            # Its .numpy() returns a DICT of arrays (not one array), which breaks the
            # downstream [T,E,D] assumption — so detect any keyed container (a plain
            # tensor has no .keys) and pull the "policy" group (fall back to the first
            # entry) so observations are captured as a single tensor, not dropped.
            if hasattr(t, "keys"):
                try:
                    _keys = list(t.keys())
                    t = t["policy"] if "policy" in _keys else t[_keys[0]]
                except Exception:
                    continue
            try:
                if hasattr(t, "detach"):
                    t = t.detach()
                # Slice the env axis on-device BEFORE the D2H copy. Tracing 8 of
                # 4096 envs must not pay for the other 4088.
                if hasattr(t, "shape") and len(getattr(t, "shape", ())) >= 2:
                    try:
                        if int(t.shape[1]) > self.traced_envs:
                            t = t[:, : self.traced_envs]
                    except Exception:
                        pass
                if hasattr(t, "to"):
                    arr = t.to("cpu").float().numpy()
                else:
                    arr = np.asarray(t, dtype="float32")
            except Exception:
                try:
                    arr = np.asarray(t, dtype="float32")
                except Exception:
                    continue
            out[name] = arr
        if "observations" not in out and "actions" not in out:
            return None
        return out

    def capture_rollout(self, runner_or_storage: Any, iteration: int) -> int:
        """Read the rollout buffer and emit STEP (+ EPISODE on done) events.

        Returns the number of STEP events emitted. Never raises into training.
        """
        try:
            storage = self._get_storage(runner_or_storage)
            if storage is None:
                return 0
            data = self._extract(storage)
            if not data:
                return 0
            # Align T across fields — RSL-RL often stores obs at T+1 vs actions at T.
            lengths = [int(a.shape[0]) for a in data.values() if getattr(a, "ndim", 0) >= 1]
            if lengths:
                t_min = min(lengths)
                for k, a in list(data.items()):
                    if getattr(a, "ndim", 0) >= 1 and a.shape[0] > t_min:
                        data[k] = a[:t_min]
            return self._emit(data, iteration)
        except Exception:
            return 0

    def _emit(self, data: Dict[str, Any], iteration: int) -> int:
        np = _np()
        ref = data.get("observations")
        if ref is None:
            ref = data["actions"]
        if ref.ndim < 2:
            return 0
        n_steps, n_envs = ref.shape[0], ref.shape[1]
        n_trace = min(self.traced_envs, n_envs)

        def col(name: str, t: int, e: int):
            a = data.get(name)
            if a is None:
                return None
            v = a[t, e]
            return v

        def scalar(name: str, t: int, e: int) -> Optional[float]:
            v = col(name, t, e)
            if v is None:
                return None
            try:
                return float(np.asarray(v).reshape(-1)[0])
            except Exception:
                return None

        emitted = 0
        for e in range(n_trace):
            self._ep_counter.setdefault(e, 0)
            self._ep_len.setdefault(e, 0)
            self._ep_rew.setdefault(e, 0.0)
            for t in range(n_steps):
                ep_id = f"{self.source}:e{e}:ep{self._ep_counter[e]}"
                rew = scalar("rewards", t, e)
                rew_f = float(rew) if rew is not None else 0.0
                self._ep_rew[e] += rew_f if math.isfinite(rew_f) else 0.0
                rew = rew_f if math.isfinite(rew_f) else None
                policy_state = self._policy_state(data, t, e, scalar)
                ev = PolicyTraceEvent.step_event(
                    self.run_id,
                    ep_id,
                    self._ep_len[e],
                    source=self.source,
                    iteration=iteration,
                    env_idx=e,
                    obs=col("observations", t, e),
                    action=col("actions", t, e),
                    reward=rew,
                    policy_state=policy_state,
                )
                self.writer.emit(ev)
                emitted += 1
                self._ep_len[e] += 1

                done = scalar("dones", t, e)
                if done is not None and done >= 0.5:
                    timeout = scalar("time_outs", t, e)
                    is_timeout = timeout is not None and timeout >= 0.5
                    # Timeouts are truncations, not crashes — failures() ignores them.
                    termination = "timeout" if is_timeout else "terminated"
                    success = False if not is_timeout else None
                    self.writer.emit(
                        PolicyTraceEvent.episode_event(
                            self.run_id,
                            ep_id,
                            source=self.source,
                            iteration=iteration,
                            env_idx=e,
                            ep_return=self._ep_rew[e],
                            length=self._ep_len[e],
                            success=success,
                            termination=termination,
                        )
                    )
                    self._ep_counter[e] += 1
                    self._ep_len[e] = 0
                    self._ep_rew[e] = 0.0
        return emitted

    @staticmethod
    def _policy_state(data: Dict[str, Any], t: int, e: int, scalar) -> List[float]:
        """The per-step internal state nobody else captures: [log_prob, value, return]."""
        out: List[float] = []
        for name in ("actions_log_prob", "values", "returns"):
            v = scalar(name, t, e)
            out.append(v if v is not None else 0.0)
        return out

    POLICY_STATE_NAMES = ["log_prob", "value", "return"]


def policy_state_registry(run_id: str, *, obs_names=None, action_names=None, reward_term_names=None) -> DimRegistry:
    """Build a DimRegistry with the policy_state component names always set.

    obs/action/reward-term names are best-effort (pass from the env managers when
    available — see `infer_dim_registry`); policy_state is always named.
    """
    reg = DimRegistry(run_id=run_id)
    reg.register(V_POLICY_STATE, RslRlStorageCapture.POLICY_STATE_NAMES)
    if obs_names:
        reg.register(V_OBS, list(obs_names))
    if action_names:
        reg.register(V_ACTION, list(action_names))
    if reward_term_names:
        reg.register(V_REWARD_TERMS, list(reward_term_names))
    return reg


def infer_dim_registry(env: Any, run_id: str) -> DimRegistry:
    """Best-effort extraction of named components from Isaac Lab managers.

    Reads `reward_manager.active_terms` and the policy observation group term
    names when present. Tolerates any env that lacks them (returns policy_state
    names only). Never raises.
    """
    obs_names: List[str] = []
    action_names: List[str] = []
    reward_term_names: List[str] = []
    base = getattr(env, "unwrapped", env)
    try:
        rm = getattr(base, "reward_manager", None)
        if rm is not None:
            terms = getattr(rm, "active_terms", None)
            if terms:
                reward_term_names = [str(t) for t in terms]
    except Exception:
        pass
    try:
        om = getattr(base, "observation_manager", None)
        if om is not None:
            groups = getattr(om, "active_terms", None)
            term_dims = getattr(om, "group_obs_term_dim", None)
            if isinstance(groups, dict):
                pol = groups.get("policy") or next(iter(groups.values()), [])
                dims = term_dims.get("policy") if isinstance(term_dims, dict) else None
                if dims and len(dims) == len(pol):
                    # The captured obs vector is the FLATTENED concat of all terms;
                    # expand each term name by its dim count so names line up index
                    # for index (otherwise naming silently misattributes components).
                    expanded: List[str] = []
                    for nm, shp in zip(pol, dims):
                        n = 1
                        for s in (shp if isinstance(shp, (list, tuple)) else [shp]):
                            n *= int(s)
                        if n <= 1:
                            expanded.append(str(nm))
                        else:
                            expanded.extend([f"{nm}[{i}]" for i in range(n)])
                    obs_names = expanded
                # else: leave obs unnamed (positional obs_i) rather than mis-name it.
    except Exception:
        pass
    try:
        am = getattr(base, "action_manager", None)
        if am is not None:
            terms = getattr(am, "active_terms", None)
            term_dims = getattr(am, "action_term_dim", None) or getattr(am, "term_dim", None)
            if terms:
                if term_dims and len(list(term_dims)) == len(list(terms)):
                    expanded_a: List[str] = []
                    for nm, shp in zip(terms, term_dims):
                        n = 1
                        for s in (shp if isinstance(shp, (list, tuple)) else [shp]):
                            n *= int(s)
                        if n <= 1:
                            expanded_a.append(str(nm))
                        else:
                            expanded_a.extend([f"{nm}[{i}]" for i in range(n)])
                    action_names = expanded_a
                else:
                    action_names = [str(t) for t in terms]
    except Exception:
        pass
    return policy_state_registry(
        run_id,
        obs_names=obs_names or None,
        action_names=action_names or None,
        reward_term_names=reward_term_names or None,
    )


# ── Generic per-step capture (framework-agnostic / real-robot replay) ─────────


class GymStepCapture:
    """Per-step emitter for loops without a rollout buffer (raw gym, SB3, replay).

    Call `record_step(...)` after each `env.step()`, then `end_episode(...)` on
    done. Use `source="real:<serial>"` to land real-robot logs in the same
    schema as sim rollouts — that is what makes the sim-to-real gap a query.
    """

    def __init__(self, writer: TraceWriter, run_id: str, *, source: str = SOURCE_PHYSX) -> None:
        self.writer = writer
        self.run_id = run_id
        self.source = source
        self._ep = 0
        self._step = 0
        self._return = 0.0

    def record_step(
        self,
        *,
        obs: Any = None,
        action: Any = None,
        reward: Optional[float] = None,
        reward_terms: Any = None,
        policy_state: Any = None,
        iteration: Optional[int] = None,
    ) -> None:
        ep_id = f"{self.source}:ep{self._ep}"
        self.writer.emit(
            PolicyTraceEvent.step_event(
                self.run_id,
                ep_id,
                self._step,
                source=self.source,
                iteration=iteration,
                obs=obs,
                action=action,
                reward=reward,
                reward_terms=reward_terms,
                policy_state=policy_state,
            )
        )
        if reward is not None:
            try:
                rf = float(reward)
                if math.isfinite(rf):
                    self._return += rf
            except (TypeError, ValueError):
                pass
        self._step += 1

    def end_episode(
        self,
        *,
        success: Optional[bool] = None,
        termination: Optional[str] = None,
        iteration: Optional[int] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> None:
        ep_id = f"{self.source}:ep{self._ep}"
        extra = dict(tags or {})
        self.writer.emit(
            PolicyTraceEvent.episode_event(
                self.run_id,
                ep_id,
                source=self.source,
                iteration=iteration,
                ep_return=self._return,
                length=self._step,
                success=success,
                termination=termination,
                tags=extra or None,
            )
        )
        self._ep += 1
        self._step = 0
        self._return = 0.0
