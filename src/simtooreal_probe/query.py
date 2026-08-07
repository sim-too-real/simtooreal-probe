"""
probe.query — ProbeRun: the local-first read API.

This is the researcher's notebook entrypoint. It reads the LocalSink store
(`~/.probe/<run_id>/`) with NO account and NO network — the same API whether the
trace is local Parquet/JSONL or (later) fetched from the platform v2 endpoints.

    from simtooreal_probe import ProbeRun
    run = ProbeRun("go2-flat-v3")
    run.iterations()                # -> per-iteration scalars (DataFrame or list)
    run.episodes(source="sim:physx")
    steps = run.steps(episode_id)   # the per-step policy trace
    run.failures()                  # episodes that ended badly
    run.first_abnormal(episode_id)  # earliest abnormal-signal onset (heuristic v1)

pandas is used when present; otherwise everything degrades to plain list[dict].
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .trace import DimRegistry, Grain, V_ACTION, V_POLICY_STATE
from .sinks import _desanitize


def _maybe_pandas():
    try:
        import pandas as pd  # optional
        return pd
    except Exception:
        return None


class ProbeRun:
    """Read-only view over one run's local trace store."""

    def __init__(self, run_id: str, root: Optional[str] = None) -> None:
        self.run_id = run_id
        base = root or os.environ.get("PROBE_HOME") or os.path.join(Path.home(), ".probe")
        self.dir = Path(base) / run_id
        if not self.dir.exists():
            raise FileNotFoundError(f"No probe run at {self.dir} (set PROBE_HOME or pass root=)")
        self.meta = self._read_json(self.dir / "_meta.json", {})
        self.dims = DimRegistry.from_dict(self._read_json(self.dir / "_dims.json", {"run_id": run_id}))

    # ── public reads ──────────────────────────────────────────────────────────

    def sources(self) -> List[str]:
        return [_desanitize(p.name) for p in self.dir.iterdir() if p.is_dir()]

    def iterations(self, source: Optional[str] = None):
        return self._frame(self._load(Grain.ITERATION, source))

    def episodes(self, source: Optional[str] = None):
        return self._frame(self._load(Grain.EPISODE, source))

    def steps(self, episode_id: Optional[str] = None, source: Optional[str] = None):
        rows = self._load(Grain.STEP, source)
        if episode_id is not None:
            rows = [r for r in rows if r.get("episode_id") == episode_id]
        rows.sort(key=lambda r: (
            r.get("env_idx") if r.get("env_idx") is not None else -1,
            r.get("step_idx") if r.get("step_idx") is not None else -1,
        ))
        return self._frame(rows)

    def failures(self, source: Optional[str] = None) -> List[Dict[str, Any]]:
        """Episodes that ended badly: success is False, or a non-timeout termination."""
        out = []
        for r in self._load(Grain.EPISODE, source):
            tags = r.get("tags", {}) or {}
            success = tags.get("success")
            term = str(tags.get("termination", "")).lower()
            bad = success is False or (term and term not in ("", "time_out", "timeout", "truncated"))
            if bad:
                out.append(r)
        return out

    def rewards(self, source: Optional[str] = None):
        """Per-iteration mean reward — the most common single-metric notebook query.

        Returns iteration + mean_reward columns (DataFrame when pandas present,
        list[dict] otherwise). Filters to rows that have a "mean_reward" scalar.
        """
        rows = self._load(Grain.ITERATION, source)
        out = [
            {"iteration": r.get("iteration"), "mean_reward": (r.get("scalars", {}) or {}).get("mean_reward")}
            for r in rows
            if (r.get("scalars", {}) or {}).get("mean_reward") is not None
        ]
        out.sort(key=lambda r: r.get("iteration") or 0)
        return self._frame(out)

    def reward_terms(self, episode_id: str, source: Optional[str] = None) -> List[Dict[str, float]]:
        """Per-step reward-term breakdown, resolved to names via the DimRegistry."""
        rows = self._load(Grain.STEP, source)
        rows = [r for r in rows if r.get("episode_id") == episode_id]
        rows.sort(key=lambda r: r.get("step_idx") or 0)
        out = []
        for r in rows:
            vec = (r.get("vectors", {}) or {}).get("reward_terms", [])
            out.append(self.dims.resolve("reward_terms", vec))
        return out

    def to_html(
        self,
        episode_id: str,
        *,
        source: Optional[str] = None,
        out: Optional[str] = None,
    ) -> str:
        """Write a self-contained failure-replay HTML file for one episode.

        Returns the path of the written file. Zero network — works offline.
        Thin wrapper around :func:`simtooreal_probe.viz.export_failure_html`.
        """
        from .viz import export_failure_html

        return export_failure_html(
            self.run_id,
            episode_id,
            root=str(self.dir.parent),
            source=source,
            out=out,
        )

    # ── first-abnormal-signal detector (CUSUM v1) ─────────────────────────────

    def first_abnormal(self, episode_id: str, source: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Earliest calibrated change-point across the policy's channels (DeviationEngine).

        Runs the calibrated robust CUSUM detector (`_cusum_changepoint`) over three
        per-step channels and returns the EARLIEST onset across them:
          * reward_collapse   : reward shifts DOWN  (robust CUSUM, low)
          * value_collapse    : policy_state value shifts DOWN (robust CUSUM, low)
          * action_explosion  : ||action||₂ shifts UP   (robust CUSUM, high)
        Returns {onset_step, channel, value_at_onset, severity (σ), confidence
        [0-1], detector: "cusum_v1"} or None.

        Compatible remote backends that re-implement this detector should keep
        the same calibration constants so offline and online onsets agree.
        """
        rows = self._load(Grain.STEP, source)
        rows = [r for r in rows if r.get("episode_id") == episode_id]
        if len(rows) < 8:
            return None
        rows.sort(key=lambda r: r.get("step_idx") if r.get("step_idx") is not None else -1)

        ps_names = self.dims.names(V_POLICY_STATE) or ["log_prob", "value", "return"]
        # Our capture writes policy_state as [log_prob, value, return]; index 1 is
        # the value-fn estimate when names aren't registered.
        value_idx = ps_names.index("value") if "value" in ps_names else 1

        def channel_pairs(kind: str):
            """(step_idx, value) pairs, skipping steps where the channel is absent.

            Skipping (rather than injecting 0.0) keeps the running statistics honest
            on sparse-reward tasks, and tolerates sanitized None values.
            """
            pairs = []
            for r in rows:
                si = r.get("step_idx")
                if kind == "reward":
                    sc = r.get("scalars", {}) or {}
                    if "reward" not in sc:
                        continue
                    v = sc.get("reward")
                elif kind == "value":
                    vec = (r.get("vectors", {}) or {}).get(V_POLICY_STATE, [])
                    if len(vec) <= value_idx:
                        continue
                    v = vec[value_idx]
                else:  # kind == "action_mag"
                    act = (r.get("vectors", {}) or {}).get(V_ACTION, [])
                    if not act:
                        continue
                    # L2 norm; None values are treated as 0 (sanitized NaN → None)
                    v = math.sqrt(sum((x or 0.0) ** 2 for x in act))
                if v is None:
                    continue
                pairs.append((si if si is not None else len(pairs), float(v)))
            return pairs

        cand = []  # (step_idx, channel, value_at_onset, shift_sigma)
        for name, kind, low in (
            ("reward_collapse", "reward", True),
            ("value_collapse", "value", True),
            ("action_explosion", "action_mag", False),
        ):
            pairs = channel_pairs(kind)
            series = [p[1] for p in pairs]
            res = _cusum_changepoint(series, low=low)
            if res is not None:
                onset_idx, shift = res
                cand.append((pairs[onset_idx][0], name, series[onset_idx], shift))
        if not cand:
            return None
        cand.sort(key=lambda c: c[0])  # earliest onset across channels wins
        step, channel, val, shift = cand[0]
        return {
            "episode_id": episode_id,
            "onset_step": step,
            "channel": channel,
            "value_at_onset": round(val, 6),
            "severity": round(shift, 3),
            "confidence": round(min(1.0, shift / 8.0), 3),
            "detector": "cusum_v1",
        }

    # ── internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _read_json(path: Path, default: Any) -> Any:
        try:
            return json.loads(path.read_text())
        except Exception:
            return default

    def _load(self, grain: Grain, source: Optional[str]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for src_dir in self.dir.iterdir():
            if not src_dir.is_dir():
                continue
            src = _desanitize(src_dir.name)
            if source is not None and src != source:
                continue
            rows.extend(self._read_shard(src_dir / f"{grain.value}.jsonl"))
            rows.extend(self._read_parquet(src_dir / f"{grain.value}.parquet"))
        return rows

    @staticmethod
    def _read_shard(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        out = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out.append(json.loads(line))
        except Exception:
            pass
        return out

    @staticmethod
    def _read_parquet(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        try:
            import pyarrow.parquet as pq  # type: ignore
            tbl = pq.read_table(str(path)).to_pylist()
        except Exception:
            return []
        out = []
        for r in tbl:
            for col in ("scalars", "vectors", "tags"):
                v = r.get(col)
                if isinstance(v, str):
                    try:
                        r[col] = json.loads(v)
                    except Exception:
                        r[col] = {}
            out.append({k: v for k, v in r.items() if v is not None})
        return out

    @staticmethod
    def _frame(rows: List[Dict[str, Any]]):
        pd = _maybe_pandas()
        if pd is None:
            return rows
        if not rows:
            return pd.DataFrame()
        # Flatten scalars into columns for ergonomic notebook use; keep vectors nested.
        flat = []
        for r in rows:
            base = {k: v for k, v in r.items() if k not in ("scalars", "vectors", "tags")}
            for sk, sv in (r.get("scalars", {}) or {}).items():
                base[sk] = sv
            for tk, tv in (r.get("tags", {}) or {}).items():
                base[f"tag_{tk}"] = tv
            base["_vectors"] = r.get("vectors", {})
            flat.append(base)
        return pd.DataFrame(flat)


#  ── The DeviationEngine: calibrated robust CUSUM change-point detector ────────
#  Replaces the heuristic-v1 z-score. MUST stay byte-for-byte equivalent to the
#  Rust mirror in backend/src/probe_ingest.rs (cusum_changepoint) so the SDK and
#  the platform never disagree on an onset.

# Calibrated empirically (warmup=12 / min_run=6 / min_shift=5.0): ZERO false alarms
# over pure-noise episodes while catching every real collapse (which sign-flip at
# tens of σ). The 8-sample warmup was too small (unstable σ); 4.0σ was on the noise
# boundary (a 4.08σ value-channel run slipped through), so 5.0σ gives real margin.
_CUSUM_WARMUP = 12      # samples used to calibrate the robust baseline (stable σ)
_CUSUM_K = 0.5          # slack / reference value (σ units) — optimal for a 1σ shift
_CUSUM_H = 5.0          # decision threshold (standard CUSUM calibration)
_CUSUM_INCR_CAP = 4.0   # Winsorize the per-step increment: one spike can't cross h
                        # alone (cap < h), but a SUSTAINED shift trips it in ~2 steps
_CUSUM_MIN_RUN = 6      # alarm segment must persist ≥ this many steps — the segment
                        # MEDIAN then averages out noise, so only *sustained* shifts pass.
_CUSUM_MIN_SHIFT = 5.0  # alarm segment median must be ≥ this many σ off baseline (the
                        # "how abnormal" gate). Real failures sign-flip (tens of σ).


def _median(xs: List[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return (float(s[mid - 1]) + float(s[mid])) / 2.0


def _cusum_changepoint(values: List[float], *, low: bool) -> Optional[tuple]:
    """Calibrated robust online change-point detector. Returns (onset_idx, shift_σ) or None.

    1. Calibrate a robust baseline (median μ + 1.4826·MAD → σ) on the first
       `_CUSUM_WARMUP` samples — robust to outliers and non-Gaussian noise.
    2. Run a one-sided tabular CUSUM (slack k, threshold h) on the standardized
       series. Under no change E[increment] = -k < 0, so the statistic decays to
       0; a sustained shift makes it climb past h. k=0.5/h=5 is the textbook
       calibration giving a controlled false-alarm rate (ARL0 ≈ 465).
    3. On alarm, CONFIRM the shift is sustained (run ≥ min_run) and large
       (segment median ≥ min_shift σ) — this rejects the single-sample spikes the
       old z-score fired on.
    4. Back-date + REFINE the onset to the first step in the alarm segment that
       genuinely crossed min_shift·σ (immune to CUSUM-anchor wobble under noise).
    """
    n = len(values)
    if n < _CUSUM_WARMUP + _CUSUM_MIN_RUN:
        return None
    base = values[:_CUSUM_WARMUP]
    mu = _median(base)
    mad = _median([abs(v - mu) for v in base])
    sigma = max(1.4826 * mad, 1e-3 * abs(mu) + 1e-9)

    c = 0.0
    start = _CUSUM_WARMUP  # change-point anchor: step after the last CUSUM reset
    for t in range(_CUSUM_WARMUP, n):
        z = (values[t] - mu) / sigma
        incr = (-_CUSUM_K - z) if low else (z - _CUSUM_K)
        if incr > _CUSUM_INCR_CAP:  # Winsorize: bound a single sample's influence
            incr = _CUSUM_INCR_CAP
        c = max(0.0, c + incr)
        if c == 0.0:
            start = t + 1
        if c >= _CUSUM_H and (t - start + 1) >= _CUSUM_MIN_RUN:
            seg = values[start : t + 1]
            m = _median(seg)
            shift = (mu - m) / sigma if low else (m - mu) / sigma
            if shift >= _CUSUM_MIN_SHIFT:
                # Refine onset to the first genuinely-deviating step in the segment.
                onset = start
                for j in range(start, t + 1):
                    dev = (mu - values[j]) / sigma if low else (values[j] - mu) / sigma
                    if dev >= _CUSUM_MIN_SHIFT:
                        onset = j
                        break
                return onset, shift
    return None
