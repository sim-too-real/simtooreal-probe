"""
Offline run analytics over a local probe store.

No network, no account — summary stats, failure counts, and CSV export from
``~/.probe/<run_id>/`` (or a custom root). Complements ``ProbeRun`` with a
simple "how did this run go?" facade for any training stack that emitted traces.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .query import ProbeRun


class LocalRunSummary:
    """Aggregate deep-analytics summary for one local probe run.

    Example::

        from simtooreal_probe import LocalRunSummary

        s = LocalRunSummary("go2-flat-v3")
        print(s.summary())
        s.export_csv("report.csv")
    """

    def __init__(
        self,
        run_id: str,
        *,
        root: Optional[str] = None,
    ) -> None:
        self.run_id = run_id
        self._run = ProbeRun(run_id, root=root)

    @property
    def probe(self) -> ProbeRun:
        return self._run

    def summary(self, *, source: Optional[str] = None) -> Dict[str, Any]:
        iterations = self._as_list(self._run.iterations(source=source) if source else self._run.iterations())
        try:
            episodes = self._as_list(
                self._run.episodes(source=source) if source else self._run.episodes()
            )
        except TypeError:
            episodes = self._as_list(self._run.episodes())
        try:
            failures = self._as_list(self._run.failures())
        except Exception:
            failures = []

        rewards: List[float] = []
        for row in iterations:
            if not isinstance(row, dict):
                continue
            val = self._first_float(row, ("mean_reward", "reward", "return"))
            if val is None:
                sc = row.get("scalars")
                if isinstance(sc, dict):
                    val = self._first_float(sc, ("mean_reward", "reward", "return"))
            if val is not None:
                rewards.append(val)

        ep_returns: List[float] = []
        for ep in episodes:
            if not isinstance(ep, dict):
                continue
            val = self._first_float(ep, ("return", "mean_reward"))
            if val is None:
                sc = ep.get("scalars")
                if isinstance(sc, dict):
                    val = self._first_float(sc, ("return", "mean_reward"))
            if val is not None:
                ep_returns.append(val)

        out = {
            "run_id": self.run_id,
            "iteration_count": len(iterations),
            "episode_count": len(episodes),
            "failure_count": len(failures),
            "failure_rate": (len(failures) / len(episodes)) if episodes else None,
            "best_mean_reward": max(rewards) if rewards else None,
            "final_mean_reward": rewards[-1] if rewards else None,
            "mean_reward_avg": (sum(rewards) / len(rewards)) if rewards else None,
            "best_episode_return": max(ep_returns) if ep_returns else None,
            "mean_episode_return": (sum(ep_returns) / len(ep_returns)) if ep_returns else None,
            "sources": self._run.sources() if hasattr(self._run, "sources") else [],
        }
        return out

    def export_csv(self, path: Union[str, Path], *, source: Optional[str] = None) -> Path:
        """Export per-iteration scalar rows to CSV."""
        path = Path(path)
        iterations = self._as_list(self._run.iterations(source=source) if source else self._run.iterations())
        rows: List[Dict[str, Any]] = []
        for row in iterations:
            if not isinstance(row, dict):
                continue
            flat: Dict[str, Any] = {
                "run_id": self.run_id,
                "iteration": row.get("iteration"),
                "source": row.get("source"),
            }
            sc = row.get("scalars") or {}
            if isinstance(sc, dict):
                flat.update(sc)
            rows.append(flat)

        fieldnames: List[str] = []
        seen = set()
        for r in rows:
            for k in r:
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames or ["run_id"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        return path

    def to_json(self, path: Union[str, Path], **kwargs: Any) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary(**kwargs), indent=2, default=str), encoding="utf-8")
        return path

    @staticmethod
    def _first_float(d: Dict[str, Any], keys: tuple) -> Optional[float]:
        for key in keys:
            if d.get(key) is not None:
                try:
                    return float(d[key])
                except (TypeError, ValueError):
                    continue
        return None

    @staticmethod
    def _as_list(obj: Any) -> List[Any]:
        if obj is None:
            return []
        if hasattr(obj, "to_dict"):
            try:
                obj = obj.to_dict(orient="records")
            except Exception:
                pass
        if isinstance(obj, list):
            return obj
        try:
            return list(obj)
        except TypeError:
            return []
