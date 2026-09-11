"""
LeRobot dataset → probe traces.

LeRobot v2/v3 datasets are the lingua franca of open VLAs (SmolVLA, ACT, pi0
via LeRobot, GR00T export). Importing them into the same PolicyTraceEvent
schema as RSL-RL rollouts is what makes sim-vs-real a query.

No hard dependency on `datasets` or `lerobot`. Prefers parquet+meta/info.json;
falls back to a frames JSONL if present.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from .adapt import emit_records
from .trace import real_source


def _read_info(root: Path) -> Dict[str, Any]:
    for candidate in (root / "meta" / "info.json", root / "info.json"):
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                return {}
    return {}


def _iter_parquet_rows(root: Path) -> Iterator[Dict[str, Any]]:
    files = sorted(root.glob("data/**/*.parquet")) or sorted(root.glob("**/*.parquet"))
    if not files:
        return
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:  # pragma: no cover - optional extra
        raise ImportError(
            "Reading LeRobot parquet requires pyarrow. Install simtooreal-probe[query]."
        ) from exc
    for path in files:
        try:
            table = pq.read_table(str(path))
        except Exception:
            continue
        for row in table.to_pylist():
            yield row


def _iter_jsonl_frames(root: Path) -> Iterator[Dict[str, Any]]:
    for path in sorted(root.glob("**/*.jsonl")):
        if path.name.startswith("_"):
            continue
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except Exception:
                        continue
        except Exception:
            continue


def _as_list(val: Any) -> List[float]:
    if val is None:
        return []
    if hasattr(val, "tolist"):
        try:
            val = val.tolist()
        except Exception:
            pass
    if isinstance(val, (int, float)):
        return [float(val)]
    if isinstance(val, dict):
        # nested observation.state style
        if "data" in val:
            return _as_list(val.get("data"))
        return []
    out: List[float] = []
    try:
        for x in val:
            try:
                out.append(float(x))
            except (TypeError, ValueError):
                return []
    except TypeError:
        return []
    return out


def _frame_to_record(row: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
    obs = (
        _as_list(row.get("observation.state"))
        or _as_list(row.get("observation.qpos"))
        or _as_list(row.get("obs"))
        or _as_list(row.get("state"))
    )
    action = _as_list(row.get("action")) or _as_list(row.get("actions"))
    reward = row.get("next.reward", row.get("reward"))
    done = bool(row.get("next.done") or row.get("done") or False)
    instruction = (
        row.get("task")
        or row.get("language_instruction")
        or row.get("instruction")
        or (info.get("task") if isinstance(info.get("task"), str) else None)
    )
    rec: Dict[str, Any] = {
        "obs": obs,
        "action": action,
        "reward": reward,
        "done": done,
        "episode_index": row.get("episode_index", row.get("episode")),
        "frame_index": row.get("frame_index", row.get("index")),
        "instruction": instruction,
    }
    suc = row.get("success", row.get("next.success"))
    if suc is not None:
        rec["success"] = bool(suc)
    return rec


def _records_with_episode_bounds(frames: Iterable[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    prev_ep = object()
    pending: Optional[Dict[str, Any]] = None
    for rec in frames:
        ep = rec.get("episode_index")
        if pending is not None and ep != prev_ep:
            pending["done"] = True
            yield pending
            pending = None
        if pending is not None:
            yield pending
        pending = rec
        prev_ep = ep
    if pending is not None:
        pending["done"] = True
        yield pending


def lerobot_to_traces(
    dataset_dir: str,
    *,
    run_id: str,
    serial: Optional[str] = None,
    source: Optional[str] = None,
    root: Optional[str] = None,
    http: Optional[bool] = None,
    max_frames: Optional[int] = None,
) -> Dict[str, int]:
    """Import a on-disk LeRobot dataset into ``~/.probe/<run_id>/``.

    ``serial`` becomes ``real:<serial>``. Override with ``source`` for sim
    replays (e.g. ``sim:mujoco``).
    """
    path = Path(dataset_dir)
    if not path.exists():
        raise FileNotFoundError(f"LeRobot dataset not found: {dataset_dir}")
    info = _read_info(path)
    robot = serial or info.get("robot_type") or info.get("robot") or path.name
    src = source or real_source(str(robot))

    rows: List[Dict[str, Any]] = []
    parquet_iter = _iter_parquet_rows(path)
    used = False
    for i, raw in enumerate(parquet_iter):
        used = True
        rows.append(_frame_to_record(raw, info))
        if max_frames is not None and i + 1 >= max_frames:
            break
    if not used:
        for i, raw in enumerate(_iter_jsonl_frames(path)):
            rows.append(_frame_to_record(raw, info))
            if max_frames is not None and i + 1 >= max_frames:
                break
    if not rows:
        raise FileNotFoundError(
            f"No parquet or jsonl frames under {dataset_dir}. "
            "Expected LeRobot layout: data/**/*.parquet + meta/info.json."
        )

    bounded = list(_records_with_episode_bounds(rows))
    return _emit_lerobot(bounded, run_id=run_id, source=src, root=root, http=http)


def _emit_lerobot(
    records: Sequence[Dict[str, Any]],
    *,
    run_id: str,
    source: str,
    root: Optional[str],
    http: Optional[bool],
) -> Dict[str, int]:
    from . import close as _close, gym_capture, init

    _close()
    init(run_id=run_id, http=http, root=root, capture="full")
    cap = gym_capture(source=source)
    n_steps = 0
    n_eps = 0
    pending = False
    current_instruction = None
    for rec in records:
        if rec.get("instruction"):
            current_instruction = rec.get("instruction")
        cap.record_step(
            obs=rec.get("obs"),
            action=rec.get("action"),
            reward=rec.get("reward"),
        )
        n_steps += 1
        pending = True
        if rec.get("done"):
            tags = {"instruction": current_instruction} if current_instruction else None
            cap.end_episode(
                success=rec.get("success"),
                termination="done",
                tags=tags,
            )
            n_eps += 1
            pending = False
            current_instruction = None
    if pending:
        cap.end_episode()
        n_eps += 1
    _close()
    return {"steps": n_steps, "episodes": n_eps}
