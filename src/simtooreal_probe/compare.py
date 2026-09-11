"""
Sim vs real distribution compare — the transfer-gap as a query.

Same PolicyTraceEvent schema, two `source` tags. We estimate a cheap
per-dimension shift (mean / pooled-std) plus action-magnitude KS-lite,
and roll it into a 0–100 transfer_risk. No scipy required.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .query import ProbeRun


def _rows(run: ProbeRun, grain_fn, source: str) -> List[Dict[str, Any]]:
    obj = grain_fn(source=source)
    if hasattr(obj, "to_dict"):
        try:
            obj = obj.to_dict(orient="records")
        except Exception:
            pass
    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)]
    try:
        return [r for r in list(obj) if isinstance(r, dict)]
    except TypeError:
        return []


def _vec(row: Dict[str, Any], name: str) -> List[float]:
    vec = row.get("_vectors") or row.get("vectors") or {}
    if not isinstance(vec, dict):
        return []
    raw = vec.get(name) or []
    out: List[float] = []
    for x in raw:
        if x is None:
            return []
        try:
            out.append(float(x))
        except (TypeError, ValueError):
            return []
    return out


def _mean_std(cols: List[List[float]]) -> Tuple[List[float], List[float]]:
    if not cols:
        return [], []
    dim = min(len(c) for c in cols)
    means: List[float] = []
    stds: List[float] = []
    n = len(cols)
    for d in range(dim):
        xs = [c[d] for c in cols]
        m = sum(xs) / n
        var = sum((x - m) ** 2 for x in xs) / max(1, n - 1)
        means.append(m)
        stds.append(math.sqrt(var))
    return means, stds


def _shift(a: Sequence[float], sa: Sequence[float], b: Sequence[float], sb: Sequence[float]) -> List[float]:
    n = min(len(a), len(b), len(sa), len(sb))
    out = []
    for i in range(n):
        pooled = math.sqrt(0.5 * (sa[i] ** 2 + sb[i] ** 2)) + 1e-6
        out.append(abs(a[i] - b[i]) / pooled)
    return out


def compare_sources(
    run: ProbeRun,
    sim_source: str,
    real_source: str,
    *,
    vector: str = "action",
) -> Dict[str, Any]:
    """Compare one vector field across two sources of the same run.

    Returns transfer_risk in [0, 100], plus per-dim z-shifts. Missing data
    yields transfer_risk=None rather than a fake 0.
    """
    sim_steps = _rows(run, run.steps, sim_source)
    real_steps = _rows(run, run.steps, real_source)
    sim_cols = [v for v in (_vec(r, vector) for r in sim_steps) if v]
    real_cols = [v for v in (_vec(r, vector) for r in real_steps) if v]
    if len(sim_cols) < 8 or len(real_cols) < 8:
        return {
            "sim_source": sim_source,
            "real_source": real_source,
            "vector": vector,
            "n_sim": len(sim_cols),
            "n_real": len(real_cols),
            "transfer_risk": None,
            "note": "Need ≥8 steps on both sources.",
        }
    sm, ss = _mean_std(sim_cols)
    rm, rs = _mean_std(real_cols)
    shifts = _shift(sm, ss, rm, rs)
    mean_shift = sum(shifts) / len(shifts) if shifts else 0.0
    max_shift = max(shifts) if shifts else 0.0
    # 2σ mean shift → ~70 risk; 4σ → 100.
    risk = max(0.0, min(100.0, 25.0 * mean_shift + 10.0 * max(0.0, max_shift - 1.0)))
    worst = sorted(enumerate(shifts), key=lambda kv: -kv[1])[:5]
    return {
        "sim_source": sim_source,
        "real_source": real_source,
        "vector": vector,
        "n_sim": len(sim_cols),
        "n_real": len(real_cols),
        "mean_z_shift": round(mean_shift, 4),
        "max_z_shift": round(max_shift, 4),
        "worst_dims": [{"dim": i, "z": round(z, 3)} for i, z in worst],
        "transfer_risk": round(risk, 1),
    }


def compare_run(run: ProbeRun) -> Optional[Dict[str, Any]]:
    """Auto-pick first sim:* and real:* sources on a run."""
    sources = run.sources()
    sim = next((s for s in sources if s.startswith("sim:")), None)
    real = next((s for s in sources if s.startswith("real:")), None)
    if not sim or not real:
        return None
    return compare_sources(run, sim, real)
