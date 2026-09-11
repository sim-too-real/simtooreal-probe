"""
Run-level failure intelligence for RL and VLA traces.

CUSUM onset (query.first_abnormal) answers "when did this episode die."
This module answers "what killed the run" — entropy collapse, KL spikes,
zero action-std (the VLA silent killer), reward hacking, idle policies,
instruction/skill decoupling, and sim-vs-real covariate shift.

Numpy is the only hard dependency. Designed to run inside an Isaac container
and in a notebook with no account.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .query import ProbeRun, _cusum_changepoint


# ── Public types ──────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Finding:
    """One diagnosed failure mode with evidence a researcher can act on."""

    id: str
    title: str
    family: str  # rl | vla | sim2real | data
    severity: str  # critical | high | medium | low
    confidence: float
    evidence: Dict[str, Any] = field(default_factory=dict)
    remedy: str = ""
    onset_iteration: Optional[int] = None
    detector: str = "diagnose_v1"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["confidence"] = round(float(self.confidence), 3)
        return d


# Catalog ids stay stable so the platform / Lab UI can deep-link.
CATALOG: Dict[str, Dict[str, str]] = {
    "entropy_collapse": {
        "title": "Entropy collapse",
        "family": "rl",
        "remedy": "Raise entropy bonus, cut learning rate, or restore from the last checkpoint before collapse. PPO is dead once std → 0.",
    },
    "kl_spike": {
        "title": "KL spike",
        "family": "rl",
        "remedy": "Lower clip range / lr, or enable adaptive KL. A sustained KL jump is a policy step the value function cannot follow.",
    },
    "action_std_zero": {
        "title": "Action std is zero",
        "family": "vla",
        "remedy": "A frozen action dimension (often gripper roll/pitch) divides by zero during VLA denormalization. Replace zero-std with ε=2e-5 and drop constant dims from the stats file.",
    },
    "reward_hacking": {
        "title": "Reward hacking",
        "family": "rl",
        "remedy": "Proxy reward is rising while success falls. Reweight the true success term, cap energy/velocity bonuses, and inspect timeout-as-success.",
    },
    "idle_policy": {
        "title": "Idle / near-zero actions",
        "family": "vla",
        "remedy": "Typical of tiny datasets, mismatched action stats, or a VLA that ignored the instruction. Replay demos, verify stats.json, collect 50–100 more diverse episodes.",
    },
    "gradient_explosion": {
        "title": "Gradient explosion",
        "family": "rl",
        "remedy": "Clip grads, drop lr, check NaNs in observations. A single non-finite batch can poison the run.",
    },
    "reward_plateau": {
        "title": "Reward plateau",
        "family": "rl",
        "remedy": "Curriculum stuck or exploration dead. Increase domain randomization, reset entropy, or advance the curriculum only on success rate — not iteration count.",
    },
    "instruction_decoupling": {
        "title": "Skill without grounding",
        "family": "vla",
        "remedy": "The manipulation primitive succeeds more than the instruction-conditioned task. Collect language-diverse demos and evaluate N paraphrases before calling the VLA done.",
    },
    "torque_saturation": {
        "title": "Action / torque saturation",
        "family": "rl",
        "remedy": "Commands sit on the limits. Scale the action space, raise actuator limits in sim to match hardware, or add an action-rate cost.",
    },
    "covariate_shift": {
        "title": "Sim vs real covariate shift",
        "family": "sim2real",
        "remedy": "Obs/action distributions disagree across sources. Calibrate cameras and actuator delay first (often +20–40% zero-shot), then Bayesian DR from hardware telemetry.",
    },
    "early_termination": {
        "title": "Early termination cluster",
        "family": "rl",
        "remedy": "Episodes die far below median length — usually contact/fall. Inspect onset via ProbeRun.first_abnormal and the termination tag, not the mean reward.",
    },
}


SEVERITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}


# ── Series helpers ────────────────────────────────────────────────────────────


def _as_rows(obj: Any) -> List[Dict[str, Any]]:
    if obj is None:
        return []
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


def _scalar(row: Mapping[str, Any], *keys: str) -> Optional[float]:
    for k in keys:
        if k in row and row[k] is not None:
            try:
                v = float(row[k])
            except (TypeError, ValueError):
                continue
            if math.isfinite(v):
                return v
        sc = row.get("scalars")
        if isinstance(sc, dict) and sc.get(k) is not None:
            try:
                v = float(sc[k])
            except (TypeError, ValueError):
                continue
            if math.isfinite(v):
                return v
    return None


def _tag(row: Mapping[str, Any], key: str) -> Any:
    if key in row and not key.startswith("tag_"):
        if key in ("success", "termination", "instruction", "primitive_success"):
            return row.get(key)
    tags = row.get("tags")
    if isinstance(tags, dict) and key in tags:
        return tags.get(key)
    return row.get(f"tag_{key}")


def _mean(xs: Sequence[float]) -> Optional[float]:
    return (sum(xs) / len(xs)) if xs else None


def _std(xs: Sequence[float]) -> Optional[float]:
    if len(xs) < 2:
        return 0.0 if xs else None
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)


def _window(xs: Sequence[float], frac: float, *, head: bool) -> List[float]:
    n = max(1, int(round(len(xs) * frac)))
    return list(xs[:n] if head else xs[-n:])


def _finding(fid: str, *, severity: str, confidence: float, evidence: Dict[str, Any],
             onset: Optional[int] = None) -> Finding:
    meta = CATALOG[fid]
    return Finding(
        id=fid,
        title=meta["title"],
        family=meta["family"],
        severity=severity,
        confidence=max(0.0, min(1.0, confidence)),
        evidence=evidence,
        remedy=meta["remedy"],
        onset_iteration=onset,
    )


# ── Detectors ─────────────────────────────────────────────────────────────────


def _detect_entropy(iters: List[Dict[str, Any]]) -> Optional[Finding]:
    series = [(_scalar(r, "iteration"), _scalar(r, "entropy", "mean_entropy")) for r in iters]
    series = [(i, v) for i, v in series if v is not None]
    if len(series) < 12:
        return None
    values = [v for _, v in series]
    early = _mean(_window(values, 0.25, head=True))
    late = _mean(_window(values, 0.25, head=False))
    if early is None or late is None or early <= 1e-6:
        return None
    ratio = late / early
    if late < 0.01 or ratio < 0.2:
        onset = None
        res = _cusum_changepoint(values, low=True)
        if res:
            onset = int(series[res[0]][0] or res[0])
        return _finding(
            "entropy_collapse",
            severity="critical" if late < 0.01 else "high",
            confidence=min(1.0, (0.2 - min(ratio, 0.2)) / 0.2 + (0.4 if late < 0.01 else 0.0)),
            evidence={"early_mean": round(early, 5), "late_mean": round(late, 5), "ratio": round(ratio, 4)},
            onset=onset,
        )
    return None


def _detect_kl(iters: List[Dict[str, Any]]) -> Optional[Finding]:
    series = [(_scalar(r, "iteration"), _scalar(r, "kl_divergence", "kl", "approx_kl")) for r in iters]
    series = [(i, v) for i, v in series if v is not None]
    if len(series) < 8:
        return None
    values = [v for _, v in series]
    baseline = values[: max(5, len(values) // 3)]
    med = sorted(baseline)[len(baseline) // 2]
    threshold = max(0.2, 3.0 * (med if med > 1e-6 else 0.02))
    streak = 0
    onset_i = None
    worst = 0.0
    for it, v in series:
        if v >= threshold:
            streak += 1
            worst = max(worst, v)
            if onset_i is None:
                onset_i = int(it) if it is not None else None
        else:
            streak = 0
            onset_i = None
        if streak >= 3:
            return _finding(
                "kl_spike",
                severity="high" if worst >= 0.5 else "medium",
                confidence=min(1.0, worst / max(threshold, 1e-6)),
                evidence={"threshold": round(threshold, 4), "peak": round(worst, 4), "median": round(med, 4)},
                onset=onset_i,
            )
    return None


def _detect_grad(iters: List[Dict[str, Any]]) -> Optional[Finding]:
    series = [(_scalar(r, "iteration"), _scalar(r, "gradient_norm", "grad_norm", "gradnorm")) for r in iters]
    series = [(i, v) for i, v in series if v is not None]
    if not series:
        return None
    peak_i, peak = max(series, key=lambda p: p[1])
    if peak >= 100:
        return _finding(
            "gradient_explosion",
            severity="critical" if peak >= 1e3 else "high",
            confidence=min(1.0, math.log10(peak + 1) / 4.0),
            evidence={"peak": round(peak, 3)},
            onset=int(peak_i) if peak_i is not None else None,
        )
    return None


def _detect_plateau(iters: List[Dict[str, Any]]) -> Optional[Finding]:
    series = [(_scalar(r, "iteration"), _scalar(r, "mean_reward", "reward", "return")) for r in iters]
    series = [(i, v) for i, v in series if v is not None]
    if len(series) < 24:
        return None
    values = [v for _, v in series]
    early = _mean(_window(values, 0.3, head=True))
    mid = _mean(values[len(values) // 3 : 2 * len(values) // 3])
    late = _mean(_window(values, 0.3, head=False))
    if early is None or mid is None or late is None:
        return None
    gained = mid - early
    stalled = abs(late - mid) <= 0.05 * (abs(mid) + 1.0)
    if gained > 0.15 * (abs(early) + 1.0) and stalled:
        return _finding(
            "reward_plateau",
            severity="medium",
            confidence=0.55,
            evidence={
                "early_mean": round(early, 4),
                "mid_mean": round(mid, 4),
                "late_mean": round(late, 4),
            },
        )
    return None


def _detect_reward_hacking(iters: List[Dict[str, Any]], episodes: List[Dict[str, Any]]) -> Optional[Finding]:
    rewards = [_scalar(r, "mean_reward", "reward") for r in iters]
    rewards = [v for v in rewards if v is not None]
    if len(rewards) < 12 or len(episodes) < 8:
        return None
    successes: List[Tuple[Optional[float], bool]] = []
    for ep in episodes:
        it = _scalar(ep, "iteration")
        suc = _tag(ep, "success")
        if suc is None:
            continue
        successes.append((it, bool(suc)))
    if len(successes) < 8:
        return None
    split = max(3, len(rewards) // 2)
    rew_early, rew_late = _mean(rewards[:split]), _mean(rewards[split:])
    suc_vals = [1.0 if s else 0.0 for _, s in successes]
    suc_split = max(3, len(suc_vals) // 2)
    suc_early, suc_late = _mean(suc_vals[:suc_split]), _mean(suc_vals[suc_split:])
    if None in (rew_early, rew_late, suc_early, suc_late):
        return None
    rew_up = (rew_late - rew_early) / (abs(rew_early) + 1.0)
    suc_down = suc_early - suc_late
    if rew_up > 0.15 and suc_down > 0.1:
        return _finding(
            "reward_hacking",
            severity="critical",
            confidence=min(1.0, rew_up + suc_down),
            evidence={
                "reward_early": round(rew_early, 4),
                "reward_late": round(rew_late, 4),
                "success_early": round(suc_early, 4),
                "success_late": round(suc_late, 4),
            },
        )
    return None


def _action_vectors(steps: List[Dict[str, Any]]) -> List[List[float]]:
    out: List[List[float]] = []
    for r in steps:
        vec = r.get("vectors") or r.get("_vectors") or {}
        act = vec.get("action") if isinstance(vec, dict) else None
        if not act:
            continue
        row = []
        ok = True
        for x in act:
            if x is None:
                ok = False
                break
            try:
                row.append(float(x))
            except (TypeError, ValueError):
                ok = False
                break
        if ok and row:
            out.append(row)
    return out


def _detect_action_std_zero(steps: List[Dict[str, Any]], iters: List[Dict[str, Any]]) -> Optional[Finding]:
    # Prefer explicit scalar if the trainer logged it.
    logged = [_scalar(r, "action_std", "std", "mean_action_std") for r in iters]
    logged = [v for v in logged if v is not None]
    if logged and min(logged) < 1e-5:
        return _finding(
            "action_std_zero",
            severity="critical",
            confidence=0.95,
            evidence={"logged_min_std": min(logged)},
        )
    acts = _action_vectors(steps)
    if len(acts) < 16:
        return None
    dim = min(len(a) for a in acts)
    if dim == 0:
        return None
    zero_dims: List[int] = []
    stds: List[float] = []
    for d in range(dim):
        col = [a[d] for a in acts]
        s = _std(col) or 0.0
        stds.append(s)
        if s < 1e-5:
            zero_dims.append(d)
    if zero_dims:
        return _finding(
            "action_std_zero",
            severity="critical",
            confidence=0.9,
            evidence={
                "zero_dims": zero_dims,
                "dim": dim,
                "min_std": round(min(stds), 8),
            },
        )
    return None


def _detect_idle(steps: List[Dict[str, Any]]) -> Optional[Finding]:
    acts = _action_vectors(steps)
    if len(acts) < 16:
        return None
    mags = [math.sqrt(sum(x * x for x in a)) for a in acts]
    m = _mean(mags)
    if m is not None and m < 0.02:
        return _finding(
            "idle_policy",
            severity="high",
            confidence=min(1.0, (0.02 - m) / 0.02),
            evidence={"mean_action_l2": round(m, 5), "n_steps": len(mags)},
        )
    return None


def _detect_saturation(steps: List[Dict[str, Any]], limit: float = 0.95) -> Optional[Finding]:
    acts = _action_vectors(steps)
    if len(acts) < 16:
        return None
    n = 0
    sat = 0
    for a in acts:
        n += len(a)
        sat += sum(1 for x in a if abs(x) >= limit)
    frac = sat / n if n else 0.0
    if frac >= 0.25:
        return _finding(
            "torque_saturation",
            severity="high" if frac >= 0.5 else "medium",
            confidence=min(1.0, frac),
            evidence={"saturated_fraction": round(frac, 4), "limit": limit},
        )
    return None


def _detect_instruction(episodes: List[Dict[str, Any]]) -> Optional[Finding]:
    prim: List[float] = []
    inst: List[float] = []
    for ep in episodes:
        p = _tag(ep, "primitive_success")
        s = _tag(ep, "success")
        if p is None or s is None:
            continue
        prim.append(1.0 if p else 0.0)
        inst.append(1.0 if s else 0.0)
    if len(prim) < 8:
        return None
    mp, mi = _mean(prim), _mean(inst)
    if mp is None or mi is None:
        return None
    gap = mp - mi
    if mp >= 0.6 and gap >= 0.25:
        return _finding(
            "instruction_decoupling",
            severity="high",
            confidence=min(1.0, gap / 0.5),
            evidence={
                "primitive_success": round(mp, 4),
                "instruction_success": round(mi, 4),
                "n": len(prim),
            },
        )
    return None


def _detect_early_term(episodes: List[Dict[str, Any]]) -> Optional[Finding]:
    lengths: List[float] = []
    terms: Dict[str, int] = {}
    for ep in episodes:
        ln = _scalar(ep, "episode_length", "length")
        if ln is None:
            continue
        lengths.append(ln)
        t = str(_tag(ep, "termination") or "unknown")
        if t.lower() not in ("", "time_out", "timeout", "truncated"):
            terms[t] = terms.get(t, 0) + 1
    if len(lengths) < 8:
        return None
    med = sorted(lengths)[len(lengths) // 2]
    short = sum(1 for x in lengths if x < 0.4 * med) / len(lengths)
    if short >= 0.3 and terms:
        top = max(terms.items(), key=lambda kv: kv[1])
        return _finding(
            "early_termination",
            severity="high",
            confidence=min(1.0, short),
            evidence={
                "median_length": round(med, 2),
                "early_fraction": round(short, 4),
                "top_termination": top[0],
                "top_count": top[1],
            },
        )
    return None


def _detect_covariate(run: Optional[ProbeRun], steps: List[Dict[str, Any]]) -> Optional[Finding]:
    if run is None:
        return None
    try:
        sources = [s for s in run.sources() if s]
    except Exception:
        sources = []
    sim = [s for s in sources if s.startswith("sim:")]
    real = [s for s in sources if s.startswith("real:")]
    if not sim or not real:
        # Also allow mixed sources already loaded in `steps`.
        srcs = {str(r.get("source") or "") for r in steps}
        sim = [s for s in srcs if s.startswith("sim:")]
        real = [s for s in srcs if s.startswith("real:")]
    if not sim or not real:
        return None
    from .compare import compare_sources

    try:
        report = compare_sources(run, sim[0], real[0])
    except Exception:
        return None
    risk = float(report.get("transfer_risk") or 0.0)
    if risk >= 35:
        return _finding(
            "covariate_shift",
            severity="critical" if risk >= 70 else "high",
            confidence=min(1.0, risk / 100.0),
            evidence=report,
        )
    return None


# ── Public API ────────────────────────────────────────────────────────────────


def diagnose(
    *,
    iterations: Optional[Iterable[Any]] = None,
    episodes: Optional[Iterable[Any]] = None,
    steps: Optional[Iterable[Any]] = None,
    run: Optional[ProbeRun] = None,
) -> List[Finding]:
    """Return ranked findings for one run.

    Pass already-loaded rows, or a ``ProbeRun``. Never raises — a detector that
    cannot fire simply omits its finding.
    """
    if run is not None:
        try:
            iterations = iterations if iterations is not None else run.iterations()
        except Exception:
            iterations = iterations or []
        try:
            episodes = episodes if episodes is not None else run.episodes()
        except Exception:
            episodes = episodes or []
        try:
            steps = steps if steps is not None else run.steps()
        except Exception:
            steps = steps or []
    iters = _as_rows(iterations)
    eps = _as_rows(episodes)
    stps = _as_rows(steps)

    found: List[Finding] = []
    for fn in (
        lambda: _detect_entropy(iters),
        lambda: _detect_kl(iters),
        lambda: _detect_grad(iters),
        lambda: _detect_plateau(iters),
        lambda: _detect_reward_hacking(iters, eps),
        lambda: _detect_action_std_zero(stps, iters),
        lambda: _detect_idle(stps),
        lambda: _detect_saturation(stps),
        lambda: _detect_instruction(eps),
        lambda: _detect_early_term(eps),
        lambda: _detect_covariate(run, stps),
    ):
        try:
            hit = fn()
        except Exception:
            hit = None
        if hit is not None:
            found.append(hit)
    found.sort(key=lambda f: (SEVERITY_RANK.get(f.severity, 9), -f.confidence, f.id))
    return found


def diagnose_run(run: ProbeRun) -> List[Finding]:
    """Convenience: diagnose a local ProbeRun store."""
    return diagnose(run=run)


def health_score(findings: Sequence[Finding], *, failure_rate: Optional[float] = None) -> Dict[str, Any]:
    """0–100 run health. Critical findings dominate; used as a pre-certificate signal."""
    penalty = 0.0
    weights = {"critical": 35.0, "high": 18.0, "medium": 8.0, "low": 3.0}
    for f in findings:
        penalty += weights.get(f.severity, 5.0) * f.confidence
    if failure_rate is not None:
        penalty += 25.0 * max(0.0, min(1.0, failure_rate))
    score = max(0.0, min(100.0, 100.0 - penalty))
    if score >= 80:
        band = "high"
    elif score >= 55:
        band = "moderate"
    else:
        band = "low"
    return {
        "score": round(score, 1),
        "band": band,
        "finding_count": len(findings),
        "critical": sum(1 for f in findings if f.severity == "critical"),
    }
