"""
probe.viz — local-first Failure Reconstruction viewer.

Renders one episode's per-step trace into a SINGLE self-contained HTML file:
synchronized obs / action / reward-term / policy-state timelines with a scrubber
and the first-abnormal-signal onset marked. Zero dependencies (vanilla JS + SVG,
no CDN), so it opens in any browser, works air-gapped, and is shareable as one
file — the researcher-first counterpart to the platform's React replay panel.

    from simtooreal_probe import ProbeRun
    run = ProbeRun("go2-flat-v3")
    ep = run.failures()[0]["episode_id"]
    path = run.to_html(ep)          # -> ~/.probe/<run>/replay_<ep>.html
    # or: from simtooreal_probe.viz import export_failure_html

The HTML embeds the episode data inline as JSON; nothing is fetched at view time.
"""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .trace import Grain, V_ACTION, V_OBS, V_POLICY_STATE


def _rows_for_episode(run, episode_id: str, source: Optional[str]) -> List[Dict[str, Any]]:
    """Raw per-step dicts for one episode, sorted by step (no pandas dependency)."""
    rows = [r for r in run._load(Grain.STEP, source) if r.get("episode_id") == episode_id]
    rows.sort(key=lambda r: r.get("step_idx") or 0)
    return rows


def build_payload(run, episode_id: str, source: Optional[str] = None) -> Dict[str, Any]:
    """Assemble the JSON payload the viewer renders. Pure data — no HTML."""
    rows = _rows_for_episode(run, episode_id, source)
    if not rows:
        # Return a minimal valid payload so the viewer renders an empty state
        # instead of crashing on DATA.steps[0] or DATA.obs[0] accesses.
        return {
            "run_id": run.run_id,
            "episode_id": episode_id,
            "source": source or "",
            "n_steps": 0,
            "termination": None,
            "success": None,
            "ep_return": None,
            "onset": None,
            "steps": [],
            "obs": [],
            "action": [],
            "reward_terms": [],
            "obs_names": [],
            "action_names": [],
            "reward_term_names": [],
        }
    ps_names = run.dims.names(V_POLICY_STATE) or ["log_prob", "value", "return"]
    obs_names = run.dims.names(V_OBS)
    act_names = run.dims.names(V_ACTION)
    rt_names = run.dims.names("reward_terms")

    def ps_index(name: str) -> int:
        return ps_names.index(name) if name in ps_names else -1

    i_val, i_lp, i_ret = ps_index("value"), ps_index("log_prob"), ps_index("return")

    steps: List[Dict[str, Any]] = []
    obs_mat: List[List[float]] = []
    act_mat: List[List[float]] = []
    rt_mat: List[Dict[str, float]] = []

    for r in rows:
        vec = r.get("vectors", {}) or {}
        sc = r.get("scalars", {}) or {}
        ps = vec.get(V_POLICY_STATE, [])
        steps.append(
            {
                "i": r.get("step_idx") or 0,
                "reward": float(sc.get("reward", 0.0)),
                "value": float(ps[i_val]) if 0 <= i_val < len(ps) else None,
                "log_prob": float(ps[i_lp]) if 0 <= i_lp < len(ps) else None,
                "ret": float(ps[i_ret]) if 0 <= i_ret < len(ps) else None,
            }
        )
        obs_mat.append([float(x) for x in vec.get(V_OBS, [])])
        act_mat.append([float(x) for x in vec.get(V_ACTION, [])])
        rt = vec.get("reward_terms", [])
        rt_mat.append(run.dims.resolve("reward_terms", rt) if rt else {})

    onset = None
    try:
        onset = run.first_abnormal(episode_id, source)
    except Exception:
        onset = None

    ep_meta = next((e for e in run._load(Grain.EPISODE, source) if e.get("episode_id") == episode_id), {})
    tags = ep_meta.get("tags", {}) or {}

    return {
        "run_id": run.run_id,
        "episode_id": episode_id,
        "source": rows[0].get("source") if rows else (source or ""),
        "n_steps": len(steps),
        "termination": tags.get("termination"),
        "success": tags.get("success"),
        "ep_return": (ep_meta.get("scalars", {}) or {}).get("return"),
        "onset": onset,
        "steps": steps,
        "obs": obs_mat,
        "action": act_mat,
        "reward_terms": rt_mat,
        "obs_names": obs_names,
        "action_names": act_names,
        "reward_term_names": rt_names,
    }


def render_failure_replay_html(run, episode_id: str, source: Optional[str] = None) -> str:
    """Return a full standalone HTML document for one episode's replay."""
    payload = build_payload(run, episode_id, source)
    # `</` inside the embedded JSON would prematurely close the <script>; escape it.
    data_json = json.dumps(payload, separators=(",", ":"), default=lambda o: None).replace("</", "<\\/")
    title = html.escape(f"{payload['run_id']} · {payload['episode_id']}")
    return _TEMPLATE.replace("__TITLE__", title).replace("__DATA__", data_json)


def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in s)


def export_failure_html(
    run_id: str,
    episode_id: str,
    *,
    root: Optional[str] = None,
    source: Optional[str] = None,
    out: Optional[str] = None,
) -> str:
    """Write a standalone replay HTML file for one episode; return its path.

    Convenience entrypoint usable without the platform: it opens the local
    ProbeRun store and renders. `ProbeRun.to_html(...)` wraps this.
    """
    from .query import ProbeRun

    run = ProbeRun(run_id, root=root)
    doc = render_failure_replay_html(run, episode_id, source)
    out_path = Path(out) if out else (run.dir / f"replay_{_safe_name(episode_id)}.html")
    out_path.write_text(doc, encoding="utf-8")
    return str(out_path)


# ── The viewer template ───────────────────────────────────────────────────────
# Self-contained: inline CSS + vanilla JS + SVG. `__DATA__` is replaced with the
# JSON payload, `__TITLE__` with the document title. No external requests.

_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Failure Replay · __TITLE__</title>
<style>
  :root{
    --bg:#0b0e14; --panel:#121722; --panel2:#0f131c; --line:#1e2633;
    --txt:#e6edf3; --muted:#8b97a7; --accent:#5b9dff; --good:#3fb950;
    --bad:#ff6b6b; --warn:#e3b341; --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--txt);
    font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;font-size:14px}
  header{padding:16px 22px;border-bottom:1px solid var(--line);
    display:flex;align-items:baseline;gap:16px;flex-wrap:wrap}
  header h1{font-size:15px;margin:0;font-weight:600;letter-spacing:.2px}
  .pill{font:600 11px/1 var(--mono);padding:4px 8px;border-radius:999px;
    background:#1b2230;color:var(--muted);border:1px solid var(--line)}
  .pill.bad{color:var(--bad);border-color:#3a2030;background:#1f1518}
  .pill.good{color:var(--good);border-color:#1f3a26;background:#101a13}
  .pill.onset{color:var(--warn);border-color:#3a3320;background:#1c1810}
  main{display:grid;grid-template-columns:1fr 360px;gap:14px;padding:14px 18px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
  .card h2{font-size:11px;text-transform:uppercase;letter-spacing:.6px;color:var(--muted);
    margin:0 0 8px;font-weight:600}
  .chart{width:100%;height:96px;display:block}
  .chart + .chart{margin-top:6px}
  .scrubwrap{position:sticky;bottom:0;background:var(--panel2);border:1px solid var(--line);
    border-radius:10px;padding:12px 16px;margin:0 18px 18px;display:flex;gap:14px;align-items:center}
  input[type=range]{flex:1;accent-color:var(--accent)}
  .stepnum{font:600 13px var(--mono);min-width:120px}
  .bars{display:flex;flex-direction:column;gap:5px}
  .bar{display:grid;grid-template-columns:120px 1fr 64px;align-items:center;gap:8px;font-size:12px}
  .bar .nm{color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font:11px var(--mono)}
  .track{height:9px;background:#0c1018;border-radius:5px;position:relative;overflow:hidden}
  .fill{position:absolute;top:0;bottom:0;left:50%;border-radius:5px}
  .val{font:11px var(--mono);text-align:right;color:var(--txt)}
  .legend{display:flex;gap:12px;flex-wrap:wrap;color:var(--muted);font:11px var(--mono)}
  .legend span::before{content:"";display:inline-block;width:9px;height:9px;border-radius:2px;
    margin-right:5px;vertical-align:middle;background:var(--c)}
  .muted{color:var(--muted)}
  .grid-span{grid-column:1 / -1}
</style>
</head>
<body>
<header>
  <h1 id="title">Failure Replay</h1>
  <span class="pill" id="src"></span>
  <span class="pill" id="ret"></span>
  <span class="pill" id="term"></span>
  <span class="pill onset" id="onset"></span>
</header>

<main>
  <section class="card">
    <h2>Policy timeline <span class="muted" id="cursorlbl"></span></h2>
    <div class="legend">
      <span style="--c:var(--accent)">reward</span>
      <span style="--c:var(--good)">value</span>
      <span style="--c:var(--warn)">log_prob</span>
      <span style="--c:var(--bad)">onset</span>
    </div>
    <svg class="chart" id="c_reward"></svg>
    <svg class="chart" id="c_value"></svg>
    <svg class="chart" id="c_logprob"></svg>
  </section>

  <aside class="card">
    <h2>Reward terms @ step <span id="rt_step" class="muted"></span></h2>
    <div class="bars" id="rt_bars"></div>
    <h2 style="margin-top:14px">Action</h2>
    <div class="bars" id="act_bars"></div>
    <h2 style="margin-top:14px">Observation</h2>
    <div class="bars" id="obs_bars"></div>
  </aside>
</main>

<div class="scrubwrap">
  <div class="stepnum">step <span id="s_idx">0</span> / <span id="s_max">0</span></div>
  <input type="range" id="scrub" min="0" max="0" value="0" step="1"/>
  <button id="play" style="background:#1b2230;color:var(--txt);border:1px solid var(--line);
    border-radius:7px;padding:6px 12px;cursor:pointer;font:600 12px var(--mono)">▶ play</button>
</div>

<script>
const DATA = __DATA__;
const S = DATA.steps, N = S.length;
const onsetStep = DATA.onset ? DATA.onset.onset_step : null;
let cur = 0, timer = null;

// header
document.getElementById('title').textContent = 'Failure Replay · ' + DATA.episode_id;
document.getElementById('src').textContent = DATA.source || 'sim';
const ret = DATA.ep_return;
document.getElementById('ret').textContent = (ret==null?'return —':'return ' + (+ret).toFixed(2));
const term = document.getElementById('term');
if (DATA.success === false || (DATA.termination && !/time|trunc/i.test(DATA.termination))){
  term.textContent = DATA.termination || 'failed'; term.className='pill bad';
} else { term.textContent = DATA.termination || 'ok'; term.className='pill good'; }
const onsetEl = document.getElementById('onset');
if (DATA.onset){ onsetEl.textContent = 'onset: step ' + DATA.onset.onset_step + ' · ' + DATA.onset.channel; }
else { onsetEl.style.display='none'; }
document.getElementById('s_max').textContent = N-1;
document.getElementById('scrub').max = N-1;

// ── tiny SVG line chart ──────────────────────────────────────────────────────
function series(key){ return S.map(s => s[key]); }
function esc(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }
function pctRange(vals){
  // Percentile-based scale so a single spike at failure doesn't flatten the
  // entire pre-failure timeline (the most common complaint in failure viewers).
  const sorted = [...vals].filter(v=>v!=null&&isFinite(v)).sort((a,b)=>a-b);
  if (!sorted.length) return [0,1];
  const p5  = sorted[Math.max(0, Math.floor(sorted.length*0.05))];
  const p95 = sorted[Math.min(sorted.length-1, Math.ceil(sorted.length*0.95-1))];
  const lo = p5 - Math.abs(p5)*0.05;   // 5% headroom below p5
  const hi = p95 + Math.abs(p95)*0.05; // 5% headroom above p95
  return [lo, hi];
}
function drawChart(svgId, vals, color){
  const svg = document.getElementById(svgId);
  const W = svg.clientWidth || 640, H = svg.clientHeight || 96, pad = 6;
  const xs = i => pad + (N<=1?0:i*(W-2*pad)/(N-1));
  let [lo, hi] = pctRange(vals);
  if (!isFinite(lo)||!isFinite(hi)){ lo=0; hi=1; }
  if (Math.abs(hi-lo) < 1e-9){ hi=lo+1; lo=lo-1; }
  const ys = v => H-pad - ((v-lo)/(hi-lo))*(H-2*pad);
  let d=''; let started=false;
  vals.forEach((v,i)=>{ if(v==null||!isFinite(v)){started=false;return;}
    d += (started?' L':' M') + xs(i).toFixed(1) + ' ' + ys(v).toFixed(1); started=true; });
  let svgInner =
    `<rect x="0" y="0" width="100%" height="100%" fill="var(--panel2)" rx="6"/>` +
    `<line x1="${pad}" y1="${ys(0).toFixed(1)}" x2="${W-pad}" y2="${ys(0).toFixed(1)}" stroke="#243049" stroke-dasharray="2 3"/>`;
  if (onsetStep!=null && onsetStep>=0 && onsetStep<N)
    svgInner += `<line class="onsetline" x1="${xs(onsetStep)}" y1="0" x2="${xs(onsetStep)}" y2="${H}" stroke="var(--bad)" stroke-dasharray="3 3" opacity=".7"/>`;
  svgInner += `<path d="${d}" fill="none" stroke="${color}" stroke-width="1.6"/>`;
  svgInner += `<line class="cursor" x1="0" y1="0" x2="0" y2="${H}" stroke="#fff" opacity=".35"/>`;
  svg.innerHTML = svgInner;
  svg.dataset.lo = lo; svg.dataset.hi = hi;
}
function moveCursor(svgId){
  const svg = document.getElementById(svgId);
  const W = svg.clientWidth || 640, pad=6;
  const x = pad + (N<=1?0:cur*(W-2*pad)/(N-1));
  const c = svg.querySelector('.cursor');
  if (c){ c.setAttribute('x1',x); c.setAttribute('x2',x); }
}

// ── bar panels ───────────────────────────────────────────────────────────────
function bipolarBar(v, scale){
  // returns {left,width,color} for a centered bar in [-1,1]*scale range
  const f = Math.max(-1, Math.min(1, v/(scale||1)));
  const w = Math.abs(f)*50;
  const left = f>=0 ? 50 : 50-w;
  const color = f>=0 ? 'var(--good)' : 'var(--bad)';
  return {left, w, color};
}
function renderBars(elId, names, vec, scale){
  const el = document.getElementById(elId);
  if (!vec || !vec.length){ el.innerHTML = '<span class="muted">—</span>'; return; }
  let h='';
  for (let i=0;i<vec.length;i++){
    const raw = (names && names[i]) ? names[i] : (elId.replace('_bars','')+'['+i+']');
    const nm = esc(raw);
    const b = bipolarBar(vec[i], scale);
    h += `<div class="bar"><div class="nm" title="${nm}">${nm}</div>`+
         `<div class="track"><div class="fill" style="left:${b.left}%;width:${b.w}%;background:${b.color}"></div></div>`+
         `<div class="val">${(+vec[i]).toFixed(3)}</div></div>`;
  }
  el.innerHTML = h;
}
function renderRewardTerms(step){
  const el = document.getElementById('rt_bars');
  const obj = DATA.reward_terms[step] || {};
  const keys = Object.keys(obj);
  if (!keys.length){ el.innerHTML='<span class="muted">no reward-term breakdown captured</span>'; return; }
  const mx = Math.max.apply(null, keys.map(k=>Math.abs(obj[k]))) || 1;
  let h='';
  keys.forEach(k=>{ const b=bipolarBar(obj[k], mx);
    h += `<div class="bar"><div class="nm" title="${k}">${k}</div>`+
         `<div class="track"><div class="fill" style="left:${b.left}%;width:${b.w}%;background:${b.color}"></div></div>`+
         `<div class="val">${(+obj[k]).toFixed(3)}</div></div>`; });
  el.innerHTML=h;
}

function update(step){
  cur = Math.max(0, Math.min(N-1, step|0));
  document.getElementById('s_idx').textContent = cur;
  document.getElementById('rt_step').textContent = cur;
  document.getElementById('scrub').value = cur;
  const s = S[cur] || {};
  document.getElementById('cursorlbl').textContent =
    `· reward ${(s.reward||0).toFixed(3)}  value ${s.value==null?'—':(+s.value).toFixed(3)}  log_prob ${s.log_prob==null?'—':(+s.log_prob).toFixed(3)}`;
  ['c_reward','c_value','c_logprob'].forEach(moveCursor);
  renderRewardTerms(cur);
  renderBars('act_bars', DATA.action_names, DATA.action[cur], 1.0);
  renderBars('obs_bars', DATA.obs_names, DATA.obs[cur], 3.0);
}

function drawAll(){
  drawChart('c_reward', series('reward'), 'var(--accent)');
  drawChart('c_value', series('value'), 'var(--good)');
  drawChart('c_logprob', series('log_prob'), 'var(--warn)');
  update(cur);
}
if (N === 0) {
  // No step data captured for this episode — show an empty state.
  document.querySelector('main').innerHTML =
    '<div class="card grid-span" style="padding:32px;text-align:center;color:var(--muted)">' +
    'No per-step trace captured for this episode.<br>' +
    'Set <code>PROBE_CAPTURE=full</code> to enable per-step recording.</div>';
  document.querySelector('.scrubwrap').style.display='none';
} else {
  document.getElementById('scrub').addEventListener('input', e => update(+e.target.value));
  document.getElementById('play').addEventListener('click', e=>{
    if (timer){ clearInterval(timer); timer=null; e.target.textContent='▶ play'; return; }
    e.target.textContent='⏸ pause';
    timer = setInterval(()=>{ if(cur>=N-1){clearInterval(timer);timer=null;document.getElementById('play').textContent='▶ play';return;} update(cur+1); }, 80);
  });
  window.addEventListener('resize', drawAll);
  // jump to a few steps before the onset for immediate context
  if (onsetStep!=null) cur = Math.max(0, onsetStep-5);
  drawAll();
}
</script>
</body>
</html>"""
