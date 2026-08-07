"""
probe.sinks — ordered output destinations for trace events.

A `TraceWriter` fans out every batch to one or more sinks:

  * `LocalSink`  — local-first. Writes `~/.probe/<run_id>/...` with zero account
                   and zero network. Parquet when pyarrow is available, newline
                   JSON (`.jsonl`) otherwise — so it works inside the bare Isaac
                   container. This is what `ProbeRun(...)` reads back in a notebook.
  * `HttpSink`   — optional network sink. Posts JSON batches to a configurable
                   base URL (stdlib urllib only). Pass any object with
                   ``.post(path, payload) -> dict`` (e.g. a platform UploadClient)
                   to reuse custom auth/retry machinery.

Both degrade gracefully: a sink that fails never raises into the training loop.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import List, Optional, Sequence

from .trace import DimRegistry, Grain, PolicyTraceEvent, _san_scalars, _san_vectors

# pyarrow is optional (the `probe` extra). Detected once.
try:  # pragma: no cover - import guard
    import pyarrow as _pa  # type: ignore
    import pyarrow.parquet as _pq  # type: ignore
    _HAVE_ARROW = True
except Exception:  # pragma: no cover
    _pa = None  # type: ignore
    _pq = None  # type: ignore
    _HAVE_ARROW = False


def _sanitize(source: str) -> str:
    """Encode a source into a reversible, filesystem-safe directory name.

    Only `:` `/` `\\` are problematic on Windows; percent-encode exactly those
    (and `%` itself) so underscores in serials (e.g. `real:robot_01`) survive the
    round-trip — a lossy `_` swap silently corrupted such sources.
    """
    return (source.replace("%", "%25").replace(":", "%3A")
            .replace("/", "%2F").replace("\\", "%5C"))


def _desanitize(name: str) -> str:
    return (name.replace("%5C", "\\").replace("%2F", "/")
            .replace("%3A", ":").replace("%25", "%"))


class Sink:
    """Base class. Subclasses override `write`; `write_dims`/`close` are optional."""

    def write(self, events: Sequence[PolicyTraceEvent]) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def write_dims(self, registry: DimRegistry) -> None:
        pass

    def close(self) -> None:
        pass


# ── LocalSink ─────────────────────────────────────────────────────────────────


class LocalSink(Sink):
    """Local-first trace store. Layout under `root/<run_id>/`:

        _meta.json                       run metadata
        _dims.json                       DimRegistry (named vector components)
        <source>/iteration.jsonl|parquet
        <source>/episode.jsonl|parquet
        <source>/step.jsonl|parquet      (the high-volume layer)

    JSONL is append-only and crash-safe (one event per line). Parquet is written
    in row-group flushes when pyarrow is present.
    """

    def __init__(
        self,
        run_id: str,
        root: Optional[str] = None,
        *,
        prefer_parquet: bool = True,
        meta: Optional[dict] = None,
    ) -> None:
        self.run_id = run_id
        base = root or os.environ.get("PROBE_HOME") or os.path.join(Path.home(), ".probe")
        self.dir = Path(base) / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._use_parquet = bool(prefer_parquet and _HAVE_ARROW)
        self._lock = threading.Lock()
        # Open JSONL append handles lazily, keyed by (source, grain).
        self._handles: dict = {}
        self._format = "parquet" if self._use_parquet else "jsonl"
        self._write_meta(meta or {})

    # -- public ---------------------------------------------------------------

    @property
    def format(self) -> str:
        return self._format

    def write(self, events: Sequence[PolicyTraceEvent]) -> None:
        if not events:
            return
        with self._lock:
            if self._use_parquet:
                self._write_parquet(events)
            else:
                self._write_jsonl(events)

    def write_dims(self, registry: DimRegistry) -> None:
        try:
            (self.dir / "_dims.json").write_text(json.dumps(registry.to_dict(), indent=1))
        except Exception:
            pass

    def close(self) -> None:
        with self._lock:
            for fh in self._handles.values():
                try:
                    fh.close()
                except Exception:
                    pass
            self._handles.clear()

    # -- internals ------------------------------------------------------------

    def _write_meta(self, meta: dict) -> None:
        try:
            payload = {"run_id": self.run_id, "format": self._format, **meta}
            (self.dir / "_meta.json").write_text(json.dumps(payload, indent=1, default=str))
        except Exception:
            pass

    def _shard_path(self, source: str, grain: Grain, ext: str) -> Path:
        d = self.dir / _sanitize(source)
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{grain.value}.{ext}"

    def _write_jsonl(self, events: Sequence[PolicyTraceEvent]) -> None:
        for ev in events:
            key = (ev.source, ev.grain)
            fh = self._handles.get(key)
            if fh is None:
                fh = open(self._shard_path(ev.source, ev.grain, "jsonl"), "a", encoding="utf-8")
                self._handles[key] = fh
            fh.write(ev.to_json() + "\n")
        for fh in self._handles.values():
            try:
                fh.flush()
            except Exception:
                pass

    def _write_parquet(self, events: Sequence[PolicyTraceEvent]) -> None:
        # Group by (source, grain); append a row-group per flush. Each grain gets
        # a stable, flattened column set so DuckDB can scan it directly.
        groups: dict = {}
        for ev in events:
            groups.setdefault((ev.source, ev.grain), []).append(ev)

        pending: list = []   # events not durably written to parquet (route to JSONL)
        failed = False
        for (source, grain), evs in groups.items():
            if failed:
                pending.extend(evs)
                continue
            try:
                table = _events_to_table(evs)
                key = (source, grain, "pq")
                writer = self._handles.get(key)
                if writer is None:
                    path = self._shard_path(source, grain, "parquet")
                    writer = _pq.ParquetWriter(str(path), table.schema)  # type: ignore[union-attr]
                    self._handles[key] = writer
                writer.write_table(table)
            except Exception:
                # Schema drift / write error -> abandon parquet for the rest of the
                # run. Close + finalise every open parquet writer (so already-written
                # shards stay readable) and re-route ONLY the un-written events to
                # JSONL. Already-succeeded groups are NOT re-written (no duplication).
                failed = True
                pending.extend(evs)
        if failed:
            self._use_parquet = False
            self._format = "jsonl"
            for k in list(self._handles):
                if isinstance(k, tuple) and len(k) == 3 and k[2] == "pq":
                    w = self._handles.pop(k)
                    try:
                        w.close()  # finalise footer -> shard remains readable
                    except Exception:
                        pass
            if pending:
                self._write_jsonl(pending)


def _events_to_table(events: Sequence[PolicyTraceEvent]):  # pragma: no cover - needs pyarrow
    """Flatten events into a columnar Arrow table.

    Scalars and vectors are kept as JSON-encoded columns plus the flat
    coordinate columns; this keeps the schema stable across heterogeneous
    grains/sources while still letting DuckDB extract fields with json functions
    or `read_parquet` + struct access.
    """
    cols = {
        "schema_version": [e.schema_version for e in events],
        "grain": [e.grain.value for e in events],
        "run_id": [e.run_id for e in events],
        "source": [e.source for e in events],
        "iteration": [e.iteration for e in events],
        "episode_id": [e.episode_id for e in events],
        "step_idx": [e.step_idx for e in events],
        "env_idx": [e.env_idx for e in events],
        "t_wall": [e.t_wall for e in events],
        "scalars": [json.dumps(_san_scalars(e.scalars), allow_nan=False) for e in events],
        "vectors": [json.dumps(_san_vectors(e.vectors), allow_nan=False) for e in events],
        "tags": [json.dumps(e.tags, default=str) for e in events],
    }
    return _pa.table(cols)  # type: ignore[union-attr]


# ── HttpSink ──────────────────────────────────────────────────────────────────


class _UrllibPostClient:
    """Minimal POST client (stdlib only). Protocol-compatible with platform clients.

    Any object with ``post(path, payload) -> dict`` can replace this when richer
    auth/retry is needed (e.g. SIMTOOREAL UploadClient).
    """

    def __init__(
        self,
        base_url: str,
        *,
        headers: Optional[dict] = None,
        timeout: float = 15.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = dict(headers or {})
        self.timeout = timeout

    def post(self, path: str, payload: dict) -> dict:
        import urllib.error
        import urllib.request

        url = f"{self.base_url}{path if path.startswith('/') else '/' + path}"
        body = json.dumps(payload, default=str).encode("utf-8")
        headers = {"Content-Type": "application/json", **self.headers}
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace") or "{}"
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    return {"success": True, "raw": raw[:200]}
                return data if isinstance(data, dict) else {"success": True, "data": data}
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:200]
            return {"success": False, "error": f"HTTP {e.code}: {detail}"}
        except Exception as e:  # noqa: BLE001
            return {"success": False, "error": str(e)}


class HttpSink(Sink):
    """Optional network sink for trace batches.

    Configuration (first non-empty wins for base URL):
      - constructor ``base_url``
      - env ``PROBE_HTTP_URL``
      - env ``SIMTOOREAL_URL`` (platform convenience; not required)

    Auth headers (optional):
      - constructor ``headers``
      - ``Authorization: Bearer $PROBE_HTTP_TOKEN`` or ``$SIMTOOREAL_API_KEY``
      - ``X-Ingest-Token: $SIMTOOREAL_INGEST_TOKEN`` when set

    Pass ``client=`` any object with ``.post(path, payload)`` to plug in custom
    transport (SIMTOOREAL tooling does this with UploadClient). Failures never
    raise; LocalSink remains the durable copy.
    """

    INGEST_PATH = "/api/v2/trace"
    DIMS_PATH = "/api/v2/trace/dims"
    MAX_CONSECUTIVE_FAILURES = 5

    def __init__(
        self,
        run_id: str,
        client: Optional[object] = None,
        *,
        base_url: Optional[str] = None,
        headers: Optional[dict] = None,
        ingest_path: Optional[str] = None,
        dims_path: Optional[str] = None,
    ) -> None:
        self.run_id = run_id
        self._client = client
        self._base_url = (
            base_url
            or os.environ.get("PROBE_HTTP_URL")
            or os.environ.get("SIMTOOREAL_URL")
            or ""
        ).rstrip("/")
        self._headers = dict(headers or {})
        self._ingest_path = ingest_path or self.INGEST_PATH
        self._dims_path = dims_path or self.DIMS_PATH
        self._enabled = True
        self._fail_streak = 0
        self._bootstrap_headers()

    def _bootstrap_headers(self) -> None:
        if "Authorization" not in self._headers:
            token = (
                os.environ.get("PROBE_HTTP_TOKEN")
                or os.environ.get("SIMTOOREAL_API_KEY")
                or ""
            )
            if token:
                self._headers["Authorization"] = f"Bearer {token}"
                if not token.startswith("eyJ"):
                    self._headers.setdefault("X-API-Key", token)
        if "X-Ingest-Token" not in self._headers:
            ingest = os.environ.get("SIMTOOREAL_INGEST_TOKEN") or os.environ.get(
                "PROBE_INGEST_TOKEN", ""
            )
            if ingest:
                self._headers["X-Ingest-Token"] = ingest

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self._base_url:
            self._enabled = False
            return None
        self._client = _UrllibPostClient(self._base_url, headers=self._headers)
        return self._client

    def write(self, events: Sequence[PolicyTraceEvent]) -> None:
        if not self._enabled or not events:
            return
        client = self._get_client()
        if client is None:
            return
        try:
            payload = {"run_id": self.run_id, "events": [e.to_dict() for e in events]}
            resp = client.post(self._ingest_path, payload)
            ok = bool(resp) and resp.get("success") is not False and not resp.get("error")
        except Exception:
            ok = False  # LocalSink remains the durable copy
        if ok:
            self._fail_streak = 0
            return
        self._fail_streak += 1
        if self._fail_streak >= self.MAX_CONSECUTIVE_FAILURES:
            self._enabled = False
            print(
                f"  [probe] HTTP trace sink disabled after {self._fail_streak} "
                f"consecutive failures — traces remain in the local store",
                flush=True,
            )

    def write_dims(self, registry: DimRegistry) -> None:
        if not self._enabled:
            return
        client = self._get_client()
        if client is None:
            return
        try:
            client.post(self._dims_path, registry.to_dict())
        except Exception:
            pass
