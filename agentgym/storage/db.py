"""AgentGymStore: SQLite-backed persistence for variants and traces. Thread-local connections in
WAL mode — the direct fix for the earlier project's single-connection design, which was never
safe for concurrent writers even though run_experiment.py used a ThreadPoolExecutor against it."""

from __future__ import annotations

import json
import sqlite3
import threading
import time

from agentgym.core.artifact import Artifact, ArtifactShape, FineTuneTechnique
from agentgym.core.diagnosis import Diagnosis
from agentgym.core.scope import Lever, Scope
from agentgym.core.score import Score
from agentgym.core.trace import Span, Trace
from agentgym.core.variant import AgentVariant

_SCHEMA = """
CREATE TABLE IF NOT EXISTS variants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    artifacts_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id TEXT NOT NULL,
    scope_str TEXT NOT NULL,
    trace_json TEXT NOT NULL,
    scores_json TEXT NOT NULL,
    diagnosis_json TEXT NOT NULL
);
"""


def _scope_to_str(scope: Scope) -> str:
    return str(scope)


def _scope_from_str(s: str) -> Scope:
    lever_str, *segments = s.split(".")
    return Scope(Lever(lever_str), tuple(segments))


def _artifact_to_dict(artifact: Artifact) -> dict:
    value = artifact.value
    if isinstance(value, list) and value and isinstance(value[0], Artifact):
        value = [_artifact_to_dict(v) for v in value]
        value_is_nested = True
    else:
        value_is_nested = False
    return {
        "scope": _scope_to_str(artifact.scope),
        "shape": artifact.shape.value,
        "value": value,
        "value_is_nested": value_is_nested,
        "optimizer_binding": artifact.optimizer_binding,
        "technique": artifact.technique.value if artifact.technique else None,
        "source": artifact.source,
        "provenance": artifact.provenance,
    }


def _artifact_from_dict(d: dict) -> Artifact:
    value = d["value"]
    if d.get("value_is_nested"):
        value = [_artifact_from_dict(v) for v in value]
    return Artifact(
        scope=_scope_from_str(d["scope"]),
        shape=ArtifactShape(d["shape"]),
        value=value,
        optimizer_binding=d["optimizer_binding"],
        technique=FineTuneTechnique(d["technique"]) if d["technique"] else None,
        source=d["source"],
        provenance=d["provenance"],
    )


def _variant_to_json(variant: AgentVariant) -> str:
    return json.dumps([_artifact_to_dict(a) for a in variant.artifacts.values()])


def _variant_from_json(s: str) -> AgentVariant:
    artifacts = [_artifact_from_dict(d) for d in json.loads(s)]
    return AgentVariant(artifacts={a.scope: a for a in artifacts})


def _span_to_dict(span: Span) -> dict:
    return span.to_dict()


def _trace_to_json(trace: Trace) -> str:
    return json.dumps({
        "trace_id": trace.trace_id,
        "task_id": trace.task_id,
        "spans": [_span_to_dict(s) for s in trace.spans],
        "metadata": trace.metadata,
    })


def _trace_from_json(s: str) -> Trace:
    d = json.loads(s)
    return Trace(
        trace_id=d["trace_id"], task_id=d["task_id"],
        spans=[Span.from_dict(sd) for sd in d["spans"]], metadata=d["metadata"],
    )


class AgentGymStore:
    def __init__(self, path: str):
        self._path = path
        self._local = threading.local()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(self._path, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            self._local.conn = conn
        return self._local.conn

    def _init_schema(self) -> None:
        conn = self._connect()
        conn.executescript(_SCHEMA)
        conn.commit()

    def save_variant(self, variant: AgentVariant) -> None:
        conn = self._connect()
        conn.execute(
            "INSERT INTO variants (created_at, artifacts_json) VALUES (?, ?)",
            (time.time(), _variant_to_json(variant)),
        )
        conn.commit()

    def latest_variant(self) -> AgentVariant | None:
        conn = self._connect()
        row = conn.execute(
            "SELECT artifacts_json FROM variants ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return _variant_from_json(row[0]) if row else None

    def save_trace(self, trace: Trace, scores: list[Score], diagnosis: Diagnosis) -> None:
        conn = self._connect()
        scope_str = _scope_to_str(diagnosis.suggested_scope) if diagnosis.suggested_scope else ""
        scores_json = json.dumps([
            {"metric_name": s.metric_name, "value": s.value, "kind": s.kind, "evidence": s.evidence}
            for s in scores
        ])
        diagnosis_json = json.dumps({
            "trace_id": diagnosis.trace_id,
            "lever": diagnosis.lever.value if diagnosis.lever else None,
            "evidence": diagnosis.evidence,
            "confidence": diagnosis.confidence,
            "suggested_scope": scope_str or None,
        })
        conn.execute(
            "INSERT INTO traces (trace_id, scope_str, trace_json, scores_json, diagnosis_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (trace.trace_id, scope_str, _trace_to_json(trace), scores_json, diagnosis_json),
        )
        conn.commit()

    def traces_for_scope(self, scope: Scope) -> list[Trace]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT trace_json FROM traces WHERE scope_str = ? ORDER BY id ASC",
            (_scope_to_str(scope),),
        ).fetchall()
        return [_trace_from_json(r[0]) for r in rows]
