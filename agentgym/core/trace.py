"""The common capture record for every lever, aligned to OpenTelemetry's GenAI semantic
conventions rather than a bespoke schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class Span:
    span_id: str
    parent_span_id: str | None
    trace_id: str
    name: str
    kind: Literal["llm", "tool", "retrieval", "other"]
    input: dict
    output: dict
    start_time: float
    end_time: float
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "trace_id": self.trace_id,
            "name": self.name,
            "kind": self.kind,
            "input": self.input,
            "output": self.output,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Span":
        return cls(
            span_id=d["span_id"],
            parent_span_id=d.get("parent_span_id"),
            trace_id=d["trace_id"],
            name=d["name"],
            kind=d["kind"],
            input=d.get("input", {}),
            output=d.get("output", {}),
            start_time=d["start_time"],
            end_time=d["end_time"],
            attributes=d.get("attributes", {}),
        )


@dataclass
class Trace:
    trace_id: str
    task_id: str
    spans: list[Span]
    metadata: dict

    def to_otel(self) -> list[dict]:
        return [s.to_dict() for s in self.spans]

    @classmethod
    def from_otel_spans(cls, otel_spans: list[dict], task_id: str, metadata: dict) -> "Trace":
        if not otel_spans:
            raise ValueError("from_otel_spans requires at least one span")
        spans = [Span.from_dict(d) for d in otel_spans]
        trace_ids = {s.trace_id for s in spans}
        if len(trace_ids) > 1:
            raise ValueError(f"all spans must share one trace_id, got {trace_ids}")
        return cls(trace_id=spans[0].trace_id, task_id=task_id, spans=spans, metadata=metadata)
