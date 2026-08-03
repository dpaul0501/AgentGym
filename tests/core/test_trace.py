import pytest

from agentgym.core.trace import Span, Trace


def make_span(span_id, parent_id, trace_id, name="search_flights", kind="tool"):
    return Span(
        span_id=span_id,
        parent_span_id=parent_id,
        trace_id=trace_id,
        name=name,
        kind=kind,
        input={"q": "GVA-LIS"},
        output={"n": 3},
        start_time=0.0,
        end_time=0.3,
        attributes={"gen_ai.system": "openai", "gen_ai.usage.input_tokens": 120},
    )


def test_span_construction_all_fields():
    s = make_span("s1", None, "t1")
    assert s.name == "search_flights"
    assert s.attributes["gen_ai.usage.input_tokens"] == 120


def test_trace_holds_ordered_spans():
    s1 = make_span("s1", None, "t1")
    s2 = make_span("s2", "s1", "t1")
    trace = Trace(trace_id="t1", task_id="task1", spans=[s1, s2], metadata={})
    assert [s.span_id for s in trace.spans] == ["s1", "s2"]


def test_otel_roundtrip():
    s1 = make_span("s1", None, "t1")
    s2 = make_span("s2", "s1", "t1")
    trace = Trace(trace_id="t1", task_id="task1", spans=[s1, s2], metadata={"domain": "travel"})
    otel = trace.to_otel()
    rebuilt = Trace.from_otel_spans(otel, task_id="task1", metadata={"domain": "travel"})
    assert rebuilt.trace_id == trace.trace_id
    assert [s.span_id for s in rebuilt.spans] == ["s1", "s2"]


def test_otel_roundtrip_preserves_parent_child():
    s1 = make_span("s1", None, "t1")
    s2 = make_span("s2", "s1", "t1")
    trace = Trace(trace_id="t1", task_id="task1", spans=[s1, s2], metadata={})
    rebuilt = Trace.from_otel_spans(trace.to_otel(), task_id="task1", metadata={})
    assert rebuilt.spans[1].parent_span_id == "s1"


def test_span_attributes_accept_arbitrary_otel_keys():
    s = make_span("s1", None, "t1")
    s.attributes["gen_ai.request.model"] = "gpt-4o-mini"
    assert s.attributes["gen_ai.request.model"] == "gpt-4o-mini"


def test_from_otel_spans_rejects_mismatched_trace_id():
    s1 = make_span("s1", None, "t1")
    s_other = make_span("s2", None, "t-different")
    with pytest.raises(ValueError):
        Trace.from_otel_spans([s1.to_dict(), s_other.to_dict()], task_id="task1", metadata={})
