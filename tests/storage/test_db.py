import threading

import pytest

from agentgym.core.artifact import Artifact, ArtifactShape
from agentgym.core.diagnosis import Diagnosis
from agentgym.core.scope import GRAPH, PROMPT, Lever
from agentgym.core.score import Score
from agentgym.core.trace import Span, Trace
from agentgym.core.variant import AgentVariant
from agentgym.storage.db import AgentGymStore


@pytest.fixture()
def store(tmp_path):
    return AgentGymStore(path=str(tmp_path / "agentgym.db"))


def _variant():
    artifact = Artifact(
        scope=PROMPT, shape=ArtifactShape.PARAMETER, value="be terse",
        optimizer_binding=None, technique=None, source="manual", provenance={},
    )
    return AgentVariant(artifacts={PROMPT: artifact})


def _trace(trace_id="t1"):
    span = Span(
        span_id="s1", parent_span_id=None, trace_id=trace_id, name="reason", kind="llm",
        input={"q": "x"}, output={"a": "y"}, start_time=0.0, end_time=0.1, attributes={},
    )
    return Trace(trace_id=trace_id, task_id="task1", spans=[span], metadata={})


def _diagnosis(trace_id="t1", scope=PROMPT):
    return Diagnosis(trace_id=trace_id, lever=scope.lever, evidence=["e"], confidence=0.8,
                      suggested_scope=scope)


def test_save_and_load_variant_roundtrip(store):
    variant = _variant()
    store.save_variant(variant)
    loaded = store.latest_variant()
    assert loaded.artifacts[PROMPT].value == "be terse"


def test_latest_variant_none_when_store_empty(store):
    assert store.latest_variant() is None


def test_latest_variant_returns_most_recent(store):
    store.save_variant(_variant())
    second = AgentVariant(artifacts={
        PROMPT: Artifact(scope=PROMPT, shape=ArtifactShape.PARAMETER, value="second",
                          optimizer_binding=None, technique=None, source="manual", provenance={})
    })
    store.save_variant(second)
    assert store.latest_variant().artifacts[PROMPT].value == "second"


def test_traces_for_scope_filters_correctly(store):
    store.save_trace(_trace("t1"), [Score(metric_name="x", value=1.0, kind="verifiable", evidence="")],
                      _diagnosis("t1", PROMPT))
    store.save_trace(_trace("t2"), [Score(metric_name="x", value=0.5, kind="verifiable", evidence="")],
                      _diagnosis("t2", GRAPH))
    prompt_traces = store.traces_for_scope(PROMPT)
    assert [t.trace_id for t in prompt_traces] == ["t1"]


def test_concurrent_writes_no_corruption(store):
    n_threads, n_per_thread = 8, 25

    def writer(thread_id):
        for i in range(n_per_thread):
            store.save_trace(
                _trace(f"t-{thread_id}-{i}"),
                [Score(metric_name="x", value=1.0, kind="verifiable", evidence="")],
                _diagnosis(f"t-{thread_id}-{i}", PROMPT),
            )

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(store.traces_for_scope(PROMPT)) == n_threads * n_per_thread
