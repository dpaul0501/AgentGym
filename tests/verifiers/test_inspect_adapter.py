"""Real test against the actual inspect_ai library — exact()/includes() are genuine, unmodified
Inspect scorers, not mocked. No LLM/network call needed since these are deterministic text
scorers, so this runs in the fast suite.

Skipped when inspect_ai isn't installed: it's declared to conflict with the `examples` extra in
pyproject.toml (inspect-ai and langchain-aws pull incompatible datasets/aiobotocore version
chains, confirmed via a real `uv sync` resolution failure) — the two are never installed together.
"""

import pytest

pytest.importorskip("inspect_ai")

from inspect_ai.scorer import exact, includes  # noqa: E402

from agentgym.core.protocols import Task  # noqa: E402
from agentgym.core.trace import Span, Trace  # noqa: E402
from agentgym.verifiers.inspect_adapter import InspectVerifier  # noqa: E402


def _trace_with_answer(answer: str) -> Trace:
    span = Span(
        span_id="s1", parent_span_id=None, trace_id="t1", name="reason", kind="llm",
        input={"q": "what is 2+2?"}, output={"answer": answer},
        start_time=0.0, end_time=0.1, attributes={},
    )
    return Trace(trace_id="t1", task_id="task1", spans=[span], metadata={})


def test_inspect_exact_scorer_correct():
    verifier = InspectVerifier(scorer=exact())
    task = Task(task_id="task1", instruction="what is 2+2?", metadata={"target": "4"})
    scores = verifier.score(_trace_with_answer("4"), task)
    assert scores[0].value == 1.0
    assert scores[0].kind == "verifiable"


def test_inspect_exact_scorer_incorrect():
    verifier = InspectVerifier(scorer=exact())
    task = Task(task_id="task1", instruction="what is 2+2?", metadata={"target": "4"})
    scores = verifier.score(_trace_with_answer("5"), task)
    assert scores[0].value == 0.0


def test_inspect_includes_scorer_partial_match():
    verifier = InspectVerifier(scorer=includes(), metric_name="contains_target")
    task = Task(task_id="task1", instruction="name the capital of france",
                metadata={"target": "paris"})
    scores = verifier.score(_trace_with_answer("The capital of France is Paris."), task)
    assert scores[0].value == 1.0
    assert scores[0].metric_name == "contains_target"


def test_evidence_carries_real_inspect_explanation_or_value():
    verifier = InspectVerifier(scorer=exact())
    task = Task(task_id="task1", instruction="x", metadata={"target": "4"})
    scores = verifier.score(_trace_with_answer("4"), task)
    assert scores[0].evidence  # not empty — real scorer output, not a placeholder
