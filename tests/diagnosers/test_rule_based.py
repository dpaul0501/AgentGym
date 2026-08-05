from agentgym.core.scope import GRAPH, GUARDRAILS, INTERRUPT, MEMORY, MODEL_ROUTING, MODEL_WEIGHTS, PROMPT, RETRIEVAL, TOOLS
from agentgym.core.score import Score
from agentgym.core.trace import Span, Trace
from agentgym.diagnosers.rule_based import RuleBasedDiagnoser


def _span(name="s", kind="llm", input=None, output=None, attributes=None):
    return Span(span_id=name, parent_span_id=None, trace_id="t1", name=name, kind=kind,
                input=input or {}, output=output or {}, start_time=0.0, end_time=0.1,
                attributes=attributes or {})


def _trace(spans, metadata=None):
    return Trace(trace_id="t1", task_id="task1", spans=spans, metadata=metadata or {})


def _score(name="task_success", value=1.0, evidence="ok"):
    return [Score(metric_name=name, value=value, kind="verifiable", evidence=evidence)]


def test_empty_scores_is_insufficient_evidence():
    d = RuleBasedDiagnoser().diagnose(_trace([_span()]), [])
    assert d.lever is None and d.suggested_scope is None
    assert d.evidence


def test_high_reward_is_insufficient_evidence_no_failure():
    d = RuleBasedDiagnoser().diagnose(_trace([_span()]), _score(value=0.9))
    assert d.lever is None and d.suggested_scope is None


def test_guardrail_violation_takes_priority_even_over_high_reward():
    scores = _score(value=0.9) + [Score(metric_name="prompt_injection_free", value=0.0,
                                          kind="verifiable", evidence="violated")]
    d = RuleBasedDiagnoser().diagnose(_trace([_span()]), scores)
    assert d.suggested_scope == GUARDRAILS
    assert d.confidence == 0.95


def test_tool_error_span_diagnosed_as_tools():
    trace = _trace([_span(name="search", kind="tool", attributes={"error": "timeout"})])
    d = RuleBasedDiagnoser().diagnose(trace, _score(value=0.1))
    assert d.suggested_scope == TOOLS


def test_missing_escalation_diagnosed_as_interrupt():
    trace = _trace([_span(kind="llm")], metadata={"requires_human": True})
    d = RuleBasedDiagnoser().diagnose(trace, _score(value=0.1))
    assert d.suggested_scope == INTERRUPT


def test_repeated_action_loop_diagnosed_as_graph():
    trace = _trace([
        _span(name="retry", kind="tool"), _span(name="retry", kind="tool"), _span(name="retry", kind="tool"),
    ])
    d = RuleBasedDiagnoser().diagnose(trace, _score(value=0.1))
    assert d.suggested_scope == GRAPH


def test_empty_retrieval_diagnosed_as_retrieval():
    trace = _trace([_span(name="search_docs", kind="retrieval", output={})])
    d = RuleBasedDiagnoser().diagnose(trace, _score(value=0.1))
    assert d.suggested_scope == RETRIEVAL


def test_oversized_context_diagnosed_as_memory():
    big = "x" * 5000
    trace = _trace([_span(kind="llm", input={"text": big})])
    d = RuleBasedDiagnoser().diagnose(trace, _score(value=0.1))
    assert d.suggested_scope == MEMORY


def test_routing_mismatch_diagnosed_as_model_routing():
    trace = _trace(
        [_span(kind="llm", attributes={"routed_model_tier": "small"})],
        metadata={"expected_model_tier": "large"},
    )
    d = RuleBasedDiagnoser().diagnose(trace, _score(value=0.1))
    assert d.suggested_scope == MODEL_ROUTING


def test_partial_failure_with_llm_span_and_no_structural_signal_diagnosed_as_prompt():
    trace = _trace([_span(kind="llm")])
    d = RuleBasedDiagnoser().diagnose(trace, _score(value=0.5))
    assert d.suggested_scope == PROMPT


def test_severe_failure_with_no_structural_signal_diagnosed_as_model_weights():
    trace = _trace([_span(kind="llm")])
    d = RuleBasedDiagnoser().diagnose(trace, _score(value=0.05))
    assert d.suggested_scope == MODEL_WEIGHTS
