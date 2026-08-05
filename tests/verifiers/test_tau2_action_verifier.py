from agentgym.benchmarks.tau2_bench import Tau2Benchmark
from agentgym.core.trace import Span, Trace
from agentgym.verifiers.tau2_action_verifier import Tau2ActionVerifier


def _tool_span(name, arguments):
    return Span(
        span_id="s1", parent_span_id=None, trace_id="t1", name=name, kind="tool",
        input=arguments, output={}, start_time=0.0, end_time=0.1, attributes={},
    )


def test_full_match_scores_1():
    benchmark = Tau2Benchmark(domain="mock")
    task = {t.task_id: t for t in benchmark.cases()}["create_task_1"]
    trace = Trace(
        trace_id="t1", task_id="create_task_1",
        spans=[_tool_span("create_task", {"user_id": "user_1", "title": "Important Meeting"})],
        metadata={},
    )
    scores = Tau2ActionVerifier().score(trace, task)
    assert scores[0].value == 1.0


def test_no_matching_action_scores_0():
    benchmark = Tau2Benchmark(domain="mock")
    task = {t.task_id: t for t in benchmark.cases()}["create_task_1"]
    trace = Trace(
        trace_id="t1", task_id="create_task_1",
        spans=[_tool_span("delete_task", {"task_id": "999"})],
        metadata={},
    )
    scores = Tau2ActionVerifier().score(trace, task)
    assert scores[0].value == 0.0


def test_wrong_arguments_does_not_count_as_match():
    benchmark = Tau2Benchmark(domain="mock")
    task = {t.task_id: t for t in benchmark.cases()}["create_task_1"]
    trace = Trace(
        trace_id="t1", task_id="create_task_1",
        spans=[_tool_span("create_task", {"user_id": "user_1", "title": "Wrong Title"})],
        metadata={},
    )
    scores = Tau2ActionVerifier().score(trace, task)
    assert scores[0].value == 0.0
