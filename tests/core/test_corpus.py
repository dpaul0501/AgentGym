from agentgym.core.corpus import TrainingCorpus
from agentgym.core.diagnosis import Diagnosis
from agentgym.core.score import Score
from agentgym.core.scope import GRAPH, PROMPT, Lever
from agentgym.core.trace import Span, Trace


def make_trace(trace_id, task_id, tool=False):
    spans = [
        Span(
            span_id=f"{trace_id}-s1", parent_span_id=None, trace_id=trace_id,
            name="search_flights" if tool else "reason",
            kind="tool" if tool else "llm",
            input={"q": "GVA-LIS"}, output={"n": 3},
            start_time=0.0, end_time=0.2, attributes={},
        )
    ]
    return Trace(trace_id=trace_id, task_id=task_id, spans=spans, metadata={})


def make_diagnosis(trace_id, scope):
    return Diagnosis(
        trace_id=trace_id, lever=scope.lever, evidence=["evidence"],
        confidence=0.9, suggested_scope=scope,
    )


def make_scores(value):
    return [Score(metric_name="task_success", value=value, kind="verifiable", evidence="")]


def test_add_trace_grows_corpus():
    corpus = TrainingCorpus()
    corpus.add_trace(make_trace("t1", "task1"), make_scores(1.0), make_diagnosis("t1", PROMPT))
    assert len(corpus) == 1


def test_empty_corpus_returns_empty_not_none():
    corpus = TrainingCorpus()
    assert corpus.sft_rows() == []
    assert corpus.dpo_pairs() == []
    assert corpus.artifact_candidates(PROMPT) == []


def test_artifact_candidates_filters_by_scope():
    corpus = TrainingCorpus()
    corpus.add_trace(make_trace("t1", "task1"), make_scores(1.0), make_diagnosis("t1", PROMPT))
    corpus.add_trace(make_trace("t2", "task2"), make_scores(0.3), make_diagnosis("t2", GRAPH))
    prompt_candidates = corpus.artifact_candidates(PROMPT)
    assert len(prompt_candidates) == 1
    assert prompt_candidates[0]["trace"].trace_id == "t1"


def test_sft_rows_native_shape_is_trajectory():
    corpus = TrainingCorpus()
    corpus.add_trace(make_trace("t1", "task1", tool=True), make_scores(0.8), make_diagnosis("t1", PROMPT))
    rows = corpus.sft_rows()
    assert len(rows) == 1
    row = rows[0]
    assert "steps" in row and "reward" in row
    assert "role" not in row and "content" not in row
    step = row["steps"][0]
    assert set(step.keys()) >= {"state", "action", "observation", "kind"}


def test_to_chat_messages_converts_trajectory():
    corpus = TrainingCorpus()
    corpus.add_trace(make_trace("t1", "task1"), make_scores(0.8), make_diagnosis("t1", PROMPT))
    rows = corpus.sft_rows()
    messages = corpus.to_chat_messages(rows[0])
    assert all("role" in m and "content" in m for m in messages)


def test_to_chat_messages_preserves_tool_calls():
    corpus = TrainingCorpus()
    corpus.add_trace(make_trace("t1", "task1", tool=True), make_scores(0.8), make_diagnosis("t1", PROMPT))
    rows = corpus.sft_rows()
    messages = corpus.to_chat_messages(rows[0])
    assistant_with_tool_call = [m for m in messages if m["role"] == "assistant" and m.get("tool_calls")]
    tool_result_messages = [m for m in messages if m["role"] == "tool"]
    assert assistant_with_tool_call
    assert tool_result_messages


def test_dpo_pairs_requires_both_chosen_and_rejected():
    corpus = TrainingCorpus()
    # only one trajectory for task1 -> no pair possible
    corpus.add_trace(make_trace("t1", "task1"), make_scores(0.9), make_diagnosis("t1", PROMPT))
    assert corpus.dpo_pairs() == []
    # add a second, worse trajectory for the same task -> now a pair exists
    corpus.add_trace(make_trace("t2", "task1"), make_scores(0.2), make_diagnosis("t2", PROMPT))
    pairs = corpus.dpo_pairs()
    assert len(pairs) == 1
    assert pairs[0]["chosen"]["reward"] == 0.9
    assert pairs[0]["rejected"]["reward"] == 0.2
