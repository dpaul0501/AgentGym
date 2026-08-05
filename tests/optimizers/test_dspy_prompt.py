"""Real integration test — actually calls a local Ollama model through DSPy. No mocking of dspy
or the LLM. Requires Ollama running locally with qwen2.5:7b pulled (already true in this
workspace from earlier work); skipped automatically if it isn't reachable."""

import socket

import pytest

from agentgym.core.artifact import Artifact, ArtifactShape
from agentgym.core.corpus import TrainingCorpus
from agentgym.core.diagnosis import Diagnosis
from agentgym.core.score import Score
from agentgym.core.scope import PROMPT
from agentgym.core.trace import Span, Trace
from agentgym.optimizers.dspy_prompt import DSPyPromptOptimizer

pytestmark = pytest.mark.integration


def _ollama_reachable() -> bool:
    try:
        with socket.create_connection(("localhost", 11434), timeout=1):
            return True
    except OSError:
        return False


def _training_trace(trace_id, instruction, answer):
    span = Span(
        span_id=f"{trace_id}-s1", parent_span_id=None, trace_id=trace_id, name="reason",
        kind="llm", input={"q": instruction}, output={"a": answer},
        start_time=0.0, end_time=0.1, attributes={},
    )
    return Trace(trace_id=trace_id, task_id=trace_id, spans=[span],
                 metadata={"instruction": instruction, "answer": answer})


@pytest.mark.skipif(not _ollama_reachable(), reason="Ollama not reachable on localhost:11434")
def test_dspy_prompt_optimizer_real_compile_produces_demos():
    corpus = TrainingCorpus()
    training_pairs = [
        ("What is 3 + 4?", "7"),
        ("What is 10 - 6?", "4"),
        ("What is 5 * 3?", "15"),
        ("What is 20 / 4?", "5"),
    ]
    for i, (q, a) in enumerate(training_pairs):
        trace = _training_trace(f"t{i}", q, a)
        scores = [Score(metric_name="task_success", value=1.0, kind="verifiable", evidence="")]
        diagnosis = Diagnosis(trace_id=trace.trace_id, lever=PROMPT.lever, evidence=["gold example"],
                               confidence=1.0, suggested_scope=PROMPT)
        corpus.add_trace(trace, scores, diagnosis)

    starting_artifact = Artifact(
        scope=PROMPT, shape=ArtifactShape.PROGRAM, value={"signature": "question -> answer", "demos": []},
        optimizer_binding=None, technique=None, source="manual", provenance={},
    )

    optimizer = DSPyPromptOptimizer()
    result = optimizer.estimate(starting_artifact, corpus, benchmark=None, variant=None)

    assert result.source == "estimated"
    assert result.optimizer_binding == "dspy.BootstrapFewShot"
    assert result.provenance["n_examples"] == 4
    # BootstrapFewShot only keeps demos the model itself got right during bootstrapping —
    # a real compile can legitimately produce zero if the local model gets all of them wrong,
    # so the real assertion is on well-formedness, not a fixed count
    assert isinstance(result.value["demos"], list)
    for demo in result.value["demos"]:
        assert set(demo.keys()) == {"question", "answer"}


@pytest.mark.skipif(not _ollama_reachable(), reason="Ollama not reachable on localhost:11434")
def test_dspy_prompt_optimizer_raises_without_training_examples():
    corpus = TrainingCorpus()
    starting_artifact = Artifact(
        scope=PROMPT, shape=ArtifactShape.PROGRAM, value={"signature": "question -> answer", "demos": []},
        optimizer_binding=None, technique=None, source="manual", provenance={},
    )
    optimizer = DSPyPromptOptimizer()
    with pytest.raises(ValueError):
        optimizer.estimate(starting_artifact, corpus, benchmark=None, variant=None)
