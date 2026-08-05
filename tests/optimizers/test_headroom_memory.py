"""HeadroomMemoryOptimizer against the real, installed headroom-ai package — real compress()
calls, real Optuna study, no fakes standing in for either."""

import json

import pytest

from agentgym.core.artifact import Artifact, ArtifactShape
from agentgym.core.corpus import TrainingCorpus
from agentgym.core.diagnosis import Diagnosis
from agentgym.core.scope import MEMORY
from agentgym.core.score import Score
from agentgym.core.trace import Span, Trace
from agentgym.optimizers.headroom_memory import HeadroomMemoryOptimizer

pytestmark = pytest.mark.integration


def _corpus_with_large_tool_output() -> TrainingCorpus:
    corpus = TrainingCorpus()
    big_results = {
        "results": [
            {"title": f"Result {i}", "snippet": f"Description {i}" * 5, "score": 100 - i}
            for i in range(300)
        ]
    }
    trace = Trace(
        trace_id="t1", task_id="task1",
        spans=[
            Span(span_id="s1", parent_span_id=None, trace_id="t1", name="search", kind="tool",
                 input={"q": "python"}, output=json.dumps(big_results), start_time=0.0, end_time=0.1,
                 attributes={}),
        ],
        metadata={},
    )
    corpus.add_trace(
        trace, [Score(metric_name="task_success", value=1.0, kind="verifiable", evidence="ok")],
        Diagnosis(trace_id="t1", lever=MEMORY.lever, evidence=["large tool output"],
                   confidence=0.9, suggested_scope=MEMORY),
    )
    return corpus


def _starting_artifact() -> Artifact:
    return Artifact(scope=MEMORY, shape=ArtifactShape.CONFIG, value={}, optimizer_binding=None,
                     technique=None, source="manual")


def test_estimate_returns_config_artifact_with_real_measured_savings():
    optimizer = HeadroomMemoryOptimizer(n_trials=5, seed=0)
    result = optimizer.estimate(_starting_artifact(), _corpus_with_large_tool_output(), benchmark=None, variant=None)

    assert result.scope == MEMORY
    assert result.shape == ArtifactShape.CONFIG
    assert result.source == "estimated"
    assert result.optimizer_binding == "headroom.compress"
    assert result.provenance["best_tokens_saved"] > 0
    assert "protect_recent" in result.value
    assert "min_tokens_to_compress" in result.value


def test_estimate_raises_on_empty_corpus():
    optimizer = HeadroomMemoryOptimizer(n_trials=2)
    with pytest.raises(ValueError):
        optimizer.estimate(_starting_artifact(), TrainingCorpus(), benchmark=None, variant=None)
