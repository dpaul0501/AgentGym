"""RetrieverConfigOptimizer against the real, installed headroom-ai BM25Scorer — real ranking,
real Optuna study, no fakes."""

import pytest

from agentgym.core.artifact import Artifact, ArtifactShape
from agentgym.core.protocols import Task
from agentgym.core.scope import RETRIEVAL
from agentgym.optimizers.retriever_search import RetrieverConfigOptimizer

pytestmark = pytest.mark.integration

DOCS = [
    {"id": "d1", "text": "A guide to python programming for absolute beginners"},
    {"id": "d2", "text": "How to bake a chocolate cake at home"},
    {"id": "d3", "text": "Advanced python tutorials covering async and typing"},
    {"id": "d4", "text": "A history of the roman empire"},
    {"id": "d5", "text": "Python beginner exercises and practice problems"},
]


class _RetrievalBenchmark:
    def cases(self):
        return [
            Task(task_id="t1", instruction="python tutorial for beginners",
                 metadata={"documents": DOCS, "relevant_doc_ids": ["d1", "d5"]}),
            Task(task_id="t2", instruction="advanced python async programming",
                 metadata={"documents": DOCS, "relevant_doc_ids": ["d3"]}),
        ]


def _starting_artifact():
    return Artifact(
        scope=RETRIEVAL, shape=ArtifactShape.CONFIG, value={"search_space": {"top_k": [1, 5]}},
        optimizer_binding=None, technique=None, source="manual",
    )


def test_estimate_finds_a_top_k_that_recalls_relevant_docs():
    optimizer = RetrieverConfigOptimizer(n_trials=15, seed=0)
    result = optimizer.estimate(_starting_artifact(), corpus=None, benchmark=_RetrievalBenchmark(), variant=None)
    assert 1 <= result.value["top_k"] <= 5


def test_estimate_returns_config_shaped_estimated_artifact():
    optimizer = RetrieverConfigOptimizer(n_trials=10, seed=0)
    result = optimizer.estimate(_starting_artifact(), corpus=None, benchmark=_RetrievalBenchmark(), variant=None)
    assert result.scope == RETRIEVAL
    assert result.shape == ArtifactShape.CONFIG
    assert result.source == "estimated"
    assert result.optimizer_binding == "headroom.BM25Scorer+optuna.TPESampler"
