"""CavemanMemoryOptimizer against a real local clone of wilpel/caveman-compression (real spaCy-
backed compress_text) — skipped if the clone isn't present locally, since it's not pip-installable
and this repo deliberately doesn't vendor it."""

import os

import pytest

from agentgym.core.artifact import Artifact, ArtifactShape
from agentgym.core.corpus import TrainingCorpus
from agentgym.core.diagnosis import Diagnosis
from agentgym.core.scope import MEMORY
from agentgym.core.score import Score
from agentgym.core.trace import Span, Trace
from agentgym.optimizers.caveman_memory import CavemanMemoryOptimizer

CAVEMAN_PATH = os.environ.get(
    "CAVEMAN_COMPRESSION_PATH",
    "/private/tmp/claude-501/-Users-debjyotipaul-projects-RL-agents/"
    "21db2fce-1878-49ff-bbe3-556ba86957b5/scratchpad/caveman-compression",
)
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.path.exists(os.path.join(CAVEMAN_PATH, "caveman_compress_nlp.py")),
        reason="wilpel/caveman-compression not cloned locally; not pip-installable, not vendored",
    ),
]


def _corpus_with_wordy_text() -> TrainingCorpus:
    corpus = TrainingCorpus()
    verbose_text = (
        "The quick brown fox jumps over the lazy dog. It was a very sunny afternoon and the "
        "fox was extremely happy about the pleasant weather that had finally arrived after a "
        "long and rather tedious winter season."
    )
    trace = Trace(
        trace_id="t1", task_id="task1",
        spans=[
            Span(span_id="s1", parent_span_id=None, trace_id="t1", name="respond", kind="llm",
                 input=verbose_text, output=verbose_text, start_time=0.0, end_time=0.1, attributes={}),
        ],
        metadata={},
    )
    corpus.add_trace(
        trace, [Score(metric_name="task_success", value=1.0, kind="verifiable", evidence="ok")],
        Diagnosis(trace_id="t1", lever=MEMORY.lever, evidence=["verbose content"],
                   confidence=0.9, suggested_scope=MEMORY),
    )
    return corpus


def _starting_artifact() -> Artifact:
    return Artifact(scope=MEMORY, shape=ArtifactShape.CONFIG, value={}, optimizer_binding=None,
                     technique=None, source="manual")


def test_estimate_returns_config_artifact_with_real_measured_compression():
    optimizer = CavemanMemoryOptimizer(caveman_path=CAVEMAN_PATH)
    result = optimizer.estimate(_starting_artifact(), _corpus_with_wordy_text(), benchmark=None, variant=None)

    assert result.scope == MEMORY
    assert result.shape == ArtifactShape.CONFIG
    assert result.optimizer_binding == "caveman_compress_nlp.compress_text"
    assert result.provenance["tokens_saved"] > 0
    assert result.provenance["tokens_after"] < result.provenance["tokens_before"]


def test_missing_local_clone_raises_clear_error():
    with pytest.raises(FileNotFoundError):
        CavemanMemoryOptimizer(caveman_path="/nonexistent/path")


def test_estimate_raises_on_empty_corpus():
    optimizer = CavemanMemoryOptimizer(caveman_path=CAVEMAN_PATH)
    with pytest.raises(ValueError):
        optimizer.estimate(_starting_artifact(), TrainingCorpus(), benchmark=None, variant=None)
