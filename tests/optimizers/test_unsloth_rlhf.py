"""UnslothRLHFOptimizer against a fake CloudGPUJobRunner — this file must never launch real
hardware. The real runner (AWSGPUJobRunner) is exercised only manually via scripts/provision_gpu.py."""

import pytest

from agentgym.core.artifact import Artifact, ArtifactShape, FineTuneTechnique
from agentgym.core.corpus import TrainingCorpus
from agentgym.core.diagnosis import Diagnosis
from agentgym.core.scope import MODEL_WEIGHTS
from agentgym.core.score import Score
from agentgym.core.trace import Span, Trace
from agentgym.optimizers.unsloth_rlhf import UnslothRLHFOptimizer


class _FakeJobRunner:
    def __init__(self):
        self.launched_with = None

    def launch(self, script_path, data, base_model, lora_rank):
        self.launched_with = {
            "script_path": script_path, "data": data, "base_model": base_model, "lora_rank": lora_rank,
        }
        return "fake-rlhf-run-1"

    def wait_and_fetch(self, run_id):
        return {"adapter_path": f"s3://fake-bucket/{run_id}/adapter", "metrics": {"train_loss": 0.1}}


def _starting_artifact() -> Artifact:
    return Artifact(
        scope=MODEL_WEIGHTS, shape=ArtifactShape.WEIGHTS, value={"base_model": "Qwen/Qwen2.5-3B-Instruct"},
        optimizer_binding=None, technique=None, source="manual",
    )


def _add_trace(corpus, task_id, reward, output="Paris is the capital of France."):
    trace = Trace(
        trace_id=f"t-{task_id}", task_id=task_id,
        spans=[Span(span_id="s1", parent_span_id=None, trace_id=f"t-{task_id}", name="respond",
                    kind="llm", input="What is the capital of France?", output=output,
                    start_time=0.0, end_time=0.1, attributes={})],
        metadata={},
    )
    corpus.add_trace(
        trace, [Score(metric_name="task_success", value=reward, kind="verifiable", evidence="ok")],
        Diagnosis(trace_id=f"t-{task_id}", lever=MODEL_WEIGHTS.lever, evidence=["ok"],
                   confidence=0.9, suggested_scope=MODEL_WEIGHTS),
    )


def test_fake_runner_satisfies_cloud_gpu_job_runner_protocol():
    from agentgym.optimizers.unsloth_lora import CloudGPUJobRunner
    assert isinstance(_FakeJobRunner(), CloudGPUJobRunner)


def test_estimate_only_uses_verified_good_rows_as_reward_targets():
    corpus = TrainingCorpus()
    _add_trace(corpus, "good", reward=1.0, output="Paris is the capital of France.")
    _add_trace(corpus, "bad", reward=0.0, output="I don't know.")
    runner = _FakeJobRunner()
    optimizer = UnslothRLHFOptimizer(job_runner=runner, lora_rank=8)

    optimizer.estimate(_starting_artifact(), corpus, benchmark=None, variant=None)

    rows = runner.launched_with["data"]
    assert len(rows) == 1
    assert rows[0]["expected_answer"] == "Paris is the capital of France."
    assert runner.launched_with["script_path"].endswith("_unsloth_rlhf_train.py")


def test_estimate_returns_rlhf_weights_artifact_with_provenance():
    corpus = TrainingCorpus()
    _add_trace(corpus, "good", reward=1.0)
    optimizer = UnslothRLHFOptimizer(job_runner=_FakeJobRunner())

    result = optimizer.estimate(_starting_artifact(), corpus, benchmark=None, variant=None)

    assert result.scope == MODEL_WEIGHTS
    assert result.shape == ArtifactShape.WEIGHTS
    assert result.technique == FineTuneTechnique.RLHF
    assert result.optimizer_binding == "trl.GRPOTrainer"
    assert result.provenance["run_id"] == "fake-rlhf-run-1"
    assert result.provenance["train_loss"] == 0.1


def test_estimate_raises_when_no_verified_good_rows_exist():
    corpus = TrainingCorpus()
    _add_trace(corpus, "bad", reward=0.2)
    optimizer = UnslothRLHFOptimizer(job_runner=_FakeJobRunner())

    with pytest.raises(ValueError):
        optimizer.estimate(_starting_artifact(), corpus, benchmark=None, variant=None)


def test_estimate_raises_without_base_model_declared():
    corpus = TrainingCorpus()
    _add_trace(corpus, "good", reward=1.0)
    optimizer = UnslothRLHFOptimizer(job_runner=_FakeJobRunner())
    artifact = Artifact(scope=MODEL_WEIGHTS, shape=ArtifactShape.WEIGHTS, value={},
                         optimizer_binding=None, technique=None, source="manual")

    with pytest.raises(ValueError):
        optimizer.estimate(artifact, corpus, benchmark=None, variant=None)
