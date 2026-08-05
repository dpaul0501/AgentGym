"""UnslothLoRAOptimizer against a fake CloudGPUJobRunner — this file must never launch real
hardware. The real runner (AWSGPUJobRunner) is exercised only manually via scripts/provision_gpu.py,
never from pytest."""

import pytest

from agentgym.core.artifact import Artifact, ArtifactShape, FineTuneTechnique
from agentgym.core.corpus import TrainingCorpus
from agentgym.core.diagnosis import Diagnosis
from agentgym.core.scope import MODEL_WEIGHTS
from agentgym.core.score import Score
from agentgym.core.trace import Span, Trace
from agentgym.optimizers.unsloth_lora import CloudGPUJobRunner, UnslothLoRAOptimizer


class _FakeJobRunner:
    def __init__(self):
        self.launched_with = None

    def launch(self, script_path, data, base_model, lora_rank):
        self.launched_with = {
            "script_path": script_path, "data": data, "base_model": base_model, "lora_rank": lora_rank,
        }
        return "fake-run-id-1"

    def wait_and_fetch(self, run_id):
        return {"adapter_path": f"s3://fake-bucket/{run_id}/adapter", "metrics": {"train_loss": 0.42}}


def _starting_artifact() -> Artifact:
    return Artifact(
        scope=MODEL_WEIGHTS, shape=ArtifactShape.WEIGHTS, value={"base_model": "Qwen/Qwen2.5-3B-Instruct"},
        optimizer_binding=None, technique=None, source="manual",
    )


def _corpus_with_one_trace() -> TrainingCorpus:
    corpus = TrainingCorpus()
    trace = Trace(
        trace_id="t1", task_id="task1",
        spans=[Span(span_id="s1", parent_span_id=None, trace_id="t1", name="respond", kind="llm",
                    input="hi", output="hello", start_time=0.0, end_time=0.1, attributes={})],
        metadata={},
    )
    corpus.add_trace(
        trace, [Score(metric_name="task_success", value=1.0, kind="verifiable", evidence="ok")],
        Diagnosis(trace_id="t1", lever=MODEL_WEIGHTS.lever, evidence=["ok"],
                   confidence=0.9, suggested_scope=MODEL_WEIGHTS),
    )
    return corpus


def test_fake_runner_satisfies_cloud_gpu_job_runner_protocol():
    assert isinstance(_FakeJobRunner(), CloudGPUJobRunner)


def test_estimate_launches_job_with_chat_message_rows_and_base_model():
    runner = _FakeJobRunner()
    optimizer = UnslothLoRAOptimizer(job_runner=runner, lora_rank=8)
    optimizer.estimate(_starting_artifact(), _corpus_with_one_trace(), benchmark=None, variant=None)

    assert runner.launched_with["base_model"] == "Qwen/Qwen2.5-3B-Instruct"
    assert runner.launched_with["lora_rank"] == 8
    assert runner.launched_with["data"][0][0]["role"] == "user"


def test_estimate_returns_weights_artifact_with_adapter_path_and_provenance():
    optimizer = UnslothLoRAOptimizer(job_runner=_FakeJobRunner())
    result = optimizer.estimate(_starting_artifact(), _corpus_with_one_trace(), benchmark=None, variant=None)

    assert result.scope == MODEL_WEIGHTS
    assert result.shape == ArtifactShape.WEIGHTS
    assert result.technique == FineTuneTechnique.LORA_ADAPTER
    assert result.optimizer_binding == "unsloth"
    assert result.value["adapter_path"] == "s3://fake-bucket/fake-run-id-1/adapter"
    assert result.provenance["run_id"] == "fake-run-id-1"
    assert result.provenance["train_loss"] == 0.42


def test_estimate_raises_on_empty_corpus():
    optimizer = UnslothLoRAOptimizer(job_runner=_FakeJobRunner())
    with pytest.raises(ValueError):
        optimizer.estimate(_starting_artifact(), TrainingCorpus(), benchmark=None, variant=None)


def test_estimate_raises_without_base_model_declared():
    optimizer = UnslothLoRAOptimizer(job_runner=_FakeJobRunner())
    artifact = Artifact(scope=MODEL_WEIGHTS, shape=ArtifactShape.WEIGHTS, value={},
                         optimizer_binding=None, technique=None, source="manual")
    with pytest.raises(ValueError):
        optimizer.estimate(artifact, _corpus_with_one_trace(), benchmark=None, variant=None)
