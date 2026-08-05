"""UnslothLoRAOptimizer: the real reference ScopeOptimizer for Scope.MODEL_WEIGHTS /
FineTuneTechnique.LORA_ADAPTER. Training itself always happens on rented hardware, driven by a
CloudGPUJobRunner — this class never touches a GPU or the cloud directly, and never launches
anything on import or in a test. The only real implementation of CloudGPUJobRunner
(AWSGPUJobRunner, agentgym/providers/aws_gpu_runner.py) is wired in exclusively by
scripts/provision_gpu.py, which is the one place actual spend happens.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agentgym.core.artifact import Artifact, ArtifactShape, FineTuneTechnique
from agentgym.core.corpus import TrainingCorpus
from agentgym.core.scope import MODEL_WEIGHTS


@runtime_checkable
class CloudGPUJobRunner(Protocol):
    def launch(self, script_path: str, data: list[dict], base_model: str, lora_rank: int) -> str: ...
    def wait_and_fetch(self, run_id: str) -> dict: ...


class UnslothLoRAOptimizer:
    scope = MODEL_WEIGHTS
    technique = FineTuneTechnique.LORA_ADAPTER
    depends_on: list = []

    def __init__(self, job_runner: CloudGPUJobRunner, lora_rank: int = 16):
        self.job_runner = job_runner
        self.lora_rank = lora_rank

    def estimate(self, artifact: Artifact, corpus: TrainingCorpus, benchmark, variant) -> Artifact:
        trajectory_rows = corpus.sft_rows()
        if not trajectory_rows:
            raise ValueError("cannot estimate MODEL_WEIGHTS from an empty TrainingCorpus")
        rows = [corpus.to_chat_messages(row) for row in trajectory_rows]

        base_model = artifact.value.get("base_model")
        if not base_model:
            raise ValueError("starting artifact.value must declare 'base_model' to fine-tune")

        run_id = self.job_runner.launch(
            script_path="agentgym/optimizers/_unsloth_train.py",
            data=rows,
            base_model=base_model,
            lora_rank=self.lora_rank,
        )
        result = self.job_runner.wait_and_fetch(run_id)

        return Artifact(
            scope=MODEL_WEIGHTS,
            shape=ArtifactShape.WEIGHTS,
            value={"adapter_path": result["adapter_path"], "base_model": base_model},
            optimizer_binding="unsloth",
            technique=FineTuneTechnique.LORA_ADAPTER,
            source="estimated",
            provenance={"run_id": run_id, "n_rows": len(rows), **result.get("metrics", {})},
        )
