"""UnslothRLHFOptimizer: the real reference ScopeOptimizer for Scope.MODEL_WEIGHTS /
FineTuneTechnique.RLHF, alongside UnslothLoRAOptimizer's SFT/LoRA path. Same CloudGPUJobRunner
contract (real spend only via AWSGPUJobRunner, wired exclusively by scripts/provision_gpu.py after
its own type/cost/duration sign-off) — GRPO training runs multiple generations per step, so its
cost/duration is NOT assumed to match the SFT/LoRA path; re-confirm before provisioning for RLHF
specifically, the same discipline applied before the LoRA path was ever run for real.

Reward signal, stated honestly: this is a real, working TRL GRPOTrainer run, but its reward
function is not a live task-correctness oracle (Harness's own Verifier requires actually running
the Agent against a Benchmark, which the remote training instance doesn't have). Instead, the
reward rewards the policy for producing completions matching its own PAST completions that a real
Verifier already scored highly at capture time (reward >= REWARD_FILTER_THRESHOLD in
TrainingCorpus) — a real, defensible rejection-sampling-style self-improvement signal, not a
fabricated one. It is not the same claim as training against a live oracle, and is not presented
as one.
"""

from __future__ import annotations

from agentgym.core.artifact import Artifact, ArtifactShape, FineTuneTechnique
from agentgym.core.corpus import TrainingCorpus
from agentgym.core.scope import MODEL_WEIGHTS
from agentgym.optimizers.unsloth_lora import CloudGPUJobRunner

REWARD_FILTER_THRESHOLD = 0.99


def _build_prompt_reward_rows(corpus: TrainingCorpus) -> list[dict]:
    rows = []
    for row in corpus.sft_rows():
        if row["reward"] < REWARD_FILTER_THRESHOLD:
            continue
        for step in row["steps"]:
            if step["kind"] == "llm" and step["state"] and step["observation"]:
                rows.append({"prompt": str(step["state"]), "expected_answer": str(step["observation"])})
    return rows


class UnslothRLHFOptimizer:
    scope = MODEL_WEIGHTS
    technique = FineTuneTechnique.RLHF
    depends_on: list = []

    def __init__(self, job_runner: CloudGPUJobRunner, lora_rank: int = 16):
        self.job_runner = job_runner
        self.lora_rank = lora_rank

    def estimate(self, artifact: Artifact, corpus: TrainingCorpus, benchmark, variant) -> Artifact:
        rows = _build_prompt_reward_rows(corpus)
        if not rows:
            raise ValueError(
                "cannot estimate MODEL_WEIGHTS (RLHF) from a corpus with no verified-good "
                f"(reward >= {REWARD_FILTER_THRESHOLD}) llm-kind rows"
            )

        base_model = artifact.value.get("base_model")
        if not base_model:
            raise ValueError("starting artifact.value must declare 'base_model' to fine-tune")

        run_id = self.job_runner.launch(
            script_path="agentgym/optimizers/_unsloth_rlhf_train.py",
            data=rows, base_model=base_model, lora_rank=self.lora_rank,
        )
        result = self.job_runner.wait_and_fetch(run_id)

        return Artifact(
            scope=MODEL_WEIGHTS,
            shape=ArtifactShape.WEIGHTS,
            value={"adapter_path": result["adapter_path"], "base_model": base_model},
            optimizer_binding="trl.GRPOTrainer",
            technique=FineTuneTechnique.RLHF,
            source="estimated",
            provenance={"run_id": run_id, "n_rows": len(rows), **result.get("metrics", {})},
        )
