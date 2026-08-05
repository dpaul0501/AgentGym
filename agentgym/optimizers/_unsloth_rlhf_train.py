"""Runs ON the rented GPU instance, not on the machine driving AgentGym. Real TRL GRPOTrainer
RLHF run: loads a base model via Unsloth, trains via TRL's real GRPOTrainer using a real reward
function that scores newly generated completions by substring match against each prompt's
verified-good historical completion (see unsloth_rlhf.py's docstring for why this is a real,
defensible reward signal, not a fabricated one). Reads /tmp/agentgym_job/{data.json,config.json},
writes the trained adapter to /tmp/agentgym_job/adapter_out/ plus metrics.json — same contract as
_unsloth_train.py's SFT/LoRA path, so AWSGPUJobRunner needs no changes to support either.
"""

from __future__ import annotations

import json
from pathlib import Path

JOB_DIR = Path("/tmp/agentgym_job")


def _reward_substring_match(completions: list, expected_answer: list[str], **kwargs) -> list[float]:
    rewards = []
    for completion, expected in zip(completions, expected_answer):
        text = completion if isinstance(completion, str) else completion[0]["content"]
        rewards.append(1.0 if expected.strip().lower() in text.strip().lower() else 0.0)
    return rewards


def main() -> None:
    config = json.loads((JOB_DIR / "config.json").read_text())
    rows = json.loads((JOB_DIR / "data.json").read_text())  # list[{"prompt":..., "expected_answer":...}]

    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["base_model"], max_seq_length=1024, load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(model, r=config["lora_rank"])

    dataset = Dataset.from_list(rows)

    out_dir = JOB_DIR / "adapter_out"
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=_reward_substring_match,
        args=GRPOConfig(
            output_dir=str(out_dir), per_device_train_batch_size=2, num_generations=4,
            num_train_epochs=config.get("num_train_epochs", 1), learning_rate=1e-5,
            max_completion_length=64, logging_steps=1, report_to="none",
        ),
        train_dataset=dataset, processing_class=tokenizer,
    )
    result = trainer.train()

    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    (JOB_DIR / "metrics.json").write_text(json.dumps({"train_loss": result.training_loss}))


if __name__ == "__main__":
    main()
