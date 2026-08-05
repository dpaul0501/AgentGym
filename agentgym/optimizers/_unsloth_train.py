"""Runs ON the rented GPU instance, not on the machine driving AgentGym. Loads a base model via
Unsloth's real FastLanguageModel, applies a real LoRA adapter, trains via TRL's real SFTTrainer on
chat-messages rows, and saves the adapter to disk. Reads /tmp/agentgym_job/{data.json,config.json},
writes the trained adapter to /tmp/agentgym_job/adapter_out/ plus metrics.json — the AWSGPUJobRunner
uploads both back to S3 and signals completion. Not imported anywhere in the agentgym package
itself; it is deployed to the instance and invoked there as a standalone script, since it depends
on unsloth/trl/peft, which are only installed on the training instance.
"""

from __future__ import annotations

import json
from pathlib import Path

JOB_DIR = Path("/tmp/agentgym_job")


def main() -> None:
    config = json.loads((JOB_DIR / "config.json").read_text())
    rows = json.loads((JOB_DIR / "data.json").read_text())  # list[list[chat-message dict]]

    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["base_model"], max_seq_length=2048, load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=config["lora_rank"], target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=config["lora_rank"] * 2, lora_dropout=0.0, bias="none",
    )

    dataset = [
        {"text": tokenizer.apply_chat_template(messages, tokenize=False)} for messages in rows
    ]

    out_dir = JOB_DIR / "adapter_out"
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=dataset,
        args=SFTConfig(
            output_dir=str(out_dir), per_device_train_batch_size=2, gradient_accumulation_steps=4,
            num_train_epochs=config.get("num_train_epochs", 3), learning_rate=2e-4,
            logging_steps=1, report_to="none",
        ),
    )
    result = trainer.train()

    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    (JOB_DIR / "metrics.json").write_text(
        json.dumps({"train_loss": result.training_loss})
    )


if __name__ == "__main__":
    main()
