#!/usr/bin/env python3
"""The ONE place a cloud GPU instance gets created for UnslothLoRAOptimizer. Never invoked
automatically by Harness.cycle() or any recipe — run this manually, and only after the type/
cost/duration checkpoint has been confirmed with whoever is paying for it.

Usage:
    uv run scripts/provision_gpu.py --corpus data/corpus.json --base-model Qwen/Qwen2.5-3B-Instruct --yes

Without --yes, this prints the confirmation banner and the launch plan, then exits without
touching AWS — a dry run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

INSTANCE_TYPE = "g6.xlarge"
REGION = "us-east-1"
ESTIMATED_RATE_PER_HOUR = 0.805
ESTIMATED_DURATION_HOURS = 1.0
BUCKET = "agentgym-gpu-runs"


def confirmation_banner() -> str:
    est_cost = ESTIMATED_RATE_PER_HOUR * ESTIMATED_DURATION_HOURS
    return (
        f"Cloud GPU provisioning checkpoint:\n"
        f"  Instance type : {INSTANCE_TYPE} (1x NVIDIA L4, 24GB VRAM)\n"
        f"  Region        : {REGION}\n"
        f"  Rate          : ~${ESTIMATED_RATE_PER_HOUR:.3f}/hr on-demand\n"
        f"  Est. duration : ~{ESTIMATED_DURATION_HOURS:.1f}hr wall-clock\n"
        f"  Est. cost     : ~${est_cost:.2f}\n"
        f"  S3 bucket     : {BUCKET} (created if it doesn't exist)\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, help="path to a JSON file of chat-message rows")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--yes", action="store_true", help="actually launch (real spend)")
    args = parser.parse_args()

    print(confirmation_banner())
    if not args.yes:
        print("Dry run — pass --yes to actually launch and start real billing.")
        return

    from agentgym.core.artifact import Artifact, ArtifactShape
    from agentgym.core.scope import MODEL_WEIGHTS
    from agentgym.optimizers.unsloth_lora import UnslothLoRAOptimizer
    from agentgym.providers.aws_gpu_runner import AWSGPUJobRunner

    rows = json.loads(Path(args.corpus).read_text())

    # Rows on disk are already chat-messages (produced upstream by TrainingCorpus.to_chat_messages
    # during a real Harness cycle); feed them straight through rather than round-tripping through
    # TrainingCorpus's trajectory shape, which this script never populates.
    class _DirectCorpus:
        def sft_rows(self):
            return list(range(len(rows)))

        def to_chat_messages(self, row):
            return rows[row]

    runner = AWSGPUJobRunner(bucket=BUCKET, region=REGION, instance_type=INSTANCE_TYPE)
    optimizer = UnslothLoRAOptimizer(job_runner=runner, lora_rank=args.lora_rank)

    artifact = Artifact(
        scope=MODEL_WEIGHTS, shape=ArtifactShape.WEIGHTS, value={"base_model": args.base_model},
        optimizer_binding=None, technique=None, source="manual",
    )

    print(f"Launching {INSTANCE_TYPE} in {REGION} now — billing starts immediately.")
    result = optimizer.estimate(artifact, _DirectCorpus(), benchmark=None, variant=None)
    print(json.dumps({
        "adapter_path": result.value["adapter_path"],
        "provenance": result.provenance,
    }, indent=2))


if __name__ == "__main__":
    main()
