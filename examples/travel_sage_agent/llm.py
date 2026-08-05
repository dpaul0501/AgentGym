"""Two LLM backends for the Travel Sage agent, chosen per-demo:

- Bedrock Claude (build_bedrock_llm) — the production-grade model, used for the MEMORY/context-
  lever demo. Wired from RL-agents/.env, the same credentials/model IDs already used elsewhere in
  this workspace.
- Local Ollama (build_ollama_llm) — a stand-in for a cheaper/self-hosted model tier, used for the
  GUARDRAILS/harness-lever demo. garak audits this model directly, so the demo reflects a measured
  vulnerability rather than a scripted one.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from langchain_ollama import ChatOllama

_ENV_PATH = Path(__file__).resolve().parents[3] / ".env"


def build_bedrock_llm(model_env_var: str = "BEDROCK_MODEL_SONNET", temperature: float = 0.0) -> ChatBedrockConverse:
    load_dotenv(_ENV_PATH)
    return ChatBedrockConverse(
        model=os.environ[model_env_var],
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        temperature=temperature,
    )


def build_ollama_llm(model_name: str = "qwen2.5:7b", temperature: float = 0.0) -> ChatOllama:
    return ChatOllama(model=model_name, temperature=temperature)
