# AgentGym

Diagnose which layer of an LLM agent is broken — context, harness, or the model's own weights —
route the fix to a real optimizer, and prove the improvement actually improved anything.

AgentGym doesn't build its own evaluation engine, orchestration framework, or training loop. It
defines a small set of protocols that let real, actively-maintained tools plug into one shared
representation, and ships reference adapters proving those protocols actually work end to end.

**Status: v0 and v1 complete.** All nine scopes have real reference `ScopeOptimizer`s, the full
eight-stage `Harness` lifecycle (Existing → Evaluate → Diagnose → Improve → Evaluate → Release →
A/B → Launch) is implemented end to end, and 169 tests pass (155 fast + 14 integration). See
`docs/DESIGN.md` for the full design and `docs/IMPLEMENTATION_PLAN.md` for the class-by-class,
test-driven build plan.

| Scope | Lever | Real optimizer |
|---|---|---|
| `PROMPT` | context | DSPy (BootstrapFewShot / MIPRO) |
| `TOOLS` | context | Optuna tool-subset search |
| `MEMORY` | context | headroom-ai + caveman-compression (two comparable bindings) |
| `RETRIEVAL` | context | headroom's BM25Scorer + Optuna |
| `MODEL_ROUTING` | harness | Optuna threshold search |
| `INTERRUPT` | harness | Optuna threshold search |
| `GRAPH` | harness | LangGraph + Optuna |
| `GUARDRAILS` | harness | garak + Invariant Labs' `invariant` |
| `MODEL_WEIGHTS` | fine_tune | Unsloth LoRA + TRL GRPO (RLHF) |

Plus a full `RuleBasedDiagnoser`, `OfflineDeployer` (benchmark replay) and `LiveDeployer` (real
routed-traffic A/B), three named `LearningRecipe`s, and a `CompositeRecipe` scheduler.

## Install (development)

```bash
uv venv --python 3.12
uv sync --group dev
```

Each real integration is opt-in via extras, so installing the core never pulls in every
third-party tool: `dspy`, `langgraph`, `finetune` (Unsloth/TRL), `providers` (boto3), `memory`
(headroom-ai), `security` (garak/invariant-ai), `examples` (the LangChain/LangGraph demo agent —
see below). e.g.:

```bash
uv sync --group dev --extra dspy --extra langgraph --extra memory --extra security
```

`examples` and `inspect` are declared mutually exclusive (`[tool.uv] conflicts` in
`pyproject.toml`) — a real, confirmed dependency clash between `langchain-aws` and `inspect-ai`'s
`datasets` pin. Don't request both in the same sync.

## Test

```bash
uv run pytest                    # fast suite
uv run pytest -m integration     # real calls to external tools/services
```

## Examples

[`examples/travel_sage_agent/`](examples/travel_sage_agent/README.md) — a real LangGraph
travel-booking agent (real Claude via Bedrock or local Ollama) onboarded onto AgentGym, with two
runnable demos: real `MEMORY`-scope context compression, and real `GUARDRAILS`-scope security
hardening driven by a real `garak` audit.
