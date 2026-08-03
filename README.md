# AgentGym

Diagnose which layer of an LLM agent is broken — context, harness, or the model's own weights —
route the fix to a real optimizer, and prove the improvement actually improved anything.

AgentGym doesn't build its own evaluation engine, orchestration framework, or training loop. It
defines a small set of protocols that let real, actively-maintained tools (DSPy, LangGraph, Unsloth,
Inspect AI, and others) plug into one shared representation, and ships reference adapters proving
those protocols actually work end to end.

Status: pre-v0, under active development. See `docs/DESIGN.md` for the full design and
`docs/IMPLEMENTATION_PLAN.md` for the class-by-class, test-driven build plan.

## Install (development)

```bash
uv venv --python 3.12
uv sync --group dev
```

## Test

```bash
uv run pytest
```
