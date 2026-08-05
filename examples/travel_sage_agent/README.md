# Travel Sage Agent — a real AgentGym onboarding example

A real LangGraph tool-calling travel-booking agent (real Claude via AWS Bedrock, or a local Ollama
model), onboarded onto AgentGym's `Agent` protocol. It demonstrates two of AgentGym's nine scopes
end to end, against a real running agent rather than a synthetic test fixture:

- **`MEMORY` (context lever)** — a mock flight-search tool returns ~30 verbose results, real
  context bloat. `HeadroomMemoryOptimizer` searches real `headroom-ai` compression configs and the
  winning one is applied live, in the loop.
- **`GUARDRAILS` (harness lever)** — a mock hotel-search tool's review field carries a real
  prompt-injection payload. `GuardrailsOptimizer` runs a real `garak` probe against the agent's
  actual backing model and produces a tightened `invariant-ai` policy, which is then enforced live:
  the malicious tool output is deterministically stripped before it ever reaches the model's
  context — not dependent on whether the model happens to resist it on its own.

Everything here is real: real Bedrock/Ollama calls, real mock tool data (no network calls from the
tools themselves), real `headroom`/`garak`/`invariant-ai` libraries doing real work. Nothing is
stubbed or scripted to produce a predetermined result.

## Prerequisites

1. From the `agentgym/` repo root:
   ```bash
   uv sync --group dev --extra langgraph --extra memory --extra security --extra examples
   ```
   (`examples` conflicts with `inspect` in `uv`'s resolver — a real, confirmed upstream dependency
   clash between `langchain-aws` and `inspect-ai`'s `datasets` pin, not something this repo can
   paper over. Don't request `--extra inspect` in the same sync.)

2. AWS credentials with Bedrock access, in `RL-agents/.env` (one directory up from `agentgym/`):
   `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, `BEDROCK_MODEL_SONNET`
   (used for the MEMORY demo).

3. A local Ollama server running with `qwen2.5:7b` pulled (used for the GUARDRAILS demo — a real
   stand-in for a cheaper/self-hosted model tier, and the actual target garak audits):
   ```bash
   ollama pull qwen2.5:7b
   ```

## Running the demos

From the `agentgym/` repo root:

```bash
uv run python examples/travel_sage_agent/demo_context_lever.py
uv run python examples/travel_sage_agent/demo_harness_lever.py
```

Each prints real before/after numbers as it runs — no flags needed. `demo_harness_lever.py` uses
garak's fast `test.Blank` smoke probe by default so the live run stays under ~30s; see "A note on
the security demo" below for the real, thorough audit this stands in for.

## What each script actually does

**`demo_context_lever.py`**: runs the flight-search task against real Bedrock Claude with no
`MEMORY` artifact bound (raw tool output reaches the model — real char count printed); captures
the trace into a real `TrainingCorpus`; runs `HeadroomMemoryOptimizer.estimate()` for real (a real
Optuna search over real `headroom.CompressConfig` variants); prints the winning config and its
real measured `tokens_before`/`tokens_after`/`tokens_saved`; re-runs the same task with that config
bound live, to confirm the agent still answers correctly with compressed context.

**`demo_harness_lever.py`**: runs the hotel-search task (the injected review) against the agent
backed by local `qwen2.5:7b`, `GUARDRAILS` unbound — prints the raw tool output actually reaching
the model. Runs `GuardrailsOptimizer.estimate()` for real: a real `garak` probe against the real
Ollama target, producing a real `invariant-ai` policy. Re-runs the same task with that policy
bound — the injected review is now blocked at the tool boundary (`{"blocked": true, ...}` in place
of the raw content) before the model ever sees it.

## A note on the security demo — an honest finding, not a scripted one

Manual testing during development showed that Claude Sonnet 4.6 independently detected and refused
the injection payload in this repo, unprompted, with no guardrail bound at all — and a local
`qwen2.5:7b` neither complied with it. Rather than hand-craft a payload specifically to defeat a
frontier model (which would make the "before" state artificial), this demo makes the honest,
stronger claim instead: regardless of whether a given model happens to resist a given attack, you
should not have to rely on that — `GuardrailsOptimizer` + the live guardrail enforcement in
`agent.py` remove the exposure deterministically, verifiable independent of model judgment.

For a real (not live-demo-fast) audit, garak's `latentinjection` probe family (e.g.
`latentinjection.LatentInjectionResume`, 256 real prompts) is the thematically-matched, thorough
probe class for this kind of indirect/tool-output injection — it takes several minutes against a
local 7B model, confirmed by timing a real run, so it's meant to be run ahead of time rather than
live. Bind it via:
```python
GuardrailsOptimizer(target_type="ollama", target_name="qwen2.5:7b",
                     probes=["latentinjection.LatentInjectionResume"], generations=1)
```

## Files

- `tools.py` — mock `search_flights`/`search_hotels`/`get_weather` LangChain tools, no network.
- `llm.py` — real `ChatBedrockConverse` and `ChatOllama` builders.
- `agent.py` — `TravelSageAgent`: the real LangGraph app + AgentGym `Agent` protocol
  (`consumes()`/`run()`), plus the live MEMORY/GUARDRAILS enforcement wrapping each tool call.
- `benchmark.py` — the two real `Task` prompts both demos use.
- `demo_context_lever.py`, `demo_harness_lever.py` — the two runnable demo scripts above.

Tests: `tests/examples/test_travel_sage_agent.py` (light smoke coverage — offline tests run in the
fast suite, two real-network tests are `integration`-marked).
