# Travel Sage Agent — an AgentGym onboarding example

A LangGraph tool-calling travel-booking agent (Claude via AWS Bedrock, or a local Ollama model),
onboarded onto AgentGym's `Agent` protocol. It demonstrates two of AgentGym's nine scopes end to
end, against a running agent rather than a synthetic test fixture. Both demos track a metric
across several independent trials before and after binding an artifact — not a single before/after
anecdote — so the improvement is a measured rate change, the same shape of evidence
`Harness.cycle()`'s own offline/A-B gates require internally:

- **`MEMORY` (context lever)** — 4 independent flight-search routes, each returning ~30 verbose
  mock results and causing real context bloat. `HeadroomMemoryOptimizer` trains on all 4 trials'
  captured traces, searches `headroom-ai` compression configs, and the winning one is applied live
  across the same 4 routes. Tracked metric: total context size and a correctness check (does the
  response still name a real flight/price) across all 4 trials, before and after.
- **`GUARDRAILS` (harness lever)** — 3 independent hotel-search cities, each with a distinct
  prompt-injection payload in a review field. `GuardrailsOptimizer` runs a `garak` probe against
  the agent's backing model and produces a tightened `invariant-ai` policy, enforced live across
  the same 3 cities. Tracked metric: how many of the 3 trials have malicious tool content reach the
  model's context, before and after.

The tools are mocked (no network calls from them), but the LLM calls, and the `headroom`/`garak`/
`invariant-ai` libraries, all do their actual work. Nothing here is scripted to produce a
predetermined result.

## Prerequisites

1. From the `agentgym/` repo root:
   ```bash
   uv sync --group dev --extra langgraph --extra memory --extra security --extra examples
   ```
   (`examples` conflicts with `inspect` in `uv`'s resolver — `langchain-aws` and `inspect-ai` pin
   incompatible `datasets` versions. Don't request `--extra inspect` in the same sync.)

2. AWS credentials with Bedrock access, in `RL-agents/.env` (one directory up from `agentgym/`):
   `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, `BEDROCK_MODEL_SONNET`
   (used for the MEMORY demo).

3. A local Ollama server running with `qwen2.5:7b` pulled (used for the GUARDRAILS demo — a
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

Each prints before/after numbers as it runs — no flags needed. `demo_harness_lever.py` uses
garak's fast `test.Blank` smoke probe by default so the live run stays under ~30s; see "A note on
the security demo" below for the thorough audit this stands in for.

## What each script does

**`demo_context_lever.py`**: runs all 4 flight-search routes against Bedrock Claude with no
`MEMORY` artifact bound, recording each trial's context size and a correctness check — a real
example run measured 93,061 total chars across the 4 trials, correctness 4/4. Captures all 4
traces into one `TrainingCorpus`; runs `HeadroomMemoryOptimizer.estimate()` (an Optuna search over
`headroom.CompressConfig` variants trained on all 4 trajectories); re-runs the same 4 routes with
the winning config bound live — that same run measured 37,513 total chars (a 59.7% reduction),
correctness still 4/4.

**`demo_harness_lever.py`**: runs all 3 hotel-search cities (each with a distinct injected review)
against the agent backed by local `qwen2.5:7b`, `GUARDRAILS` unbound, recording how many trials
have the malicious content actually reach the model — a real example run measured 2/3. Runs
`GuardrailsOptimizer.estimate()`: a `garak` probe against the Ollama target, producing an
`invariant-ai` policy. Re-runs the same 3 cities with that policy bound — that same run measured
0/3, every trial blocked at the tool boundary (`{"blocked": true, ...}` in place of the raw
content) before the model ever saw it.

## A note on the security demo

Manual testing during development showed that Claude Sonnet 4.6 independently detected and refused
the injection payloads in this repo, unprompted, with no guardrail bound at all — and local
`qwen2.5:7b` didn't comply with them either. Rather than hand-craft payloads specifically to defeat
a frontier model (which would make the "before" state artificial), this demo makes the stronger
claim instead: regardless of whether a given model happens to resist a given attack, you shouldn't
have to rely on that — `GuardrailsOptimizer` plus the live guardrail enforcement in `agent.py`
remove the exposure deterministically, independent of model judgment. That's also why the tracked
metric is "did the malicious content reach the model's context," not "did the model comply" — the
former is measurable and enforceable regardless of the model; the latter would depend on which
model happens to be behind the agent that day.

For a thorough (not live-demo-fast) audit, garak's `latentinjection` probe family (e.g.
`latentinjection.LatentInjectionResume`, 256 prompts) is the thematically-matched probe class for
this kind of indirect/tool-output injection — it takes several minutes against a local 7B model,
so it's meant to run ahead of time rather than live. Bind it via:
```python
GuardrailsOptimizer(target_type="ollama", target_name="qwen2.5:7b",
                     probes=["latentinjection.LatentInjectionResume"], generations=1)
```

## Files

- `tools.py` — mock `search_flights`/`search_hotels`/`get_weather` LangChain tools, no network.
  `INJECTION_PAYLOADS` maps each of the 3 demo cities to a distinct injection payload.
- `llm.py` — `ChatBedrockConverse` and `ChatOllama` builders.
- `agent.py` — `TravelSageAgent`: the LangGraph app + AgentGym `Agent` protocol
  (`consumes()`/`run()`), plus the live MEMORY/GUARDRAILS enforcement wrapping each tool call.
- `benchmark.py` — `TravelBenchmark`: 4 flight-search routes and 3 hotel-search cities, the task
  set both demos loop over.
- `demo_context_lever.py`, `demo_harness_lever.py` — the two runnable demo scripts above.

Tests: `tests/examples/test_travel_sage_agent.py` (light smoke coverage — offline tests run in the
fast suite, two network-hitting tests are `integration`-marked).
