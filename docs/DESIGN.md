# AgentGym — Overall Design

## Problem

An LLM agent can be improved in exactly three ways: change what it sees, change how the system
around it runs it, or change the model's own weights. Every team doing this today does it by hand
and by instinct — someone looks at a failure, guesses which of the three is responsible, and
reaches for whichever tool they already know (rewrite a prompt, add a retry rule, kick off a
fine-tune). Three things are consistently missing when that happens:

1. **No standard way to diagnose which of the three is actually responsible** for a given failure,
   as opposed to guessing.
2. **No standard way to represent what's tunable** in a given agent, so that different tools
   (a prompt optimizer, an orchestration framework, a training library) can be swapped in against
   the same representation instead of each inventing its own config format.
3. **No standard way to prove an "improvement" actually improved anything** — a measured
   before/after comparison, not a demo that happens to look better once.

This is true independent of any one framework's terminology. Evaluation is well covered (Inspect
AI, τ²-bench and similar benchmarks), tracing is well covered (OpenTelemetry, Langfuse),
orchestration is well covered (LangGraph, CrewAI), training is well covered (Hugging Face TRL,
Unsloth, ART). None of them decide *which* of the three levers is broken, and none of them treat
context-changes, orchestration-changes, and weight-changes as instances of one underlying thing
that can be measured, compared, and automated the same way. That gap is what AgentGym exists to
fill.

## Design philosophy

Two commitments — actual choices among real alternatives:

1. **Standardize agent improvement as an RL formulation, not an ad hoc pipeline.** The thing being
   learned isn't just "which prompt is better" — it's a policy over which lever to touch and how,
   evaluated against a reward, updated, and rolled out. Naming this as RL isn't decoration: it's
   what makes it possible to reuse RL's own vocabulary for the parts that were previously informal
   — state, action, reward, policy, transition, deployment — instead of inventing bespoke language
   per lever.
2. **Every stage of that loop is an interface over existing OSS tools — interoperate, don't
   reinvent.** AgentGym does not build its own evaluation engine, its own orchestration framework,
   its own training loop, or its own experimentation platform. It defines the interfaces that let
   actively-maintained tools plug into a shared representation, and ships reference adapters
   proving those interfaces work end to end.

Separately, a standard the whole project is held to, not a third "commitment" — because treating
honesty as a commitment implies faking it was ever a legitimate alternative on the table: every
reference implementation described here is an integration against a real tool, doing real work,
scored by a real verifier. Nothing is stubbed, mocked, or a placeholder standing in for a result
that was never actually computed.

## The lifecycle — the actual harness

The standard shape, named plainly: **Existing → Evaluate → Diagnose → Improve → Evaluate →
Release → A/B → Launch.** This is the RL loop stated as an MLOps lifecycle instead of stopping at
"the optimizer produced a better offline number":

| Stage | RL term | What it is |
|---|---|---|
| Existing | state | the current `AgentVariant` — a baseline, possibly the production one |
| Evaluate | reward computation | run the `Benchmark`, score via the `Verifier` — this is *where
  benchmark definition lives*, not a separate free-floating concept |
| Diagnose | credit assignment | the `Diagnoser` maps a scored trace to the responsible lever/scope |
| Improve | policy update / action | the `LearningRecipe` picks a scope, the bound `ScopeOptimizer`
  estimates a new `Artifact` — the actual state transition |
| Evaluate (again) | offline reward re-check | re-score the *candidate* variant on the same
  benchmark before it touches anything real — a sanity gate, not the final word |
| Release | canary deployment | ship the candidate to a small slice |
| A/B | online policy comparison | compare candidate vs. current baseline with statistical
  evidence — distinct from the offline re-check above, because a benchmark win and a production
  win are not the same claim |
| Launch | policy promotion | candidate becomes the new baseline; a losing A/B rolls back instead |

**Data collection is its own interface**, not folded into Evaluate: every stage produces traces,
and how those traces get captured — via tracing (the OTEL-aligned `Trace`/`Span` types) or from
external/already-existing data — is what `TrainingCorpus` standardizes. Its *native* representation
stays trajectory-based — state, action, observation, reward per step, the same shape used for the
SFT/DPO/GRPO rows throughout this design, and close to the `Trajectory` abstraction OpenPipe's ART
uses for agent RL specifically. That's a deliberate choice, not an assumption of free compatibility:
there is no single established "agent training format" the way OpenTelemetry is a standard for
tracing — TRL and Unsloth consume a *chat-messages* format (role/content, tool_calls on assistant
messages, a tool role for results), which represents one flattened conversation, not a trajectory
with per-step state and reward. Projecting a trajectory down into chat-messages is lossy, named
work — `TrainingCorpus.to_chat_messages()` — not a translator-free claim.

`Harness` (the orchestrator) implements this literally — `cycle()` runs all eight stages in order,
and a `Deployer` protocol owns Release/A-B/Launch/rollback, separately from the `Verifier` that
owns both Evaluate stages. Two reference `Deployer`s exist: `OfflineDeployer` simulates
Release/A-B/Launch via benchmark replay — useful for offline development, honestly labeled as a
simulation — and `LiveDeployer` routes real traffic between baseline and candidate and computes
its A/B statistics from accumulated real outcomes rather than a replayed benchmark.

## The Scope & Artifact model — the core abstraction

The three levers decompose further. "What the model sees" isn't one thing — a prompt template, a
memory policy, and a tool-selection rule are different problems that happen to share a lever.
Calling any of them a "hyperparameter" would also be dishonest: a prompt-assembly pipeline is a
composed program, model weights are tensors, a tool-selection policy might be an ensemble of
strategies, not a scalar knob. The model needs to represent all of that without forcing everything
into one shape.

**`Scope`** — the named area where a tunable thing lives (an area is not itself a parameter), and
**recursive by construction**: a lever is the root, named scopes are its children, and any scope
can have children of its own if a finer grain is ever useful (`context.retrieval.reranker` under
`context.retrieval`, without redesigning the type). This is the literal answer to "is there a
context-optimizer scope containing a prompt-optimizer scope" — yes, `Lever.CONTEXT` is that root:

```python
class Lever(str, Enum):
    CONTEXT   = "context"     # what the model is shown
    HARNESS   = "harness"     # how the system runs it
    FINE_TUNE = "fine_tune"   # the model's own weights

@dataclass(frozen=True)
class Scope:
    lever: Lever
    segments: tuple[str, ...]     # ("prompt",), or ("retrieval","reranker") for a nested child

    def __str__(self) -> str:
        return f"{self.lever.value}.{'.'.join(self.segments)}"

    def is_child_of(self, other: "Scope") -> bool:
        return (self.lever == other.lever
                and self.segments[:len(other.segments)] == other.segments
                and len(self.segments) > len(other.segments))

# the named scopes, expressed as Scope instances:
PROMPT         = Scope(Lever.CONTEXT, ("prompt",))
TOOLS          = Scope(Lever.CONTEXT, ("tools",))
MEMORY         = Scope(Lever.CONTEXT, ("memory",))
RETRIEVAL      = Scope(Lever.CONTEXT, ("retrieval",))
MODEL_ROUTING  = Scope(Lever.HARNESS, ("model_routing",))
INTERRUPT      = Scope(Lever.HARNESS, ("interrupt",))       # escalation / human-in-the-loop
GRAPH          = Scope(Lever.HARNESS, ("graph",))            # topology / control-flow
GUARDRAILS     = Scope(Lever.HARNESS, ("guardrails",))       # security policy: injection/PII/
                                                               #   secret/jailbreak detection and
                                                               #   response — a runtime-verification
                                                               #   concern, so it lives under harness
                                                               #   rather than as a fourth lever
MODEL_WEIGHTS  = Scope(Lever.FINE_TUNE, ("model_weights",))  # a technique sits above the raw weights
```

**Composing across levers** is a namespaced union, not an AND/OR to resolve: because every `Scope`
carries its `lever` as part of its identity, a `context.prompt` scope can never collide with
anything under `harness` or `fine_tune`, so a full agent configuration is simply the union of
whatever scopes are bound under each lever — see `AgentVariant` below.

**Composing within a lever** is a declared dependency, not implicit coupling — a prompt optimizer
may genuinely need to know what's currently in the tools scope to write instructions that match it.
That's `ScopeOptimizer.depends_on` (below): an explicit list of other scopes an optimizer reads
while estimating its own artifact, distinct from `optimizer_binding` (which tool owns the scope).

**`ArtifactShape`** — what *kind* of thing occupies a scope right now, declared honestly instead of
forced into one mold:

```python
class ArtifactShape(str, Enum):
    PARAMETER = "parameter"   # a scalar/discrete value — retry_limit=3, LoRA rank=16
    CONFIG    = "config"      # a flat named-field structure
    PROGRAM   = "program"     # a composed pipeline; each stage is itself an Artifact, recursively
                               #   — this is literally what a DSPy Module is
    ENSEMBLE  = "ensemble"    # multiple strategies combined or voted, not one thing selected
    WEIGHTS   = "weights"     # a numeric tensor / adapter reference
```

**`Artifact`** — the thing that actually occupies a `Scope`, carrying two separate manual/estimated
decisions:

```python
@dataclass
class Artifact:
    scope: Scope
    shape: ArtifactShape
    value: Any                            # interpretation depends on shape; a PROGRAM's value can
                                           #   itself be a list[Artifact] — composition is native
    optimizer_binding: str | None         # e.g. "dspy.MIPRO" — which tool handles this scope
    technique: FineTuneTechnique | None   # only meaningful when scope == MODEL_WEIGHTS
    source: Literal["manual", "estimated"]
    provenance: dict                      # which run/corpus produced this value, if estimated

class FineTuneTechnique(str, Enum):
    LORA_ADAPTER = "lora_adapter"
    FULL_FINETUNE = "full_finetune"
    RLHF = "rlhf"
```

Worth being explicit that there are **two separate recursion mechanisms**, not one, so they don't
get conflated: `Scope` nesting (above) is about the *namespace* of tunable areas — it's how
"context" decomposes into "prompt," "tools," "memory," and could decompose further. A
`PROGRAM`-shaped artifact's own `value` being `list[Artifact]` is about *one artifact's* internal
pipeline structure — e.g. the `context.prompt` scope's artifact might internally be [instruction
template, few-shot selector, formatter], which are pipeline stages, not separate named scopes of
their own. A scope tree organizes *what's tunable*; a `PROGRAM` artifact organizes *how one tunable
thing is built*.

Two decisions stack on top of each other, and both are visible in the type. **Which optimizer is
bound to a scope** is almost always a manual, human, one-time choice — "DSPy handles my prompt
scope," "Unsloth-LoRA handles my model-weights scope" — that's `optimizer_binding`. **The actual
value living in the scope** is what's manual-or-estimated: whatever was hand-set until an optimizer
is invoked, at which point it's estimated and carries provenance. Nothing gets estimated until a
scope is declared with a starting artifact and a binding — and because the protocol is defined on
`Scope` + `ArtifactShape`, any tool that can produce the right shape for a scope can be swapped in
for the binding without touching anything else.

## The protocol — standardized per scope

`typing.Protocol` structural types throughout — no inheritance required to implement one. One
optimizer protocol *per scope*, not one per lever, since prompt optimization and memory
optimization are genuinely different problems that happen to both live under the context lever.

- **`Trace`/`Span`** — aligned to OpenTelemetry's GenAI semantic conventions, not a bespoke schema,
  so any OTEL-instrumented agent produces a valid trace without a custom adapter.
- **`Score`** — metric_name, value, kind: verifiable|plausible, evidence.
- **`Diagnosis`** — `{trace_id, lever, evidence, confidence, suggested_scope}`, with
  `insufficient_evidence` as a first-class outcome, not a fallback.
- **`TrainingCorpus`** — accumulator: `add_trace(trace, scores, diagnosis)`, `sft_rows()`,
  `dpo_pairs()`, `artifact_candidates(scope)` — filterable by scope.
- **`Agent`** — the contract layer, made explicit: `consumes() -> dict[Scope, Artifact | None]`
  declares which scopes the agent reads when it runs, and what its default is for each (`None`
  meaning no default exists); `run(task, artifacts: dict[Scope, Artifact]) -> Trace` is then
  guaranteed to receive a bound value or an explicit declared default for every scope in
  `consumes()` — an agent never silently runs with a gap, and never has to guess whether a scope
  it needs is present.
- **`Benchmark`** — `cases() -> Iterable[Task]` — one benchmark, shared across every scope and every
  lever, deliberately: it's the only way a change to context and a change to weights are ever
  actually comparable.
- **`Verifier`** — `score(trace, task) -> list[Score]`. `Verifier`s compose: a `CompositeVerifier`
  calls several and concatenates their `Score` lists onto the same trace — this is how a
  security-probing pass adds `prompt_injection_free`-style scores alongside task-correctness
  scores, without forking off a second benchmark. The benchmark stays one; what scores it, per
  trace, doesn't have to be.
- **`Diagnoser`** — `diagnose(trace, scores) -> Diagnosis`. Reference: rule-based, swappable later
  for a learned version without touching anything else.
- **`ScopeOptimizer`** — the standardized interface tools plug into, per scope: `scope: Scope`;
  `depends_on: list[Scope]` (other scopes read, not written, while estimating — the explicit
  answer to "how do scopes within a lever compose"); `estimate(artifact, corpus, benchmark,
  variant: AgentVariant) -> Artifact`, where `variant` is how an optimizer reads its declared
  dependencies' current values.
- **`AgentVariant`** — a **partial** mapping `dict[Scope, Artifact]`, not a total one: not every
  scope needs to be bound (an agent variant with context and harness configured but no fine-tune
  applied yet is simply one where no `fine_tune.*` scope is present, falling back to each
  unbound scope's declared default). A full variant's scopes are the namespaced union of whatever
  is bound per lever — collision-free by construction, since `Scope` carries its `lever`.
- **`LearningRecipe`** — the pluggable policy: `next_action(report_history, variant) -> Action`.
  Reference recipes named after the automation patterns they follow: a recipe that auto-searches
  context scopes, one that auto-searches harness scopes, one that iterates fine-tune experiments
  via propose → apply → run → keep/revert, and a composite default that sequences all three
  cheapest-scope-first.
- **`Deployer`** — owns Release/A-B/Launch/rollback, kept separate from the `Verifier` that owns
  both Evaluate stages, because "scored better on the benchmark" and "won in production" are
  different claims: `release(variant, traffic_pct) -> Release`, `ab_compare(baseline, candidate,
  release) -> ABResult`, `launch(release) -> None`, `rollback(release) -> None`.
- **`Harness`** — the orchestrator implementing the full lifecycle: `cycle(baseline) ->
  CycleReport`. Existing → Evaluate → Diagnose → Improve → Evaluate → Release → A/B → Launch, in
  order, every run.

## Scopes and their optimizers

Every scope has a working reference `ScopeOptimizer`, each binding a real, independently-verified
external tool rather than a hand-rolled heuristic:

| Scope | Lever | Optimizer | Shape |
|---|---|---|---|
| `PROMPT` | context | DSPy (BootstrapFewShot / MIPRO) | `PROGRAM` |
| `TOOLS` | context | Optuna search over which tools to expose | `ENSEMBLE` |
| `MEMORY` | context | headroom-ai, and a second, independently-swappable binding via
  caveman-compression — proof that the interface is genuinely tool-agnostic, not just
  single-vendor | `CONFIG` |
| `RETRIEVAL` | context | headroom's BM25 scorer + Optuna top-k search | `CONFIG` |
| `MODEL_ROUTING` | harness | Optuna threshold search over a routing rule | `PARAMETER` |
| `INTERRUPT` | harness | Optuna threshold search over an escalation trigger | `PARAMETER` |
| `GRAPH` | harness | a LangGraph app + Optuna search over its config space | `CONFIG` |
| `GUARDRAILS` | harness | garak (vulnerability probing) driving an Invariant Labs `invariant`
  policy | `CONFIG` |
| `MODEL_WEIGHTS` | fine_tune | Unsloth LoRA (SFT) and TRL's GRPOTrainer (RLHF-style), both
  dispatched via `FineTuneTechnique` and run on a rented GPU instance | `WEIGHTS` |

Two scope pairings are worth calling out specifically because they double as proof the
`ScopeOptimizer` interface is genuinely swappable, not single-vendor:

- **`MEMORY`** has two independent, comparable bindings — headroom-ai and caveman-compression —
  run against the same failure corpus and the same benchmark, with directly comparable `ABResult`s.
  The same scope, two competing tools, one shared measurement.
- **`GUARDRAILS`** composes security scoring onto the same benchmark task set via
  `CompositeVerifier` rather than forking a second benchmark: a trace carries both "did it
  complete the task" and "could it be attacked" scores, and a guardrail policy earns its way to
  production through the same Release/A-B/Launch stages as any other scope.

`MODEL_WEIGHTS` is the one scope with a real cost/infrastructure dependency: training runs on a
rented cloud GPU instance via a `CloudGPUJobRunner` the optimizer is handed, not something it
provisions itself. **Before any cloud GPU instance is actually provisioned, the instance type,
estimated cost, and estimated duration are confirmed explicitly** — this is the one place in the
system real money gets spent, and it's never triggered automatically by a recipe or a run loop.

`RuleBasedDiagnoser` covers all nine scopes with a real evidence table (a guardrail violation, a
tool-call error, a missing escalation, a repeated-action loop, an empty retrieval, oversized
context, a routing mismatch, a partial vs. severe task failure), checked in order of how directly
diagnostic each signal is.

## Verification

- `pytest` over `tests/` — protocol conformance, artifact/scope-binding tests, a `PROGRAM`-shaped
  artifact composition test (nesting artifacts recursively), and a per-optimizer test suite that
  exercises each real tool integration directly.
- A full end-to-end `Harness.cycle()` test exercises all eight lifecycle stages against a real
  search and a real statistical A/B, not fakes standing in for the pipeline.
- Slower, network- or cost-adjacent tests (LLM calls, cloud-GPU-adjacent work) are marked
  `integration` and run separately from the fast suite.
