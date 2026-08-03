# AgentGym — Design Document

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
3. **No standard way to prove an "improvement" actually improved anything** — a real, measured
   before/after comparison, not a demo that happens to look better once.

This is true independent of any one framework's terminology. Evaluation is well covered (Inspect
AI, τ²-bench and similar benchmarks), tracing is well covered (OpenTelemetry, Langfuse),
orchestration is well covered (LangGraph, CrewAI), training is well covered (Hugging Face TRL,
Unsloth, ART). None of them decide *which* of the three levers is broken, and none of them treat
context-changes, orchestration-changes, and weight-changes as instances of one underlying thing
that can be measured, compared, and automated the same way. That gap — verified directly against
these projects' current state, not assumed — is what AgentGym exists to fill.

## Design philosophy

Two real commitments — actual choices among real alternatives:

1. **Standardize agent improvement as a real RL formulation, not an ad hoc pipeline.** The thing
   being learned isn't just "which prompt is better" — it's a policy over which lever to touch and
   how, evaluated against a reward, updated, and rolled out. Naming this as RL isn't decoration:
   it's what makes it possible to reuse RL's own vocabulary for the parts that were previously
   informal — state, action, reward, policy, transition, deployment — instead of inventing bespoke
   language per lever.
2. **Every stage of that loop is an interface over existing OSS tools — interoperate, don't
   reinvent.** AgentGym does not build its own evaluation engine, its own orchestration framework,
   its own training loop, or its own experimentation platform. It defines the interfaces that let
   real, actively-maintained tools plug into a shared representation, and ships reference adapters
   proving those interfaces actually work end to end.

Separately, a standard the whole document is held to, not a third "commitment" — because treating
honesty as a commitment implies faking it was ever a legitimate alternative on the table: every
reference implementation described here is a real integration against a real tool, doing real
work, scored by a real verifier. Nothing here is stubbed, mocked, or a placeholder standing in for
a result that was never actually computed.

## The lifecycle — the actual harness

The standard shape, named plainly: **Existing → Evaluate → Diagnose → Improve → Evaluate →
Release → A/B → Launch.** This is the RL loop stated as a real MLOps lifecycle instead of stopping
at "the optimizer produced a better offline number":

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
| Release | canary deployment | ship the candidate to a small slice, for real or realistically |
| A/B | online policy comparison | compare candidate vs. current baseline with real evidence,
  statistically — distinct from the offline re-check above, because a benchmark win and a
  production win are not the same claim |
| Launch | policy promotion | candidate becomes the new baseline; a losing A/B rolls back instead |

**Data collection is its own interface**, not folded into Evaluate: every stage produces traces,
and how those traces get captured — via tracing (the OTEL-aligned `Trace`/`Span` types) or from
external/already-existing data — is what `TrainingCorpus` standardizes. Its *native* representation
stays trajectory-based — state, action, observation, reward per step, the same shape used for the
SFT/DPO/GRPO rows throughout this design, and close to the `Trajectory` abstraction OpenPipe's ART
uses for agent RL specifically. That's a deliberate choice, not an assumption of free compatibility:
there is no single established "agent training format" the way OpenTelemetry is a real standard for
tracing — TRL and Unsloth consume a *chat-messages* format (role/content, tool_calls on assistant
messages, a tool role for results), which represents one flattened conversation, not a trajectory
with per-step state and reward. Projecting a trajectory down into chat-messages is real, lossy,
named work — `TrainingCorpus.to_chat_messages()` — not a translator-free claim. The exact current
tool-calling chat-template conventions in TRL/Unsloth should be confirmed at implementation time,
not assumed — added to the same "verify the real API, don't guess" list as the Inspect AI and
benchmark-format items above.

`Harness` (the orchestrator; see Low-Level Design) implements this literally — `cycle()` runs all
eight stages in order, and a `Deployer` protocol owns Release/A-B/Launch/rollback, separately from
the offline `Verifier` that owns both Evaluate stages. v0's reference `Deployer` is honestly an
offline simulation (benchmark replay standing in for real traffic, clearly labeled as such) — a
`Deployer` backed by real production traffic is real infrastructure work in its own right and isn't
pretended to be solved by this design.

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

# the starting set of named scopes, expressed as Scope instances:
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
    optimizer_binding: str | None         # e.g. "dspy.MIPRO" — which real tool handles this scope
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
- **`Agent`** — the contract layer, made explicit (this was previously underspecified):
  `consumes() -> list[Scope]` declares which scopes the agent actually reads when it runs; `run(
  task, artifacts: dict[Scope, Artifact]) -> Trace` is then guaranteed by `AgentGym` to receive a
  bound value or an explicit declared default for every scope in `consumes()` — an agent never
  silently runs with a gap, and never has to guess whether a scope it needs is present.
- **`Benchmark`** — `cases() -> Iterable[Task]` — one benchmark, shared across every scope and every
  lever, deliberately: it's the only way a change to context and a change to weights are ever
  actually comparable.
- **`Verifier`** — `score(trace, task) -> list[Score]`. Reference implementation wraps Inspect AI's
  own scorer machinery — real verification, not a hand-rolled heuristic. `Verifier`s compose: a
  `CompositeVerifier` calls several and concatenates their `Score` lists onto the same trace —
  this is how a security-probing pass (garak, promptfoo) adds `prompt_injection_resistance`-style
  scores alongside task-correctness scores, without forking off a second benchmark. The benchmark
  stays one; what scores it, per trace, doesn't have to be.
- **`Diagnoser`** — `diagnose(trace, scores) -> Diagnosis`. Reference: rule-based, swappable later
  for a learned version without touching anything else.
- **`ScopeOptimizer`** — the standardized interface real tools plug into, per scope:
  `scope: Scope`; `depends_on: list[Scope]` (other scopes read, not written, while estimating —
  the explicit answer to "how do scopes within a lever compose"); `estimate(artifact, corpus,
  benchmark, variant: AgentVariant) -> Artifact`, where `variant` is how an optimizer actually
  reads its declared dependencies' current values.
  - `PROMPT` → DSPy (MIPRO / BootstrapFewShot) — produces a `PROGRAM`-shaped artifact; typically
    `depends_on=[TOOLS]` so generated instructions match the currently-bound tool set
  - `MEMORY` → a memory-store-backed or custom compressor optimizer — `CONFIG` or `PROGRAM`
  - `TOOLS` → a tool-schema/selection optimizer — `CONFIG` or `ENSEMBLE`
  - `RETRIEVAL` → reranker/retriever-config search — `CONFIG` or `PROGRAM`
  - `MODEL_ROUTING` → a routing-policy learner (bandit/classifier) — `PARAMETER` or `CONFIG`
  - `INTERRUPT` → escalation-threshold search — `PARAMETER`
  - `GRAPH` → topology search (Optuna or a contextual bandit) over a declared config space —
    `CONFIG`
  - `MODEL_WEIGHTS` → dispatches by `technique`: a LoRA optimizer, a full-finetune optimizer, an
    RLHF optimizer — all `WEIGHTS`-shaped
- **`AgentVariant`** — a **partial** mapping `dict[Scope, Artifact]`, not a total one: not every
  scope needs to be bound (an agent variant with context and harness configured but no fine-tune
  applied yet is simply one where no `fine_tune.*` scope is present, falling back to each
  unbound scope's declared default). A full variant's scopes are the namespaced union of whatever
  is bound per lever — collision-free by construction, since `Scope` carries its `lever`.
- **`LearningRecipe`** — the pluggable policy: `next_action(report_history, variant) -> Action`.
  Reference recipes named after the real automation patterns they follow: a recipe that
  auto-searches context scopes, one that auto-searches harness scopes, one that iterates fine-tune
  experiments via propose → apply → run → keep/revert, and a composite default that sequences all
  three cheapest-scope-first.
- **`Deployer`** — owns Release/A-B/Launch/rollback, kept separate from the offline `Verifier`
  that owns both Evaluate stages, because "scored better on the benchmark" and "won in production"
  are different claims: `release(variant, traffic_pct) -> Release`, `ab_compare(baseline,
  candidate, release) -> ABResult`, `launch(release) -> None`, `rollback(release) -> None`.
  Reference implementation for v0 (`OfflineDeployer`) honestly simulates this via benchmark replay
  — real production-traffic deployment is real infrastructure work, not something this design
  pretends to solve.
- **`Harness`** — the orchestrator implementing the full lifecycle: `cycle(baseline) ->
  CycleReport`. Existing → Evaluate → Diagnose → Improve → Evaluate → Release → A/B → Launch, in
  order, every run.

## Low-Level Design

### Package layout

```
agentgym/
  pyproject.toml                   # MIT, Python 3.12, installable/importable package
  README.md
  agentgym/
    __init__.py
    core/
      scope.py                     # Lever, Scope
      artifact.py                  # ArtifactShape, Artifact, FineTuneTechnique
      trace.py                     # Span, Trace, OTEL GenAI semconv converters
      score.py                     # Score
      diagnosis.py                 # Diagnosis
      corpus.py                    # TrainingCorpus
      variant.py                   # AgentVariant
      ab.py                        # ABComparison, ABResult, bootstrap-CI delta
      protocols.py                 # Agent, Benchmark, Verifier, Diagnoser, ScopeOptimizer,
                                    #   LearningRecipe, Deployer — all typing.Protocol
      harness.py                   # Harness orchestrator (the 8-stage cycle), CycleReport
      registry.py                  # register_optimizer / register_recipe / register_deployer
    deployers/
      offline_deployer.py          # OfflineDeployer — v0 reference, benchmark-replay simulation
                                    #   of Release/A-B/Launch, honestly labeled as such
    diagnosers/
      rule_based.py                # RuleBasedDiagnoser
    optimizers/
      dspy_prompt.py               # v0 — Scope.PROMPT
      langgraph_search.py          # v0 — Scope.GRAPH, extended to MODEL_ROUTING/INTERRUPT in v1
      unsloth_lora.py              # v0 — Scope.MODEL_WEIGHTS
      _unsloth_train.py            # the script that RUNS on the rented GPU instance
      headroom_memory.py           # Milestone B — Scope.MEMORY, wraps headroomlabs-ai/headroom
      caveman_memory.py            # Milestone B — Scope.MEMORY, alternate binding (caveman-
                                    #   compression and/or cavemem — confirm which, see note above)
      security_guardrails.py       # Milestone C — Scope.GUARDRAILS, wraps garak + promptfoo +
                                    #   invariantlabs-ai/invariant policy shape
      tool_selector.py              # v1 — Scope.TOOLS
      retriever_search.py           # v1 — Scope.RETRIEVAL
      trl_rlhf.py                   # v1 — Scope.MODEL_WEIGHTS, FineTuneTechnique.RLHF
    verifiers/
      inspect_adapter.py           # v0 — wraps Inspect AI's Scorer machinery
      security_verifier.py         # Milestone C — re-runs garak/promptfoo probes as scoring
      composite_verifier.py        # Milestone C — composes multiple Verifiers onto one trace
    benchmarks/
      tau2_bench.py                # a real tool-use benchmark
    recipes/
      context_search_recipe.py     # auto-searches context scopes
      harness_search_recipe.py     # auto-searches harness scopes
      finetune_recipe.py           # propose/apply/run/keep-revert on fine-tune experiments
      composite.py                 # v1 — cheapest-scope-first scheduler across all three
    storage/
      db.py                        # AgentGymStore — SQLite, thread-local connections, WAL mode
    providers/
      llm_adapter.py               # provider registry: Ollama/OpenAI/Bedrock/Anthropic/Groq
  scripts/
    run_gym.py                     # CLI entrypoint: agentgym run --recipe ... --iterations ...
    provision_gpu.py               # the ONE place a cloud GPU instance gets created — always run
                                    #   manually, never auto-invoked by run_gym.py
  tests/
    test_protocols.py              # structural-typing conformance
    test_artifact_composition.py   # PROGRAM-shaped artifacts nesting other artifacts
    test_corpus.py
    test_ab.py
    test_diagnoser.py
```

### Core types, in full

`agentgym/core/trace.py`:
```python
@dataclass
class Span:
    span_id: str
    parent_span_id: str | None
    trace_id: str
    name: str
    kind: Literal["llm", "tool", "retrieval", "other"]
    input: dict
    output: dict
    start_time: float
    end_time: float
    attributes: dict[str, Any]   # OTEL GenAI semconv keys for llm-kind spans:
                                  #   gen_ai.system, gen_ai.request.model,
                                  #   gen_ai.usage.input_tokens/output_tokens, etc.

@dataclass
class Trace:
    trace_id: str
    task_id: str
    spans: list[Span]
    metadata: dict

    @classmethod
    def from_otel_spans(cls, otel_spans: list[dict]) -> "Trace": ...
    def to_otel(self) -> list[dict]: ...
```

`agentgym/core/harness.py` — the orchestrator, implementing all eight stages, real control flow:
```python
class Harness:
    def __init__(self, agent: Agent, benchmark: Benchmark, verifier: Verifier,
                 diagnoser: Diagnoser, optimizers: dict[Scope, ScopeOptimizer],
                 recipe: LearningRecipe, deployer: Deployer, store: AgentGymStore): ...

    def _evaluate(self, variant: AgentVariant) -> tuple[list[Trace], list[list[Score]]]:
        traces, scores = [], []
        for task in self.benchmark.cases():
            resolved = variant.resolve_for(self.agent.consumes())   # bound value or declared
                                                                       #   default per consumed scope
            trace = self.agent.run(task, resolved)
            traces.append(trace)
            scores.append(self.verifier.score(trace, task))
        return traces, scores

    def cycle(self, baseline: AgentVariant) -> CycleReport:
        current = baseline                                            # 1. EXISTING
        traces, scores = self._evaluate(current)                      # 2. EVALUATE
        diagnoses = [self.diagnoser.diagnose(t, s) for t, s in zip(traces, scores)]  # 3. DIAGNOSE
        for t, s, d in zip(traces, scores, diagnoses):
            self.corpus.add_trace(t, s, d)

        action = self.recipe.next_action(self.history, current)       # 4. IMPROVE (choose)
        if action is None:
            return CycleReport(status="no_action", variant=current)
        optimizer = self.optimizers[action.scope]
        base_artifact = current.resolve(action.scope, default=optimizer.default_artifact())
        new_artifact = optimizer.estimate(base_artifact, self.corpus, self.benchmark, current)
        candidate = current.with_artifact(action.scope, new_artifact)  # 4. IMPROVE (apply)

        _, candidate_scores = self._evaluate(candidate)                # 5. EVALUATE (candidate)
        if not offline_improved(scores, candidate_scores):
            return CycleReport(status="rejected_offline", variant=current)

        release = self.deployer.release(candidate, traffic_pct=SMALL_CANARY)  # 6. RELEASE
        ab = self.deployer.ab_compare(baseline=current, candidate=candidate, release=release)  # 7. A/B
        if not ab.candidate_wins:
            self.deployer.rollback(release)
            return CycleReport(status="rejected_ab", variant=current, ab_result=ab)

        self.deployer.launch(release)                                  # 8. LAUNCH
        self.store.save_variant(candidate)
        self.history.append(CycleReport(status="launched", variant=candidate, ab_result=ab))
        return self.history[-1]
```

`agentgym/deployers/offline_deployer.py` — the v0 reference `Deployer`, honestly a simulation:
```python
class OfflineDeployer:
    """Release/A-B/Launch implemented as benchmark replay, not real traffic. Exists so the full
    eight-stage lifecycle is exercisable end-to-end in v0 without requiring a live deployment
    target — clearly not the same claim as a production A/B, and never presented as one."""
    def release(self, variant, traffic_pct): return Release(variant=variant)
    def ab_compare(self, baseline, candidate, release) -> ABResult:
        scores_a = self._run_all(baseline, self.benchmark, self.verifier)
        scores_b = self._run_all(release.variant, self.benchmark, self.verifier)
        delta, ci = bootstrap_delta(scores_a, scores_b)
        return ABResult(delta=delta, ci=ci, candidate_wins=(delta > 0 and not ci.contains_zero))
    def launch(self, release): pass       # v0: no real rollout target to promote to
    def rollback(self, release): pass
```

### Real integration sketches

`agentgym/optimizers/dspy_prompt.py`:
```python
class DSPyPromptOptimizer:
    scope = PROMPT
    depends_on = [TOOLS]                                          # reads the current tool set so
                                                                    #   generated instructions match it
    def estimate(self, artifact: Artifact, corpus: TrainingCorpus, benchmark: Benchmark,
                 variant: AgentVariant) -> Artifact:
        tools = variant.resolve(TOOLS, default=empty_tools_artifact())
        examples = corpus.artifact_candidates(PROMPT)             # real (task, good_trajectory) pairs
        program = build_dspy_module(artifact.value, tools=tools.value)
        compiled = dspy.MIPROv2(metric=make_metric(benchmark)).compile(
            program, trainset=to_dspy_examples(examples))
        return Artifact(scope=PROMPT, shape=ArtifactShape.PROGRAM,
                         value=serialize_dspy_module(compiled), optimizer_binding="dspy.MIPROv2",
                         source="estimated", provenance={"n_examples": len(examples)})
```

`agentgym/optimizers/langgraph_search.py`:
```python
class LangGraphConfigOptimizer:
    scope = GRAPH
    depends_on = []
    def estimate(self, artifact, corpus, benchmark, variant) -> Artifact:
        space = artifact.value["search_space"]                   # declared tunable ranges
        study = optuna.create_study(direction="maximize")
        def objective(trial):
            cfg = sample_config(trial, space)
            graph = build_langgraph_app(cfg)                      # a real LangGraph app
            return run_benchmark_subset(graph, benchmark)          # scored via the same Verifier
        study.optimize(objective, n_trials=20)
        return Artifact(scope=GRAPH, shape=ArtifactShape.CONFIG, value=study.best_params,
                         optimizer_binding="optuna.TPESampler", source="estimated",
                         provenance={"n_trials": 20, "best_score": study.best_value})
```

`agentgym/optimizers/unsloth_lora.py` — real spend, flagged again below:
```python
class UnslothLoRAOptimizer:
    scope = Scope.MODEL_WEIGHTS
    technique = FineTuneTechnique.LORA_ADAPTER
    depends_on = []
    def estimate(self, artifact, corpus, benchmark, variant) -> Artifact:
        trajectories = corpus.sft_rows()       # native trajectory shape: state/action/observation/
                                                #   reward per step, not yet chat-messages
        sft_rows = to_chat_messages(trajectories)   # real, lossy, named conversion — confirm the
                                                      #   current TRL/Unsloth tool-call chat-template
                                                      #   convention at implementation time
        run_id = launch_cloud_gpu_job(                             # stops for sign-off, see checkpoint below
            script="agentgym/optimizers/_unsloth_train.py",
            data=sft_rows, base_model=artifact.value.get("base_model"), lora_rank=16)
        adapter_ref = wait_and_fetch_adapter(run_id)
        return Artifact(scope=Scope.MODEL_WEIGHTS, shape=ArtifactShape.WEIGHTS, value=adapter_ref,
                         technique=FineTuneTechnique.LORA_ADAPTER, optimizer_binding="unsloth",
                         source="estimated", provenance={"run_id": run_id, "n_rows": len(sft_rows)})
```
`_unsloth_train.py` runs on the provisioned instance: loads the base model via Unsloth's
`FastLanguageModel`, applies the LoRA config, trains via `SFTTrainer` on `sft_rows`, saves the
adapter, reports metrics back — a real training job, not a mocked return value.

`agentgym/verifiers/inspect_adapter.py` — one open design question, flagged rather than guessed:
Inspect AI is normally the one *driving* execution (`inspect eval`), and `Harness` is also the one
driving execution (`Agent.run()`, inside `_evaluate`). Two ways to reconcile, resolved by reading
Inspect's actual current API during implementation rather than assuming:
  (a) treat Inspect as a **scoring library only** — `Harness` drives `Agent.run()`, then hands the
      resulting trace to an Inspect scorer. Simpler, keeps `Harness` as the one orchestrator.
      Current lean.
  (b) wrap `Agent` as an Inspect solver and let `inspect eval` drive execution, with `Harness`'s
      diagnose/improve/deploy stages running as a post-processing pass over Inspect's eval logs.

`agentgym/benchmarks/tau2_bench.py` — first real implementation task: read the benchmark's actual
current task/scenario format and confirm the conversion into `Task`, rather than assume
compatibility.

## Scope v0 — a real, minimal, vertical proof

One scope per lever, each with a genuinely working reference optimizer — enough to prove every
`ArtifactShape` (`PROGRAM`, `CONFIG`, `WEIGHTS`) and every lever is real, without yet covering
every scope:

- Full `agentgym/core/` — every type and protocol, complete.
- **Context: `Scope.PROMPT`** via `DSPyPromptOptimizer` — real MIPROv2 search.
- **Harness: `Scope.GRAPH`** via `LangGraphConfigOptimizer` — a real LangGraph app, real Optuna
  search.
- **Fine-tune: `Scope.MODEL_WEIGHTS`** via `UnslothLoRAOptimizer` — one real, small,
  explicitly-approved LoRA training run on a rented cloud GPU instance. All three scopes run for
  real in v0, narrower in breadth rather than any of them faked.
- `RuleBasedDiagnoser` — distinguishes these three scopes' failure signatures; the full evidence
  table across all eight scopes is a v1 extension.
- `InspectVerifier` + a real tool-use benchmark adapter — real verification, no hand-rolled
  heuristic standing in for one.
- `Harness.cycle()` executing all eight stages end-to-end, one recipe manually selected per run,
  with `OfflineDeployer` standing in for Release/A-B/Launch — a real benchmark-replay comparison,
  clearly labeled as not the same claim as a production A/B.
- `AgentGymStore` (SQLite, thread-safe).
- **Not in v0**: the remaining five scopes (they exist in the type system, no optimizer bound yet),
  the composite autonomous scheduler, a learned diagnoser, RLHF/full-finetune techniques (LoRA is
  the one real fine-tune path in v0), a `Deployer` backed by real production traffic.
- **Success criteria**: `cycle()` completes all eight stages against the real benchmark, and each
  of the three v0 optimizers drives at least one real, measured before/after result with a
  confidence interval through to a `launched` or honestly-`rejected` outcome — not "ran without
  error."

## Path to v1 — two more milestones, each proving a real use case

v0 already proves the first use case: **prompt optimization**, real, via DSPy. Two more milestones
close the gap to v1, each anchored on a real external tool rather than another internal exercise.

### Milestone B — memory/context optimization

Adds `MEMORY` and `RETRIEVAL` for real, moved ahead of the rest of v1's remaining scopes because it
doubles as the clearest proof that the `ScopeOptimizer` interface is genuinely swappable — two
different, independently real tools bound to the same scope:

- `agentgym/optimizers/headroom_memory.py` — `HeadroomMemoryOptimizer` wraps `headroomlabs-ai/
  headroom` (Apache-2.0, confirmed real): compresses tool outputs/RAG chunks/conversation history
  before they reach the model, and its cross-agent `SharedContext` store maps directly onto the
  `MEMORY` scope. `headroom learn` — mining failed sessions into corrections — is a second, natural
  real data source for `TrainingCorpus`, distinct from trace capture.
- `agentgym/optimizers/caveman_memory.py` — an alternative `MEMORY`-scope binding. Two real
  sub-projects solve different parts of this, worth keeping distinct rather than picked between
  blindly: `wilpel/caveman-compression`'s semantic compression technique (strips predictable
  grammar while preserving factual content) is closer to a `CONFIG`/`PROGRAM`-shaped context
  compressor; `JuliusBrussee/caveman`'s sibling **cavemem** project is closer to cross-session
  memory retention, though it's thinly documented on the main repo and needs its own read before
  committing to an integration shape. First real implementation task: read `cavemem` directly and
  decide which of the two caveman sub-projects this binding actually targets, rather than
  conflating them.
- **Proof scenario**: reuses the context-rot failure pattern already established as a real,
  recognizable agent failure — a stale early detail outweighing a corrected constraint. Diagnose
  routes it to `MEMORY`; both `HeadroomMemoryOptimizer` and `caveman_memory`'s binding get run
  against the same failure corpus and the same benchmark, and their `ABResult`s are directly
  comparable — the same scope, two real competing tools, one shared measurement. That comparability
  *is* the proof the interface standardization works, not just that either tool works alone.

### Milestone C — agentic security

Adds `GUARDRAILS` (under harness, added to the `Scope` set above) for real:

- `agentgym/optimizers/security_guardrails.py` — `GuardrailsOptimizer`: runs garak (confirmed real,
  open-source LLM vulnerability scanner) and promptfoo's red-team probes against the current agent,
  and produces a `GUARDRAILS` artifact — a policy config close to what Invariant Labs' open-source
  `invariant` project (confirmed real and active, Apache-2.0, now operating under Snyk Labs)
  expresses as its own guardrail rules: PII/secret/prompt-injection/jailbreak detection and
  response.
- `agentgym/verifiers/security_verifier.py` — a `Verifier` that re-runs the garak/promptfoo probe
  suite as its scoring mechanism, composed via `CompositeVerifier` alongside the main task-
  correctness verifier — the benchmark's task set doesn't fork, but every trace now carries both
  "did it complete the task" and "could it be attacked" scores.
- **Proof scenario**: onboard a fresh agent with no guardrail artifact bound (a real case of an
  unbound scope, exercising the default-fallback path in `Agent.consumes()`), run the probe suite
  as the first Evaluate stage, diagnose the vulnerability pattern, estimate a guardrail policy,
  re-run the same probes as the second Evaluate stage, and report the real reduction in successful
  probes through the same Release/A-B/Launch stages as any other scope — security config earns its
  way to production the same way a prompt or a graph topology does, not through a separate
  special-cased path.

## Scope v1 — full breadth

Builds on Milestones A/B/C:

- Real optimizers for the remaining three scopes not yet covered — tools, model routing,
  interrupt/escalation — reusing the config-search machinery where the shape fits.
- All three named recipes independently functional, plus the composite autonomous scheduler that
  picks across every scope using accumulated A/B history, not a human choosing one recipe per run.
- RLHF added alongside LoRA for the fine-tune lever, via a real PPO/GRPO training path, same
  cloud-GPU sign-off discipline as v0.
- `RuleBasedDiagnoser` extended to the full evidence table across every scope, including the
  security-specific evidence patterns Milestone C surfaces.
- A `Deployer` backed by real (or realistically staged) traffic, replacing `OfflineDeployer` —
  the honest boundary v0 draws around itself. This is likely the single largest v1 item, since it's
  real infrastructure, not just another optimizer.
- Stretch, can slip past v1: a first-pass learned diagnoser.

## Repo setup

New repository at `/Users/debjyotipaul/projects/RL-agents/agentgym/`. `pyproject.toml`, MIT
license, Python 3.12. `providers/llm_adapter.py` wires all provider keys already present in
`RL-agents/.env` (Ollama, OpenAI, Bedrock, Anthropic, Groq).

## Verification

- `pytest` over `tests/` — protocol conformance (a minimal implementation satisfies each `Protocol`
  structurally), training-corpus row-shape tests, artifact/scope-binding tests, a `PROGRAM`-shaped
  artifact composition test (nesting artifacts recursively).
- v0: an end-to-end run against the real benchmark, with the three-optimizer real-delta success
  criteria above.
- v1: the same run with the composite scheduler making its own scope choices across a longer run,
  plus each named recipe independently verified to drive at least one real iteration on its own.
- Cost/compute checkpoint, both v0 and v1: before any cloud GPU instance is provisioned, stop and
  confirm instance type, estimated cost, and estimated duration. `scripts/provision_gpu.py` is the
  only place this happens, and it is never invoked automatically by the run loop.
