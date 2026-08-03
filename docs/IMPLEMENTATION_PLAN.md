# Implementation Plan — class by class, test-driven

Companion to `docs/DESIGN.md`. Scope: everything needed for v0 (the core protocol suite plus the
`PROMPT`/`GRAPH`/`MODEL_WEIGHTS` reference optimizers), in build order. Every class below gets its
tests written first — red, then the minimum implementation to go green, then refactor — not tests
retrofitted after the fact.

## Rules this plan holds itself to

- One test file per module (`tests/core/test_scope.py` for `agentgym/core/scope.py`, etc.), so a
  failing test always points at exactly one file to open.
- Protocols are `@runtime_checkable` so structural conformance can be asserted with `isinstance()`
  in tests — with the caveat that `runtime_checkable` only checks method *presence*, not
  signatures, so conformance tests also call the methods, not just `isinstance()` them.
- No test depends on a live LLM call, a network request, or the cloud GPU path. Reference
  optimizers (`DSPyPromptOptimizer`, `LangGraphConfigOptimizer`, `UnslothLoRAOptimizer`) get their
  own integration tests, gated behind a `-m integration` marker, run separately from the fast unit
  suite that runs on every change.
- Real design decisions this plan pins down that the design doc left implicit — called out inline
  below, each with the test that locks it in:
  1. `Scope` is frozen/hashable (it's used as a dict key throughout).
  2. `Artifact.technique` set on any scope other than `MODEL_WEIGHTS` is a construction-time error,
     not a silently ignored field.
  3. `AgentVariant` is immutable — `with_artifact` returns a new instance, never mutates in place.
  4. `Harness._evaluate` resolving a scope an `Agent.consumes()` but that has neither a bound
     artifact nor a declared default raises `MissingScopeError` — it does not run with a silent gap.
  5. `AgentGymStore` uses thread-local SQLite connections in WAL mode — fixing the old
     single-connection pattern directly, tested with real concurrent writers, not asserted.
  6. `AgentGymRegistry.register_optimizer` raises on overwriting an existing scope binding unless
     `force=True` — silent rebinding is a real footgun, not a convenience worth defaulting to.
  7. `ABResult.candidate_wins` requires both a positive delta *and* a confidence interval that
     excludes zero — a positive point estimate alone is not enough to call a winner.

## `agentgym/core/scope.py` — `Lever`, `Scope`

**Purpose:** the three-lever root and the recursive, namespaced path type everything else keys on.

| Test | Locks in |
|---|---|
| `test_scope_str_is_dotted_path` | `Scope(Lever.CONTEXT, ("prompt",))` → `"context.prompt"` |
| `test_scope_str_nested` | `Scope(Lever.CONTEXT, ("retrieval","reranker"))` → `"context.retrieval.reranker"` |
| `test_is_child_of_true_for_direct_child` | `retrieval.reranker.is_child_of(retrieval)` is `True` |
| `test_is_child_of_false_across_levers` | `harness.graph.is_child_of(context.retrieval)` is `False` |
| `test_is_child_of_false_for_self` | a scope is not a child of itself (equal length segments) |
| `test_scope_equality_by_value` | two `Scope`s with equal `lever`+`segments` are `==` |
| `test_scope_hashable_usable_as_dict_key` | a `dict[Scope, X]` correctly deduplicates/looks up by value |
| `test_scope_is_frozen` | mutating `.segments` after construction raises `FrozenInstanceError` |

## `agentgym/core/artifact.py` — `ArtifactShape`, `FineTuneTechnique`, `Artifact`

**Purpose:** the typed value occupying a scope, with the manual/estimated provenance split.

| Test | Locks in |
|---|---|
| `test_artifact_manual_construction` | `source="manual"` artifact constructs with no provenance required |
| `test_artifact_estimated_requires_provenance` | `source="estimated"` with empty `provenance` raises `ValueError` |
| `test_technique_rejected_on_non_model_weights_scope` | constructing an `Artifact(scope=PROMPT, technique=LORA_ADAPTER, ...)` raises — decision #2 |
| `test_technique_allowed_on_model_weights_scope` | same construction with `scope=MODEL_WEIGHTS` succeeds |
| `test_program_artifact_nests_artifacts` | `Artifact(shape=PROGRAM, value=[Artifact(...), Artifact(...)])` round-trips; nested artifacts are independently inspectable |
| `test_artifact_shape_enum_membership` | all five shapes (`PARAMETER,CONFIG,PROGRAM,ENSEMBLE,WEIGHTS`) exist with the documented string values |

## `agentgym/core/trace.py` — `Span`, `Trace`

**Purpose:** the OTEL GenAI-semconv-aligned capture format.

| Test | Locks in |
|---|---|
| `test_span_construction_all_fields` | basic construction, all fields readable |
| `test_trace_holds_ordered_spans` | `Trace.spans` preserves insertion order |
| `test_otel_roundtrip` | `Trace.from_otel_spans(trace.to_otel())` reconstructs an equivalent `Trace` |
| `test_otel_roundtrip_preserves_parent_child` | `parent_span_id` links survive the round trip |
| `test_span_attributes_accept_arbitrary_otel_keys` | `gen_ai.usage.input_tokens` etc. stored/read with no fixed schema |
| `test_from_otel_spans_rejects_span_with_unknown_trace_id` | a span whose `trace_id` doesn't match the batch raises, rather than silently building a malformed `Trace` |

## `agentgym/core/score.py` — `Score`

| Test | Locks in |
|---|---|
| `test_score_verifiable_construction` | `kind="verifiable"` constructs normally |
| `test_score_plausible_construction` | `kind="plausible"` constructs normally |
| `test_score_invalid_kind_rejected` | any other `kind` string raises at construction |

## `agentgym/core/diagnosis.py` — `Diagnosis`

| Test | Locks in |
|---|---|
| `test_diagnosis_with_scope_requires_lever_match` | `suggested_scope.lever` must equal `lever`, or construction raises — a diagnosis can't name a context scope while claiming the harness lever is at fault |
| `test_insufficient_evidence_has_no_scope` | `lever=None`/`suggested_scope=None` is legal *only* when `evidence` explains why (non-empty) |
| `test_confidence_bounded` | `confidence` outside `[0.0, 1.0]` raises |

## `agentgym/core/corpus.py` — `TrainingCorpus`

| Test | Locks in |
|---|---|
| `test_add_trace_grows_corpus` | `len(corpus)` or an equivalent count increases per `add_trace` |
| `test_artifact_candidates_filters_by_scope` | traces diagnosed to `PROMPT` vs `GRAPH` are correctly separated by `artifact_candidates(scope)` |
| `test_empty_corpus_returns_empty_not_none` | every accessor on a fresh corpus returns `[]`, never `None` |
| `test_sft_rows_native_shape_is_trajectory` | `sft_rows()` entries have `state`/`action`/`observation`/`reward` keys — **not** `role`/`content` — locking in the corrected native-format decision |
| `test_to_chat_messages_converts_trajectory` | the separate, named converter produces a `role`/`content` list from a trajectory row |
| `test_to_chat_messages_preserves_tool_calls` | a trajectory step with a tool call survives conversion as an assistant message with `tool_calls`, plus a `tool`-role message for the observation |
| `test_dpo_pairs_requires_both_chosen_and_rejected` | a trajectory with only a chosen response and no rejected counterpart is excluded from `dpo_pairs()`, not included with a null rejected field |

## `agentgym/core/variant.py` — `AgentVariant`

| Test | Locks in |
|---|---|
| `test_resolve_returns_bound_artifact` | a bound scope resolves to its artifact |
| `test_resolve_falls_back_to_default_when_unbound` | an unbound scope resolves to the caller-supplied default, not an exception |
| `test_with_artifact_returns_new_instance` | `variant.with_artifact(...)` does not mutate `variant`; the original's scopes are unchanged — decision #3 |
| `test_union_across_levers_no_collision` | binding `context.prompt` and `harness.graph` in the same variant, both independently resolvable |
| `test_scopes_under_lever_filters_correctly` | `scopes_under(Lever.CONTEXT)` returns only context-lever bindings |
| `test_resolve_for_returns_all_requested_scopes` | given a `consumes()` list, `resolve_for` returns a dict covering every requested scope |
| `test_resolve_for_raises_missing_scope_error` | a requested scope with no binding and no declared default raises `MissingScopeError` — decision #4, this is the actual Agent↔Scope contract enforcement |

## `agentgym/core/ab.py` — `ABComparison`, `ABResult`

| Test | Locks in |
|---|---|
| `test_positive_delta_and_ci_excludes_zero_wins` | the straightforward win case |
| `test_positive_delta_but_ci_contains_zero_does_not_win` | decision #7 — a positive point estimate with an inconclusive interval is not a win |
| `test_negative_delta_never_wins` | sanity check on the obvious case |
| `test_identical_variant_against_itself_never_wins` | comparing a variant to a copy of itself never reports a significant win — catches a broken delta/CI computation that would otherwise pass silently |
| `test_bootstrap_delta_reproducible_with_seed` | fixed RNG seed produces the same CI across runs — required for test determinism, not just nice-to-have |

## `agentgym/core/protocols.py` — `Agent`, `Benchmark`, `Verifier`, `Diagnoser`, `ScopeOptimizer`, `LearningRecipe`, `Deployer`

| Test | Locks in |
|---|---|
| `test_minimal_agent_satisfies_protocol` | a hand-written class with just `consumes()`/`run()` passes `isinstance(x, Agent)` (runtime-checkable) and both methods are callable with the documented signature |
| `test_minimal_scope_optimizer_satisfies_protocol` | same pattern for `ScopeOptimizer`, including the `scope`/`depends_on` attributes, not just the `estimate()` method |
| `test_minimal_deployer_satisfies_protocol` | same pattern for `Deployer`, covering all four methods (`release`/`ab_compare`/`launch`/`rollback`) |
| *(repeat the same conformance pattern for `Benchmark`, `Verifier`, `Diagnoser`, `LearningRecipe`)* | every protocol has at least one structurally-conforming fake proving it's genuinely implementable without inheriting from `agentgym` code |
| `test_composite_verifier_concatenates_scores` | `CompositeVerifier([verifier_a, verifier_b]).score(trace, task)` returns the union of both verifiers' `Score` lists on the same trace |

## `agentgym/core/registry.py` — `AgentGymRegistry`

| Test | Locks in |
|---|---|
| `test_register_and_get_optimizer` | round-trip register/retrieve for a scope |
| `test_register_optimizer_conflict_raises_without_force` | registering a second optimizer for an already-bound scope raises unless `force=True` — decision #6 |
| `test_register_optimizer_force_overwrites` | `force=True` replaces the existing binding |
| `test_get_missing_scope_raises_clear_error` | looking up an unregistered scope raises a `KeyError` subtype with a message naming the scope, not a bare `KeyError` |

## `agentgym/core/harness.py` — `Harness`, `CycleReport`

The highest-value tests in v0 — everything above is a building block for these. All collaborators
(`Agent`, `Benchmark`, `Verifier`, `Diagnoser`, `ScopeOptimizer`, `LearningRecipe`, `Deployer`,
`AgentGymStore`) are hand-written fakes here, not a mocking library — the protocols are simple
enough that fakes are more honest about what's actually being exercised.

| Test | Locks in |
|---|---|
| `test_cycle_no_action_short_circuits` | when the fake recipe's `next_action` returns `None`, no optimizer/deployer method is ever called, and `status == "no_action"` |
| `test_cycle_evaluates_twice_baseline_then_candidate` | the fake benchmark/verifier are invoked once for `current` and once for the post-improve `candidate` — exactly two `_evaluate` passes, not one |
| `test_cycle_rejects_offline_when_candidate_scores_worse` | a fake `ScopeOptimizer` that deliberately produces a worse artifact results in `status == "rejected_offline"`, and the fake `Deployer`'s `release`/`ab_compare` are **never called** — proves the offline gate actually gates |
| `test_cycle_rejects_ab_and_rolls_back` | offline improves, but the fake `Deployer.ab_compare` reports no win → `status == "rejected_ab"`, `rollback` is called, `launch` is not |
| `test_cycle_launches_on_full_success` | full path to `status == "launched"`, `store.save_variant` called once with the new candidate, not the baseline |
| `test_cycle_appends_report_to_history` | two sequential `cycle()` calls: the second call's recipe receives a `report_history` containing the first call's `CycleReport` |
| `test_evaluate_uses_agent_consumes_not_full_variant` | the fake `Agent.consumes()` declares two scopes out of five bound in the variant; assert `Agent.run()` is called with exactly those two, not all five — proves the contract layer is enforced, not just decorative |
| `test_evaluate_raises_missing_scope_before_calling_agent` | a fake `Agent` that consumes a scope with no binding and no default causes `MissingScopeError` to propagate out of `_evaluate` *before* `Agent.run()` is invoked — fail closed, not after a partial run |

## `agentgym/storage/db.py` — `AgentGymStore`

| Test | Locks in |
|---|---|
| `test_save_and_load_variant_roundtrip` | basic persistence correctness |
| `test_latest_variant_none_when_store_empty` | fresh store returns `None`, not an exception |
| `test_traces_for_scope_filters_correctly` | stored traces filtered by their diagnosed scope |
| `test_concurrent_writes_no_corruption` | N threads each writing M traces via thread-local connections in WAL mode complete with the correct total row count and no `database is locked` errors — the direct regression test for the old single-connection design's known flaw |

## Build order

1. `scope.py` → `artifact.py` (no dependencies on anything else; purely structural)
2. `trace.py`, `score.py`, `diagnosis.py` (independent of each other, can be done in parallel)
3. `corpus.py`, `variant.py`, `ab.py` (depend on the above three)
4. `protocols.py` (depends on everything above — it's the interface layer over all of it)
5. `registry.py`, `storage/db.py` (independent of each other, depend on `protocols.py`/`variant.py`)
6. `harness.py` (depends on everything — built and tested last, against fakes for every collaborator)
7. Only after `harness.py`'s fake-based tests are green: the real reference optimizers
   (`dspy_prompt.py`, `langgraph_search.py`, `unsloth_lora.py`) and their integration tests, which
   are allowed to be slower and to actually call DSPy/LangGraph/a rented GPU — never part of the
   fast suite.
