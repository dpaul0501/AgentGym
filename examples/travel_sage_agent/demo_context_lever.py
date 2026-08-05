#!/usr/bin/env python3
"""Context/memory lever demo: a tracked metric across multiple trials, not a single anecdote.
Four independent flight-search tasks (different routes), each producing real context bloat from
the mock flight-search tool (this deck's own "40 flight results pasted in full" framing, with
measured numbers instead of illustrative ones). The metric is total context size across all four
trials, measured before MEMORY is bound (baseline) and after (post-optimization) — plus a real
correctness check (does the response still name real flights/prices) on every trial, both before
and after, so the win is "less context, same correctness across N sessions," not "it did something
different once."

HeadroomMemoryOptimizer trains on the combined corpus from all four baseline trials (more
realistic than a single trace) and searches real headroom.CompressConfig variants; the winning
config is then applied live across the same four trials to measure the after state.
"""

from __future__ import annotations

import sys

from agentgym.core.artifact import Artifact, ArtifactShape
from agentgym.core.corpus import TrainingCorpus
from agentgym.core.diagnosis import Diagnosis
from agentgym.core.scope import MEMORY
from agentgym.core.score import Score
from agentgym.optimizers.headroom_memory import HeadroomMemoryOptimizer
from examples.travel_sage_agent.agent import TravelSageAgent
from examples.travel_sage_agent.benchmark import TravelBenchmark
from examples.travel_sage_agent.tools import AIRLINES

CONTEXT_BLOAT_CHARS = 4000  # same threshold agentgym.diagnosers.rule_based.RuleBasedDiagnoser uses


def _flight_span(trace):
    """The agent doesn't always call tools in the same order — find search_flights specifically
    rather than assuming it's the first tool-kind span."""
    return next((s for s in trace.spans if s.kind == "tool" and s.name == "search_flights"), None)


def _context_chars(trace) -> int:
    span = _flight_span(trace)
    if span is None:
        return 0
    return len(str(span.input)) + len(str(span.output))


def _looks_correct(trace) -> bool:
    response = trace.metadata["final_response"]
    return "$" in response and any(airline in response for airline in AIRLINES)


def main() -> None:
    tasks = TravelBenchmark().flight_search_cases()
    agent = TravelSageAgent()
    corpus = TrainingCorpus()

    print(f"=== BEFORE: MEMORY unbound, {len(tasks)} independent flight-search trials ===")
    before_chars, before_correct = [], 0
    for task in tasks:
        trace = agent.run(task, {MEMORY: None})
        chars = _context_chars(trace)
        before_chars.append(chars)
        correct = _looks_correct(trace)
        before_correct += correct
        print(f"  [{task.metadata['origin']} -> {task.metadata['destination']}] "
              f"{chars} chars, response {'looks correct' if correct else 'MISSING flight details'}")
        corpus.add_trace(
            trace,
            [Score(metric_name="task_success", value=1.0 if correct else 0.0, kind="verifiable",
                   evidence="mentions a real airline and price" if correct else "no flight details found")],
            Diagnosis(
                trace_id=trace.trace_id, lever=MEMORY.lever, suggested_scope=MEMORY, confidence=0.65,
                evidence=[f"trace context size {chars} chars > {CONTEXT_BLOAT_CHARS}"],
            ),
        )
    print(f"Metric: {sum(before_chars)} total chars, {before_correct}/{len(tasks)} correct\n")

    print("=== SEARCH: HeadroomMemoryOptimizer.estimate() over CompressConfig variants "
          f"(trained on all {len(tasks)} trials) ===")
    optimizer = HeadroomMemoryOptimizer(n_trials=10, seed=0)
    starting_artifact = Artifact(
        scope=MEMORY, shape=ArtifactShape.CONFIG, value={}, optimizer_binding=None,
        technique=None, source="manual",
    )
    memory_artifact = optimizer.estimate(starting_artifact, corpus, benchmark=None, variant=None)
    print(f"Winning config: {memory_artifact.value}")
    print(f"Provenance: {memory_artifact.provenance}\n")

    print(f"=== AFTER: MEMORY bound, same {len(tasks)} trials re-run live ===")
    after_chars, after_correct = [], 0
    for task in tasks:
        trace = agent.run(task, {MEMORY: memory_artifact})
        chars = _context_chars(trace)
        after_chars.append(chars)
        correct = _looks_correct(trace)
        after_correct += correct
        print(f"  [{task.metadata['origin']} -> {task.metadata['destination']}] "
              f"{chars} chars, response {'looks correct' if correct else 'MISSING flight details'}")
    print(f"Metric: {sum(after_chars)} total chars, {after_correct}/{len(tasks)} correct\n")

    reduction = 1 - (sum(after_chars) / sum(before_chars))
    print(f"=== RESULT: context size {sum(before_chars)} -> {sum(after_chars)} chars "
          f"({reduction:.1%} reduction), correctness {before_correct}/{len(tasks)} -> "
          f"{after_correct}/{len(tasks)} ===")


if __name__ == "__main__":
    sys.exit(main())
