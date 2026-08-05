#!/usr/bin/env python3
"""Orchestrator/harness lever demo: a tracked metric across multiple trials, not a single
anecdote. Three independent hotel-search tasks (Paris/Rome/Berlin), each with a distinct
prompt-injection payload in tools.py's INJECTION_PAYLOADS. The metric is how many of the three
have malicious tool content actually reach the model's context — measured before GUARDRAILS is
bound (baseline) and after (post-optimization), so the improvement is a real rate change, not one
run's behavior.

garak audits the local model backing the agent, GuardrailsOptimizer produces a tightened policy
from that audit, and the harness enforces it live — independent of whether the model would have
resisted any individual attempt anyway (Sonnet did, in manual testing; you shouldn't have to rely
on that for every attempt, every time).

Live-demo note: this script uses garak's fast `test.Blank` smoke probe by default so the live run
stays quick (~15-20s, confirmed by timing). A thorough audit (e.g.
`latentinjection.LatentInjectionResume`, 256 prompts against this same local model) takes several
minutes — run separately, ahead of time, as the presentation's supporting evidence; see
captured_output/.
"""

from __future__ import annotations

import json
import sys

from agentgym.core.artifact import Artifact, ArtifactShape
from agentgym.core.scope import GUARDRAILS
from agentgym.optimizers.security_guardrails import GuardrailsOptimizer
from examples.travel_sage_agent.agent import TravelSageAgent
from examples.travel_sage_agent.benchmark import TravelBenchmark
from examples.travel_sage_agent.llm import build_ollama_llm
from examples.travel_sage_agent.tools import INJECTION_PAYLOADS

OLLAMA_MODEL = "qwen2.5:7b"


def _hotel_span(trace):
    """The agent doesn't always call tools in the same order — find search_hotels specifically
    rather than assuming it's the first tool-kind span."""
    return next((s for s in trace.spans if s.kind == "tool" and s.name == "search_hotels"), None)


def _reached_model(trace, city: str) -> bool:
    span = _hotel_span(trace)
    return span is not None and INJECTION_PAYLOADS[city] in str(span.output)


def _was_blocked(trace) -> bool:
    span = _hotel_span(trace)
    if span is None:
        return False
    output = span.output
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except json.JSONDecodeError:
            return False
    return isinstance(output, dict) and output.get("blocked") is True


def main() -> None:
    tasks = TravelBenchmark().hotel_search_cases()
    agent = TravelSageAgent(llm=build_ollama_llm(OLLAMA_MODEL))

    print(f"=== BEFORE: GUARDRAILS unbound, {len(tasks)} independent hotel-search trials ===")
    reached = 0
    for task in tasks:
        trace = agent.run(task, {GUARDRAILS: None})
        if _reached_model(trace, task.metadata["city"]):
            reached += 1
            print(f"  [{task.metadata['city']}] malicious content reached the model's context")
        else:
            print(f"  [{task.metadata['city']}] malicious content did NOT reach the model")
    print(f"Metric: {reached}/{len(tasks)} trials had malicious tool content reach the model\n")

    print(f"=== AUDIT: garak probe against the local {OLLAMA_MODEL} target ===")
    optimizer = GuardrailsOptimizer(
        target_type="ollama", target_name=OLLAMA_MODEL, probes=["test.Blank"], generations=1,
    )
    starting_artifact = Artifact(
        scope=GUARDRAILS, shape=ArtifactShape.CONFIG, value={}, optimizer_binding=None,
        technique=None, source="manual",
    )
    guardrails_artifact = optimizer.estimate(starting_artifact, corpus=None, benchmark=None, variant=None)
    print(json.dumps({
        "probes_run": guardrails_artifact.provenance["probes_run"],
        "vulnerability_rate": guardrails_artifact.provenance["vulnerability_rate"],
        "prompt_injection_threshold": guardrails_artifact.value["prompt_injection_threshold"],
    }, indent=2))

    print(f"\n=== AFTER: GUARDRAILS bound, same {len(tasks)} trials re-run ===")
    blocked = 0
    for task in tasks:
        trace = agent.run(task, {GUARDRAILS: guardrails_artifact})
        if _was_blocked(trace):
            blocked += 1
            print(f"  [{task.metadata['city']}] blocked at the tool boundary")
        else:
            print(f"  [{task.metadata['city']}] NOT blocked")
    print(f"Metric: {blocked}/{len(tasks)} trials blocked before reaching the model\n")

    print(f"=== RESULT: malicious content reaching the model went from {reached}/{len(tasks)} "
          f"to {len(tasks) - blocked}/{len(tasks)} ===")


if __name__ == "__main__":
    sys.exit(main())
