#!/usr/bin/env python3
"""Orchestrator/harness lever demo: garak audits the real local model backing the agent, real
GuardrailsOptimizer produces a tightened policy from that audit, and the harness enforces it live
— the malicious tool-output content is deterministically stripped before it ever reaches the
model's context, independent of whether the model would have resisted it anyway (Sonnet did, in
manual testing; you should not have to rely on that).

Live-demo note: this script uses garak's fast `test.Blank` smoke probe by default so the live run
stays quick (~15-20s, confirmed by timing). A real, thorough audit (e.g.
`latentinjection.LatentInjectionResume`, 256 real prompts against this same local model) takes
several minutes — run separately, ahead of time, as the presentation's supporting evidence; see
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

OLLAMA_MODEL = "qwen2.5:7b"


def main() -> None:
    task = next(t for t in TravelBenchmark().cases() if t.task_id == "hotel_search_paris")
    agent = TravelSageAgent(llm=build_ollama_llm(OLLAMA_MODEL))

    print(f"=== BEFORE: GUARDRAILS unbound, agent backed by local {OLLAMA_MODEL} ===")
    before_trace = agent.run(task, {GUARDRAILS: None})
    tool_span = next(s for s in before_trace.spans if s.kind == "tool")
    print(f"Raw tool output reaching the model's context (first 200 chars):\n  {str(tool_span.output)[:200]}")
    print(f"\nFinal response:\n{before_trace.metadata['final_response']}\n")

    print(f"=== AUDIT: real garak probe against the real local {OLLAMA_MODEL} target ===")
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

    print(f"\n=== AFTER: GUARDRAILS bound to the policy produced above ===")
    after_trace = agent.run(task, {GUARDRAILS: guardrails_artifact})
    tool_span_after = next(s for s in after_trace.spans if s.kind == "tool")
    print(f"Tool output actually delivered to the model (first 200 chars):\n  {str(tool_span_after.output)[:200]}")
    print(f"\nFinal response:\n{after_trace.metadata['final_response']}")


if __name__ == "__main__":
    sys.exit(main())
