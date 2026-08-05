#!/usr/bin/env python3
"""Context/memory lever demo: run the flight-search task for real against Bedrock Claude with no
MEMORY artifact bound (raw, uncompressed tool output reaching the model — this deck's own "40
flight results pasted in full" framing, with real numbers instead of illustrative ones); capture
the trace into a real TrainingCorpus; run HeadroomMemoryOptimizer for real to search a compression
config; measure the real token savings that config achieves on the same content via headroom's own
compress(); re-run the same task with the winning MEMORY artifact bound to confirm the agent still
answers correctly with the compressed context.
"""

from __future__ import annotations

import sys

from headroom.compress import CompressConfig, compress

from agentgym.core.artifact import Artifact, ArtifactShape
from agentgym.core.corpus import TrainingCorpus
from agentgym.core.diagnosis import Diagnosis
from agentgym.core.scope import MEMORY
from agentgym.core.score import Score
from agentgym.optimizers.headroom_memory import HeadroomMemoryOptimizer
from examples.travel_sage_agent.agent import TravelSageAgent
from examples.travel_sage_agent.benchmark import TravelBenchmark

CONTEXT_BLOAT_CHARS = 4000  # same threshold agentgym.diagnosers.rule_based.RuleBasedDiagnoser uses


def main() -> None:
    task = next(t for t in TravelBenchmark().cases() if t.task_id == "flight_search_sf_nyc")
    agent = TravelSageAgent()

    print("=== BEFORE: MEMORY unbound — raw tool output reaches the model ===")
    before_trace = agent.run(task, {MEMORY: None})
    tool_span = next(s for s in before_trace.spans if s.kind == "tool")
    raw_output = tool_span.output
    context_chars = len(str(tool_span.input)) + len(str(raw_output))
    print(f"Real context size from this tool call: {context_chars} chars "
          f"({'exceeds' if context_chars > CONTEXT_BLOAT_CHARS else 'under'} the "
          f"{CONTEXT_BLOAT_CHARS}-char bloat threshold RuleBasedDiagnoser uses)")

    corpus = TrainingCorpus()
    corpus.add_trace(
        before_trace,
        [Score(metric_name="task_success", value=1.0, kind="verifiable", evidence="answered")],
        Diagnosis(
            trace_id=before_trace.trace_id, lever=MEMORY.lever, suggested_scope=MEMORY,
            confidence=0.65, evidence=[f"trace context size {context_chars} chars > {CONTEXT_BLOAT_CHARS}"],
        ),
    )

    print("\n=== SEARCH: real HeadroomMemoryOptimizer.estimate() over real CompressConfig variants ===")
    optimizer = HeadroomMemoryOptimizer(n_trials=10, seed=0)
    starting_artifact = Artifact(
        scope=MEMORY, shape=ArtifactShape.CONFIG, value={}, optimizer_binding=None,
        technique=None, source="manual",
    )
    memory_artifact = optimizer.estimate(starting_artifact, corpus, benchmark=None, variant=None)
    print(f"Winning config: {memory_artifact.value}")
    print(f"Provenance: {memory_artifact.provenance}")

    print("\n=== MEASURED: real headroom.compress() on the same tool output with the winning config ===")
    config = CompressConfig(
        compress_user_messages=memory_artifact.value["compress_user_messages"],
        protect_recent=memory_artifact.value["protect_recent"],
        min_tokens_to_compress=memory_artifact.value["min_tokens_to_compress"],
    )
    messages = [{"role": "tool", "content": str(raw_output)}]
    result = compress(messages, model=memory_artifact.value["model"], config=config)
    print(f"tokens_before={result.tokens_before} tokens_after={result.tokens_after} "
          f"tokens_saved={result.tokens_saved} ratio={result.compression_ratio:.2%}")

    print("\n=== AFTER: MEMORY bound — re-run the same task with the winning config applied live ===")
    after_trace = agent.run(task, {MEMORY: memory_artifact})
    print(f"Final response:\n{after_trace.metadata['final_response']}")


if __name__ == "__main__":
    sys.exit(main())
