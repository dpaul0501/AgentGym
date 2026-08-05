"""RuleBasedDiagnoser: the v1 reference Diagnoser — an ordered evidence table across all nine
scopes currently defined in agentgym.core.scope. Each rule below checks a real, structural signal
in the trace (a tool-call error, a repeated-action loop, an empty retrieval, an oversized context,
a failed guardrail score, a missing escalation) rather than guessing from the reward number alone.
Rules are checked in order of how directly diagnostic their signal is — a guardrail violation is
unambiguous (confidence 0.95); "nothing structural fired, so it's probably the model's own
capability" is the least specific signal and sits last (confidence 0.35), used only once every
narrower explanation has been ruled out.

This is a real, swappable reference implementation, not the only possible one — the whole point of
Diagnoser being a Protocol (agentgym/core/protocols.py) is that a learned diagnoser can replace
this later without touching Harness.
"""

from __future__ import annotations

from agentgym.core.diagnosis import Diagnosis
from agentgym.core.scope import (
    GRAPH,
    GUARDRAILS,
    INTERRUPT,
    MEMORY,
    MODEL_ROUTING,
    MODEL_WEIGHTS,
    PROMPT,
    RETRIEVAL,
    TOOLS,
)
from agentgym.core.score import Score, primary_reward
from agentgym.core.trace import Trace

FAIL_THRESHOLD = 0.7
SEVERE_FAIL_THRESHOLD = 0.2
CONTEXT_BLOAT_CHARS = 4000
LOOP_REPEAT_COUNT = 3
ESCALATION_SPAN_NAMES = {"escalate", "human_review", "request_human", "human_handoff"}


def _guardrail_violation(trace: Trace, scores: list[Score]) -> Score | None:
    for s in scores:
        if ("injection" in s.metric_name or "guardrail" in s.metric_name or "jailbreak" in s.metric_name) and s.value < 1.0:
            return s
    return None


def _tool_error_span(trace: Trace):
    for span in trace.spans:
        if span.kind != "tool":
            continue
        if span.attributes.get("error") or (isinstance(span.output, dict) and span.output.get("error")):
            return span
    return None


def _missing_escalation(trace: Trace) -> bool:
    if not trace.metadata.get("requires_human"):
        return False
    return not any(span.name in ESCALATION_SPAN_NAMES for span in trace.spans)


def _repeated_action_loop(trace: Trace) -> str | None:
    names = [span.name for span in trace.spans if span.kind == "tool"]
    run_name, run_len = None, 0
    for name in names:
        if name == run_name:
            run_len += 1
        else:
            run_name, run_len = name, 1
        if run_len >= LOOP_REPEAT_COUNT:
            return run_name
    return None


def _empty_retrieval_span(trace: Trace):
    for span in trace.spans:
        if span.kind == "retrieval" and not span.output:
            return span
    return None


def _context_size_chars(trace: Trace) -> int:
    return sum(len(str(span.input)) + len(str(span.output)) for span in trace.spans)


def _routing_mismatch(trace: Trace) -> bool:
    expected = trace.metadata.get("expected_model_tier")
    if expected is None:
        return False
    return any(
        span.attributes.get("routed_model_tier") not in (None, expected) for span in trace.spans
    )


class RuleBasedDiagnoser:
    def diagnose(self, trace: Trace, scores: list[Score]) -> Diagnosis:
        if not scores:
            return Diagnosis(
                trace_id=trace.trace_id, lever=None, evidence=["no scores available to diagnose"],
                confidence=0.0, suggested_scope=None,
            )

        reward = primary_reward(scores)

        violation = _guardrail_violation(trace, scores)
        if violation is not None:
            return Diagnosis(
                trace_id=trace.trace_id, lever=GUARDRAILS.lever,
                evidence=[f"security score '{violation.metric_name}'={violation.value}: {violation.evidence}"],
                confidence=0.95, suggested_scope=GUARDRAILS,
            )

        if reward >= FAIL_THRESHOLD:
            return Diagnosis(
                trace_id=trace.trace_id, lever=None,
                evidence=[f"primary_reward={reward} >= {FAIL_THRESHOLD}; no failure to diagnose"],
                confidence=0.0, suggested_scope=None,
            )

        tool_error = _tool_error_span(trace)
        if tool_error is not None:
            return Diagnosis(
                trace_id=trace.trace_id, lever=TOOLS.lever,
                evidence=[f"tool span '{tool_error.name}' reported an error"],
                confidence=0.85, suggested_scope=TOOLS,
            )

        if _missing_escalation(trace):
            return Diagnosis(
                trace_id=trace.trace_id, lever=INTERRUPT.lever,
                evidence=["task metadata requires_human=True but no escalation span was found"],
                confidence=0.8, suggested_scope=INTERRUPT,
            )

        looped_action = _repeated_action_loop(trace)
        if looped_action is not None:
            return Diagnosis(
                trace_id=trace.trace_id, lever=GRAPH.lever,
                evidence=[f"action '{looped_action}' repeated >= {LOOP_REPEAT_COUNT} times consecutively"],
                confidence=0.75, suggested_scope=GRAPH,
            )

        empty_retrieval = _empty_retrieval_span(trace)
        if empty_retrieval is not None:
            return Diagnosis(
                trace_id=trace.trace_id, lever=RETRIEVAL.lever,
                evidence=[f"retrieval span '{empty_retrieval.name}' returned no results"],
                confidence=0.7, suggested_scope=RETRIEVAL,
            )

        context_chars = _context_size_chars(trace)
        if context_chars > CONTEXT_BLOAT_CHARS:
            return Diagnosis(
                trace_id=trace.trace_id, lever=MEMORY.lever,
                evidence=[f"trace context size {context_chars} chars > {CONTEXT_BLOAT_CHARS} bloat threshold"],
                confidence=0.65, suggested_scope=MEMORY,
            )

        if _routing_mismatch(trace):
            return Diagnosis(
                trace_id=trace.trace_id, lever=MODEL_ROUTING.lever,
                evidence=["a span's routed_model_tier did not match task metadata's expected_model_tier"],
                confidence=0.6, suggested_scope=MODEL_ROUTING,
            )

        has_llm_span = any(span.kind == "llm" for span in trace.spans)
        if has_llm_span and reward > SEVERE_FAIL_THRESHOLD:
            return Diagnosis(
                trace_id=trace.trace_id, lever=PROMPT.lever,
                evidence=[
                    f"no structural harness/context defect found; reward={reward} is a partial "
                    "failure, consistent with an instruction-following gap"
                ],
                confidence=0.5, suggested_scope=PROMPT,
            )

        return Diagnosis(
            trace_id=trace.trace_id, lever=MODEL_WEIGHTS.lever,
            evidence=[
                f"no structural harness/context defect found; reward={reward} is a severe "
                f"failure (<= {SEVERE_FAIL_THRESHOLD}), consistent with a capability gap"
            ],
            confidence=0.35, suggested_scope=MODEL_WEIGHTS,
        )
