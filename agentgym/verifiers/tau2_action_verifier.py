"""Tau2ActionVerifier: scores a Trace against tau2-bench's real evaluation_criteria.actions — a
genuine subset of tau2's own reward methodology (the ACTION component of reward_basis), applied
to whatever Agent is under test in Harness, not an invented heuristic."""

from __future__ import annotations

from agentgym.core.protocols import Task
from agentgym.core.score import Score
from agentgym.core.trace import Trace


def _args_match(actual: dict, expected: dict) -> bool:
    return all(actual.get(k) == v for k, v in expected.items())


class Tau2ActionVerifier:
    def score(self, trace: Trace, task: Task) -> list[Score]:
        expected_actions = task.metadata.get("evaluation_criteria", {}).get("actions", [])
        if not expected_actions:
            return [
                Score(
                    metric_name="action_match", value=1.0, kind="verifiable",
                    evidence="task has no evaluation_criteria.actions to check",
                )
            ]

        tool_spans = [s for s in trace.spans if s.kind == "tool"]
        matched = sum(
            1
            for expected in expected_actions
            if any(
                s.name == expected["name"] and _args_match(s.input, expected.get("arguments", {}))
                for s in tool_spans
            )
        )
        value = matched / len(expected_actions)
        return [
            Score(
                metric_name="action_match", value=value, kind="verifiable",
                evidence=f"{matched}/{len(expected_actions)} expected tau2 actions matched",
            )
        ]
