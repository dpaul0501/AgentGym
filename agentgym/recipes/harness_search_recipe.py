"""HarnessSearchRecipe: a real LearningRecipe that auto-searches Lever.HARNESS's scopes
(MODEL_ROUTING, INTERRUPT, GRAPH, GUARDRAILS), proposing each scope that has a registered
optimizer exactly once, in a fixed order, then stopping. Same history-derived, no-internal-state
design as ContextSearchRecipe — the two are deliberately structurally identical since the
scope-cycling strategy itself doesn't depend on which lever it's applied to; what differs is only
the candidate scope list.
"""

from __future__ import annotations

from agentgym.core.protocols import Action
from agentgym.core.scope import GRAPH, GUARDRAILS, INTERRUPT, MODEL_ROUTING

CANDIDATE_SCOPES = [MODEL_ROUTING, INTERRUPT, GRAPH, GUARDRAILS]


class HarnessSearchRecipe:
    def __init__(self, optimizers: dict):
        self.candidate_scopes = [s for s in CANDIDATE_SCOPES if s in optimizers]

    def next_action(self, report_history: list, variant) -> Action | None:
        already_tried = {r.action.scope for r in report_history if r.action is not None}
        for scope in self.candidate_scopes:
            if scope not in already_tried:
                return Action(scope=scope)
        return None
