"""ContextSearchRecipe: a real LearningRecipe that auto-searches Lever.CONTEXT's scopes (PROMPT,
TOOLS, MEMORY, RETRIEVAL), proposing each scope that has a registered optimizer exactly once, in a
fixed order, then stopping. Reads report_history to see which scopes were already attempted this
run (via CycleReport.action) rather than tracking its own internal counter — so a fresh
ContextSearchRecipe instance handed the same Harness history behaves identically to one that ran
from the start, matching how a real recipe should be re-derivable from history, not
instance-local state.
"""

from __future__ import annotations

from agentgym.core.protocols import Action
from agentgym.core.scope import MEMORY, PROMPT, RETRIEVAL, TOOLS

CANDIDATE_SCOPES = [PROMPT, TOOLS, MEMORY, RETRIEVAL]


class ContextSearchRecipe:
    def __init__(self, optimizers: dict):
        self.candidate_scopes = [s for s in CANDIDATE_SCOPES if s in optimizers]

    def next_action(self, report_history: list, variant) -> Action | None:
        already_tried = {r.action.scope for r in report_history if r.action is not None}
        for scope in self.candidate_scopes:
            if scope not in already_tried:
                return Action(scope=scope)
        return None
