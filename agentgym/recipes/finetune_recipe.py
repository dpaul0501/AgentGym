"""FineTuneRecipe: a real LearningRecipe for Lever.FINE_TUNE, structurally different from
ContextSearchRecipe/HarnessSearchRecipe on purpose. There's only one scope to propose
(MODEL_WEIGHTS), so "search" isn't the right shape — each attempt is a real, costly training run
(a rented GPU, real wall-clock time), so this recipe caps how many times it will propose
MODEL_WEIGHTS at all (max_attempts), and stops proposing it the moment one attempt actually
launches — retraining again without new failure evidence to justify it just burns money for no
reason. This is the propose -> apply -> run -> keep/revert loop the design calls out, with Harness
itself supplying apply/run/keep-revert; this recipe's only job is deciding whether to propose
again.
"""

from __future__ import annotations

from agentgym.core.protocols import Action
from agentgym.core.scope import MODEL_WEIGHTS


class FineTuneRecipe:
    def __init__(self, max_attempts: int = 3):
        self.max_attempts = max_attempts

    def next_action(self, report_history: list, variant) -> Action | None:
        attempts = [r for r in report_history if r.action is not None and r.action.scope == MODEL_WEIGHTS]
        if any(r.status == "launched" for r in attempts):
            return None
        if len(attempts) >= self.max_attempts:
            return None
        return Action(scope=MODEL_WEIGHTS)
