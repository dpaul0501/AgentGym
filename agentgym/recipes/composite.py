"""CompositeRecipe: the v1 default scheduler. Sequences the three named recipes cheapest-lever-
first — context (prompt/memory/tool search, cheap and fast) before harness (config/topology
search, medium) before fine-tune (real GPU training runs, expensive) — falling through to the
next recipe only once the current one has nothing left to propose. All real decision logic stays
in the three named recipes; this class's only job is picking which recipe gets to propose next.
"""

from __future__ import annotations

from agentgym.core.protocols import Action


class CompositeRecipe:
    def __init__(self, context_recipe, harness_recipe, finetune_recipe):
        self.recipes_cheapest_first = [context_recipe, harness_recipe, finetune_recipe]

    def next_action(self, report_history: list, variant) -> Action | None:
        for recipe in self.recipes_cheapest_first:
            action = recipe.next_action(report_history, variant)
            if action is not None:
                return action
        return None
