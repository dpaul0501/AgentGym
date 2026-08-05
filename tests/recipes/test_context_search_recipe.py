from agentgym.core.harness import CycleReport
from agentgym.core.protocols import Action
from agentgym.core.scope import MEMORY, PROMPT, RETRIEVAL, TOOLS
from agentgym.recipes.context_search_recipe import ContextSearchRecipe


def _launched_report(scope):
    return CycleReport(status="launched", variant=None, action=Action(scope=scope))


def test_proposes_scopes_with_registered_optimizers_only():
    recipe = ContextSearchRecipe(optimizers={PROMPT: object(), MEMORY: object()})
    assert recipe.candidate_scopes == [PROMPT, MEMORY]


def test_proposes_first_untried_candidate_scope():
    recipe = ContextSearchRecipe(optimizers={PROMPT: object(), TOOLS: object(), MEMORY: object()})
    action = recipe.next_action(report_history=[], variant=None)
    assert action == Action(scope=PROMPT)


def test_skips_scopes_already_tried_in_history():
    recipe = ContextSearchRecipe(optimizers={PROMPT: object(), TOOLS: object(), MEMORY: object()})
    history = [_launched_report(PROMPT)]
    action = recipe.next_action(report_history=history, variant=None)
    assert action == Action(scope=TOOLS)


def test_returns_none_once_every_candidate_tried():
    recipe = ContextSearchRecipe(optimizers={PROMPT: object(), TOOLS: object()})
    history = [_launched_report(PROMPT), _launched_report(TOOLS)]
    assert recipe.next_action(report_history=history, variant=None) is None


def test_ignores_no_action_reports_when_checking_tried_scopes():
    recipe = ContextSearchRecipe(optimizers={PROMPT: object(), TOOLS: object()})
    history = [CycleReport(status="no_action", variant=None, action=None)]
    action = recipe.next_action(report_history=history, variant=None)
    assert action == Action(scope=PROMPT)
