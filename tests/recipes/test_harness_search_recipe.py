from agentgym.core.harness import CycleReport
from agentgym.core.protocols import Action
from agentgym.core.scope import GRAPH, GUARDRAILS, INTERRUPT, MODEL_ROUTING
from agentgym.recipes.harness_search_recipe import HarnessSearchRecipe


def _launched_report(scope):
    return CycleReport(status="launched", variant=None, action=Action(scope=scope))


def test_proposes_scopes_with_registered_optimizers_only():
    recipe = HarnessSearchRecipe(optimizers={GRAPH: object(), GUARDRAILS: object()})
    assert recipe.candidate_scopes == [GRAPH, GUARDRAILS]


def test_proposes_first_untried_candidate_scope():
    recipe = HarnessSearchRecipe(
        optimizers={MODEL_ROUTING: object(), INTERRUPT: object(), GRAPH: object(), GUARDRAILS: object()}
    )
    action = recipe.next_action(report_history=[], variant=None)
    assert action == Action(scope=MODEL_ROUTING)


def test_skips_scopes_already_tried_in_history():
    recipe = HarnessSearchRecipe(optimizers={MODEL_ROUTING: object(), GRAPH: object()})
    history = [_launched_report(MODEL_ROUTING)]
    action = recipe.next_action(report_history=history, variant=None)
    assert action == Action(scope=GRAPH)


def test_returns_none_once_every_candidate_tried():
    recipe = HarnessSearchRecipe(optimizers={GRAPH: object()})
    history = [_launched_report(GRAPH)]
    assert recipe.next_action(report_history=history, variant=None) is None
