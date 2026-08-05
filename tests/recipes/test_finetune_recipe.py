from agentgym.core.harness import CycleReport
from agentgym.core.protocols import Action
from agentgym.core.scope import MODEL_WEIGHTS, PROMPT
from agentgym.recipes.finetune_recipe import FineTuneRecipe


def _report(status, scope=MODEL_WEIGHTS):
    return CycleReport(status=status, variant=None, action=Action(scope=scope))


def test_proposes_model_weights_when_no_history():
    recipe = FineTuneRecipe(max_attempts=3)
    assert recipe.next_action(report_history=[], variant=None) == Action(scope=MODEL_WEIGHTS)


def test_keeps_proposing_after_rejected_attempts_until_cap():
    recipe = FineTuneRecipe(max_attempts=3)
    history = [_report("rejected_offline"), _report("rejected_ab")]
    action = recipe.next_action(report_history=history, variant=None)
    assert action == Action(scope=MODEL_WEIGHTS)


def test_stops_once_max_attempts_reached_without_a_launch():
    recipe = FineTuneRecipe(max_attempts=2)
    history = [_report("rejected_offline"), _report("rejected_ab")]
    assert recipe.next_action(report_history=history, variant=None) is None


def test_stops_immediately_once_a_launch_succeeded():
    recipe = FineTuneRecipe(max_attempts=5)
    history = [_report("rejected_offline"), _report("launched")]
    assert recipe.next_action(report_history=history, variant=None) is None


def test_ignores_attempts_targeting_other_scopes():
    recipe = FineTuneRecipe(max_attempts=1)
    history = [_report("launched", scope=PROMPT)]
    assert recipe.next_action(report_history=history, variant=None) == Action(scope=MODEL_WEIGHTS)
