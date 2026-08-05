from agentgym.core.protocols import Action
from agentgym.core.scope import GRAPH, MODEL_WEIGHTS, PROMPT
from agentgym.recipes.composite import CompositeRecipe


class _FakeRecipe:
    def __init__(self, action):
        self._action = action
        self.calls = 0

    def next_action(self, report_history, variant):
        self.calls += 1
        return self._action


def test_delegates_to_context_recipe_first_when_it_has_an_action():
    context = _FakeRecipe(Action(scope=PROMPT))
    harness = _FakeRecipe(Action(scope=GRAPH))
    finetune = _FakeRecipe(Action(scope=MODEL_WEIGHTS))
    composite = CompositeRecipe(context, harness, finetune)

    action = composite.next_action(report_history=[], variant=None)

    assert action == Action(scope=PROMPT)
    assert harness.calls == 0
    assert finetune.calls == 0


def test_falls_through_to_harness_when_context_exhausted():
    context = _FakeRecipe(None)
    harness = _FakeRecipe(Action(scope=GRAPH))
    finetune = _FakeRecipe(Action(scope=MODEL_WEIGHTS))
    composite = CompositeRecipe(context, harness, finetune)

    action = composite.next_action(report_history=[], variant=None)

    assert action == Action(scope=GRAPH)
    assert finetune.calls == 0


def test_falls_through_to_finetune_when_both_context_and_harness_exhausted():
    context = _FakeRecipe(None)
    harness = _FakeRecipe(None)
    finetune = _FakeRecipe(Action(scope=MODEL_WEIGHTS))
    composite = CompositeRecipe(context, harness, finetune)

    action = composite.next_action(report_history=[], variant=None)

    assert action == Action(scope=MODEL_WEIGHTS)


def test_returns_none_when_all_three_exhausted():
    composite = CompositeRecipe(_FakeRecipe(None), _FakeRecipe(None), _FakeRecipe(None))
    assert composite.next_action(report_history=[], variant=None) is None
