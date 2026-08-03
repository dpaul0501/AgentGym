import pytest

from agentgym.core.registry import AgentGymRegistry
from agentgym.core.scope import GRAPH, PROMPT


class _FakeOptimizer:
    def __init__(self, scope):
        self.scope = scope
        self.depends_on = []

    def estimate(self, artifact, corpus, benchmark, variant):
        return artifact


def test_register_and_get_optimizer():
    registry = AgentGymRegistry()
    opt = _FakeOptimizer(PROMPT)
    registry.register_optimizer(opt)
    assert registry.get_optimizer(PROMPT) is opt


def test_register_optimizer_conflict_raises_without_force():
    registry = AgentGymRegistry()
    registry.register_optimizer(_FakeOptimizer(PROMPT))
    with pytest.raises(ValueError):
        registry.register_optimizer(_FakeOptimizer(PROMPT))


def test_register_optimizer_force_overwrites():
    registry = AgentGymRegistry()
    registry.register_optimizer(_FakeOptimizer(PROMPT))
    second = _FakeOptimizer(PROMPT)
    registry.register_optimizer(second, force=True)
    assert registry.get_optimizer(PROMPT) is second


def test_get_missing_scope_raises_clear_error():
    registry = AgentGymRegistry()
    with pytest.raises(KeyError, match=str(GRAPH)):
        registry.get_optimizer(GRAPH)


def test_register_and_get_recipe():
    registry = AgentGymRegistry()
    recipe = object()
    registry.register_recipe("context_search", recipe)
    assert registry.get_recipe("context_search") is recipe


def test_register_and_get_deployer():
    registry = AgentGymRegistry()
    deployer = object()
    registry.register_deployer("offline", deployer)
    assert registry.get_deployer("offline") is deployer
