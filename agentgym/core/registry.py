"""A plugin registry: which real tool is bound to which scope, which named recipe, which
deployer. Rebinding an already-registered scope is a real footgun, so it's rejected by default
rather than silently overwritten."""

from __future__ import annotations

from agentgym.core.scope import Scope


class AgentGymRegistry:
    def __init__(self):
        self._optimizers: dict[Scope, object] = {}
        self._recipes: dict[str, object] = {}
        self._deployers: dict[str, object] = {}

    def register_optimizer(self, optimizer, force: bool = False) -> None:
        if optimizer.scope in self._optimizers and not force:
            raise ValueError(
                f"{optimizer.scope} already has an optimizer bound; pass force=True to rebind"
            )
        self._optimizers[optimizer.scope] = optimizer

    def get_optimizer(self, scope: Scope):
        if scope not in self._optimizers:
            raise KeyError(f"no optimizer registered for {scope}")
        return self._optimizers[scope]

    def register_recipe(self, name: str, recipe, force: bool = False) -> None:
        if name in self._recipes and not force:
            raise ValueError(f"recipe {name!r} already registered; pass force=True to rebind")
        self._recipes[name] = recipe

    def get_recipe(self, name: str):
        if name not in self._recipes:
            raise KeyError(f"no recipe registered under {name!r}")
        return self._recipes[name]

    def register_deployer(self, name: str, deployer, force: bool = False) -> None:
        if name in self._deployers and not force:
            raise ValueError(f"deployer {name!r} already registered; pass force=True to rebind")
        self._deployers[name] = deployer

    def get_deployer(self, name: str):
        if name not in self._deployers:
            raise KeyError(f"no deployer registered under {name!r}")
        return self._deployers[name]
