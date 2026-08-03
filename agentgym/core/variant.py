"""AgentVariant: a partial mapping of Scope -> Artifact, one complete-or-partial versioned agent
configuration. Composing across levers is a namespaced union, always collision-free, since a
Scope carries its lever as part of its identity."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from agentgym.core.artifact import Artifact
from agentgym.core.scope import Lever, Scope


class MissingScopeError(KeyError):
    """Raised when a scope an Agent declares it consumes has neither a bound artifact in the
    variant nor a declared default — fail closed, never run with a silent gap."""


@dataclass(frozen=True)
class AgentVariant:
    artifacts: dict[Scope, Artifact] = field(default_factory=dict)

    def resolve(self, scope: Scope, default: Artifact) -> Artifact:
        return self.artifacts.get(scope, default)

    def with_artifact(self, scope: Scope, artifact: Artifact) -> "AgentVariant":
        return replace(self, artifacts={**self.artifacts, scope: artifact})

    def scopes_under(self, lever: Lever) -> dict[Scope, Artifact]:
        return {s: a for s, a in self.artifacts.items() if s.lever == lever}

    def resolve_for(self, consumes: dict[Scope, Artifact | None]) -> dict[Scope, Artifact]:
        """`consumes` maps each scope an Agent needs to its fallback default (or None if the
        scope has no legal default and must already be bound)."""
        resolved: dict[Scope, Artifact] = {}
        for scope, default in consumes.items():
            if scope in self.artifacts:
                resolved[scope] = self.artifacts[scope]
            elif default is not None:
                resolved[scope] = default
            else:
                raise MissingScopeError(
                    f"{scope} is required but has no bound artifact and no declared default"
                )
        return resolved
