"""The namespace of tunable areas in an agent: a lever (context/harness/fine_tune) as the root,
scopes as its children, recursively nestable."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Lever(str, Enum):
    CONTEXT = "context"
    HARNESS = "harness"
    FINE_TUNE = "fine_tune"


@dataclass(frozen=True)
class Scope:
    lever: Lever
    segments: tuple[str, ...]

    def __str__(self) -> str:
        return f"{self.lever.value}.{'.'.join(self.segments)}"

    def is_child_of(self, other: "Scope") -> bool:
        return (
            self.lever == other.lever
            and self.segments[: len(other.segments)] == other.segments
            and len(self.segments) > len(other.segments)
        )


PROMPT = Scope(Lever.CONTEXT, ("prompt",))
TOOLS = Scope(Lever.CONTEXT, ("tools",))
MEMORY = Scope(Lever.CONTEXT, ("memory",))
RETRIEVAL = Scope(Lever.CONTEXT, ("retrieval",))
MODEL_ROUTING = Scope(Lever.HARNESS, ("model_routing",))
INTERRUPT = Scope(Lever.HARNESS, ("interrupt",))
GRAPH = Scope(Lever.HARNESS, ("graph",))
GUARDRAILS = Scope(Lever.HARNESS, ("guardrails",))
MODEL_WEIGHTS = Scope(Lever.FINE_TUNE, ("model_weights",))
