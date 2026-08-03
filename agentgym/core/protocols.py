"""The interoperability layer: typing.Protocol structural types, so any real OSS tool can bind to
a scope, provide a benchmark, verify a trace, or diagnose a failure without inheriting from
agentgym code — matching how real interoperable libraries are designed, not app frameworks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from agentgym.core.artifact import Artifact
from agentgym.core.diagnosis import Diagnosis
from agentgym.core.scope import Scope
from agentgym.core.score import Score
from agentgym.core.trace import Trace


@dataclass(frozen=True)
class Task:
    task_id: str
    instruction: str
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class Agent(Protocol):
    def consumes(self) -> dict[Scope, Artifact | None]: ...
    def run(self, task: Task, artifacts: dict[Scope, Artifact]) -> Trace: ...


@runtime_checkable
class Benchmark(Protocol):
    def cases(self) -> list[Task]: ...


@runtime_checkable
class Verifier(Protocol):
    def score(self, trace: Trace, task: Task) -> list[Score]: ...


@runtime_checkable
class Diagnoser(Protocol):
    def diagnose(self, trace: Trace, scores: list[Score]) -> Diagnosis: ...


@runtime_checkable
class ScopeOptimizer(Protocol):
    scope: Scope
    depends_on: list[Scope]

    def estimate(self, artifact: Artifact, corpus: Any, benchmark: Benchmark, variant: Any) -> Artifact: ...


@dataclass(frozen=True)
class Action:
    scope: Scope


@runtime_checkable
class LearningRecipe(Protocol):
    def next_action(self, report_history: list[Any], variant: Any) -> Action | None: ...


@dataclass
class Release:
    variant: Any


@runtime_checkable
class Deployer(Protocol):
    def release(self, variant: Any, traffic_pct: float) -> Any: ...
    def ab_compare(self, baseline: Any, candidate: Any, release: Any) -> Any: ...
    def launch(self, release: Any) -> None: ...
    def rollback(self, release: Any) -> None: ...


class CompositeVerifier:
    """Composes several Verifiers onto the same trace — how a security-probing pass adds scores
    alongside task-correctness scores without forking a second benchmark."""

    def __init__(self, verifiers: list[Verifier]):
        self.verifiers = verifiers

    def score(self, trace: Trace, task: Task) -> list[Score]:
        result: list[Score] = []
        for v in self.verifiers:
            result.extend(v.score(trace, task))
        return result
