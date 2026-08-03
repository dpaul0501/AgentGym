"""The output of a Diagnoser: which lever/scope a scored trace's failure belongs to, or an
explicit insufficient_evidence outcome."""

from __future__ import annotations

from dataclasses import dataclass

from agentgym.core.scope import Lever, Scope


@dataclass
class Diagnosis:
    trace_id: str
    lever: Lever | None
    evidence: list[str]
    confidence: float
    suggested_scope: Scope | None

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")
        if self.lever is None or self.suggested_scope is None:
            if self.lever is not None or self.suggested_scope is not None:
                raise ValueError(
                    "lever and suggested_scope must both be set, or both be None "
                    "(insufficient_evidence) — not a mix"
                )
            if not self.evidence:
                raise ValueError(
                    "insufficient_evidence (lever=None) must still explain why in `evidence`"
                )
        elif self.suggested_scope.lever != self.lever:
            raise ValueError(
                f"suggested_scope {self.suggested_scope} belongs to lever "
                f"{self.suggested_scope.lever}, not the diagnosed lever {self.lever}"
            )
