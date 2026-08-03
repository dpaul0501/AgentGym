"""A single scored metric on a trace, from one Verifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

_VALID_KINDS = {"verifiable", "plausible"}


@dataclass
class Score:
    metric_name: str
    value: float
    kind: Literal["verifiable", "plausible"]
    evidence: str

    def __post_init__(self) -> None:
        if self.kind not in _VALID_KINDS:
            raise ValueError(f"kind must be one of {_VALID_KINDS}, got {self.kind!r}")


def primary_reward(scores: list["Score"]) -> float:
    """The one number a scope's before/after comparison is judged on: the first verifiable
    score if one exists, else the first score of any kind, else 0.0 for an empty list."""
    for s in scores:
        if s.kind == "verifiable":
            return s.value
    return scores[0].value if scores else 0.0
