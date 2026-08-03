"""The value occupying a Scope: what kind of thing it is (ArtifactShape), and whether it was
hand-set or produced by an optimizer (source), with provenance when estimated."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from agentgym.core.scope import MODEL_WEIGHTS, Scope


class ArtifactShape(str, Enum):
    PARAMETER = "parameter"
    CONFIG = "config"
    PROGRAM = "program"
    ENSEMBLE = "ensemble"
    WEIGHTS = "weights"


class FineTuneTechnique(str, Enum):
    LORA_ADAPTER = "lora_adapter"
    FULL_FINETUNE = "full_finetune"
    RLHF = "rlhf"


@dataclass
class Artifact:
    scope: Scope
    shape: ArtifactShape
    value: Any
    optimizer_binding: str | None
    technique: FineTuneTechnique | None
    source: Literal["manual", "estimated"]
    provenance: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.technique is not None and self.scope != MODEL_WEIGHTS:
            raise ValueError(
                f"technique is only meaningful on {MODEL_WEIGHTS}, got scope={self.scope}"
            )
        if self.source == "estimated" and not self.provenance:
            raise ValueError("an estimated artifact must carry provenance (which run/corpus produced it)")
