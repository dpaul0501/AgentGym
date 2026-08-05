"""CavemanMemoryOptimizer: the real alternate reference binding for Scope.MEMORY, wrapping
wilpel/caveman-compression's real NLP-based compress_text/count_tokens (spaCy-backed, offline,
free). Unlike headroom, this tool exposes no tunable config beyond a language code — so
"estimating" here means running the real technique against the corpus's actual captured content
and reporting real, measured compression, not a hyperparameter search. That's an honest
difference between the two tools' shapes, not a shortcut, and exactly the kind of side-by-side
Milestone B's comparison is meant to surface: same scope, two real competing tools, one shared
measurement.

Not pip-installable upstream (no packaging metadata, no LICENSE file in the source repo) — this
module imports it via sys.path from a local clone the caller points at, deliberately not vendored
into this repo. Clone it yourself first:
    git clone https://github.com/wilpel/caveman-compression
"""

from __future__ import annotations

import sys
from pathlib import Path

from agentgym.core.artifact import Artifact, ArtifactShape
from agentgym.core.corpus import TrainingCorpus
from agentgym.core.scope import MEMORY


def _flatten_to_text(row: dict) -> str:
    parts = []
    for step in row["steps"]:
        if step["state"]:
            parts.append(str(step["state"]))
        if step["observation"]:
            parts.append(str(step["observation"]))
    return " ".join(parts)


class CavemanMemoryOptimizer:
    scope = MEMORY
    depends_on: list = []

    def __init__(self, caveman_path: str | Path, lang: str = "en"):
        self.caveman_path = Path(caveman_path)
        self.lang = lang
        if not (self.caveman_path / "caveman_compress_nlp.py").exists():
            raise FileNotFoundError(
                f"caveman-compression not found at {self.caveman_path} — clone it manually: "
                "git clone https://github.com/wilpel/caveman-compression"
            )
        if str(self.caveman_path) not in sys.path:
            sys.path.insert(0, str(self.caveman_path))

    def estimate(self, artifact: Artifact, corpus: TrainingCorpus, benchmark, variant) -> Artifact:
        from caveman_compress_nlp import compress_text, count_tokens

        trajectory_rows = corpus.sft_rows()
        if not trajectory_rows:
            raise ValueError("cannot estimate MEMORY from an empty TrainingCorpus")
        texts = [t for t in (_flatten_to_text(row) for row in trajectory_rows) if t.strip()]
        if not texts:
            raise ValueError("corpus produced no non-empty text content to compress")

        tokens_before = tokens_after = 0
        for text in texts:
            compressed = compress_text(text, lang=self.lang)
            tokens_before += count_tokens(text)
            tokens_after += count_tokens(compressed)

        return Artifact(
            scope=MEMORY,
            shape=ArtifactShape.CONFIG,
            value={"tool": "caveman-compression", "variant": "nlp", "lang": self.lang},
            optimizer_binding="caveman_compress_nlp.compress_text",
            technique=None,
            source="estimated",
            provenance={
                "n_trajectories": len(texts),
                "tokens_before": tokens_before,
                "tokens_after": tokens_after,
                "tokens_saved": tokens_before - tokens_after,
            },
        )
