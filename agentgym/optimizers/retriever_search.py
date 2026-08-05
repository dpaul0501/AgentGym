"""RetrieverConfigOptimizer: the real reference ScopeOptimizer for Scope.RETRIEVAL (part of
Milestone B alongside the MEMORY optimizers, since retrieval-config search shares headroom-ai's
real BM25Scorer rather than a hand-rolled ranking heuristic). Real Optuna search over top_k —
how many BM25-ranked documents to retrieve per task — maximizing recall of each task's real
relevant_doc_ids while penalizing a larger top_k (retrieving more docs than needed bloats context,
the same cost consideration ToolSelectorOptimizer applies to tool count). Produces a CONFIG-shaped
artifact, matching this design's own ArtifactShape taxonomy for this scope.
"""

from __future__ import annotations

import optuna
from headroom import BM25Scorer

from agentgym.core.artifact import Artifact, ArtifactShape
from agentgym.core.scope import RETRIEVAL

TOP_K_PENALTY = 0.02


class RetrieverConfigOptimizer:
    scope = RETRIEVAL
    depends_on: list = []

    def __init__(self, n_trials: int = 20, seed: int = 0):
        self.n_trials = n_trials
        self.seed = seed
        self.scorer = BM25Scorer()

    def estimate(self, artifact: Artifact, corpus, benchmark, variant) -> Artifact:
        lo, hi = artifact.value["search_space"]["top_k"]
        cases = benchmark.cases()

        def _retrieve(query: str, documents: list[dict], top_k: int) -> set[str]:
            texts = [d["text"] for d in documents]
            scores = self.scorer.score_batch(texts, query)
            ranked = sorted(zip(documents, scores), key=lambda p: p[1].score, reverse=True)
            return {doc["id"] for doc, _ in ranked[:top_k]}

        def objective(trial: optuna.Trial) -> float:
            top_k = trial.suggest_int("top_k", lo, hi)
            recalls = []
            for task in cases:
                relevant = set(task.metadata["relevant_doc_ids"])
                retrieved = _retrieve(task.instruction, task.metadata["documents"], top_k)
                recalls.append(len(relevant & retrieved) / len(relevant) if relevant else 1.0)
            return (sum(recalls) / len(recalls)) - TOP_K_PENALTY * top_k

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.seed)
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        return Artifact(
            scope=RETRIEVAL,
            shape=ArtifactShape.CONFIG,
            value={"top_k": study.best_params["top_k"]},
            optimizer_binding="headroom.BM25Scorer+optuna.TPESampler",
            technique=None,
            source="estimated",
            provenance={"n_trials": self.n_trials, "best_score": study.best_value},
        )
