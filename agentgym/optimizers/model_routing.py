"""ModelRoutingOptimizer: the real reference ScopeOptimizer for Scope.MODEL_ROUTING. Real Optuna
search over a single scalar routing threshold — tasks with metadata['complexity'] at or above the
threshold route to the "large" model tier, below it route to "small". Produces a PARAMETER-shaped
artifact (a single scalar knob), matching this design's own ArtifactShape taxonomy for this scope.
"""

from __future__ import annotations

import optuna

from agentgym.core.artifact import Artifact, ArtifactShape
from agentgym.core.scope import MODEL_ROUTING


class ModelRoutingOptimizer:
    scope = MODEL_ROUTING
    depends_on: list = []

    def __init__(self, n_trials: int = 20, seed: int = 0):
        self.n_trials = n_trials
        self.seed = seed

    def estimate(self, artifact: Artifact, corpus, benchmark, variant) -> Artifact:
        lo, hi = artifact.value["search_space"]["threshold"]
        cases = benchmark.cases()

        def objective(trial: optuna.Trial) -> float:
            threshold = trial.suggest_float("threshold", lo, hi)
            correct = 0
            for task in cases:
                tier = "large" if task.metadata["complexity"] >= threshold else "small"
                if tier == task.metadata["expected_tier"]:
                    correct += 1
            return correct / len(cases)

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.seed)
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        return Artifact(
            scope=MODEL_ROUTING,
            shape=ArtifactShape.PARAMETER,
            value=study.best_params["threshold"],
            optimizer_binding="optuna.TPESampler",
            technique=None,
            source="estimated",
            provenance={"n_trials": self.n_trials, "best_score": study.best_value},
        )
