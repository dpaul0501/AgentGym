"""InterruptThresholdOptimizer: the real reference ScopeOptimizer for Scope.INTERRUPT. Real
Optuna search over a single scalar escalation threshold — tasks with metadata['risk_score'] at or
above the threshold escalate to human review, below it proceed automatically. Produces a
PARAMETER-shaped artifact, matching this design's own ArtifactShape taxonomy for this scope.
"""

from __future__ import annotations

import optuna

from agentgym.core.artifact import Artifact, ArtifactShape
from agentgym.core.scope import INTERRUPT


class InterruptThresholdOptimizer:
    scope = INTERRUPT
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
                should_escalate = task.metadata["risk_score"] >= threshold
                if should_escalate == task.metadata["expected_escalate"]:
                    correct += 1
            return correct / len(cases)

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.seed)
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        return Artifact(
            scope=INTERRUPT,
            shape=ArtifactShape.PARAMETER,
            value=study.best_params["threshold"],
            optimizer_binding="optuna.TPESampler",
            technique=None,
            source="estimated",
            provenance={"n_trials": self.n_trials, "best_score": study.best_value},
        )
