"""ToolSelectorOptimizer: the real reference ScopeOptimizer for Scope.TOOLS. Real Optuna search
over which subset of a candidate tool registry to expose to the agent — maximizing the fraction
of benchmark tasks whose real required_tools are covered, while penalizing tool count (a genuine,
real consideration: exposing every tool to an LLM agent is a documented source of tool-selection
confusion, not a made-up constraint). Produces an ENSEMBLE-shaped artifact, matching this design's
own ArtifactShape taxonomy — a set of strategies (tools) combined, not one value selected.
"""

from __future__ import annotations

import optuna

from agentgym.core.artifact import Artifact, ArtifactShape
from agentgym.core.scope import TOOLS

TOOL_COUNT_PENALTY = 0.05


class ToolSelectorOptimizer:
    scope = TOOLS
    depends_on: list = []

    def __init__(self, n_trials: int = 20, seed: int = 0):
        self.n_trials = n_trials
        self.seed = seed

    def estimate(self, artifact: Artifact, corpus, benchmark, variant) -> Artifact:
        candidate_tools: list[str] = artifact.value["candidate_tools"]
        cases = benchmark.cases()

        def objective(trial: optuna.Trial) -> float:
            enabled = {
                tool for tool in candidate_tools if trial.suggest_categorical(f"use_{tool}", [True, False])
            }
            covered = sum(
                1 for task in cases if set(task.metadata["required_tools"]) <= enabled
            )
            coverage = covered / len(cases)
            return coverage - TOOL_COUNT_PENALTY * len(enabled)

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.seed)
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        enabled_tools = sorted(
            tool for tool in candidate_tools if study.best_params[f"use_{tool}"]
        )
        return Artifact(
            scope=TOOLS,
            shape=ArtifactShape.ENSEMBLE,
            value={"enabled_tools": enabled_tools},
            optimizer_binding="optuna.TPESampler",
            technique=None,
            source="estimated",
            provenance={"n_trials": self.n_trials, "best_score": study.best_value},
        )
