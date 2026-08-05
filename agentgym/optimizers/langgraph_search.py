"""LangGraphConfigOptimizer: the real reference ScopeOptimizer for Scope.GRAPH. Wraps a real
LangGraph StateGraph (conditional routing) and a real Optuna study — this file does not
reimplement graph orchestration or hyperparameter search, it binds LangGraph's real `compile()`/
`invoke()` and Optuna's real `Study.optimize()` to the Artifact/TrainingCorpus/Benchmark protocol.

Concrete scenario, matching the harness/routing failure used throughout this project's own
worked examples: a sales-workflow graph deciding whether a lead is auto-processed or routed to
human review, based on a deal-value threshold — exactly the "a chain can't branch" fix (moving to
a Graph paradigm with a conditional edge on deal value).
"""

from __future__ import annotations

from typing import TypedDict

import optuna
from langgraph.graph import END, START, StateGraph

from agentgym.core.artifact import Artifact, ArtifactShape
from agentgym.core.scope import GRAPH


class _LeadState(TypedDict):
    deal_value: float
    route: str


def _auto_process(state: _LeadState) -> dict:
    return {"route": "auto"}


def _human_review(state: _LeadState) -> dict:
    return {"route": "human"}


def build_lead_routing_graph(threshold: float):
    """A real, compiled LangGraph app. `threshold` is the tunable parameter this optimizer
    searches over — deals at or above it route to human review, matching the exact failure/fix
    already established: 'a chain can't branch' -> a conditional edge on deal value."""

    def _router(state: _LeadState) -> str:
        return "human_review" if state["deal_value"] >= threshold else "auto_process"

    g = StateGraph(_LeadState)
    g.add_node("auto_process", _auto_process)
    g.add_node("human_review", _human_review)
    g.add_conditional_edges(
        START, _router, {"human_review": "human_review", "auto_process": "auto_process"}
    )
    g.add_edge("auto_process", END)
    g.add_edge("human_review", END)
    return g.compile()


class LangGraphConfigOptimizer:
    scope = GRAPH
    depends_on: list = []

    def __init__(self, n_trials: int = 20, seed: int = 0):
        self.n_trials = n_trials
        self.seed = seed

    def estimate(self, artifact: Artifact, corpus, benchmark, variant) -> Artifact:
        space = artifact.value["search_space"]  # e.g. {"threshold": [0, 100000]}
        lo, hi = space["threshold"]
        cases = benchmark.cases()  # each Task.metadata carries deal_value + expected_route

        def objective(trial: optuna.Trial) -> float:
            threshold = trial.suggest_float("threshold", lo, hi)
            graph = build_lead_routing_graph(threshold)
            correct = 0
            for task in cases:
                result = graph.invoke(
                    {"deal_value": task.metadata["deal_value"], "route": ""}
                )
                if result["route"] == task.metadata["expected_route"]:
                    correct += 1
            return correct / len(cases)

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.seed)
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        return Artifact(
            scope=GRAPH,
            shape=ArtifactShape.CONFIG,
            value={"threshold": study.best_params["threshold"]},
            optimizer_binding="optuna.TPESampler",
            technique=None,
            source="estimated",
            provenance={"n_trials": self.n_trials, "best_score": study.best_value},
        )
