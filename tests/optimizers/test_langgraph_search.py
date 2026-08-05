"""LangGraphConfigOptimizer against a real, compiled LangGraph StateGraph and a real Optuna
study — no fakes standing in for either library."""

from agentgym.core.artifact import Artifact, ArtifactShape
from agentgym.core.protocols import Task
from agentgym.core.scope import GRAPH
from agentgym.optimizers.langgraph_search import LangGraphConfigOptimizer, build_lead_routing_graph


class _LeadRoutingBenchmark:
    def cases(self) -> list[Task]:
        return [
            Task(task_id="t1", instruction="", metadata={"deal_value": 500, "expected_route": "auto"}),
            Task(task_id="t2", instruction="", metadata={"deal_value": 2000, "expected_route": "auto"}),
            Task(task_id="t3", instruction="", metadata={"deal_value": 80000, "expected_route": "human"}),
            Task(task_id="t4", instruction="", metadata={"deal_value": 95000, "expected_route": "human"}),
        ]


def _starting_artifact() -> Artifact:
    return Artifact(
        scope=GRAPH,
        shape=ArtifactShape.CONFIG,
        value={"search_space": {"threshold": [1000, 90000]}},
        optimizer_binding=None,
        technique=None,
        source="manual",
    )


def test_build_lead_routing_graph_routes_by_threshold():
    graph = build_lead_routing_graph(threshold=50000)
    assert graph.invoke({"deal_value": 60000, "route": ""})["route"] == "human"
    assert graph.invoke({"deal_value": 1000, "route": ""})["route"] == "auto"


def test_estimate_finds_a_threshold_that_perfectly_separates_the_cases():
    optimizer = LangGraphConfigOptimizer(n_trials=15, seed=0)
    result = optimizer.estimate(
        artifact=_starting_artifact(), corpus=None, benchmark=_LeadRoutingBenchmark(), variant=None
    )
    threshold = result.value["threshold"]
    assert 2000 < threshold <= 80000


def test_estimate_returns_config_shaped_estimated_artifact_with_provenance():
    optimizer = LangGraphConfigOptimizer(n_trials=15, seed=0)
    result = optimizer.estimate(
        artifact=_starting_artifact(), corpus=None, benchmark=_LeadRoutingBenchmark(), variant=None
    )
    assert result.scope == GRAPH
    assert result.shape == ArtifactShape.CONFIG
    assert result.source == "estimated"
    assert result.optimizer_binding == "optuna.TPESampler"
    assert result.provenance["n_trials"] == 15
    assert result.provenance["best_score"] == 1.0
