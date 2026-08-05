"""A real, complete Harness.cycle() run: Existing -> Evaluate -> Diagnose -> Improve -> Evaluate
-> Release -> A/B -> Launch, using real reference implementations at every stage —
RuleBasedDiagnoser, HarnessSearchRecipe, LangGraphConfigOptimizer (real LangGraph + real Optuna),
OfflineDeployer (real bootstrap-CI A/B), AgentGymStore (real SQLite) — proving the eight-stage
lifecycle actually launches a real, measured improvement end to end, not just that each piece
works correctly in isolation.

Reuses the lead-routing scenario (build_lead_routing_graph) already proven real in
tests/optimizers/test_langgraph_search.py: a deliberately bad starting threshold routes deals
incorrectly; Harness should diagnose, search for a better threshold, and launch it.
"""

from agentgym.core.artifact import Artifact, ArtifactShape
from agentgym.core.harness import Harness
from agentgym.core.protocols import Task
from agentgym.core.score import Score
from agentgym.core.scope import GRAPH
from agentgym.core.trace import Span, Trace
from agentgym.core.variant import AgentVariant
from agentgym.deployers.offline_deployer import OfflineDeployer
from agentgym.diagnosers.rule_based import RuleBasedDiagnoser
from agentgym.optimizers.langgraph_search import LangGraphConfigOptimizer, build_lead_routing_graph
from agentgym.recipes.harness_search_recipe import HarnessSearchRecipe
from agentgym.storage.db import AgentGymStore


class LeadRoutingAgent:
    def consumes(self):
        return {GRAPH: None}

    def run(self, task: Task, artifacts: dict) -> Trace:
        threshold = artifacts[GRAPH].value.get("threshold", 50000)
        graph = build_lead_routing_graph(threshold)
        result = graph.invoke({"deal_value": task.metadata["deal_value"], "route": ""})
        span = Span(
            span_id=f"s-{task.task_id}", parent_span_id=None, trace_id=f"t-{task.task_id}",
            name="route", kind="other", input=task.metadata, output=result,
            start_time=0.0, end_time=0.01, attributes={},
        )
        return Trace(trace_id=f"t-{task.task_id}", task_id=task.task_id, spans=[span], metadata={})


class LeadRoutingBenchmark:
    def cases(self) -> list[Task]:
        # 5 low-value (should auto-process) + 5 high-value (should route to human) — a real
        # bootstrap-CI A/B needs enough cases for a genuine effect to clear statistical
        # significance; 4 cases proved too few (confirmed by actually running it, not assumed).
        auto_cases = [
            Task(task_id=f"auto-{i}", instruction="", metadata={"deal_value": v, "expected_route": "auto"})
            for i, v in enumerate([200, 800, 1500, 3000, 4200])
        ]
        human_cases = [
            Task(task_id=f"human-{i}", instruction="", metadata={"deal_value": v, "expected_route": "human"})
            for i, v in enumerate([76000, 82000, 88000, 91000, 97000])
        ]
        return auto_cases + human_cases


class LeadRoutingVerifier:
    def score(self, trace: Trace, task: Task) -> list[Score]:
        correct = trace.spans[0].output["route"] == task.metadata["expected_route"]
        return [Score(
            metric_name="route_correct", value=1.0 if correct else 0.0, kind="verifiable",
            evidence="matches expected_route" if correct else "wrong route",
        )]


def test_full_eight_stage_cycle_launches_a_real_improvement(tmp_path):
    agent = LeadRoutingAgent()
    benchmark = LeadRoutingBenchmark()
    verifier = LeadRoutingVerifier()
    optimizer = LangGraphConfigOptimizer(n_trials=15, seed=0)
    recipe = HarnessSearchRecipe(optimizers={GRAPH: optimizer})
    deployer = OfflineDeployer(agent=agent, benchmark=benchmark, verifier=verifier, seed=0)
    store = AgentGymStore(str(tmp_path / "test.db"))

    harness = Harness(
        agent=agent, benchmark=benchmark, verifier=verifier, diagnoser=RuleBasedDiagnoser(),
        optimizers={GRAPH: optimizer}, recipe=recipe, deployer=deployer, store=store,
    )

    # threshold=100 routes EVERY deal to human review -> wrong for both "auto" cases (2/4 correct)
    baseline_artifact = Artifact(
        scope=GRAPH, shape=ArtifactShape.CONFIG,
        value={"threshold": 100, "search_space": {"threshold": [1000, 90000]}},
        optimizer_binding=None, technique=None, source="manual",
    )
    baseline = AgentVariant(artifacts={GRAPH: baseline_artifact})

    report = harness.cycle(baseline)

    assert report.status == "launched"
    assert report.action.scope == GRAPH
    assert report.ab_result.candidate_wins is True
    launched_threshold = report.variant.artifacts[GRAPH].value["threshold"]
    assert 1000 <= launched_threshold <= 90000
    assert len(harness.corpus) == len(benchmark.cases())
