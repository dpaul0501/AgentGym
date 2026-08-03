from agentgym.core.artifact import Artifact, ArtifactShape
from agentgym.core.corpus import TrainingCorpus
from agentgym.core.diagnosis import Diagnosis
from agentgym.core.harness import Harness
from agentgym.core.protocols import Action, Task
from agentgym.core.score import Score
from agentgym.core.scope import PROMPT, Lever
from agentgym.core.trace import Trace
from agentgym.core.variant import AgentVariant


def _artifact(value, source="manual", provenance=None):
    return Artifact(
        scope=PROMPT, shape=ArtifactShape.PARAMETER, value=value,
        optimizer_binding=None, technique=None, source=source,
        provenance=provenance or ({} if source == "manual" else {"run": "x"}),
    )


class FakeAgent:
    def __init__(self):
        self.run_calls: list[dict] = []

    def consumes(self):
        return {PROMPT: None}

    def run(self, task, artifacts):
        self.run_calls.append({"task": task, "artifacts": artifacts})
        return Trace(
            trace_id=f"trace-{task.task_id}", task_id=task.task_id, spans=[],
            metadata={"quality": artifacts[PROMPT].value},
        )


class FakeBenchmark:
    def cases(self):
        return [Task(task_id="t1", instruction="x", metadata={}),
                Task(task_id="t2", instruction="y", metadata={})]


class FakeVerifier:
    def score(self, trace, task):
        return [Score(metric_name="task_success", value=trace.metadata["quality"],
                       kind="verifiable", evidence="")]


class FakeDiagnoser:
    def diagnose(self, trace, scores):
        return Diagnosis(trace_id=trace.trace_id, lever=Lever.CONTEXT, evidence=["low score"],
                          confidence=0.9, suggested_scope=PROMPT)


class FakeOptimizer:
    scope = PROMPT
    depends_on: list = []

    def __init__(self, produces_value):
        self.produces_value = produces_value

    def estimate(self, artifact, corpus, benchmark, variant):
        return _artifact(self.produces_value, source="estimated")


class FakeRecipe:
    def __init__(self, actions):
        self._actions = list(actions)
        self.calls: list[list] = []

    def next_action(self, report_history, variant):
        self.calls.append(list(report_history))
        return self._actions.pop(0) if self._actions else None


class FakeDeployer:
    def __init__(self, ab_wins=True):
        self.ab_wins = ab_wins
        self.calls: list[str] = []

    def release(self, variant, traffic_pct):
        self.calls.append("release")
        return {"variant": variant}

    def ab_compare(self, baseline, candidate, release):
        self.calls.append("ab_compare")
        from agentgym.core.ab import ABResult
        return ABResult(delta=0.1 if self.ab_wins else -0.1,
                         ci=(0.05, 0.15) if self.ab_wins else (-0.2, -0.02),
                         candidate_wins=self.ab_wins)

    def launch(self, release):
        self.calls.append("launch")

    def rollback(self, release):
        self.calls.append("rollback")


class FakeStore:
    def __init__(self):
        self.saved: list = []

    def save_variant(self, variant):
        self.saved.append(variant)


def make_harness(recipe, optimizer, deployer, agent=None):
    agent = agent or FakeAgent()
    return Harness(
        agent=agent, benchmark=FakeBenchmark(), verifier=FakeVerifier(),
        diagnoser=FakeDiagnoser(), optimizers={PROMPT: optimizer},
        recipe=recipe, deployer=deployer, store=FakeStore(),
        corpus=TrainingCorpus(),
    ), agent


def test_cycle_no_action_short_circuits():
    recipe = FakeRecipe(actions=[])  # always returns None
    deployer = FakeDeployer()
    harness, _ = make_harness(recipe, FakeOptimizer(0.9), deployer)
    baseline = AgentVariant(artifacts={PROMPT: _artifact(0.5)})

    report = harness.cycle(baseline)

    assert report.status == "no_action"
    assert deployer.calls == []


def test_cycle_evaluates_twice_baseline_then_candidate():
    recipe = FakeRecipe(actions=[Action(scope=PROMPT)])
    deployer = FakeDeployer(ab_wins=True)
    harness, agent = make_harness(recipe, FakeOptimizer(0.9), deployer)
    baseline = AgentVariant(artifacts={PROMPT: _artifact(0.5)})

    harness.cycle(baseline)

    # 2 benchmark cases x 2 evaluate passes (baseline, candidate) = 4 agent.run calls
    assert len(agent.run_calls) == 4


def test_cycle_rejects_offline_when_candidate_scores_worse():
    recipe = FakeRecipe(actions=[Action(scope=PROMPT)])
    deployer = FakeDeployer()
    # optimizer deliberately produces a WORSE artifact (0.1 < baseline's 0.5)
    harness, _ = make_harness(recipe, FakeOptimizer(0.1), deployer)
    baseline = AgentVariant(artifacts={PROMPT: _artifact(0.5)})

    report = harness.cycle(baseline)

    assert report.status == "rejected_offline"
    assert deployer.calls == []  # release/ab_compare never reached


def test_cycle_rejects_ab_and_rolls_back():
    recipe = FakeRecipe(actions=[Action(scope=PROMPT)])
    deployer = FakeDeployer(ab_wins=False)
    harness, _ = make_harness(recipe, FakeOptimizer(0.9), deployer)
    baseline = AgentVariant(artifacts={PROMPT: _artifact(0.5)})

    report = harness.cycle(baseline)

    assert report.status == "rejected_ab"
    assert deployer.calls == ["release", "ab_compare", "rollback"]
    assert "launch" not in deployer.calls


def test_cycle_launches_on_full_success():
    recipe = FakeRecipe(actions=[Action(scope=PROMPT)])
    deployer = FakeDeployer(ab_wins=True)
    harness, _ = make_harness(recipe, FakeOptimizer(0.9), deployer)
    baseline = AgentVariant(artifacts={PROMPT: _artifact(0.5)})

    report = harness.cycle(baseline)

    assert report.status == "launched"
    assert deployer.calls == ["release", "ab_compare", "launch"]
    assert harness.store.saved[-1].artifacts[PROMPT].value == 0.9


def test_cycle_appends_report_to_history():
    recipe = FakeRecipe(actions=[Action(scope=PROMPT), None])
    deployer = FakeDeployer(ab_wins=True)
    harness, _ = make_harness(recipe, FakeOptimizer(0.9), deployer)
    baseline = AgentVariant(artifacts={PROMPT: _artifact(0.5)})

    first = harness.cycle(baseline)
    harness.cycle(first.variant)

    assert len(recipe.calls[1]) == 1
    assert recipe.calls[1][0].status == "launched"


def test_evaluate_uses_agent_consumes_not_full_variant():
    from agentgym.core.scope import GRAPH

    recipe = FakeRecipe(actions=[])
    deployer = FakeDeployer()
    harness, agent = make_harness(recipe, FakeOptimizer(0.9), deployer)
    # variant has TWO bound scopes; agent.consumes() only asks for PROMPT
    variant = AgentVariant(artifacts={PROMPT: _artifact(0.5), GRAPH: _artifact("g")})

    harness.cycle(variant)

    resolved_keys = set(agent.run_calls[0]["artifacts"].keys())
    assert resolved_keys == {PROMPT}


def test_evaluate_raises_missing_scope_before_calling_agent():
    from agentgym.core.variant import MissingScopeError
    import pytest

    recipe = FakeRecipe(actions=[])
    deployer = FakeDeployer()
    harness, agent = make_harness(recipe, FakeOptimizer(0.9), deployer)
    empty_variant = AgentVariant(artifacts={})  # PROMPT unbound, no default from consumes()

    with pytest.raises(MissingScopeError):
        harness.cycle(empty_variant)

    assert agent.run_calls == []
