from agentgym.core.artifact import Artifact, ArtifactShape
from agentgym.core.diagnosis import Diagnosis
from agentgym.core.protocols import (
    Agent,
    Benchmark,
    Deployer,
    Diagnoser,
    LearningRecipe,
    ScopeOptimizer,
    Task,
    Verifier,
)
from agentgym.core.score import Score
from agentgym.core.scope import PROMPT
from agentgym.core.trace import Trace


class _MinimalAgent:
    def consumes(self):
        return {PROMPT: None}

    def run(self, task, artifacts):
        return Trace(trace_id="t1", task_id=task.task_id, spans=[], metadata={})


def test_minimal_agent_satisfies_protocol():
    agent = _MinimalAgent()
    assert isinstance(agent, Agent)
    trace = agent.run(Task(task_id="x", instruction="do it", metadata={}), {})
    assert isinstance(trace, Trace)


class _MinimalBenchmark:
    def cases(self):
        return [Task(task_id="t1", instruction="x", metadata={})]


def test_minimal_benchmark_satisfies_protocol():
    b = _MinimalBenchmark()
    assert isinstance(b, Benchmark)
    assert list(b.cases())[0].task_id == "t1"


class _MinimalVerifier:
    def score(self, trace, task):
        return [Score(metric_name="x", value=1.0, kind="verifiable", evidence="")]


def test_minimal_verifier_satisfies_protocol():
    v = _MinimalVerifier()
    assert isinstance(v, Verifier)
    assert v.score(None, None)[0].value == 1.0


class _MinimalDiagnoser:
    def diagnose(self, trace, scores):
        return Diagnosis(
            trace_id="t1", lever=None, evidence=["insufficient"], confidence=0.0, suggested_scope=None
        )


def test_minimal_diagnoser_satisfies_protocol():
    d = _MinimalDiagnoser()
    assert isinstance(d, Diagnoser)


class _MinimalScopeOptimizer:
    scope = PROMPT
    depends_on: list = []

    def estimate(self, artifact, corpus, benchmark, variant):
        return artifact


def test_minimal_scope_optimizer_satisfies_protocol():
    opt = _MinimalScopeOptimizer()
    assert isinstance(opt, ScopeOptimizer)
    assert opt.scope == PROMPT
    assert opt.depends_on == []


class _MinimalLearningRecipe:
    def next_action(self, report_history, variant):
        return None


def test_minimal_learning_recipe_satisfies_protocol():
    r = _MinimalLearningRecipe()
    assert isinstance(r, LearningRecipe)
    assert r.next_action([], None) is None


class _MinimalDeployer:
    def release(self, variant, traffic_pct):
        return {"variant": variant}

    def ab_compare(self, baseline, candidate, release):
        return None

    def launch(self, release):
        pass

    def rollback(self, release):
        pass


def test_minimal_deployer_satisfies_protocol():
    d = _MinimalDeployer()
    assert isinstance(d, Deployer)


def test_composite_verifier_concatenates_scores():
    from agentgym.core.protocols import CompositeVerifier

    class _Verifier:
        def __init__(self, name):
            self.name = name

        def score(self, trace, task):
            return [Score(metric_name=self.name, value=1.0, kind="verifiable", evidence="")]

    composite = CompositeVerifier([_Verifier("task_success"), _Verifier("prompt_injection_resistance")])
    scores = composite.score(trace=None, task=None)
    assert {s.metric_name for s in scores} == {"task_success", "prompt_injection_resistance"}
