from agentgym.core.artifact import Artifact, ArtifactShape
from agentgym.core.protocols import Task
from agentgym.core.scope import PROMPT
from agentgym.core.score import Score
from agentgym.core.trace import Trace
from agentgym.core.variant import AgentVariant
from agentgym.deployers.offline_deployer import OfflineDeployer


def _artifact(value):
    return Artifact(scope=PROMPT, shape=ArtifactShape.PARAMETER, value=value,
                     optimizer_binding=None, technique=None, source="manual")


class _FakeAgent:
    def consumes(self):
        return {PROMPT: None}

    def run(self, task, artifacts):
        return Trace(trace_id=f"t-{task.task_id}", task_id=task.task_id, spans=[],
                      metadata={"quality": artifacts[PROMPT].value})


class _FakeBenchmark:
    def cases(self):
        return [Task(task_id="t1", instruction="x", metadata={}),
                Task(task_id="t2", instruction="y", metadata={})]


class _FakeVerifier:
    def score(self, trace, task):
        return [Score(metric_name="task_success", value=trace.metadata["quality"],
                       kind="verifiable", evidence="")]


def test_release_wraps_variant_without_touching_it():
    deployer = OfflineDeployer(agent=_FakeAgent(), benchmark=_FakeBenchmark(), verifier=_FakeVerifier())
    variant = AgentVariant(artifacts={PROMPT: _artifact(0.5)})

    release = deployer.release(variant, traffic_pct=0.05)

    assert release.variant is variant


def test_ab_compare_runs_real_bootstrap_comparison_and_candidate_wins():
    deployer = OfflineDeployer(agent=_FakeAgent(), benchmark=_FakeBenchmark(), verifier=_FakeVerifier(), seed=0)
    baseline = AgentVariant(artifacts={PROMPT: _artifact(0.2)})
    candidate = AgentVariant(artifacts={PROMPT: _artifact(0.9)})
    release = deployer.release(candidate, traffic_pct=0.05)

    result = deployer.ab_compare(baseline, candidate, release)

    assert result.delta > 0
    assert result.candidate_wins is True


def test_ab_compare_candidate_loses_when_worse():
    deployer = OfflineDeployer(agent=_FakeAgent(), benchmark=_FakeBenchmark(), verifier=_FakeVerifier(), seed=0)
    baseline = AgentVariant(artifacts={PROMPT: _artifact(0.9)})
    candidate = AgentVariant(artifacts={PROMPT: _artifact(0.2)})
    release = deployer.release(candidate, traffic_pct=0.05)

    result = deployer.ab_compare(baseline, candidate, release)

    assert result.delta < 0
    assert result.candidate_wins is False


def test_launch_and_rollback_are_real_no_ops_in_v0():
    deployer = OfflineDeployer(agent=_FakeAgent(), benchmark=_FakeBenchmark(), verifier=_FakeVerifier())
    variant = AgentVariant(artifacts={PROMPT: _artifact(0.5)})
    release = deployer.release(variant, traffic_pct=0.05)

    assert deployer.launch(release) is None
    assert deployer.rollback(release) is None
