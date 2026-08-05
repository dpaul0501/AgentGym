from agentgym.core.artifact import Artifact, ArtifactShape
from agentgym.core.protocols import Task
from agentgym.core.scope import INTERRUPT
from agentgym.optimizers.interrupt_threshold import InterruptThresholdOptimizer


class _EscalationBenchmark:
    def cases(self):
        return [
            Task(task_id="t1", instruction="", metadata={"risk_score": 0.1, "expected_escalate": False}),
            Task(task_id="t2", instruction="", metadata={"risk_score": 0.2, "expected_escalate": False}),
            Task(task_id="t3", instruction="", metadata={"risk_score": 0.8, "expected_escalate": True}),
            Task(task_id="t4", instruction="", metadata={"risk_score": 0.9, "expected_escalate": True}),
        ]


def _starting_artifact():
    return Artifact(
        scope=INTERRUPT, shape=ArtifactShape.PARAMETER, value={"search_space": {"threshold": [0.0, 1.0]}},
        optimizer_binding=None, technique=None, source="manual",
    )


def test_estimate_finds_a_threshold_that_separates_escalation_correctly():
    optimizer = InterruptThresholdOptimizer(n_trials=20, seed=0)
    result = optimizer.estimate(_starting_artifact(), corpus=None, benchmark=_EscalationBenchmark(), variant=None)
    assert 0.2 < result.value <= 0.8


def test_estimate_returns_parameter_shaped_estimated_artifact():
    optimizer = InterruptThresholdOptimizer(n_trials=10, seed=0)
    result = optimizer.estimate(_starting_artifact(), corpus=None, benchmark=_EscalationBenchmark(), variant=None)
    assert result.scope == INTERRUPT
    assert result.shape == ArtifactShape.PARAMETER
    assert result.source == "estimated"
    assert result.provenance["best_score"] == 1.0
