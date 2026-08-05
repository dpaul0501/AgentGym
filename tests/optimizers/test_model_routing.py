from agentgym.core.artifact import Artifact, ArtifactShape
from agentgym.core.protocols import Task
from agentgym.core.scope import MODEL_ROUTING
from agentgym.optimizers.model_routing import ModelRoutingOptimizer


class _RoutingBenchmark:
    def cases(self):
        return [
            Task(task_id="t1", instruction="", metadata={"complexity": 1, "expected_tier": "small"}),
            Task(task_id="t2", instruction="", metadata={"complexity": 2, "expected_tier": "small"}),
            Task(task_id="t3", instruction="", metadata={"complexity": 8, "expected_tier": "large"}),
            Task(task_id="t4", instruction="", metadata={"complexity": 9, "expected_tier": "large"}),
        ]


def _starting_artifact():
    return Artifact(
        scope=MODEL_ROUTING, shape=ArtifactShape.PARAMETER, value={"search_space": {"threshold": [0, 10]}},
        optimizer_binding=None, technique=None, source="manual",
    )


def test_estimate_finds_a_threshold_that_separates_tiers_correctly():
    optimizer = ModelRoutingOptimizer(n_trials=20, seed=0)
    result = optimizer.estimate(_starting_artifact(), corpus=None, benchmark=_RoutingBenchmark(), variant=None)
    assert 2 < result.value <= 8


def test_estimate_returns_parameter_shaped_estimated_artifact():
    optimizer = ModelRoutingOptimizer(n_trials=10, seed=0)
    result = optimizer.estimate(_starting_artifact(), corpus=None, benchmark=_RoutingBenchmark(), variant=None)
    assert result.scope == MODEL_ROUTING
    assert result.shape == ArtifactShape.PARAMETER
    assert result.source == "estimated"
    assert result.provenance["best_score"] == 1.0
