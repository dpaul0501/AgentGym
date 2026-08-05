from agentgym.core.artifact import Artifact, ArtifactShape
from agentgym.core.protocols import Task
from agentgym.core.scope import TOOLS
from agentgym.optimizers.tool_selector import ToolSelectorOptimizer


class _ToolBenchmark:
    def cases(self):
        return [
            Task(task_id="t1", instruction="", metadata={"required_tools": ["search"]}),
            Task(task_id="t2", instruction="", metadata={"required_tools": ["search", "calculator"]}),
            Task(task_id="t3", instruction="", metadata={"required_tools": ["calculator"]}),
        ]


def _starting_artifact():
    return Artifact(
        scope=TOOLS, shape=ArtifactShape.ENSEMBLE,
        value={"candidate_tools": ["search", "calculator", "weather", "unused_tool"]},
        optimizer_binding=None, technique=None, source="manual",
    )


def test_estimate_selects_the_tools_that_cover_every_task():
    optimizer = ToolSelectorOptimizer(n_trials=40, seed=0)
    result = optimizer.estimate(_starting_artifact(), corpus=None, benchmark=_ToolBenchmark(), variant=None)

    assert result.scope == TOOLS
    assert result.shape == ArtifactShape.ENSEMBLE
    assert "search" in result.value["enabled_tools"]
    assert "calculator" in result.value["enabled_tools"]


def test_estimate_excludes_unneeded_tools_due_to_count_penalty():
    optimizer = ToolSelectorOptimizer(n_trials=40, seed=0)
    result = optimizer.estimate(_starting_artifact(), corpus=None, benchmark=_ToolBenchmark(), variant=None)
    assert "unused_tool" not in result.value["enabled_tools"]
    assert "weather" not in result.value["enabled_tools"]


def test_estimate_returns_estimated_artifact_with_provenance():
    optimizer = ToolSelectorOptimizer(n_trials=10, seed=0)
    result = optimizer.estimate(_starting_artifact(), corpus=None, benchmark=_ToolBenchmark(), variant=None)
    assert result.source == "estimated"
    assert result.optimizer_binding == "optuna.TPESampler"
    assert result.provenance["n_trials"] == 10
