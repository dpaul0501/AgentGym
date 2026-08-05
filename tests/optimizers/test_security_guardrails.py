"""GuardrailsOptimizer against the real, installed garak package, run against a real local Ollama
target. Uses garak's own designated fast smoke probe (test.Blank) to keep this test's wall-clock
time reasonable — timing a real vulnerability probe (promptinject.HijackKillHumans) against a
local 7B model showed it takes multiple minutes, confirmed directly rather than assumed; that
probe is GuardrailsOptimizer's real default for an actual security audit, just not for this test.
"""

import pytest

from agentgym.core.artifact import Artifact, ArtifactShape
from agentgym.core.scope import GUARDRAILS
from agentgym.optimizers.security_guardrails import GuardrailsOptimizer

pytestmark = pytest.mark.integration


def _starting_artifact() -> Artifact:
    return Artifact(scope=GUARDRAILS, shape=ArtifactShape.CONFIG, value={}, optimizer_binding=None,
                     technique=None, source="manual")


def test_estimate_runs_real_garak_probe_and_returns_config_artifact():
    optimizer = GuardrailsOptimizer(
        target_type="ollama", target_name="qwen2.5:7b", probes=["test.Blank"], generations=1,
    )
    result = optimizer.estimate(_starting_artifact(), corpus=None, benchmark=None, variant=None)

    assert result.scope == GUARDRAILS
    assert result.shape == ArtifactShape.CONFIG
    assert result.source == "estimated"
    assert result.optimizer_binding == "garak+invariant"
    assert "policy_dsl" in result.value
    assert "prompt_injection_threshold" in result.value
    assert result.provenance["probes_run"] == ["test.Blank"]
    assert len(result.provenance["evals"]) >= 1
    assert result.provenance["evals"][0]["probe"] == "test.Blank"
