from agentgym.core.ab import ABComparison, ABResult, bootstrap_delta
from agentgym.core.artifact import Artifact, ArtifactShape
from agentgym.core.scope import PROMPT
from agentgym.core.score import Score
from agentgym.core.variant import AgentVariant


def test_bootstrap_delta_reproducible_with_seed():
    a = [0.5, 0.6, 0.55, 0.52, 0.58]
    b = [0.7, 0.72, 0.68, 0.71, 0.69]
    delta1, ci1 = bootstrap_delta(a, b, seed=42)
    delta2, ci2 = bootstrap_delta(a, b, seed=42)
    assert delta1 == delta2
    assert ci1 == ci2


def test_ab_result_wins_when_delta_positive_and_ci_excludes_zero():
    result = ABResult(delta=0.15, ci=(0.05, 0.25), candidate_wins=None)
    assert result.compute_wins() is True


def test_ab_result_does_not_win_when_ci_contains_zero():
    result = ABResult(delta=0.05, ci=(-0.02, 0.12), candidate_wins=None)
    assert result.compute_wins() is False


def test_ab_result_does_not_win_when_delta_negative():
    result = ABResult(delta=-0.1, ci=(-0.2, -0.02), candidate_wins=None)
    assert result.compute_wins() is False


def test_identical_variant_against_itself_never_wins():
    scores = [0.6, 0.62, 0.58, 0.61, 0.59]
    delta, ci = bootstrap_delta(scores, list(scores), seed=1)
    result = ABResult(delta=delta, ci=ci, candidate_wins=None)
    assert result.compute_wins() is False


class _FakeAgent:
    """Its 'performance' is entirely determined by the resolved PROMPT artifact's value —
    the only honest way to make a fake agent's behavior actually depend on which variant
    it was run with, matching how a real Agent would read its resolved artifacts."""

    def consumes(self):
        return {PROMPT: None}

    def run(self, task, artifacts):
        return {"task": task, "prompt_quality": artifacts[PROMPT].value}


class _FakeBenchmark:
    def cases(self):
        return ["task1", "task2", "task3", "task4", "task5"]


class _FakeVerifier:
    def score(self, trace, task):
        return [Score(metric_name="task_success", value=trace["prompt_quality"], kind="verifiable", evidence="")]


def _prompt_artifact(value):
    return Artifact(
        scope=PROMPT, shape=ArtifactShape.PARAMETER, value=value,
        optimizer_binding=None, technique=None, source="manual", provenance={},
    )


def test_ab_comparison_end_to_end_with_fakes():
    baseline = AgentVariant(artifacts={PROMPT: _prompt_artifact(0.5)})
    candidate = AgentVariant(artifacts={PROMPT: _prompt_artifact(0.9)})
    comparison = ABComparison(agent=_FakeAgent())
    result = comparison.compare(
        variant_a=baseline, variant_b=candidate,
        benchmark=_FakeBenchmark(), verifier=_FakeVerifier(),
    )
    assert result.delta > 0
    assert result.compute_wins() is True
