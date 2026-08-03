import pytest

from agentgym.core.score import Score


def test_score_verifiable_construction():
    s = Score(metric_name="slot_fill_f1", value=0.92, kind="verifiable", evidence="exact match")
    assert s.kind == "verifiable"


def test_score_plausible_construction():
    s = Score(metric_name="tone", value=0.7, kind="plausible", evidence="judge rubric")
    assert s.kind == "plausible"


def test_score_invalid_kind_rejected():
    with pytest.raises(ValueError):
        Score(metric_name="x", value=1.0, kind="vibes", evidence="")
