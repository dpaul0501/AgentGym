import pytest

from agentgym.core.diagnosis import Diagnosis
from agentgym.core.scope import GRAPH, Lever, PROMPT


def test_diagnosis_with_scope_requires_lever_match():
    with pytest.raises(ValueError):
        Diagnosis(
            trace_id="t1",
            lever=Lever.HARNESS,
            evidence=["retries without a verify span"],
            confidence=0.8,
            suggested_scope=PROMPT,  # a CONTEXT scope, but lever says HARNESS
        )


def test_diagnosis_with_matching_lever_and_scope_ok():
    d = Diagnosis(
        trace_id="t1",
        lever=Lever.HARNESS,
        evidence=["retries without a verify span"],
        confidence=0.8,
        suggested_scope=GRAPH,
    )
    assert d.suggested_scope is GRAPH


def test_insufficient_evidence_has_no_scope():
    d = Diagnosis(
        trace_id="t1",
        lever=None,
        evidence=["trace missing tool-call spans, cannot localize failure"],
        confidence=0.0,
        suggested_scope=None,
    )
    assert d.suggested_scope is None
    assert d.evidence


def test_insufficient_evidence_requires_a_reason():
    with pytest.raises(ValueError):
        Diagnosis(
            trace_id="t1",
            lever=None,
            evidence=[],
            confidence=0.0,
            suggested_scope=None,
        )


def test_confidence_bounded_below():
    with pytest.raises(ValueError):
        Diagnosis(
            trace_id="t1", lever=Lever.CONTEXT, evidence=["x"], confidence=-0.1,
            suggested_scope=PROMPT,
        )


def test_confidence_bounded_above():
    with pytest.raises(ValueError):
        Diagnosis(
            trace_id="t1", lever=Lever.CONTEXT, evidence=["x"], confidence=1.1,
            suggested_scope=PROMPT,
        )
