import dataclasses

import pytest

from agentgym.core.scope import Lever, Scope


def test_scope_str_is_dotted_path():
    assert str(Scope(Lever.CONTEXT, ("prompt",))) == "context.prompt"


def test_scope_str_nested():
    assert str(Scope(Lever.CONTEXT, ("retrieval", "reranker"))) == "context.retrieval.reranker"


def test_is_child_of_true_for_direct_child():
    retrieval = Scope(Lever.CONTEXT, ("retrieval",))
    reranker = Scope(Lever.CONTEXT, ("retrieval", "reranker"))
    assert reranker.is_child_of(retrieval) is True


def test_is_child_of_false_across_levers():
    context_retrieval = Scope(Lever.CONTEXT, ("retrieval",))
    harness_graph = Scope(Lever.HARNESS, ("graph",))
    assert harness_graph.is_child_of(context_retrieval) is False


def test_is_child_of_false_for_self():
    prompt = Scope(Lever.CONTEXT, ("prompt",))
    assert prompt.is_child_of(prompt) is False


def test_is_child_of_false_for_unrelated_sibling():
    prompt = Scope(Lever.CONTEXT, ("prompt",))
    memory = Scope(Lever.CONTEXT, ("memory",))
    assert memory.is_child_of(prompt) is False


def test_scope_equality_by_value():
    a = Scope(Lever.CONTEXT, ("prompt",))
    b = Scope(Lever.CONTEXT, ("prompt",))
    assert a == b


def test_scope_hashable_usable_as_dict_key():
    a = Scope(Lever.CONTEXT, ("prompt",))
    b = Scope(Lever.CONTEXT, ("prompt",))
    d = {a: "value"}
    assert d[b] == "value"


def test_scope_is_frozen():
    s = Scope(Lever.CONTEXT, ("prompt",))
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.segments = ("tools",)  # type: ignore[misc]


def test_lever_has_three_members():
    assert {Lever.CONTEXT.value, Lever.HARNESS.value, Lever.FINE_TUNE.value} == {
        "context",
        "harness",
        "fine_tune",
    }
