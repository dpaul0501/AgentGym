import pytest

from agentgym.core.artifact import Artifact, ArtifactShape
from agentgym.core.scope import GRAPH, PROMPT, Lever
from agentgym.core.variant import AgentVariant, MissingScopeError


def make_artifact(scope, value="v", source="manual"):
    return Artifact(
        scope=scope, shape=ArtifactShape.CONFIG, value=value,
        optimizer_binding=None, technique=None, source=source,
        provenance={} if source == "manual" else {"run": "x"},
    )


def test_resolve_returns_bound_artifact():
    a = make_artifact(PROMPT, "bound")
    variant = AgentVariant(artifacts={PROMPT: a})
    default = make_artifact(PROMPT, "default")
    assert variant.resolve(PROMPT, default=default).value == "bound"


def test_resolve_falls_back_to_default_when_unbound():
    variant = AgentVariant(artifacts={})
    default = make_artifact(PROMPT, "default")
    assert variant.resolve(PROMPT, default=default).value == "default"


def test_with_artifact_returns_new_instance():
    variant = AgentVariant(artifacts={})
    new_artifact = make_artifact(PROMPT, "new")
    updated = variant.with_artifact(PROMPT, new_artifact)
    assert PROMPT not in variant.artifacts
    assert updated.artifacts[PROMPT].value == "new"


def test_union_across_levers_no_collision():
    variant = AgentVariant(artifacts={})
    variant = variant.with_artifact(PROMPT, make_artifact(PROMPT, "p"))
    variant = variant.with_artifact(GRAPH, make_artifact(GRAPH, "g"))
    assert variant.artifacts[PROMPT].value == "p"
    assert variant.artifacts[GRAPH].value == "g"


def test_scopes_under_lever_filters_correctly():
    variant = AgentVariant(artifacts={PROMPT: make_artifact(PROMPT), GRAPH: make_artifact(GRAPH)})
    context_scopes = variant.scopes_under(Lever.CONTEXT)
    assert list(context_scopes.keys()) == [PROMPT]


def test_resolve_for_returns_all_requested_scopes():
    variant = AgentVariant(artifacts={PROMPT: make_artifact(PROMPT, "bound")})
    consumes = {PROMPT: None, GRAPH: make_artifact(GRAPH, "default-graph")}
    resolved = variant.resolve_for(consumes)
    assert resolved[PROMPT].value == "bound"
    assert resolved[GRAPH].value == "default-graph"


def test_resolve_for_raises_missing_scope_error():
    variant = AgentVariant(artifacts={})
    consumes = {PROMPT: None}  # no binding, no default
    with pytest.raises(MissingScopeError):
        variant.resolve_for(consumes)
