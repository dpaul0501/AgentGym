import pytest

from agentgym.core.artifact import Artifact, ArtifactShape, FineTuneTechnique
from agentgym.core.scope import PROMPT, MODEL_WEIGHTS


def test_artifact_manual_construction():
    a = Artifact(
        scope=PROMPT,
        shape=ArtifactShape.CONFIG,
        value={"template": "answer directly"},
        optimizer_binding=None,
        technique=None,
        source="manual",
        provenance={},
    )
    assert a.source == "manual"
    assert a.value == {"template": "answer directly"}


def test_artifact_estimated_requires_provenance():
    with pytest.raises(ValueError):
        Artifact(
            scope=PROMPT,
            shape=ArtifactShape.CONFIG,
            value={},
            optimizer_binding="dspy.MIPROv2",
            technique=None,
            source="estimated",
            provenance={},
        )


def test_artifact_estimated_with_provenance_ok():
    a = Artifact(
        scope=PROMPT,
        shape=ArtifactShape.PROGRAM,
        value=[],
        optimizer_binding="dspy.MIPROv2",
        technique=None,
        source="estimated",
        provenance={"n_examples": 12},
    )
    assert a.provenance["n_examples"] == 12


def test_technique_rejected_on_non_model_weights_scope():
    with pytest.raises(ValueError):
        Artifact(
            scope=PROMPT,
            shape=ArtifactShape.CONFIG,
            value={},
            optimizer_binding=None,
            technique=FineTuneTechnique.LORA_ADAPTER,
            source="manual",
            provenance={},
        )


def test_technique_allowed_on_model_weights_scope():
    a = Artifact(
        scope=MODEL_WEIGHTS,
        shape=ArtifactShape.WEIGHTS,
        value="s3://adapters/run-1",
        optimizer_binding="unsloth",
        technique=FineTuneTechnique.LORA_ADAPTER,
        source="estimated",
        provenance={"run_id": "run-1"},
    )
    assert a.technique is FineTuneTechnique.LORA_ADAPTER


def test_program_artifact_nests_artifacts():
    stage1 = Artifact(
        scope=PROMPT, shape=ArtifactShape.PARAMETER, value="instruction template",
        optimizer_binding=None, technique=None, source="manual", provenance={},
    )
    stage2 = Artifact(
        scope=PROMPT, shape=ArtifactShape.PARAMETER, value="few-shot selector",
        optimizer_binding=None, technique=None, source="manual", provenance={},
    )
    program = Artifact(
        scope=PROMPT, shape=ArtifactShape.PROGRAM, value=[stage1, stage2],
        optimizer_binding="dspy.MIPROv2", technique=None,
        source="estimated", provenance={"n_examples": 3},
    )
    assert len(program.value) == 2
    assert program.value[0].value == "instruction template"


def test_artifact_shape_enum_membership():
    assert {s.value for s in ArtifactShape} == {
        "parameter", "config", "program", "ensemble", "weights",
    }


def test_finetune_technique_enum_membership():
    assert {t.value for t in FineTuneTechnique} == {
        "lora_adapter", "full_finetune", "rlhf",
    }
