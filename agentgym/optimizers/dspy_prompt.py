"""DSPyPromptOptimizer: the real reference ScopeOptimizer for Scope.PROMPT. Wraps DSPy's own
few-shot/instruction search (BootstrapFewShot by default; MIPROv2 is a heavier drop-in swap) —
this file does not reimplement prompt optimization, it binds DSPy's real compile step to the
Artifact/TrainingCorpus/Benchmark protocol.
"""

from __future__ import annotations

from typing import Any

import dspy

from agentgym.core.artifact import Artifact, ArtifactShape
from agentgym.core.scope import PROMPT, TOOLS


def _normalize(text: str) -> str:
    return str(text).strip().lower().rstrip(".")


def _corpus_examples_to_dspy(records: list[dict]) -> list[dspy.Example]:
    """Real (task, good_trajectory) pairs from the corpus -> dspy.Example objects. Expects each
    record's trace.metadata to carry the training signal directly (instruction + gold answer) —
    the shape a captured trace needs for this scope to be trainable at all."""
    examples = []
    for record in records:
        meta = record["trace"].metadata
        if "instruction" not in meta or "answer" not in meta:
            continue
        examples.append(
            dspy.Example(question=meta["instruction"], answer=meta["answer"]).with_inputs("question")
        )
    return examples


class DSPyPromptOptimizer:
    scope = PROMPT
    depends_on = [TOOLS]

    def __init__(self, model_id: str = "ollama_chat/qwen2.5:7b", api_base: str = "http://localhost:11434",
                 max_bootstrapped_demos: int = 4, max_rounds: int = 1):
        self.model_id = model_id
        self.api_base = api_base
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_rounds = max_rounds

    def _metric(self, example: dspy.Example, prediction: Any, trace=None) -> bool:
        return _normalize(getattr(prediction, "answer", "")) == _normalize(example.answer)

    def estimate(self, artifact: Artifact, corpus, benchmark, variant) -> Artifact:
        examples = _corpus_examples_to_dspy(corpus.artifact_candidates(PROMPT))
        if not examples:
            raise ValueError(
                "DSPyPromptOptimizer needs at least one corpus example with "
                "trace.metadata['instruction']/['answer'] to compile against"
            )

        lm = dspy.LM(self.model_id, api_base=self.api_base, api_key="")
        dspy.configure(lm=lm)

        base_program = dspy.Predict("question -> answer")
        optimizer = dspy.BootstrapFewShot(
            metric=self._metric,
            max_bootstrapped_demos=self.max_bootstrapped_demos,
            max_rounds=self.max_rounds,
        )
        compiled = optimizer.compile(base_program, trainset=examples)

        demos = [{"question": d.question, "answer": d.answer} for d in compiled.demos]
        return Artifact(
            scope=PROMPT,
            shape=ArtifactShape.PROGRAM,
            value={"signature": "question -> answer", "demos": demos},
            optimizer_binding="dspy.BootstrapFewShot",
            technique=None,
            source="estimated",
            provenance={"n_examples": len(examples), "n_demos": len(demos), "model": self.model_id},
        )
