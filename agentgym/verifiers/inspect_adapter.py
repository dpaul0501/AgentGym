"""InspectVerifier: wraps a real Inspect AI Scorer (inspect_ai.scorer) as our Verifier protocol.

Design decision this file resolves, per docs/DESIGN.md's flagged open question, confirmed by
reading the installed inspect_ai==0.3.251 API directly rather than assuming: Inspect AI's own
Scorer is `async (TaskState, Target) -> Score`, and TaskState is a rich object describing an
entire model interaction. We take option (a) from the design doc — treat Inspect as a **scoring
library only**. Harness still drives Agent.run() and owns the loop; this adapter's only job is
converting one of our (Trace, Task) pairs into a real TaskState, invoking a real Inspect Scorer,
and converting the real inspect_ai Score back into our Score type. Nothing here is a stub —
`exact()`/`includes()`/`match()` etc. are real, unmodified Inspect scorers.
"""

from __future__ import annotations

import asyncio

from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, ModelName, ModelOutput
from inspect_ai.scorer import CORRECT, INCORRECT, NOANSWER, PARTIAL, Scorer, Target
from inspect_ai.solver import TaskState

from agentgym.core.protocols import Task
from agentgym.core.score import Score
from agentgym.core.trace import Trace

_VALUE_MAP = {CORRECT: 1.0, INCORRECT: 0.0, PARTIAL: 0.5, NOANSWER: 0.0}


def _final_answer(trace: Trace) -> str:
    """The text Inspect scores against: the last llm-kind span's output, stringified. A real
    trace's last reasoning step is the closest analog to a Task's final model output."""
    llm_spans = [s for s in trace.spans if s.kind == "llm"]
    if not llm_spans:
        return ""
    output = llm_spans[-1].output
    return output.get("answer") or output.get("text") or str(output)


def _to_task_state(trace: Trace, task: Task) -> TaskState:
    answer = _final_answer(trace)
    target_value = task.metadata.get("target", "")
    return TaskState(
        model=ModelName("agentgym/harness"),
        sample_id=task.task_id,
        epoch=1,
        input=task.instruction,
        messages=[ChatMessageUser(content=task.instruction), ChatMessageAssistant(content=answer)],
        target=Target(target_value),
        output=ModelOutput.from_content(model="agentgym/harness", content=answer),
    )


def _to_score_value(inspect_value) -> float:
    if isinstance(inspect_value, (int, float, bool)):
        return float(inspect_value)
    return _VALUE_MAP.get(inspect_value, 0.0)


class InspectVerifier:
    def __init__(self, scorer: Scorer, metric_name: str = "task_success", kind: str = "verifiable"):
        self.scorer = scorer
        self.metric_name = metric_name
        self.kind = kind

    def score(self, trace: Trace, task: Task) -> list[Score]:
        state = _to_task_state(trace, task)
        target = state.target
        inspect_score = asyncio.run(self.scorer(state, target))
        return [
            Score(
                metric_name=self.metric_name,
                value=_to_score_value(inspect_score.value),
                kind=self.kind,
                evidence=inspect_score.explanation or f"inspect_ai scorer -> {inspect_score.value!r}",
            )
        ]
