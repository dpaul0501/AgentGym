"""TrainingCorpus: the data-collection interface. Native representation is trajectory-based
(state/action/observation/reward per step) — the same shape as OpenPipe ART's Trajectory
abstraction — not a chat-messages log. Converting to chat-messages (what TRL/Unsloth actually
consume) is a separate, named, lossy step: to_chat_messages()."""

from __future__ import annotations

from dataclasses import dataclass, field

from agentgym.core.diagnosis import Diagnosis
from agentgym.core.score import Score, primary_reward
from agentgym.core.scope import Scope
from agentgym.core.trace import Trace


@dataclass
class _Record:
    trace: Trace
    scores: list[Score]
    diagnosis: Diagnosis


@dataclass
class TrainingCorpus:
    _records: list[_Record] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self._records)

    def add_trace(self, trace: Trace, scores: list[Score], diagnosis: Diagnosis) -> None:
        self._records.append(_Record(trace=trace, scores=scores, diagnosis=diagnosis))

    def artifact_candidates(self, scope: Scope) -> list[dict]:
        return [
            {"trace": r.trace, "scores": r.scores, "diagnosis": r.diagnosis}
            for r in self._records
            if r.diagnosis.suggested_scope == scope
        ]

    def sft_rows(self, scope: Scope | None = None) -> list[dict]:
        """Native trajectory shape: one row per captured trace, with nested per-step state/
        action/observation, plus the trace's overall reward."""
        rows = []
        for r in self._records:
            if scope is not None and r.diagnosis.suggested_scope != scope:
                continue
            steps = [
                {
                    "state": span.input,
                    "action": span.name,
                    "observation": span.output,
                    "kind": span.kind,
                }
                for span in r.trace.spans
            ]
            rows.append(
                {
                    "task_id": r.trace.task_id,
                    "trace_id": r.trace.trace_id,
                    "steps": steps,
                    "reward": primary_reward(r.scores),
                }
            )
        return rows

    def to_chat_messages(self, row: dict) -> list[dict]:
        """Real, lossy, named conversion from the native trajectory row into the chat-messages
        format TRL/Unsloth consume (role/content, tool_calls on assistant messages, a tool role
        for results). Confirm the current exact tool-calling chat-template convention against
        TRL/Unsloth's real API before relying on this for a live training run."""
        messages: list[dict] = []
        for step in row["steps"]:
            if step["kind"] == "tool":
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{"name": step["action"], "arguments": step["state"]}],
                    }
                )
                messages.append(
                    {"role": "tool", "content": step["observation"], "name": step["action"]}
                )
            else:
                messages.append({"role": "user", "content": step["state"]})
                messages.append({"role": "assistant", "content": step["observation"]})
        return messages

    def dpo_pairs(self) -> list[dict]:
        """Pairs two trajectories on the same task with different rewards into a (chosen,
        rejected) row. A task with only one trajectory produces no pair — DPO needs a real
        comparison, not a trajectory paired with a null."""
        by_task: dict[str, list[dict]] = {}
        for row in self.sft_rows():
            by_task.setdefault(row["task_id"], []).append(row)

        pairs = []
        for rows in by_task.values():
            if len(rows) < 2:
                continue
            rows_sorted = sorted(rows, key=lambda r: r["reward"], reverse=True)
            best, worst = rows_sorted[0], rows_sorted[-1]
            if best["reward"] == worst["reward"]:
                continue
            pairs.append({"chosen": best, "rejected": worst})
        return pairs
