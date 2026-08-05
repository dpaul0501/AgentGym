"""Tau2Benchmark: real task data from Sierra's tau2-bench (formerly tau-bench/tau2-bench, now
tau3-bench upstream), sourced directly from sierra-research/tau2-bench's actual tasks.json — the
`mock` domain, which tau2-bench's own docs describe as the "lightweight test domain for
development." Not invented data.

Scope decision, made explicitly rather than silently: tau2-bench's full evaluation also runs a
live LLM user-simulator and its own tool-execution sandbox (`tau2 run`) — real infrastructure of
its own, a heavier dependency than this benchmark adapter takes on. This class exposes the real
tasks and their real evaluation_criteria for scoring via Tau2ActionVerifier, which checks the
ACTION component of tau2's own reward_basis methodology against whatever Agent is under test in
Harness — it does not reproduce tau2's user-simulator turn-taking.
"""

from __future__ import annotations

import json
from pathlib import Path

from agentgym.core.protocols import Task

_DATA_DIR = Path(__file__).parent / "data"


class Tau2Benchmark:
    def __init__(self, domain: str = "mock", path: Path | None = None):
        self.domain = domain
        data_path = path or (_DATA_DIR / f"tau2_{domain}_tasks.json")
        if not data_path.exists():
            raise FileNotFoundError(
                f"no real tau2-bench task data bundled for domain={domain!r} at {data_path}"
            )
        raw_tasks = json.loads(data_path.read_text())
        self._tasks = [self._to_task(t) for t in raw_tasks]

    def _to_task(self, raw: dict) -> Task:
        instruction = raw.get("ticket") or raw["user_scenario"]["instructions"]
        return Task(
            task_id=raw["id"],
            instruction=instruction,
            metadata={
                "domain": self.domain,
                "evaluation_criteria": raw.get("evaluation_criteria", {}),
                "description": raw.get("description", {}),
            },
        )

    def cases(self) -> list[Task]:
        return list(self._tasks)
