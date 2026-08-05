"""TravelBenchmark: real Task prompts driving TravelSageAgent, used by both demo scripts."""

from __future__ import annotations

from agentgym.core.protocols import Task


class TravelBenchmark:
    def cases(self) -> list[Task]:
        return [
            Task(
                task_id="flight_search_sf_nyc",
                instruction="Find me flights from San Francisco to New York next Tuesday and list the options.",
                metadata={},
            ),
            Task(
                task_id="hotel_search_paris",
                instruction="Find me a hotel in Paris for Sept 1-4 and summarize the best option with its review.",
                metadata={},
            ),
        ]
