"""ABComparison: first-class, not implicit. Runs two AgentVariants on the same benchmark, scores
both with the same Verifier, and reports a bootstrap-CI delta. A positive point-estimate delta
alone is not a win — the confidence interval must exclude zero too."""

from __future__ import annotations

import random
from dataclasses import dataclass

from agentgym.core.score import primary_reward


def bootstrap_delta(
    scores_a: list[float], scores_b: list[float], seed: int, n_resamples: int = 2000
) -> tuple[float, tuple[float, float]]:
    rng = random.Random(seed)
    point_delta = (sum(scores_b) / len(scores_b)) - (sum(scores_a) / len(scores_a))

    deltas = []
    for _ in range(n_resamples):
        resample_a = [rng.choice(scores_a) for _ in scores_a]
        resample_b = [rng.choice(scores_b) for _ in scores_b]
        deltas.append((sum(resample_b) / len(resample_b)) - (sum(resample_a) / len(resample_a)))
    deltas.sort()
    lo = deltas[int(0.025 * n_resamples)]
    hi = deltas[int(0.975 * n_resamples) - 1]
    return point_delta, (lo, hi)


@dataclass
class ABResult:
    delta: float
    ci: tuple[float, float]
    candidate_wins: bool | None  # None until compute_wins() is called, or set at construction

    def compute_wins(self) -> bool:
        lo, hi = self.ci
        ci_excludes_zero = lo > 0 or hi < 0
        wins = self.delta > 0 and ci_excludes_zero
        self.candidate_wins = wins
        return wins


class ABComparison:
    def __init__(self, agent):
        self.agent = agent

    def _run_all(self, variant, benchmark, verifier) -> list[float]:
        resolved = variant.resolve_for(self.agent.consumes())
        scores = []
        for task in benchmark.cases():
            trace = self.agent.run(task, resolved)
            scores.append(primary_reward(verifier.score(trace, task)))
        return scores

    def compare(self, variant_a, variant_b, benchmark, verifier, seed: int = 0) -> ABResult:
        scores_a = self._run_all(variant_a, benchmark, verifier)
        scores_b = self._run_all(variant_b, benchmark, verifier)
        delta, ci = bootstrap_delta(scores_a, scores_b, seed=seed)
        result = ABResult(delta=delta, ci=ci, candidate_wins=None)
        result.compute_wins()
        return result
