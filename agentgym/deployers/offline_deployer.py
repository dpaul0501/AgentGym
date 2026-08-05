"""OfflineDeployer: the v0/v1 reference Deployer. Implements Release/A-B/Launch as benchmark
replay rather than real traffic — honestly labeled as a simulation, not a production A/B. Reuses
the real ABComparison/bootstrap_delta machinery (agentgym/core/ab.py) rather than reimplementing
statistical comparison.

A Deployer backed by real (or realistically staged) traffic is real infrastructure work of its
own kind — see agentgym/deployers/live_deployer.py for that reference. OfflineDeployer stays the
honest, cheap default for offline development and for any Harness run without a live traffic
target wired up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentgym.core.ab import ABComparison, ABResult
from agentgym.core.protocols import Benchmark, Release, Verifier


@dataclass
class OfflineDeployer:
    agent: Any
    benchmark: Benchmark
    verifier: Verifier
    seed: int = 0

    def release(self, variant: Any, traffic_pct: float) -> Release:
        return Release(variant=variant)

    def ab_compare(self, baseline: Any, candidate: Any, release: Release) -> ABResult:
        comparison = ABComparison(self.agent)
        return comparison.compare(
            baseline, release.variant, self.benchmark, self.verifier, seed=self.seed,
        )

    def launch(self, release: Release) -> None:
        pass  # v0: no real rollout target to promote to; the candidate becomes `current` in Harness

    def rollback(self, release: Release) -> None:
        pass  # v0: nothing was actually shipped to roll back
