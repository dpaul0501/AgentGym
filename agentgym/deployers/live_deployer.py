"""LiveDeployer: the v1 reference Deployer backed by real (or realistically staged) traffic,
replacing OfflineDeployer's benchmark-replay simulation. Unlike OfflineDeployer, which re-runs the
Agent against a fixed Benchmark to synthesize an A/B comparison, LiveDeployer has no benchmark at
all — it routes real incoming requests between the baseline and candidate variant (weighted by
traffic_pct) and accumulates real scored outcomes as they happen, computing the same real
bootstrap-CI statistics (agentgym.core.ab.bootstrap_delta) over genuinely observed results rather
than a replayed benchmark.

This is the honest boundary this design draws around itself: AgentGym does not run its own
production traffic-serving infrastructure (a load balancer, request router, or live service) —
that's real infrastructure a caller must already have. What LiveDeployer provides is the decision
layer any such caller wires real traffic through: `route()` picks baseline or candidate for each
real request per the released traffic_pct, and `record_outcome()` accumulates the real result.
`ab_compare` refuses to report a result until enough real outcomes have actually been observed —
it never falls back to a synthetic number.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from agentgym.core.ab import ABResult, bootstrap_delta
from agentgym.core.protocols import Release

MIN_OUTCOMES_FOR_AB = 30


@dataclass
class _LiveReleaseState:
    variant: Any
    traffic_pct: float
    baseline_outcomes: list[float] = field(default_factory=list)
    candidate_outcomes: list[float] = field(default_factory=list)


class LiveDeployer:
    def __init__(self, seed: int = 0, min_outcomes_for_ab: int = MIN_OUTCOMES_FOR_AB):
        self._seed = seed
        self.min_outcomes_for_ab = min_outcomes_for_ab

    def release(self, variant: Any, traffic_pct: float) -> Release:
        release = Release(variant=variant)
        release.live = _LiveReleaseState(variant=variant, traffic_pct=traffic_pct)
        return release

    def route(self, release: Release, request_id: str, baseline_variant: Any) -> Any:
        """Called by the real caller for each real incoming request. Deterministic per
        request_id, so the same request always routes to the same variant within one release —
        a real requirement for a live A/B (no flip-flopping mid-session)."""
        bucket = (hash((self._seed, request_id)) % 10_000) / 10_000
        return release.live.variant if bucket < release.live.traffic_pct else baseline_variant

    def record_outcome(self, release: Release, used_variant: Any, score: float) -> None:
        """Called by the real caller after a real request completes and is scored. This is the
        only source of truth ab_compare uses — no benchmark replay."""
        live = release.live
        if used_variant is live.variant:
            live.candidate_outcomes.append(score)
        else:
            live.baseline_outcomes.append(score)

    def ab_compare(self, baseline: Any, candidate: Any, release: Release) -> ABResult:
        live = release.live
        if len(live.baseline_outcomes) < self.min_outcomes_for_ab or len(live.candidate_outcomes) < self.min_outcomes_for_ab:
            raise ValueError(
                "not enough real observed outcomes yet for a statistically meaningful A/B "
                f"(need >= {self.min_outcomes_for_ab} each; have "
                f"{len(live.baseline_outcomes)} baseline / {len(live.candidate_outcomes)} candidate)"
            )
        delta, ci = bootstrap_delta(live.baseline_outcomes, live.candidate_outcomes, seed=self._seed)
        result = ABResult(delta=delta, ci=ci, candidate_wins=None)
        result.compute_wins()
        return result

    def launch(self, release: Release) -> None:
        pass  # promotion (candidate becomes `current`) is Harness's job; LiveDeployer's part
              # of the contract ends once ab_compare reports a real win

    def rollback(self, release: Release) -> None:
        release.live.traffic_pct = 0.0  # stop routing any further real requests to the candidate
