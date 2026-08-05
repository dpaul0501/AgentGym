"""Harness: the orchestrator implementing the full lifecycle — Existing -> Evaluate -> Diagnose ->
Improve -> Evaluate -> Release -> A/B -> Launch — every stage a real interface over real tools,
never skipped or faked."""

from __future__ import annotations

from dataclasses import dataclass, field

from agentgym.core.corpus import TrainingCorpus
from agentgym.core.protocols import Task
from agentgym.core.score import Score, primary_reward
from agentgym.core.trace import Trace
from agentgym.core.variant import AgentVariant

SMALL_CANARY = 0.05


@dataclass
class CycleReport:
    status: str  # "no_action" | "rejected_offline" | "rejected_ab" | "launched"
    variant: AgentVariant
    ab_result: object | None = None
    action: object | None = None  # the Action this cycle attempted, if any — lets a LearningRecipe
                                   # inspect report_history to see which scopes were already tried


def offline_improved(scores_before: list[list[Score]], scores_after: list[list[Score]]) -> bool:
    """The sanity gate between Improve and Release: does the candidate look better on the exact
    same benchmark, before it touches anything real? A simple mean comparison — not the full
    bootstrap-CI statistical test A/B uses, since this is a cheap pre-check, not the final word."""
    before = sum(primary_reward(s) for s in scores_before) / len(scores_before)
    after = sum(primary_reward(s) for s in scores_after) / len(scores_after)
    return after > before


class Harness:
    def __init__(self, agent, benchmark, verifier, diagnoser, optimizers, recipe, deployer,
                 store, corpus: TrainingCorpus | None = None):
        self.agent = agent
        self.benchmark = benchmark
        self.verifier = verifier
        self.diagnoser = diagnoser
        self.optimizers = optimizers
        self.recipe = recipe
        self.deployer = deployer
        self.store = store
        self.corpus = corpus if corpus is not None else TrainingCorpus()
        self.history: list[CycleReport] = []

    def _evaluate(self, variant: AgentVariant) -> tuple[list[Trace], list[list[Score]]]:
        resolved = variant.resolve_for(self.agent.consumes())  # raises MissingScopeError early
        traces: list[Trace] = []
        scores: list[list[Score]] = []
        for task in self.benchmark.cases():
            trace = self.agent.run(task, resolved)
            traces.append(trace)
            scores.append(self.verifier.score(trace, task))
        return traces, scores

    def cycle(self, baseline: AgentVariant) -> CycleReport:
        current = baseline  # 1. EXISTING

        traces, scores = self._evaluate(current)  # 2. EVALUATE
        for trace, trace_scores in zip(traces, scores):  # 3. DIAGNOSE
            diagnosis = self.diagnoser.diagnose(trace, trace_scores)
            self.corpus.add_trace(trace, trace_scores, diagnosis)

        action = self.recipe.next_action(self.history, current)  # 4. IMPROVE (choose)
        if action is None:
            report = CycleReport(status="no_action", variant=current)
            self.history.append(report)
            return report

        optimizer = self.optimizers[action.scope]
        base_artifact = current.artifacts.get(action.scope)
        new_artifact = optimizer.estimate(base_artifact, self.corpus, self.benchmark, current)
        candidate = current.with_artifact(action.scope, new_artifact)  # 4. IMPROVE (apply)

        _, candidate_scores = self._evaluate(candidate)  # 5. EVALUATE (candidate)
        if not offline_improved(scores, candidate_scores):
            report = CycleReport(status="rejected_offline", variant=current, action=action)
            self.history.append(report)
            return report

        release = self.deployer.release(candidate, traffic_pct=SMALL_CANARY)  # 6. RELEASE
        ab = self.deployer.ab_compare(baseline=current, candidate=candidate, release=release)  # 7. A/B
        if not ab.candidate_wins:
            self.deployer.rollback(release)
            report = CycleReport(status="rejected_ab", variant=current, ab_result=ab, action=action)
            self.history.append(report)
            return report

        self.deployer.launch(release)  # 8. LAUNCH
        self.store.save_variant(candidate)
        report = CycleReport(status="launched", variant=candidate, ab_result=ab, action=action)
        self.history.append(report)
        return report
