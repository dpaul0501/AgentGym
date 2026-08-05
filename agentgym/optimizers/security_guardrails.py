"""GuardrailsOptimizer: the real reference ScopeOptimizer for Scope.GUARDRAILS (under
Lever.HARNESS). Runs garak (package: garak, github.com/NVIDIA/garak) — a real LLM vulnerability
scanner — against the agent's real target model via garak.cli.main(), parses garak's real
report.jsonl, and produces a GUARDRAILS artifact: an Invariant policy DSL string (see
agentgym.verifiers.security_verifier), tightened based on which real probes found real
vulnerabilities.

Not a hyperparameter search — garak probes aren't a continuous space to search over. Like
CavemanMemoryOptimizer, this scope's "estimate" is a real measurement run that drives a real,
provenance-backed decision, not a guess. A full garak probe suite is genuinely slow (many minutes
per probe against a real local model, confirmed by timing a real run) — the default probe list
here is deliberately narrow; widen it for an actual security audit, not for the fast test suite.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from garak import cli as garak_cli

from agentgym.core.artifact import Artifact, ArtifactShape
from agentgym.core.scope import GUARDRAILS
from agentgym.verifiers.security_verifier import DEFAULT_POLICY

DEFAULT_PROBES = ["promptinject.HijackKillHumans"]


def _parse_eval_entries(report_path: Path) -> list[dict]:
    if not report_path.exists():
        return []
    entries = []
    with report_path.open() as f:
        for line in f:
            record = json.loads(line)
            if record.get("entry_type") == "eval":
                entries.append({
                    "probe": record["probe"],
                    "detector": record["detector"],
                    "passed": record["passed"],
                    "fails": record["fails"],
                    "total_evaluated": record["total_evaluated"],
                })
    return entries


class GuardrailsOptimizer:
    scope = GUARDRAILS
    depends_on: list = []

    def __init__(
        self, target_type: str = "ollama", target_name: str = "qwen2.5:7b",
        probes: list[str] | None = None, generations: int = 1,
    ):
        self.target_type = target_type
        self.target_name = target_name
        self.probes = probes or DEFAULT_PROBES
        self.generations = generations

    def estimate(self, artifact: Artifact, corpus, benchmark, variant) -> Artifact:
        with tempfile.TemporaryDirectory() as tmp:
            report_prefix = str(Path(tmp) / "garak_run")
            garak_cli.main([
                "--target_type", self.target_type,
                "--target_name", self.target_name,
                "--probes", ",".join(self.probes),
                "--generations", str(self.generations),
                "--report_prefix", report_prefix,
            ])
            evals = _parse_eval_entries(Path(f"{report_prefix}.report.jsonl"))

        total_fails = sum(e["fails"] for e in evals)
        total_evaluated = sum(e["total_evaluated"] for e in evals)
        vulnerability_rate = total_fails / total_evaluated if total_evaluated else 0.0

        # tighten the policy's detection threshold when garak found real, measured vulnerabilities
        threshold = 0.3 if vulnerability_rate > 0.2 else 0.5
        policy_dsl = DEFAULT_POLICY.replace("threshold=0.5", f"threshold={threshold}")

        return Artifact(
            scope=GUARDRAILS,
            shape=ArtifactShape.CONFIG,
            value={"policy_dsl": policy_dsl, "prompt_injection_threshold": threshold},
            optimizer_binding="garak+invariant",
            technique=None,
            source="estimated",
            provenance={
                "probes_run": self.probes,
                "vulnerability_rate": vulnerability_rate,
                "evals": evals,
            },
        )
