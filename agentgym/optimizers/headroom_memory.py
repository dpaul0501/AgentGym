"""HeadroomMemoryOptimizer: the real reference ScopeOptimizer for Scope.MEMORY, wrapping
headroomlabs-ai/headroom's real compress() function and CompressConfig (package: headroom-ai on
PyPI). Searches over real CompressConfig variants against the corpus's actual captured message
trajectories via a real Optuna study, maximizing real, measured token savings — the legitimate
reward signal for this scope. Downstream task-correctness impact of the winning config is
Harness's own offline Evaluate stage's job once the artifact is bound, same as any other scope.
"""

from __future__ import annotations

import optuna
from headroom.compress import CompressConfig, compress

from agentgym.core.artifact import Artifact, ArtifactShape
from agentgym.core.corpus import TrainingCorpus
from agentgym.core.scope import MEMORY


class HeadroomMemoryOptimizer:
    scope = MEMORY
    depends_on: list = []

    def __init__(self, n_trials: int = 10, seed: int = 0, model: str = "gpt-4o"):
        self.n_trials = n_trials
        self.seed = seed
        self.model = model

    def estimate(self, artifact: Artifact, corpus: TrainingCorpus, benchmark, variant) -> Artifact:
        trajectory_rows = corpus.sft_rows()
        if not trajectory_rows:
            raise ValueError("cannot estimate MEMORY from an empty TrainingCorpus")
        message_lists = [corpus.to_chat_messages(row) for row in trajectory_rows]
        message_lists = [m for m in message_lists if m]
        if not message_lists:
            raise ValueError("corpus produced no non-empty message trajectories to compress")

        def objective(trial: optuna.Trial) -> float:
            config = CompressConfig(
                compress_user_messages=trial.suggest_categorical("compress_user_messages", [True, False]),
                protect_recent=trial.suggest_int("protect_recent", 1, 6),
                min_tokens_to_compress=trial.suggest_int("min_tokens_to_compress", 50, 500, step=50),
            )
            return sum(
                compress(messages, model=self.model, config=config).tokens_saved
                for messages in message_lists
            )

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.seed)
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        return Artifact(
            scope=MEMORY,
            shape=ArtifactShape.CONFIG,
            value={
                "compress_user_messages": study.best_params["compress_user_messages"],
                "protect_recent": study.best_params["protect_recent"],
                "min_tokens_to_compress": study.best_params["min_tokens_to_compress"],
                "model": self.model,
            },
            optimizer_binding="headroom.compress",
            technique=None,
            source="estimated",
            provenance={
                "n_trials": self.n_trials,
                "best_tokens_saved": study.best_value,
                "n_trajectories": len(message_lists),
            },
        )
