"""Primitive-turn accounting for future MaskablePPO runs."""

from __future__ import annotations

from collections import Counter

from stable_baselines3.common.callbacks import BaseCallback


class PrimitiveStepBudget(BaseCallback):
    """Stop after the first vector step that reaches the primitive-turn budget."""

    def __init__(self, budget: int, verbose: int = 0):
        super().__init__(verbose)
        if budget <= 0:
            raise ValueError("Primitive-step budget must be positive")
        self.budget = int(budget)
        self.primitive_steps = 0
        self.opponent_episode_counts: Counter[str] = Counter()
        self._seen_episodes: set[tuple[int, int]] = set()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [False] * len(infos))
        for info, done in zip(infos, dones):
            self.primitive_steps += int(info.get("primitive_steps", 0))
            key = (int(info.get("player_position", -1)), int(info.get("episode_seed", -1)))
            if done and key not in self._seen_episodes:
                self._seen_episodes.add(key)
                self.opponent_episode_counts[info.get("opponent_id", "unknown")] += 1
        return self.primitive_steps < self.budget
