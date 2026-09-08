"""Stateful submission runtime shared by future exported RL agents."""

from __future__ import annotations

from .encoder import ObservationEncoder
from .executor import SeedBuyerExecutor
from .export import NumpyActor


class SeedBuyerRuntime:
    def __init__(self, weights_path, configuration=None):
        self.actor = NumpyActor.load(weights_path)
        self.encoder = ObservationEncoder(configuration)
        self.executor = SeedBuyerExecutor(configuration)

    def reset(self) -> None:
        self.executor.reset()

    def act(self, obs: dict, configuration=None) -> dict:
        if int(obs.get("step", 0)) == 0:
            self.reset()
        else:
            self.executor.observe(obs)
        if self.executor.needs_decision(obs):
            encoded = self.encoder.encode(obs, self.executor.state)
            mask = self.executor.action_mask(obs)
            self.executor.start(self.actor.predict(encoded, mask), obs)
        action = self.executor.act(obs)
        return action
