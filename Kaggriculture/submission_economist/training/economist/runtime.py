"""Stateful submission runtime for the exported Economist actor."""

from __future__ import annotations

from .encoder import ObservationEncoder
from .executor import EconomistExecutor
from .export import NumpyActor


class EconomistRuntime:
    def __init__(self, weights_path, configuration=None):
        self.actor = NumpyActor.load(weights_path)
        self.encoder = ObservationEncoder(configuration)
        self.executor = EconomistExecutor(configuration)

    def reset(self) -> None:
        self.executor.reset()

    def act(self, obs: dict, configuration=None) -> dict:
        # `obs` here already reflects the outcome of last turn's action (or is
        # the very first observation) -- confirm/clear pending state against
        # it *before* making this turn's decision, matching how gym_env.py
        # calls `observe()` on the post-step obs right after stepping, so the
        # next `step()` call's mask/register_decision/act see already-observed
        # state. See docs/IDEAS.md IDEA-020 for the contract this mirrors.
        if int(obs.get("step", 0)) == 0:
            self.reset()
        else:
            self.executor.observe(obs)
        encoded = self.encoder.encode(obs, self.executor.state)
        mask = self.executor.action_mask(obs)
        action = self.actor.predict(encoded, mask)
        self.executor.register_decision(action, obs)
        return self.executor.act(obs)
