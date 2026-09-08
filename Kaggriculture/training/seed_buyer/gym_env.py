"""Gymnasium adapter whose step spans one complete seed-buyer decision."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from kaggle_environments import make

from training.catalog import REPO_ROOT
from training.pools import DEFAULT_POOLS_PATH, PoolSelection, sample_opponent

from .constants import ACTION_NAMES, DEFAULT_CONFIGURATION
from .encoder import FEATURE_COUNT, ObservationEncoder
from .executor import SeedBuyerExecutor


class KaggricultureSeedBuyerEnv(gym.Env):
    """Train one manager action while deterministic rules handle primitive turns."""

    metadata = {"render_modes": ["ansi", "html"]}

    def __init__(
        self,
        pool_name: str,
        master_seed: int = 8008,
        worker_id: int = 0,
        num_workers: int = 1,
        pools_path: Path | str = DEFAULT_POOLS_PATH,
        repo_root: Path | str = REPO_ROOT,
        configuration: dict | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.pool_name = pool_name
        self.master_seed = int(master_seed)
        self.worker_id = int(worker_id)
        self.num_workers = int(num_workers)
        self.pools_path = Path(pools_path)
        self.repo_root = Path(repo_root)
        self.configuration = dict(DEFAULT_CONFIGURATION)
        if configuration:
            self.configuration.update(configuration)
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(FEATURE_COUNT,), dtype=np.float32
        )
        self.encoder = ObservationEncoder(self.configuration)
        self.executor = SeedBuyerExecutor(self.configuration)
        self._episode_index = 0
        self._obs = None
        self._trainer = None
        self._kaggle_env = None
        self._selection: PoolSelection | None = None
        self.total_primitive_steps = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is None:
            episode_seed = self.worker_id + self.num_workers * self._episode_index
            self._episode_index += 1
        else:
            episode_seed = int(seed)
        self._selection = sample_opponent(
            self.pool_name,
            self.master_seed,
            episode_seed,
            self.pools_path,
            self.repo_root,
        )
        kaggle_configuration = dict(self.configuration)
        kaggle_configuration["seed"] = episode_seed
        self._kaggle_env = make("kaggriculture", configuration=kaggle_configuration, debug=False)
        players = [None, self._selection.agent.kaggle_agent_path]
        if self._selection.position == 1:
            players.reverse()
        self._trainer = self._kaggle_env.train(players)
        self._obs = self._kaggle_env.state[self._selection.position].observation
        self.executor.reset()
        info = self._info(primitive_steps=0, money_delta=0.0)
        return self.encoder.encode(self._obs, self.executor.state), info

    def step(self, action: int):
        if self._obs is None or self._trainer is None:
            raise RuntimeError("reset() must be called before step()")
        action = int(action)
        mask = self.action_masks()
        invalid_action = None
        if action < 0 or action >= len(ACTION_NAMES) or not mask[action]:
            # Generic Gymnasium checkers do not understand action masks. Keep the
            # environment total by converting an invalid external action to WAIT;
            # MaskablePPO itself should never take this branch.
            invalid_action = action
            action = len(ACTION_NAMES) - 1
        money_before = self._money(self._obs)
        self.executor.start(action, self._obs)
        primitive_steps = 0
        terminated = False
        while True:
            primitive_action = self.executor.act(self._obs)
            next_obs, _, done, _ = self._trainer.step(primitive_action)
            primitive_steps += 1
            self.total_primitive_steps += 1
            self._obs = next_obs
            self.executor.observe(self._obs)
            terminated = bool(done)
            if terminated or self.executor.needs_decision(self._obs):
                break
        money_delta = self._money(self._obs) - money_before
        reward = float(0.01 * money_delta)
        info = self._info(primitive_steps=primitive_steps, money_delta=money_delta)
        if invalid_action is not None:
            info["invalid_action"] = invalid_action
        return (
            self.encoder.encode(self._obs, self.executor.state),
            reward,
            terminated,
            False,
            info,
        )

    def action_masks(self) -> np.ndarray:
        if self._obs is None:
            return np.asarray([False] * (len(ACTION_NAMES) - 1) + [True], dtype=bool)
        return self.executor.action_mask(self._obs)

    def render(self):
        if self._kaggle_env is None or self.render_mode is None:
            return None
        return self._kaggle_env.render(mode=self.render_mode)

    def close(self):
        self._trainer = None
        self._kaggle_env = None

    def _money(self, obs) -> float:
        return float(obs["farms"][self._selection.position]["money"])

    def _info(self, primitive_steps: int, money_delta: float) -> dict:
        return {
            "primitive_steps": int(primitive_steps),
            "total_primitive_steps": int(self.total_primitive_steps),
            "money_delta": float(money_delta),
            "opponent_id": self._selection.agent.agent_id,
            "player_position": self._selection.position,
            "episode_seed": self._selection.episode_seed,
        }
