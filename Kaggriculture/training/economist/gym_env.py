"""Gymnasium adapter: one env.step == one primitive turn == one RL decision.

Unlike the semi-Markov `seed_buyer` wrapper, the Economist is asked for a
decision every turn (IDEA-020 contract point 3: "не ждать завершения полного
цикла, как в старой Gym-обёртке").
"""

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
from .executor import EconomistExecutor
from .phi import compute_phi


class KaggricultureEconomistEnv(gym.Env):
    """Train the multi-cell Economist while rules handle routes/watering/selling."""

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
        phi_enabled: bool = True,
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
        self.phi_enabled = bool(phi_enabled)
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(FEATURE_COUNT,), dtype=np.float32
        )
        self.encoder = ObservationEncoder(self.configuration)
        self.executor = EconomistExecutor(self.configuration)
        self._episode_index = 0
        self._obs = None
        self._trainer = None
        self._kaggle_env = None
        self._selection: PoolSelection | None = None
        self._phi = 0.0
        self.total_primitive_steps = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is None:
            episode_seed = self.worker_id + self.num_workers * self._episode_index
            self._episode_index += 1
        else:
            episode_seed = int(seed)
        self._selection = sample_opponent(
            self.pool_name, self.master_seed, episode_seed, self.pools_path, self.repo_root
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
        self._phi = self._compute_phi(self._obs) if self.phi_enabled else 0.0
        info = self._info(money_delta=0.0, phi_delta=0.0)
        return self.encoder.encode(self._obs, self.executor.state), info

    def step(self, action: int):
        if self._obs is None or self._trainer is None:
            raise RuntimeError("reset() must be called before step()")
        action = int(action)
        mask = self.action_masks()
        invalid_action = None
        if action < 0 or action >= len(ACTION_NAMES) or not mask[action]:
            invalid_action = action
            action = len(ACTION_NAMES) - 1  # WAIT
        money_before = self._money(self._obs)
        self.executor.register_decision(action, self._obs)
        primitive_action = self.executor.act(self._obs)
        next_obs, _, done, _ = self._trainer.step(primitive_action)
        self.total_primitive_steps += 1
        self._obs = next_obs
        self.executor.observe(self._obs)
        terminated = bool(done)

        money_delta = self._money(self._obs) - money_before
        if terminated:
            next_phi = 0.0  # A genuine season end zeroes Phi by construction.
        elif self.phi_enabled:
            next_phi = self._compute_phi(self._obs)
        else:
            next_phi = 0.0
        phi_delta = next_phi - self._phi
        self._phi = next_phi
        reward = float(0.01 * (money_delta + phi_delta))

        info = self._info(money_delta=money_delta, phi_delta=phi_delta)
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

    def _compute_phi(self, obs) -> float:
        return compute_phi(obs, self.executor.state, self.configuration)

    def _money(self, obs) -> float:
        return float(obs["farms"][self._selection.position]["money"])

    def _info(self, money_delta: float, phi_delta: float) -> dict:
        return {
            "total_primitive_steps": int(self.total_primitive_steps),
            "money_delta": float(money_delta),
            "phi_delta": float(phi_delta),
            "opponent_id": self._selection.agent.agent_id,
            "player_position": self._selection.position,
            "episode_seed": self._selection.episode_seed,
        }
