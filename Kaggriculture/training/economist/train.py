"""MaskablePPO runner for the multi-cell Economist (IDEA-020).

Unlike `training/seed_buyer/train.py`, one env.step is one primitive game
turn (no semi-Markov skipping -- see gym_env.py), so `total_timesteps` in SB3
already *is* the primitive-turn budget; no `PrimitiveStepBudget` callback is
needed here.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv

from training.catalog import REPO_ROOT
from training.manifest import build_run_manifest, write_run_manifest
from training.pools import DEFAULT_POOLS_PATH, validate_pools

from .encoder import ObservationEncoder
from .export import export_actor
from .gym_env import KaggricultureEconomistEnv
from .ppo import PPO_CONFIG, build_model


class EpisodeTracker(BaseCallback):
    """Counts completed episodes per opponent; does not stop training."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.opponent_episode_counts: Counter = Counter()
        self._seen_episodes: set[tuple[int, int]] = set()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [False] * len(infos))
        for info, done in zip(infos, dones):
            if not done:
                continue
            key = (int(info.get("player_position", -1)), int(info.get("episode_seed", -1)))
            if key not in self._seen_episodes:
                self._seen_episodes.add(key)
                self.opponent_episode_counts[info.get("opponent_id", "unknown")] += 1
        return True


class CheckpointingCallback(BaseCallback):
    """Periodically persist an SB3 checkpoint, a submission-ready NumPy actor and
    resume progress, so an interrupted run does not lose accumulated training."""

    def __init__(
        self,
        tracker: EpisodeTracker,
        baseline_timesteps: int,
        baseline_episode_counts: dict,
        output_dir: Path,
        encoder_schema: dict,
        every_timesteps: int,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.tracker = tracker
        self.baseline_timesteps = int(baseline_timesteps)
        self.baseline_episode_counts: Counter = Counter(baseline_episode_counts)
        self.output_dir = Path(output_dir)
        self.encoder_schema = encoder_schema
        self.every_timesteps = max(1, int(every_timesteps))
        self._last_saved_at = 0

    def _on_step(self) -> bool:
        if self.model.num_timesteps - self._last_saved_at >= self.every_timesteps:
            self.save()
            self._last_saved_at = self.model.num_timesteps
        return True

    def _on_training_end(self) -> None:
        self.save()

    def save(self) -> None:
        self.model.save(self.output_dir / "sb3_model")
        export_actor(
            self.model,
            self.output_dir / "actor.npz",
            self.encoder_schema,
            self.output_dir / "feature_schema.json",
        )
        merged_counts = self.baseline_episode_counts + self.tracker.opponent_episode_counts
        progress = {
            "primitive_steps": self.baseline_timesteps + int(self.model.num_timesteps),
            "sb3_num_timesteps": int(self.model.num_timesteps),
            "opponent_episode_counts": dict(merged_counts),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        (self.output_dir / "progress.json").write_text(
            json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


def load_experiment_config(path: Path | str) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"schema_version", "train_pool", "master_seed", "primitive_budget", "n_envs", "output_dir"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Experiment config is missing: {', '.join(missing)}")
    if config["schema_version"] != 1:
        raise ValueError("Experiment config must use schema_version=1")
    if int(config["primitive_budget"]) <= 0 or int(config["n_envs"]) <= 0:
        raise ValueError("primitive_budget and n_envs must be positive")
    config.setdefault("phi_enabled", True)
    return config


def build_vector_env(
    pool_name: str,
    master_seed: int,
    n_envs: int,
    phi_enabled: bool = True,
    pools_path: Path | str = DEFAULT_POOLS_PATH,
    repo_root: Path | str = REPO_ROOT,
):
    factories = []
    for worker_id in range(n_envs):
        factories.append(
            lambda worker_id=worker_id: KaggricultureEconomistEnv(
                pool_name=pool_name,
                master_seed=master_seed,
                worker_id=worker_id,
                num_workers=n_envs,
                pools_path=pools_path,
                repo_root=repo_root,
                phi_enabled=phi_enabled,
            )
        )
    return SubprocVecEnv(factories, start_method="forkserver")


def run_training(config_path: Path | str, pools_path: Path | str = DEFAULT_POOLS_PATH) -> Path:
    """Run only when explicitly called with a complete external experiment config."""
    config = load_experiment_config(config_path)
    pool_data, _, warnings = validate_pools(pools_path, REPO_ROOT)
    if config["train_pool"] not in pool_data["pools"]:
        raise ValueError(f"Unknown train_pool={config['train_pool']}")
    if pool_data["pools"][config["train_pool"]]["role"] != "train":
        raise ValueError("train_pool must reference a pool with role='train'")
    for warning in warnings:
        print(f"WARNING: {warning}")

    output_dir = (REPO_ROOT / config["output_dir"]).resolve()
    manifest_path = output_dir / "run_manifest.json"
    progress_path = output_dir / "progress.json"
    checkpoint_path = output_dir / "sb3_model.zip"
    resuming = checkpoint_path.exists() and progress_path.exists()

    if resuming:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        baseline_timesteps = int(progress["primitive_steps"])
        baseline_episode_counts = progress.get("opponent_episode_counts", {})
        print(f"Resuming from {checkpoint_path}: {baseline_timesteps} primitive steps already done")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        manifest = build_run_manifest(
            [config["train_pool"]],
            int(config["master_seed"]),
            {**config, "ppo": PPO_CONFIG},
            pools_path,
            REPO_ROOT,
        )
        write_run_manifest(manifest, manifest_path)
        baseline_timesteps = 0
        baseline_episode_counts = {}

    remaining_budget = int(config["primitive_budget"]) - baseline_timesteps
    if remaining_budget <= 0:
        print(f"Primitive budget already reached ({baseline_timesteps} steps); nothing to do.")
        return output_dir

    env = build_vector_env(
        config["train_pool"],
        int(config["master_seed"]),
        int(config["n_envs"]),
        bool(config["phi_enabled"]),
        pools_path,
        REPO_ROOT,
    )
    try:
        if resuming:
            from sb3_contrib import MaskablePPO

            model = MaskablePPO.load(checkpoint_path, env=env, device="cpu")
        else:
            model = build_model(env, seed=int(config["master_seed"]), device="cpu")
        encoder = ObservationEncoder()
        tracker = EpisodeTracker()
        checkpoint_cb = CheckpointingCallback(
            tracker,
            baseline_timesteps,
            baseline_episode_counts,
            output_dir,
            encoder.schema(),
            every_timesteps=max(2000, int(config["primitive_budget"]) // 100),
        )
        model.learn(
            total_timesteps=remaining_budget,
            callback=CallbackList([tracker, checkpoint_cb]),
            reset_num_timesteps=not resuming,
        )
        checkpoint_cb.save()
        manifest["actual_primitive_steps"] = baseline_timesteps + int(model.num_timesteps)
        manifest["opponent_episode_counts"][config["train_pool"]] = dict(
            Counter(baseline_episode_counts) + tracker.opponent_episode_counts
        )
        write_run_manifest(manifest, manifest_path)
    finally:
        env.close()
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pools", type=Path, default=DEFAULT_POOLS_PATH)
    args = parser.parse_args(argv)
    print(run_training(args.config, args.pools))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
