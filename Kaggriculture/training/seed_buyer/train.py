"""Future MaskablePPO runner. No experiment config is committed yet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3.common.vec_env import SubprocVecEnv

from training.catalog import REPO_ROOT
from training.manifest import build_run_manifest, write_run_manifest
from training.pools import DEFAULT_POOLS_PATH, validate_pools

from .budget import PrimitiveStepBudget
from .encoder import ObservationEncoder
from .export import export_actor
from .gym_env import KaggricultureSeedBuyerEnv
from .ppo import PPO_CONFIG, build_model


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
    return config


def build_vector_env(
    pool_name: str,
    master_seed: int,
    n_envs: int,
    pools_path: Path | str = DEFAULT_POOLS_PATH,
    repo_root: Path | str = REPO_ROOT,
):
    factories = []
    for worker_id in range(n_envs):
        factories.append(
            lambda worker_id=worker_id: KaggricultureSeedBuyerEnv(
                pool_name=pool_name,
                master_seed=master_seed,
                worker_id=worker_id,
                num_workers=n_envs,
                pools_path=pools_path,
                repo_root=repo_root,
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
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = build_run_manifest(
        [config["train_pool"]],
        int(config["master_seed"]),
        {**config, "ppo": PPO_CONFIG},
        pools_path,
        REPO_ROOT,
    )
    manifest_path = output_dir / "run_manifest.json"
    write_run_manifest(manifest, manifest_path)

    env = build_vector_env(
        config["train_pool"],
        int(config["master_seed"]),
        int(config["n_envs"]),
        pools_path,
        REPO_ROOT,
    )
    try:
        model = build_model(env, seed=int(config["master_seed"]), device="cpu")
        budget = PrimitiveStepBudget(int(config["primitive_budget"]))
        model.learn(total_timesteps=2_147_483_647, callback=budget)
        model.save(output_dir / "sb3_model")
        encoder = ObservationEncoder()
        export_actor(
            model,
            output_dir / "actor.npz",
            encoder.schema(),
            output_dir / "feature_schema.json",
        )
        manifest["actual_primitive_steps"] = budget.primitive_steps
        manifest["opponent_episode_counts"][config["train_pool"]] = dict(
            budget.opponent_episode_counts
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
