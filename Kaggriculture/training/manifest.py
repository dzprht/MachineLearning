"""Resolved, reproducible manifests for future training runs."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path

from .catalog import REPO_ROOT
from .pools import DEFAULT_POOLS_PATH, resolved_pool_snapshot, validate_pools


def build_run_manifest(
    pool_names: list[str],
    master_seed: int,
    config: dict,
    pools_path: Path | str = DEFAULT_POOLS_PATH,
    repo_root: Path | str = REPO_ROOT,
) -> dict:
    data, catalog, warnings = validate_pools(pools_path, repo_root)
    missing = [name for name in pool_names if name not in data["pools"]]
    if missing:
        raise ValueError(f"Unknown pools: {', '.join(missing)}")
    versions = {}
    for package in (
        "kaggle-environments",
        "numpy",
        "torch",
        "gymnasium",
        "stable-baselines3",
        "sb3-contrib",
    ):
        versions[package] = importlib.metadata.version(package)
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "master_seed": int(master_seed),
        "config": config,
        "pools": resolved_pool_snapshot(pool_names, data, catalog),
        "pool_overlap_warnings": warnings,
        "package_versions": versions,
        "opponent_episode_counts": {name: {} for name in pool_names},
    }


def write_run_manifest(manifest: dict, path: Path | str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
