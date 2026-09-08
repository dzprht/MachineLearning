"""Named, weighted opponent pools with deterministic per-episode sampling."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Iterable

from .catalog import AgentRecord, REPO_ROOT, discover_agents


DEFAULT_POOLS_PATH = REPO_ROOT / "training" / "opponent_pools.json"
VALID_ROLES = {"train", "validation", "test"}


@dataclass(frozen=True)
class PoolSelection:
    pool: str
    agent: AgentRecord
    position: int
    episode_seed: int


def load_pool_data(path: Path | str = DEFAULT_POOLS_PATH) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("pools"), dict):
        raise ValueError("Opponent pools must use schema_version=1 and contain a pools object")
    return data


def validate_pool_data(data: dict, catalog: dict[str, AgentRecord]) -> list[str]:
    errors: list[str] = []
    for pool_name, pool in data.get("pools", {}).items():
        if pool.get("role") not in VALID_ROLES:
            errors.append(f"{pool_name}: role must be one of {sorted(VALID_ROLES)}")
        members = pool.get("members")
        if not isinstance(members, list) or not members:
            errors.append(f"{pool_name}: members must be a non-empty list")
            continue
        seen: set[str] = set()
        for index, member in enumerate(members):
            label = f"{pool_name}.members[{index}]"
            agent_id = member.get("agent_id")
            if agent_id not in catalog:
                errors.append(f"{label}: unknown agent_id={agent_id!r}")
            if agent_id in seen:
                errors.append(f"{label}: duplicate agent_id={agent_id!r}")
            seen.add(agent_id)
            weight = member.get("weight")
            if not isinstance(weight, (int, float)) or isinstance(weight, bool) or weight <= 0:
                errors.append(f"{label}: weight must be positive")
            positions = member.get("positions")
            if not isinstance(positions, list) or not positions or any(p not in (0, 1) for p in positions):
                errors.append(f"{label}: positions must be a non-empty subset of [0, 1]")
            elif len(set(positions)) != len(positions):
                errors.append(f"{label}: positions contain duplicates")
    return errors


def overlap_warnings(data: dict) -> list[str]:
    by_role: dict[str, set[str]] = {role: set() for role in VALID_ROLES}
    for pool in data.get("pools", {}).values():
        role = pool.get("role")
        if role in by_role:
            by_role[role].update(member.get("agent_id") for member in pool.get("members", []))
    warnings = []
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = sorted(by_role[left] & by_role[right])
        if overlap:
            warnings.append(f"{left}/{right} pools overlap: {', '.join(overlap)}")
    return warnings


def validate_pools(
    path: Path | str = DEFAULT_POOLS_PATH,
    repo_root: Path | str = REPO_ROOT,
) -> tuple[dict, dict[str, AgentRecord], list[str]]:
    catalog = discover_agents(repo_root)
    data = load_pool_data(path)
    errors = validate_pool_data(data, catalog)
    if errors:
        raise ValueError("Invalid opponent pools:\n" + "\n".join(errors))
    return data, catalog, overlap_warnings(data)


def sample_opponent(
    pool_name: str,
    master_seed: int,
    episode_seed: int,
    path: Path | str = DEFAULT_POOLS_PATH,
    repo_root: Path | str = REPO_ROOT,
) -> PoolSelection:
    data, catalog, _ = validate_pools(path, repo_root)
    if pool_name not in data["pools"]:
        raise KeyError(f"Unknown opponent pool: {pool_name}")
    members = data["pools"][pool_name]["members"]
    combined_seed = ((int(master_seed) & ((1 << 64) - 1)) * 1_000_003) ^ (
        int(episode_seed) & ((1 << 64) - 1)
    )
    rng = random.Random(combined_seed)
    threshold = rng.random() * sum(float(member["weight"]) for member in members)
    selected = members[-1]
    cumulative = 0.0
    for member in members:
        cumulative += float(member["weight"])
        if threshold < cumulative:
            selected = member
            break
    return PoolSelection(
        pool=pool_name,
        agent=catalog[selected["agent_id"]],
        position=rng.choice(selected["positions"]),
        episode_seed=int(episode_seed),
    )


def write_pool_data_atomic(data: dict, path: Path | str = DEFAULT_POOLS_PATH) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=destination.name + ".", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def resolved_pool_snapshot(
    pool_names: Iterable[str], data: dict, catalog: dict[str, AgentRecord]
) -> dict:
    snapshot = {}
    for name in pool_names:
        pool = data["pools"][name]
        snapshot[name] = {
            "role": pool["role"],
            "members": [
                {
                    **member,
                    "artifact_sha256": catalog[member["agent_id"]].artifact_sha256,
                }
                for member in pool["members"]
            ],
        }
    return snapshot
