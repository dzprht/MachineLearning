"""Command-line management for opponent pools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .catalog import REPO_ROOT, discover_agents
from .pools import (
    DEFAULT_POOLS_PATH,
    VALID_ROLES,
    load_pool_data,
    overlap_warnings,
    sample_opponent,
    validate_pool_data,
    validate_pools,
    write_pool_data_atomic,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pools", type=Path, default=DEFAULT_POOLS_PATH)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list-agents")
    sub.add_parser("list-pools")
    create = sub.add_parser("create-pool")
    create.add_argument("pool")
    create.add_argument("--role", required=True, choices=sorted(VALID_ROLES))
    create.add_argument("--agent", dest="agent_id", required=True)
    create.add_argument("--weight", type=float, default=1.0)
    create.add_argument("--positions", type=int, nargs="+", default=[0, 1])
    add = sub.add_parser("add")
    add.add_argument("pool")
    add.add_argument("agent_id")
    add.add_argument("--weight", type=float, required=True)
    add.add_argument("--positions", type=int, nargs="+", default=[0, 1])
    remove = sub.add_parser("remove")
    remove.add_argument("pool")
    remove.add_argument("agent_id")
    weight = sub.add_parser("set-weight")
    weight.add_argument("pool")
    weight.add_argument("agent_id")
    weight.add_argument("weight", type=float)
    sub.add_parser("validate")
    sample = sub.add_parser("sample")
    sample.add_argument("pool")
    sample.add_argument("--episodes", type=int, required=True)
    sample.add_argument("--seed", type=int, required=True)
    return parser


def _mutable_pool(data: dict, pool_name: str) -> dict:
    try:
        return data["pools"][pool_name]
    except KeyError as exc:
        raise ValueError(f"Unknown pool: {pool_name}") from exc


def _validate_and_write(data: dict, args: argparse.Namespace) -> None:
    catalog = discover_agents(args.repo_root)
    errors = validate_pool_data(data, catalog)
    if errors:
        raise ValueError("Invalid opponent pools:\n" + "\n".join(errors))
    write_pool_data_atomic(data, args.pools)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    catalog = discover_agents(args.repo_root)
    if args.command == "list-agents":
        for agent_id, record in sorted(catalog.items()):
            print(f"{agent_id}\t{record.entrypoint_path}\t{record.artifact_sha256}")
        return 0

    data = load_pool_data(args.pools)
    if args.command == "list-pools":
        for name, pool in data["pools"].items():
            members = ", ".join(
                f"{member['agent_id']}@{member['weight']}" for member in pool["members"]
            )
            print(f"{name}\t{pool['role']}\t{members}")
        return 0
    if args.command == "create-pool":
        if args.pool in data["pools"]:
            raise ValueError(f"Pool already exists: {args.pool}")
        if args.agent_id not in catalog:
            raise ValueError(f"Unknown agent: {args.agent_id}")
        data["pools"][args.pool] = {
            "role": args.role,
            "members": [
                {
                    "agent_id": args.agent_id,
                    "weight": args.weight,
                    "positions": args.positions,
                }
            ],
        }
        _validate_and_write(data, args)
        return 0
    if args.command == "add":
        if args.agent_id not in catalog:
            raise ValueError(f"Unknown agent: {args.agent_id}")
        pool = _mutable_pool(data, args.pool)
        if any(member["agent_id"] == args.agent_id for member in pool["members"]):
            raise ValueError(f"Agent {args.agent_id} already belongs to {args.pool}")
        pool["members"].append(
            {"agent_id": args.agent_id, "weight": args.weight, "positions": args.positions}
        )
        _validate_and_write(data, args)
        return 0
    if args.command == "remove":
        pool = _mutable_pool(data, args.pool)
        before = len(pool["members"])
        pool["members"] = [m for m in pool["members"] if m["agent_id"] != args.agent_id]
        if len(pool["members"]) == before:
            raise ValueError(f"Agent {args.agent_id} is not in {args.pool}")
        if not pool["members"]:
            raise ValueError("A pool cannot be left empty; add a replacement before removing")
        _validate_and_write(data, args)
        return 0
    if args.command == "set-weight":
        pool = _mutable_pool(data, args.pool)
        matches = [m for m in pool["members"] if m["agent_id"] == args.agent_id]
        if not matches:
            raise ValueError(f"Agent {args.agent_id} is not in {args.pool}")
        matches[0]["weight"] = args.weight
        _validate_and_write(data, args)
        return 0
    if args.command == "validate":
        _, _, warnings = validate_pools(args.pools, args.repo_root)
        for warning in warnings:
            print(f"WARNING: {warning}", file=sys.stderr)
        print("Opponent pools are valid")
        return 0
    if args.command == "sample":
        if args.episodes <= 0:
            raise ValueError("--episodes must be positive")
        selections = [
            sample_opponent(args.pool, args.seed, episode_seed, args.pools, args.repo_root)
            for episode_seed in range(args.episodes)
        ]
        print(
            json.dumps(
                [
                    {
                        "episode_seed": item.episode_seed,
                        "agent_id": item.agent.agent_id,
                        "position": item.position,
                    }
                    for item in selections
                ],
                indent=2,
            )
        )
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
