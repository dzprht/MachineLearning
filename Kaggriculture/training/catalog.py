"""Discovery and integrity checks for frozen agent artifacts."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AgentRecord:
    agent_id: str
    artifact_dir: Path
    entrypoint_path: Path
    callable_name: str
    metadata_path: Path
    metadata: dict
    artifact_sha256: str

    @property
    def kaggle_agent_path(self) -> str:
        return str(self.entrypoint_path)


def hash_artifact(artifact_dir: Path, artifact_files: list[str]) -> str:
    """Hash an explicit, ordered-independent set of artifact files."""
    digest = hashlib.sha256()
    for relative in sorted(artifact_files):
        path = (artifact_dir / relative).resolve()
        try:
            path.relative_to(artifact_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"Artifact file escapes its directory: {relative}") from exc
        if not path.is_file():
            raise ValueError(f"Artifact file does not exist: {path}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def discover_agents(repo_root: Path | str = REPO_ROOT) -> dict[str, AgentRecord]:
    """Discover every frozen agent described by an agents/**/metadata.json file."""
    root = Path(repo_root).resolve()
    agents_root = root / "agents"
    records: dict[str, AgentRecord] = {}
    errors: list[str] = []

    for metadata_path in sorted(agents_root.glob("**/metadata.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            required = {
                "schema_version",
                "id",
                "status",
                "type",
                "entrypoint",
                "artifact_files",
                "artifact_sha256",
            }
            missing = sorted(required - metadata.keys())
            if missing:
                raise ValueError(f"missing fields: {', '.join(missing)}")
            if metadata["schema_version"] != 1:
                raise ValueError(f"unsupported schema_version={metadata['schema_version']}")
            if metadata["status"] != "frozen":
                raise ValueError("status must be 'frozen'")
            if metadata["type"] != "python":
                raise ValueError("only type='python' is supported")
            if not isinstance(metadata["artifact_files"], list) or not metadata["artifact_files"]:
                raise ValueError("artifact_files must be a non-empty list")

            agent_id = metadata["id"]
            if agent_id in records:
                raise ValueError(f"duplicate agent id: {agent_id}")
            entrypoint, separator, callable_name = metadata["entrypoint"].rpartition(":")
            if not separator or not entrypoint or not callable_name:
                raise ValueError("entrypoint must have the form path.py:callable")
            entrypoint_path = (root / entrypoint).resolve()
            if not entrypoint_path.is_file():
                raise ValueError(f"entrypoint does not exist: {entrypoint}")
            syntax = ast.parse(entrypoint_path.read_text(encoding="utf-8"), filename=str(entrypoint_path))
            top_level_callables = {
                node.name
                for node in syntax.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            }
            if callable_name not in top_level_callables:
                raise ValueError(f"entrypoint callable is not defined at module level: {callable_name}")

            actual_hash = hash_artifact(metadata_path.parent, metadata["artifact_files"])
            if actual_hash != metadata["artifact_sha256"]:
                raise ValueError(
                    f"artifact hash mismatch: expected {metadata['artifact_sha256']}, got {actual_hash}"
                )
            records[agent_id] = AgentRecord(
                agent_id=agent_id,
                artifact_dir=metadata_path.parent,
                entrypoint_path=entrypoint_path,
                callable_name=callable_name,
                metadata_path=metadata_path,
                metadata=metadata,
                artifact_sha256=actual_hash,
            )
        except (OSError, SyntaxError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{metadata_path}: {exc}")

    if errors:
        raise ValueError("Invalid agent catalog:\n" + "\n".join(errors))
    return records
