import json
from pathlib import Path
import tempfile
import unittest

from training.catalog import discover_agents, hash_artifact
from training.poolctl import main as poolctl_main
from training.manifest import build_run_manifest
from training.pools import (
    overlap_warnings,
    sample_opponent,
    validate_pool_data,
    validate_pools,
)


class OpponentPoolTest(unittest.TestCase):
    def _agent(self, root: Path, agent_id: str) -> None:
        directory = root / "agents" / "baselines" / agent_id
        directory.mkdir(parents=True)
        (directory / "main.py").write_text(
            "def agent(obs):\n    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n",
            encoding="utf-8",
        )
        metadata = {
            "schema_version": 1,
            "id": agent_id,
            "status": "frozen",
            "type": "python",
            "entrypoint": f"agents/baselines/{agent_id}/main.py:agent",
            "artifact_files": ["main.py"],
            "artifact_sha256": hash_artifact(directory, ["main.py"]),
        }
        (directory / "metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )

    def _workspace(self, root: Path) -> Path:
        self._agent(root, "agent_a")
        self._agent(root, "agent_b")
        pools = root / "pools.json"
        pools.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "pools": {
                        "train": {
                            "role": "train",
                            "members": [
                                {"agent_id": "agent_a", "weight": 1.0, "positions": [0, 1]}
                            ],
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        return pools

    def test_repository_catalog_discovers_and_verifies_baseline(self):
        catalog = discover_agents()
        self.assertIn("carrot_loop_v1", catalog)
        self.assertEqual(len(catalog["carrot_loop_v1"].artifact_sha256), 64)

    def test_run_manifest_resolves_pool_hashes_and_dependencies(self):
        manifest = build_run_manifest(["train_v1"], 8008, {"purpose": "test"})
        member = manifest["pools"]["train_v1"]["members"][0]
        self.assertEqual(member["agent_id"], "carrot_loop_v1")
        self.assertEqual(len(member["artifact_sha256"]), 64)
        self.assertEqual(manifest["package_versions"]["sb3-contrib"], "2.9.0")

    def test_pool_cli_add_weight_remove_and_create(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pools = self._workspace(root)
            base = ["--pools", str(pools), "--repo-root", str(root)]
            self.assertEqual(poolctl_main([*base, "add", "train", "agent_b", "--weight", "2"]), 0)
            self.assertEqual(poolctl_main([*base, "set-weight", "train", "agent_b", "3"]), 0)
            self.assertEqual(poolctl_main([*base, "remove", "train", "agent_b"]), 0)
            self.assertEqual(
                poolctl_main(
                    [
                        *base,
                        "create-pool",
                        "validation",
                        "--role",
                        "validation",
                        "--agent",
                        "agent_a",
                    ]
                ),
                0,
            )
            data, _, warnings = validate_pools(pools, root)
            self.assertEqual(data["pools"]["train"]["members"][0]["agent_id"], "agent_a")
            self.assertTrue(any("train/validation" in warning for warning in warnings))

    def test_sampling_is_deterministic_per_episode_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pools = self._workspace(root)
            data = json.loads(pools.read_text())
            data["pools"]["train"]["members"].append(
                {"agent_id": "agent_b", "weight": 3.0, "positions": [0, 1]}
            )
            pools.write_text(json.dumps(data), encoding="utf-8")
            forward = {
                seed: sample_opponent("train", 8008, seed, pools, root)
                for seed in range(100)
            }
            reverse = {
                seed: sample_opponent("train", 8008, seed, pools, root)
                for seed in reversed(range(100))
            }
            self.assertEqual(
                {seed: (item.agent.agent_id, item.position) for seed, item in forward.items()},
                {seed: (item.agent.agent_id, item.position) for seed, item in reverse.items()},
            )
            self.assertEqual({item.agent.agent_id for item in forward.values()}, {"agent_a", "agent_b"})

    def test_overlap_warns_but_validation_succeeds(self):
        catalog = discover_agents()
        member = {"agent_id": "carrot_loop_v1", "weight": 1.0, "positions": [0, 1]}
        data = {
            "schema_version": 1,
            "pools": {
                "train": {"role": "train", "members": [member]},
                "test": {"role": "test", "members": [dict(member)]},
            },
        }
        self.assertEqual(validate_pool_data(data, catalog), [])
        self.assertTrue(any("train/test" in warning for warning in overlap_warnings(data)))


if __name__ == "__main__":
    unittest.main()
