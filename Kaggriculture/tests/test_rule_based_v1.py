import unittest
from pathlib import Path

from kaggle_environments import make
from kaggle_environments.agent import get_last_callable

from agents.baselines.rule_based_v1 import engine
from training.catalog import REPO_ROOT, discover_agents


RULE_BASED_IDS = {
    "wheat_loop_v1", "tomato_loop_v1", "strawberry_loop_v1", "melon_loop_v1",
    "goose_loop_v1", "cow_loop_v1", "sheep_loop_v1",
    "carrot_field_v1", "tomato_field_v1", "melon_field_v1",
    "staggered_mix_v1", "town_demand_v1", "crop_follower_v1",
    "crop_avoider_v1", "crop_expander_v1", "goose_ranch_v1",
    "dairy_ranch_v1", "mixed_farm_v1", "buffered_seller_v1",
}


def passive(obs):
    farm = obs["farms"][obs["player"]]
    return {
        "farmer": ["PASS"],
        "hands": [["PASS"] for _ in farm.get("hands", [])],
        "market": [],
    }


def run(agent_path, steps=720):
    env = make(
        "kaggriculture",
        configuration={"episodeSteps": steps, "seed": 1701, "weedSpawnChance": 0},
        debug=False,
    )
    env.run([str(agent_path), passive])
    return env.state[0]


class RuleBasedV1Test(unittest.TestCase):
    def test_catalog_discovers_every_rule_based_version_and_raw_entrypoint_loads(self):
        records = discover_agents(REPO_ROOT)
        self.assertTrue(RULE_BASED_IDS.issubset(records))
        for agent_id in RULE_BASED_IDS:
            record = records[agent_id]
            source = record.entrypoint_path.read_text(encoding="utf-8")
            self.assertTrue(callable(get_last_callable(source, path=str(record.entrypoint_path))))

    def test_crop_selectors_react_only_in_the_intended_direction(self):
        tiles = [[None for _ in range(10)] for _ in range(10)]
        other_tiles = [[None for _ in range(10)] for _ in range(10)]
        for x in range(4):
            other_tiles[0][x] = {"kind": "PLANT", "crop": "CARROT", "planted_day": 0}
        obs = {
            "step": 0, "day": 0, "player": 0,
            "farms": [
                {"money": 3000, "tiles": tiles, "farmer": [4, 4], "hands": [], "unlocked_quadrants": ["NW"]},
                {"money": 3000, "tiles": other_tiles, "farmer": [4, 4], "hands": [], "unlocked_quadrants": ["NW"]},
            ],
            "market": {"prices": dict(engine.BASE_PRICE)},
            "town": {"unlocked_shops": []},
            "private": {"shed": {}, "seeds": {}, "inventories": [{}]},
        }
        follower = engine.run_agent(obs, {"mode": "crops", "selector": "follower", "cells": 1})
        avoider = engine.run_agent(obs, {"mode": "crops", "selector": "avoider", "cells": 1})
        self.assertEqual(follower["market"][-1][1], "CARROT")
        self.assertNotEqual(avoider["market"][-1][1], "CARROT")

    def test_all_entrypoints_survive_a_multiday_raw_file_smoke_run(self):
        records = discover_agents(REPO_ROOT)
        for agent_id in sorted(RULE_BASED_IDS):
            with self.subTest(agent_id=agent_id):
                state = run(records[agent_id].entrypoint_path, steps=120)
                self.assertEqual(state.status, "DONE")

    def test_crop_loops_finish_without_stranded_plants(self):
        records = discover_agents(REPO_ROOT)
        for agent_id in ("wheat_loop_v1", "tomato_loop_v1", "strawberry_loop_v1", "melon_loop_v1"):
            with self.subTest(agent_id=agent_id):
                state = run(records[agent_id].entrypoint_path)
                farm = state.observation.farms[0]
                plants = [
                    tile for row in farm.tiles for tile in row
                    if isinstance(tile, dict) and tile.get("kind") == "PLANT"
                ]
                self.assertEqual(plants, [])
                self.assertGreater(farm.money, 3000)

    def test_complex_profiles_exercise_their_distinguishing_mechanics(self):
        records = discover_agents(REPO_ROOT)
        expander = run(records["crop_expander_v1"].entrypoint_path).observation.farms[0]
        ranch = run(records["goose_ranch_v1"].entrypoint_path).observation.farms[0]
        mixed = run(records["mixed_farm_v1"].entrypoint_path).observation.farms[0]
        self.assertGreaterEqual(len(expander.unlocked_quadrants), 2)
        self.assertGreaterEqual(len(ranch.unlocked_quadrants), 2)
        self.assertGreater(sum(tile.get("animal") == "GOOSE" for row in ranch.tiles for tile in row if isinstance(tile, dict)), 1)
        self.assertGreater(sum(tile.get("animal") == "GOOSE" for row in mixed.tiles for tile in row if isinstance(tile, dict)), 1)


if __name__ == "__main__":
    unittest.main()
