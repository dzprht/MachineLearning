import unittest

from agents.baselines.carrot_loop_v1.main import agent


def observation(tile=None, *, day=0, seeds=0, carrots=0, money=3000):
    tiles = [["LOCKED" for _ in range(10)] for _ in range(10)]
    for y in range(5):
        for x in range(5):
            tiles[y][x] = None
    tiles[4][4] = tile
    return {
        "player": 0,
        "day": day,
        "farms": [
            {
                "money": money,
                "farmer": [4, 4],
                "hands": [],
                "tiles": tiles,
            }
        ],
        "private": {
            "seeds": {"CARROT": seeds},
            "shed": {"CARROT": carrots},
            "inventories": [{}],
        },
    }


class CarrotLoopTest(unittest.TestCase):
    def test_buys_one_seed_when_none_is_available(self):
        action = agent(observation())
        self.assertEqual(action["farmer"], ["PASS"])
        self.assertEqual(action["market"], [["BUY_SEED", "CARROT", 1]])

    def test_plants_an_available_seed(self):
        action = agent(observation(seeds=1))
        self.assertEqual(action["farmer"], ["PLANT", "CARROT"])

    def test_waters_before_harvesting_on_peak_day(self):
        plant = {
            "kind": "PLANT",
            "crop": "CARROT",
            "planted_day": 0,
            "watered_today": False,
        }
        self.assertEqual(agent(observation(plant, day=3))["farmer"], ["WATER"])
        plant["watered_today"] = True
        self.assertEqual(agent(observation(plant, day=3))["farmer"], ["HARVEST"])

    def test_sells_all_carrots_in_the_shed(self):
        action = agent(observation(seeds=1, carrots=3))
        self.assertEqual(action["market"], [["SELL", "CARROT", 3]])

    def test_digs_a_weed_blocking_the_work_tile(self):
        action = agent(observation({"kind": "WEED"}, seeds=1))
        self.assertEqual(action["farmer"], ["DIG"])


if __name__ == "__main__":
    unittest.main()
