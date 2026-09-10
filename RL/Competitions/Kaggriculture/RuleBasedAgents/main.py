

CROP = "CARROT"
SEED_COST = 20
HARVEST_DAY = 3


def _empty_action():
    return {"farmer": ["PASS"], "hands": [], "market": []}


def agent(obs):
    farms = obs.get("farms", [])
    player = obs.get("player", 0)
    private = obs.get("private", {}) or {}
    if not farms or player >= len(farms):
        return _empty_action()

    farm = farms[player]
    seeds = private.get("seeds", {})
    shed = private.get("shed", {})
    x, y = farm["farmer"]
    tile = farm["tiles"][y][x]

    market = []
    carrots = shed.get(CROP, 0)
    if carrots > 0:
        market.append(["SELL", CROP, carrots])
    if seeds.get(CROP, 0) == 0 and farm["money"] >= SEED_COST:
        market.append(["BUY_SEED", CROP, 1])

    farmer = ["PASS"]
    if tile is None and seeds.get(CROP, 0) > 0:
        farmer = ["PLANT", CROP]
    elif isinstance(tile, dict) and tile.get("kind") == "WEED":
        farmer = ["DIG"]
    elif (
        isinstance(tile, dict)
        and tile.get("kind") == "PLANT"
        and tile.get("crop") == CROP
    ):
        age = obs.get("day", 0) - tile["planted_day"]
        if not tile.get("watered_today", False):
            farmer = ["WATER"]
        elif age >= HARVEST_DAY:
            farmer = ["HARVEST"]

    return {"farmer": farmer, "hands": [], "market": market}
