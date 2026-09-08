"""Shared deterministic executor for the frozen rule_based_v1 agent bundle.

The policies differ only in economic choices. Movement, work assignment and
market plumbing deliberately stay identical so paired agents isolate the rule
described in IDEAS.md.
"""

from __future__ import annotations

from collections import Counter


CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
PRODUCTS = (*CROPS, "EGG", "MILK", "WOOL", "FERTILIZER")
CROP = {
    "WHEAT": {"seed": 10, "last": 4, "yield": 4},
    "CARROT": {"seed": 20, "last": 3, "yield": 3},
    "TOMATO": {"seed": 50, "last": 11, "yield": 4},
    "STRAWBERRY": {"seed": 100, "last": 16, "yield": 4},
    "MELON": {"seed": 80, "last": 10, "yield": 6},
}
ANIMAL = {
    "GOOSE": {"cost": 300, "structure": "COOP", "product": "EGG"},
    "COW": {"cost": 400, "structure": "PASTURE", "product": "MILK"},
    "SHEEP": {"cost": 500, "structure": "PASTURE", "product": "WOOL"},
}
BASE_PRICE = {
    "WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120,
    "MELON": 250, "EGG": 50, "MILK": 160, "WOOL": 200,
    "FERTILIZER": 100,
}
LAND_PRICES = (1000, 2000, 4000)
SHOPS = {
    "BAKERY": ("WHEAT",),
    "PIZZA_SHOP": ("WHEAT", "TOMATO"),
    "BRUNCH_SPOT": ("WHEAT", "STRAWBERRY"),
    "ICE_CREAM_SHOP": ("WHEAT", "STRAWBERRY"),
    "PET_CAFE": ("CARROT", "CARROT"),
    "SMOOTHIE_SHOP": ("STRAWBERRY",),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}


def _empty(farm):
    return {"farmer": ["PASS"], "hands": [["PASS"] for _ in farm.get("hands", [])], "market": []}


def _tiles(farm):
    for y, row in enumerate(farm.get("tiles", [])):
        for x, tile in enumerate(row):
            yield (x, y), tile


def _shed_tiles(size):
    half = size // 2
    return ((half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half))


def _distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _move(position, target):
    x, y = position
    tx, ty = target
    if x < tx:
        return ["EAST"]
    if x > tx:
        return ["WEST"]
    if y < ty:
        return ["SOUTH"]
    if y > ty:
        return ["NORTH"]
    return ["PASS"]


def _at_or_move(position, target, action):
    return action if tuple(position) == tuple(target) else _move(position, target)


def _nearest(position, targets):
    return min(targets, key=lambda target: (_distance(position, target), target[1], target[0]))


def _market_price(obs, item):
    market = obs.get("market", {}) or {}
    prices = market.get("prices", {}) or {}
    return float(prices.get(item, BASE_PRICE[item]))


def _remaining_days(obs):
    configuration = obs.get("configuration", {}) or {}
    turns = int(configuration.get("turnsPerDay", 24)) if isinstance(configuration, dict) else 24
    steps = int(configuration.get("episodeSteps", 720)) if isinstance(configuration, dict) else 720
    return max(0, (steps - int(obs.get("step", 0)) - 1) // turns)


def _crop_score(obs, crop):
    demand = Counter()
    for shop in (obs.get("town", {}) or {}).get("unlocked_shops", []):
        demand.update(SHOPS.get(shop, ()))
    farms = obs.get("farms", [])
    player = int(obs.get("player", 0))
    own_supply = 0
    if player < len(farms):
        own_supply = sum(
            tile.get("crop") == crop
            for _, tile in _tiles(farms[player])
            if isinstance(tile, dict) and tile.get("kind") == "PLANT"
        )
    revenue = _market_price(obs, crop) * CROP[crop]["yield"]
    supply_penalty = own_supply * BASE_PRICE[crop] * 0.08
    return (revenue - CROP[crop]["seed"] + demand[crop] * BASE_PRICE[crop] * 0.15 - supply_penalty) / (CROP[crop]["last"] + 1)


def _opponent_crop_counts(obs):
    farms = obs.get("farms", [])
    player = int(obs.get("player", 0))
    if len(farms) < 2:
        return Counter()
    other = farms[1 - player]
    return Counter(
        tile.get("crop")
        for _, tile in _tiles(other)
        if isinstance(tile, dict) and tile.get("kind") == "PLANT"
    )


def _opponent_supply_at_our_harvest(obs, crop):
    farms = obs.get("farms", [])
    player = int(obs.get("player", 0))
    if len(farms) < 2:
        return 0.0
    day = int(obs.get("day", 0))
    horizon = CROP[crop]["last"]
    supply = 0.0
    for _, tile in _tiles(farms[1 - player]):
        if not (isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile.get("crop") == crop):
            continue
        age = day - int(tile.get("planted_day", day))
        days_to_finish = max(0, CROP[crop]["last"] - age)
        if days_to_finish <= horizon:
            supply += CROP[crop]["yield"] * (1.0 - 0.25 * days_to_finish / max(1, horizon))
    return supply


def _choose_crop(obs, config, position):
    mode = config.get("selector", "demand")
    candidates = [crop for crop in CROPS if CROP[crop]["last"] + 1 <= _remaining_days(obs)]
    if not candidates:
        return None
    fixed = config.get("crop")
    if fixed:
        return fixed if fixed in candidates else None
    if mode == "staggered":
        mix = config.get("mix", ("WHEAT", "CARROT", "TOMATO"))
        return mix[(position[0] + 2 * position[1] + int(obs.get("day", 0))) % len(mix)]
    counts = _opponent_crop_counts(obs)
    if mode == "follower" and counts:
        return max(candidates, key=lambda crop: (counts[crop], _crop_score(obs, crop), crop))
    if mode == "avoider" and counts:
        return min(candidates, key=lambda crop: (_opponent_supply_at_our_harvest(obs, crop), -_crop_score(obs, crop), crop))
    return max(candidates, key=lambda crop: (_crop_score(obs, crop), crop))


def _ordered_cells(farm, include_locked=False):
    size = len(farm.get("tiles", []))
    sheds = _shed_tiles(size)
    cells = []
    for position, tile in _tiles(farm):
        if tile == "LOCKED" and not include_locked:
            continue
        distance = min(_distance(position, shed) for shed in sheds)
        cells.append((distance, position[1], position[0], position))
    return [entry[-1] for entry in sorted(cells)]


def _target_cells(farm, config):
    limit = int(config.get("cells", 1))
    return _ordered_cells(farm)[:limit]


def _crop_jobs(obs, farm, private, config):
    jobs = []
    day = int(obs.get("day", 0))
    targets = _target_cells(farm, config)
    seeds_left = Counter(private.get("seeds", {}) or {})
    planned = Counter()
    for position in targets:
        x, y = position
        tile = farm["tiles"][y][x]
        if isinstance(tile, dict) and tile.get("kind") == "WEED":
            jobs.append((3, position, ["DIG"]))
            continue
        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
            crop = tile.get("crop")
            age = day - int(tile.get("planted_day", day))
            if not tile.get("watered_today", False):
                jobs.append((0, position, ["WATER"]))
            elif tile.get("yield_units", 0) > 0 and (CROP[crop]["last"] <= age or crop in ("TOMATO", "STRAWBERRY")):
                jobs.append((1, position, ["HARVEST"]))
            elif crop in ("TOMATO", "STRAWBERRY") and age >= CROP[crop]["last"]:
                jobs.append((2, position, ["DIG"]))
            continue
        if tile is None:
            crop = _choose_crop(obs, config, position)
            if crop and seeds_left[crop] > planned[crop]:
                jobs.append((5, position, ["PLANT", crop]))
                planned[crop] += 1
    return jobs, planned


def _animal_targets(farm, config):
    target = int(config.get("animals", 1))
    offset = int(config.get("animal_offset", 0))
    return _ordered_cells(farm)[offset:offset + target]


def _animal_setup_jobs(farm, config):
    animal = config["animal"]
    structure = ANIMAL[animal]["structure"]
    jobs = []
    for position in _animal_targets(farm, config):
        x, y = position
        tile = farm["tiles"][y][x]
        if isinstance(tile, dict) and tile.get("kind") == "WEED":
            jobs.append((3, position, ["DIG"]))
        elif tile is None:
            jobs.append((4, position, ["BUILD_" + structure]))
    return jobs


def _animal_tiles(farm, animal=None):
    result = []
    for position, tile in _tiles(farm):
        if isinstance(tile, dict) and tile.get("animal") and (animal is None or tile.get("animal") == animal):
            result.append((position, tile))
    return result


def _desired_workers(farm, config):
    workload = len(_target_cells(farm, config))
    if config.get("animal"):
        workload = max(workload, len(_animal_targets(farm, config)) * 2)
    cap = int(config.get("workers", 0))
    return min(cap, max(0, (workload + 2) // 3 - 1))


def _sell_orders(obs, farm, private, config):
    shed = private.get("shed", {}) or {}
    reserve_wheat = 0
    if config.get("animal"):
        reserve_wheat = max(3, 2 * len(_animal_tiles(farm)))
    orders = []
    buffered = config.get("buffered", False)
    total_stock = sum(int(shed.get(item, 0)) for item in PRODUCTS)
    force = total_stock >= 80 or farm.get("money", 0) < 100 or _remaining_days(obs) <= 1
    for item in PRODUCTS:
        amount = int(shed.get(item, 0))
        if item == "WHEAT":
            amount = max(0, amount - reserve_wheat)
        if amount <= 0:
            continue
        if buffered and item == "CARROT" and not force:
            wait_expired = int(obs.get("day", 0)) % int(config.get("max_wait_days", 3)) == 0
            if _market_price(obs, item) < BASE_PRICE[item] * float(config.get("sell_fraction", 0.92)) and not wait_expired:
                continue
            amount = min(amount, int(config.get("sell_batch", 8)))
        orders.append(["SELL", item, amount])
    return orders


def _expansion_order(obs, farm, config):
    if not config.get("expand") or _remaining_days(obs) < 10:
        return None
    extra = len(farm.get("unlocked_quadrants", ["NW"])) - 1
    if extra >= len(LAND_PRICES):
        return None
    unlocked = sum(tile != "LOCKED" for _, tile in _tiles(farm))
    desired = int(config.get("expansion_demand", config.get("cells", config.get("animals", 1))))
    reserve = int(config.get("cash_reserve", 600))
    price = LAND_PRICES[extra]
    capacity = int(config.get("capacity_per_quadrant", unlocked - 2)) * (extra + 1)
    if desired > capacity and farm.get("money", 0) >= price + reserve:
        return ["BUY_LAND"]
    return None


def _market_orders(obs, farm, private, config, crop_jobs):
    orders = _sell_orders(obs, farm, private, config)
    desired_workers = _desired_workers(farm, config)
    missing_workers = max(0, desired_workers - len(farm.get("hands", [])))
    orders.extend([["HIRE"] for _ in range(missing_workers)])
    expansion = _expansion_order(obs, farm, config)
    if expansion:
        orders.append(expansion)

    if config.get("mode") in ("crops", "mixed"):
        seeds = Counter(private.get("seeds", {}) or {})
        needed = Counter(action[1] for _, _, action in crop_jobs if action[0] == "PLANT")
        empty_without_seed = []
        for position in _target_cells(farm, config):
            x, y = position
            if farm["tiles"][y][x] is None:
                crop = _choose_crop(obs, config, position)
                if crop:
                    empty_without_seed.append(crop)
        wanted = Counter(empty_without_seed)
        for crop in CROPS:
            buy = max(0, wanted[crop] - seeds[crop] - needed[crop])
            reserve = int(config.get("cash_reserve", 100))
            affordable = max(0, int((farm.get("money", 0) - reserve) // CROP[crop]["seed"]))
            buy = min(buy, affordable, 10)
            if buy:
                orders.append(["BUY_SEED", crop, buy])

    if config.get("animal"):
        animal = config["animal"]
        occupied = len(_animal_tiles(farm, animal))
        in_transit = int((private.get("shed", {}) or {}).get(animal, 0))
        for inventory in private.get("inventories", []) or []:
            in_transit += int(inventory.get(animal, 0))
        empty_structures = sum(
            isinstance(tile, dict) and tile.get("kind") == ANIMAL[animal]["structure"] and not tile.get("animal")
            for _, tile in _tiles(farm)
        )
        desired = min(int(config.get("animals", 1)), 1 + int(obs.get("day", 0)) // int(config.get("animal_interval", 4)))
        if occupied + in_transit < desired and empty_structures > in_transit and _remaining_days(obs) >= 8:
            reserve = int(config.get("cash_reserve", 100))
            if farm.get("money", 0) >= ANIMAL[animal]["cost"] + reserve:
                orders.append(["BUY_ANIMAL", animal, 1])
        feed_need = max(0, 2 * max(1, occupied) - int((private.get("shed", {}) or {}).get("WHEAT", 0)))
        if feed_need and farm.get("money", 0) >= feed_need * _market_price(obs, "WHEAT") + 50:
            orders.append(["BUY_PRODUCT", "WHEAT", min(feed_need, 10)])
    return orders[:10]


def _pop_nearest_job(position, jobs):
    if not jobs:
        return None
    index = min(range(len(jobs)), key=lambda i: (jobs[i][0], _distance(position, jobs[i][1]), jobs[i][1]))
    return jobs.pop(index)


def _unit_actions(obs, farm, private, config, crop_jobs):
    positions = [farm.get("farmer", [0, 0]), *farm.get("hands", [])]
    inventories = list(private.get("inventories", []) or [])
    inventories.extend({} for _ in range(max(0, len(positions) - len(inventories))))
    size = len(farm.get("tiles", []))
    sheds = _shed_tiles(size)
    jobs = list(crop_jobs)
    if config.get("animal"):
        jobs.extend(_animal_setup_jobs(farm, config))
        for target, tile in _animal_tiles(farm, config["animal"]):
            if tile.get("yield_units", 0) > 0:
                jobs.append((1, target, ["HARVEST"]))
            if tile.get("fertilizer_available", False):
                jobs.append((2, target, ["COLLECT_FERTILIZER"]))
            if tile.get("fed_today", False) and not tile.get("cared_today", False):
                jobs.append((6, target, ["CARE"]))

    animal_positions = [position for position, _ in _animal_tiles(farm, config.get("animal"))]
    feed_positions = [
        position for position, tile in _animal_tiles(farm, config.get("animal"))
        if not tile.get("fed_today", False)
    ]
    empty_structures = [
        position for position, tile in _tiles(farm)
        if isinstance(tile, dict)
        and config.get("animal")
        and tile.get("kind") == ANIMAL[config["animal"]]["structure"]
        and not tile.get("animal")
    ]
    actions = []
    pickup_feed_assigned = False
    pickup_animal_assigned = False
    for index, position in enumerate(positions):
        inventory = inventories[index] or {}
        animal = config.get("animal")
        if animal and inventory.get(animal, 0) > 0 and empty_structures:
            target = _nearest(position, empty_structures)
            empty_structures.remove(target)
            actions.append(_at_or_move(position, target, ["PLACE", animal]))
            continue
        if animal and inventory.get("WHEAT", 0) > 0 and feed_positions:
            target = _nearest(position, feed_positions)
            feed_positions.remove(target)
            actions.append(_at_or_move(position, target, ["FEED"]))
            continue
        if config.get("mode") == "mixed" and inventory.get("FERTILIZER", 0) > 0:
            profitable = []
            for target, tile in _tiles(farm):
                if not (isinstance(tile, dict) and tile.get("kind") == "PLANT"):
                    continue
                crop = tile.get("crop")
                if tile.get("fertilized_until_day", -1) < int(obs.get("day", 0)) and _market_price(obs, crop) > _market_price(obs, "FERTILIZER"):
                    profitable.append(target)
            if profitable:
                target = _nearest(position, profitable)
                actions.append(_at_or_move(position, target, ["FERTILIZE"]))
                continue
        useful = {item: count for item, count in inventory.items() if count > 0}
        if useful:
            target = _nearest(position, sheds)
            actions.append(_at_or_move(position, target, ["DROP"]))
            continue
        if animal and feed_positions and not pickup_feed_assigned and (private.get("shed", {}) or {}).get("WHEAT", 0) > 0:
            pickup_feed_assigned = True
            target = _nearest(position, sheds)
            actions.append(_at_or_move(position, target, ["PICKUP", "WHEAT", min(10, len(feed_positions))]))
            continue
        if animal and empty_structures and not pickup_animal_assigned and (private.get("shed", {}) or {}).get(animal, 0) > 0:
            pickup_animal_assigned = True
            target = _nearest(position, sheds)
            actions.append(_at_or_move(position, target, ["PICKUP", animal, 1]))
            continue
        job = _pop_nearest_job(position, jobs)
        if job:
            _, target, command = job
            actions.append(_at_or_move(position, target, command))
        else:
            target = _nearest(position, sheds)
            actions.append(_move(position, target))
    return actions


def run_agent(obs, config):
    """Return one valid Kaggriculture action from a declarative strategy config."""
    farms = obs.get("farms", [])
    player = int(obs.get("player", 0))
    if not farms or player >= len(farms):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    farm = farms[player]
    private = obs.get("private", {}) or {}
    config = dict(config)
    if config.get("mode") == "mixed":
        config["animal"] = "GOOSE"
    if config.get("animal"):
        interval = max(1, int(config.get("animal_interval", 4)))
        config["animals"] = min(int(config.get("animals", 1)), 1 + int(obs.get("day", 0)) // interval)
    crop_jobs = []
    if config.get("mode") in ("crops", "mixed"):
        crop_jobs, _ = _crop_jobs(obs, farm, private, config)
    unit_actions = _unit_actions(obs, farm, private, config, crop_jobs)
    result = _empty(farm)
    result["farmer"] = unit_actions[0]
    result["hands"] = unit_actions[1:]
    result["market"] = _market_orders(obs, farm, private, config, crop_jobs)
    return result
