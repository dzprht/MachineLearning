"""Fixed, versioned numeric observation encoding for the v1 seed buyer."""

from __future__ import annotations

import math

import numpy as np

from .constants import (
    ACTION_NAMES,
    AGE_HISTOGRAM_MAX,
    ANIMALS,
    ANIMAL_SPECS,
    CROP_SPECS,
    CROPS,
    DEFAULT_CONFIGURATION,
    FEATURE_SCHEMA_VERSION,
    MARKET_SPECS,
    MAX_DAILY_DEMAND,
    MAX_SEEDS,
    MAX_SHOP_INSTANCES,
    MAX_WORKERS,
    MONEY_SCALE,
    PRODUCTS,
    SHOPS,
    SHOP_ORDER,
)


FEATURE_COUNT = 249


def _cfg(configuration, name: str):
    default = DEFAULT_CONFIGURATION[name]
    if configuration is None:
        return default
    if isinstance(configuration, dict):
        return configuration.get(name, default)
    return getattr(configuration, name, default)


def _unit(value: float, denominator: float) -> float:
    return min(1.0, max(0.0, float(value) / max(float(denominator), 1.0)))


def _money(value: float) -> float:
    return min(1.0, math.log1p(max(0.0, float(value))) / math.log1p(MONEY_SCALE))


class ObservationEncoder:
    schema_version = FEATURE_SCHEMA_VERSION
    feature_count = FEATURE_COUNT

    def __init__(self, configuration=None):
        self.configuration = configuration
        self.feature_names = self._build_feature_names()
        if len(self.feature_names) != FEATURE_COUNT:
            raise AssertionError(
                f"Feature schema has {len(self.feature_names)} names, expected {FEATURE_COUNT}"
            )

    def encode(self, obs: dict, executor_state=None) -> np.ndarray:
        player = int(obs.get("player", 0))
        opponent = 1 - player
        farms = obs["farms"]
        own = farms[player]
        other = farms[opponent]
        private = obs.get("private", {}) or {}
        turns = int(_cfg(self.configuration, "turnsPerDay"))
        episode_steps = int(_cfg(self.configuration, "episodeSteps"))
        total_days = max(1, math.ceil(episode_steps / turns))
        board_cells = max(1, len(own["tiles"]) ** 2)
        shed_capacity = int(_cfg(self.configuration, "shedCapacity"))
        values: list[float] = []

        step = int(obs.get("step", 0))
        day = int(obs.get("day", step // turns))
        hour = int(obs.get("hour", step % turns))
        values.extend(
            (
                _unit(step, episode_steps - 1),
                _unit(day, total_days - 1),
                _unit(hour, turns - 1),
                _unit(max(0, episode_steps - 1 - step), episode_steps - 1),
            )
        )

        values.append(_money(own["money"]))
        seeds = private.get("seeds", {})
        values.extend(_unit(seeds.get(crop, 0), MAX_SEEDS) for crop in CROPS)
        shed = private.get("shed", {})
        values.extend(_unit(shed.get(item, 0), shed_capacity) for item in PRODUCTS)
        values.extend(
            (
                _unit(1 + len(own.get("hands", [])), MAX_WORKERS),
                _unit(own.get("hires_today", 0), MAX_WORKERS),
                _unit(len(own.get("unlocked_quadrants", [])), 4),
            )
        )
        plan = getattr(executor_state, "plan", None) if executor_state is not None else None
        values.extend(1.0 if plan == action else 0.0 for action in ACTION_NAMES)
        values.append(
            1.0 if executor_state is not None and executor_state.purchase_requested else 0.0
        )
        wait_remaining = getattr(executor_state, "wait_remaining", 0)
        values.append(_unit(wait_remaining, turns))

        market = obs.get("market", {})
        prices = market.get("prices", {})
        inventory = market.get("inventory", {})
        for item in PRODUCTS:
            price = max(1.0, float(prices.get(item, MARKET_SPECS[item]["base"])))
            values.append(math.tanh(math.log(price / MARKET_SPECS[item]["base"])))
        for item in PRODUCTS:
            spec = MARKET_SPECS[item]
            values.append(math.tanh((spec["I0"] - inventory.get(item, spec["I0"])) / spec["T"]))

        unlocked_shops = obs.get("town", {}).get("unlocked_shops", [])
        shop_counts = {shop: unlocked_shops.count(shop) for shop in SHOP_ORDER}
        values.extend(_unit(shop_counts[shop], MAX_SHOP_INSTANCES) for shop in SHOP_ORDER)
        demand = {item: 0.0 for item in PRODUCTS}
        center_ticks = turns / max(1, int(_cfg(self.configuration, "townCenterSellInterval")))
        for item in PRODUCTS:
            if item != "FERTILIZER":
                demand[item] += center_ticks
        shop_ticks = turns / max(1, int(_cfg(self.configuration, "townShopSellInterval")))
        for shop, count in shop_counts.items():
            multiplier = 2 if len(SHOPS[shop]) == 1 else 1
            for item in SHOPS[shop]:
                demand[item] += count * multiplier * shop_ticks
        values.extend(_unit(demand[item], MAX_DAILY_DEMAND) for item in PRODUCTS)

        values.extend(self._farm_features(own, day, board_cells))
        values.extend(self._farm_features(other, day, board_cells))
        values.extend(
            (
                _money(other["money"]),
                _unit(len(other.get("unlocked_quadrants", [])), 4),
                _unit(1 + len(other.get("hands", [])), MAX_WORKERS),
                _unit(other.get("hires_today", 0), MAX_WORKERS),
            )
        )

        encoded = np.asarray(values, dtype=np.float32)
        if encoded.shape != (FEATURE_COUNT,):
            raise AssertionError(f"Encoded shape {encoded.shape}, expected ({FEATURE_COUNT},)")
        if not np.isfinite(encoded).all():
            raise ValueError("Observation encoding contains non-finite values")
        return np.clip(encoded, -1.0, 1.0)

    @staticmethod
    def _farm_features(farm: dict, day: int, board_cells: int) -> list[float]:
        tiles = [tile for row in farm["tiles"] for tile in row]
        values: list[float] = []
        for crop in CROPS:
            maximum = AGE_HISTOGRAM_MAX[crop]
            histogram = [0] * (maximum + 2)
            for tile in tiles:
                if isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile.get("crop") == crop:
                    age = max(0, day - int(tile["planted_day"]))
                    histogram[min(age, maximum + 1)] += 1
            values.extend(_unit(count, board_cells) for count in histogram)

        for crop in CROPS:
            crop_tiles = [
                tile
                for tile in tiles
                if isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile.get("crop") == crop
            ]
            values.extend(
                (
                    _unit(
                        sum(tile.get("yield_units", 0) for tile in crop_tiles),
                        CROP_SPECS[crop]["max_yield"] * board_cells,
                    ),
                    _unit(sum(bool(tile.get("watered_today", False)) for tile in crop_tiles), board_cells),
                    _unit(
                        sum(tile.get("consecutive_unwatered", 0) >= 1 for tile in crop_tiles),
                        board_cells,
                    ),
                )
            )

        for animal in ANIMALS:
            animal_tiles = [
                tile
                for tile in tiles
                if isinstance(tile, dict) and tile.get("animal") == animal
            ]
            max_held = ANIMAL_SPECS[animal]["max_held"]
            values.extend(
                (
                    _unit(len(animal_tiles), board_cells),
                    _unit(sum(tile.get("yield_units", 0) for tile in animal_tiles), max_held * board_cells),
                    _unit(sum(not tile.get("fed_today", False) for tile in animal_tiles), board_cells),
                    _unit(sum(bool(tile.get("fertilizer_available", False)) for tile in animal_tiles), board_cells),
                    _unit(sum(tile.get("pending_care_bonus", 0) for tile in animal_tiles), max_held * board_cells),
                )
            )

        for structure in ("COOP", "PASTURE"):
            values.append(
                _unit(
                    sum(
                        isinstance(tile, dict)
                        and tile.get("kind") == structure
                        and "animal" not in tile
                        for tile in tiles
                    ),
                    board_cells,
                )
            )
        values.append(
            _unit(sum(isinstance(tile, dict) and tile.get("kind") == "WEED" for tile in tiles), board_cells)
        )
        values.append(_unit(sum(tile is None for tile in tiles), board_cells))
        if len(values) != 90:
            raise AssertionError(f"Farm feature block has {len(values)} values, expected 90")
        return values

    @staticmethod
    def _build_feature_names() -> list[str]:
        names = ["time.step", "time.day", "time.hour", "time.remaining"]
        names.append("self.money")
        names.extend(f"self.seeds.{crop}" for crop in CROPS)
        names.extend(f"self.shed.{item}" for item in PRODUCTS)
        names.extend(("self.workers", "self.hires_today", "self.unlocked_fraction"))
        names.extend(f"executor.plan.{action}" for action in ACTION_NAMES)
        names.extend(("executor.purchase_requested", "executor.wait_remaining"))
        names.extend(f"market.price.{item}" for item in PRODUCTS)
        names.extend(f"market.inventory_deviation.{item}" for item in PRODUCTS)
        names.extend(f"town.shop_count.{shop}" for shop in SHOP_ORDER)
        names.extend(f"town.daily_demand.{item}" for item in PRODUCTS)
        for role in ("self_farm", "opponent_farm"):
            for crop in CROPS:
                maximum = AGE_HISTOGRAM_MAX[crop]
                names.extend(f"{role}.{crop}.age_{age}" for age in range(maximum + 1))
                names.append(f"{role}.{crop}.age_overflow")
            for crop in CROPS:
                names.extend(
                    (
                        f"{role}.{crop}.yield_units",
                        f"{role}.{crop}.watered",
                        f"{role}.{crop}.unwatered_risk",
                    )
                )
            for animal in ANIMALS:
                names.extend(
                    (
                        f"{role}.{animal}.count",
                        f"{role}.{animal}.yield_units",
                        f"{role}.{animal}.needs_feed",
                        f"{role}.{animal}.fertilizer_available",
                        f"{role}.{animal}.pending_care_bonus",
                    )
                )
            names.extend(
                (
                    f"{role}.empty_coops",
                    f"{role}.empty_pastures",
                    f"{role}.weeds",
                    f"{role}.empty_unlocked",
                )
            )
        names.extend(
            (
                "opponent.money",
                "opponent.unlocked_fraction",
                "opponent.workers",
                "opponent.hires_today",
            )
        )
        return names

    def schema(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "dtype": "float32",
            "shape": [FEATURE_COUNT],
            "range": [-1.0, 1.0],
            "features": self.feature_names,
        }
