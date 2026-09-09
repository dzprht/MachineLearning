"""Fixed, versioned numeric observation encoding for the v1 multi-cell Economist.

Market/shop/opponent-farm blocks are the current seed-buyer encoder's own
static helpers, reused verbatim per the IDEA-020 contract point 5 ("блоки
рынка, магазинов и публичной фермы соперника из текущего encoder"); this
module only adds the new nine-cell/worker/self blocks.
"""

from __future__ import annotations

import math

import numpy as np

from training.seed_buyer.encoder import ObservationEncoder as SeedBuyerEncoder

from .constants import (
    ACTION_NAMES,
    CROP_TIMELINE,
    CROPS,
    DEFAULT_CONFIGURATION,
    FEATURE_SCHEMA_VERSION,
    MAX_HANDS,
    MAX_SEEDS,
    MAX_WORKERS,
    MONEY_SCALE,
    NUM_TARGET_CELLS,
    PRODUCTS,
    SHOP_ORDER,
)
from .executor import EconomistState, _target_cells

CELL_BLOCK_SIZE = 11
WORKER_BLOCK_SIZE = 5
MAX_AGE = 20.0
MAX_YIELD_UNITS = 6.0
FEATURE_COUNT = (
    4  # time
    + 1  # self.money
    + len(CROPS)  # self.seeds
    + len(PRODUCTS)  # self.shed
    + 3  # self.workers, self.hires_today, self.next_hire_affordable
    + (len(CROPS) + 1)  # pending crop one-hot (+ none)
    + 1  # hire_requested
    + NUM_TARGET_CELLS * CELL_BLOCK_SIZE
    + MAX_WORKERS * WORKER_BLOCK_SIZE
    + 18  # market: price + inventory deviation (seed-buyer block)
    + 17  # town: shop counts + daily demand (seed-buyer block)
    + 90  # self_farm (seed-buyer block)
    + 90  # opponent_farm (seed-buyer block)
    + 4  # opponent summary
)


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


def _fib_cost(n: int) -> int:
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


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

    def encode(self, obs: dict, executor_state: EconomistState | None = None) -> np.ndarray:
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
        step = int(obs.get("step", 0))
        day = int(obs.get("day", step // turns))
        hour = int(obs.get("hour", step % turns))
        values: list[float] = []

        values.extend(
            (
                _unit(step, episode_steps - 1),
                _unit(day, total_days - 1),
                _unit(hour, turns - 1),
                _unit(max(0, episode_steps - 1 - step), episode_steps - 1),
            )
        )

        values.append(_money(own["money"]))
        seeds = private.get("seeds", {}) or {}
        values.extend(_unit(seeds.get(crop, 0), MAX_SEEDS) for crop in CROPS)
        shed = private.get("shed", {}) or {}
        values.extend(_unit(shed.get(item, 0), shed_capacity) for item in PRODUCTS)

        hands = own.get("hands", [])
        hires_today = int(own.get("hires_today", 0))
        values.extend(
            (
                _unit(1 + len(hands), MAX_WORKERS),
                _unit(hires_today, MAX_WORKERS),
                1.0 if own["money"] >= _fib_cost(hires_today) and len(hands) < MAX_HANDS else 0.0,
            )
        )

        state = executor_state or EconomistState()
        values.extend(1.0 if state.pending_crop == crop else 0.0 for crop in CROPS)
        values.append(1.0 if state.pending_crop is None else 0.0)
        values.append(1.0 if state.hire_requested else 0.0)

        values.extend(self._cell_features(own, day))
        values.extend(self._worker_features(own, private, board_size=len(own["tiles"])))

        values.extend(SeedBuyerEncoder._market_features(obs.get("market", {})))
        values.extend(SeedBuyerEncoder._shop_features(obs.get("town", {}), turns, self.configuration))
        values.extend(SeedBuyerEncoder._farm_features(own, day, board_cells))
        values.extend(SeedBuyerEncoder._farm_features(other, day, board_cells))
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
    def _cell_features(farm: dict, day: int) -> list[float]:
        values: list[float] = []
        cells = _target_cells(len(farm["tiles"]))
        for x, y in cells:
            tile = farm["tiles"][y][x]
            is_empty = tile is None
            is_weed = isinstance(tile, dict) and tile.get("kind") == "WEED"
            crop_one_hot = [0.0] * len(CROPS)
            age_norm = 0.0
            yield_norm = 0.0
            watered = 0.0
            unwatered_risk = 0.0
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                crop = tile.get("crop")
                if crop in CROPS:
                    crop_one_hot[CROPS.index(crop)] = 1.0
                age_norm = _unit(max(0, day - int(tile.get("planted_day", day))), MAX_AGE)
                cap = CROP_TIMELINE.get(crop, {}).get("unfertilized_yield", 1)
                yield_norm = _unit(tile.get("yield_units", 0), max(cap, 1))
                watered = 1.0 if tile.get("watered_today", False) else 0.0
                unwatered_risk = 1.0 if int(tile.get("consecutive_unwatered", 0)) >= 1 else 0.0
            values.extend(
                (
                    1.0 if is_empty else 0.0,
                    1.0 if is_weed else 0.0,
                    *crop_one_hot,
                    age_norm,
                    yield_norm,
                    watered,
                    unwatered_risk,
                )
            )
        return values

    @staticmethod
    def _worker_features(farm: dict, private: dict, board_size: int) -> list[float]:
        values: list[float] = []
        positions = [farm.get("farmer", [0, 0]), *farm.get("hands", [])]
        inventories = list(private.get("inventories", []) or [])
        for index in range(MAX_WORKERS):
            if index < len(positions):
                x, y = positions[index]
                inventory = inventories[index] if index < len(inventories) else {}
                total_items = sum((inventory or {}).values())
                values.extend(
                    (
                        1.0,
                        _unit(x, max(board_size - 1, 1)),
                        _unit(y, max(board_size - 1, 1)),
                        1.0 if total_items > 0 else 0.0,
                        _unit(total_items, 20.0),
                    )
                )
            else:
                values.extend((0.0, 0.0, 0.0, 0.0, 0.0))
        return values

    @staticmethod
    def _build_feature_names() -> list[str]:
        names = ["time.step", "time.day", "time.hour", "time.remaining", "self.money"]
        names.extend(f"self.seeds.{crop}" for crop in CROPS)
        names.extend(f"self.shed.{item}" for item in PRODUCTS)
        names.extend(("self.workers", "self.hires_today", "self.next_hire_affordable"))
        names.extend(f"pending.{crop}" for crop in CROPS)
        names.extend(("pending.none", "pending.hire_requested"))
        for i in range(NUM_TARGET_CELLS):
            names.extend(
                (
                    f"cell{i}.empty",
                    f"cell{i}.weed",
                    *(f"cell{i}.crop.{crop}" for crop in CROPS),
                    f"cell{i}.age",
                    f"cell{i}.yield_units",
                    f"cell{i}.watered",
                    f"cell{i}.unwatered_risk",
                )
            )
        for i in range(MAX_WORKERS):
            names.extend((f"worker{i}.present", f"worker{i}.x", f"worker{i}.y", f"worker{i}.carrying", f"worker{i}.inventory"))
        names.extend(f"market.price.{item}" for item in PRODUCTS)
        names.extend(f"market.inventory_deviation.{item}" for item in PRODUCTS)
        names.extend(f"town.shop_count.{shop}" for shop in SHOP_ORDER)
        names.extend(f"town.daily_demand.{item}" for item in PRODUCTS)
        for role in ("self_farm", "opponent_farm"):
            names.extend(_farm_feature_names(role))
        names.extend(("opponent.money", "opponent.unlocked_fraction", "opponent.workers", "opponent.hires_today"))
        return names

    def schema(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "dtype": "float32",
            "shape": [FEATURE_COUNT],
            "range": [-1.0, 1.0],
            "features": self.feature_names,
        }


def _farm_feature_names(role: str) -> list[str]:
    from training.seed_buyer.constants import AGE_HISTOGRAM_MAX, ANIMALS

    names = []
    for crop in CROPS:
        maximum = AGE_HISTOGRAM_MAX[crop]
        names.extend(f"{role}.{crop}.age_{age}" for age in range(maximum + 1))
        names.append(f"{role}.{crop}.age_overflow")
    for crop in CROPS:
        names.extend((f"{role}.{crop}.yield_units", f"{role}.{crop}.watered", f"{role}.{crop}.unwatered_risk"))
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
    names.extend((f"{role}.empty_coops", f"{role}.empty_pastures", f"{role}.weeds", f"{role}.empty_unlocked"))
    return names
