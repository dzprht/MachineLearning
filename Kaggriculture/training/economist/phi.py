"""analytic_v1 potential function Phi(s), per IDEA-007/IDEA-020 in docs/IDEAS.md.

An engineering approximation of realizable future net worth, computed purely
from the current observation and fixed game-rule formulas -- no `env.step`
anywhere in this module (checked by tests/test_economist.py via a step-count
monkeypatch, matching the "Дополнительная приёмка analytic_v1" requirement).

Deliberately-approximate pieces, called out where they matter:
  * one day of calendar granularity (a[j]/beta[d]/tau[d] are per-day, not
    per-turn);
  * no future weed spawns, decay, or opponent actions are predicted -- only
    currently-visible state and known crop-growth rules;
  * `A`/`W`/`K_future` assume the *fixed* future-hiring rule described in the
    contract, not the RL policy's actual future hires;
  * fractional batch quantities are linearly interpolated between integer
    sale prices (see `pricing.batch_revenue`).
None of this affects the reward-shaping identity `sum(r) == 0.01 * (C_terminal
- C_initial - Phi_initial)` at gamma=1, which holds by telescoping regardless
of Phi's accuracy as an economic estimate (see IDEA-007) -- only how *useful*
a training signal Phi is.
"""

from __future__ import annotations

from collections import Counter
import math

from . import pricing
from .constants import (
    CROP_TIMELINE,
    CROPS,
    DEFAULT_CONFIGURATION,
    FARM_HAND_COST_MULT,
    MARKET_PARAMS,
    NUM_TARGET_CELLS,
    PRODUCTS,
    SHOPS,
    SHOP_ORDER,
)


def _cfg(configuration, name: str):
    default = DEFAULT_CONFIGURATION[name]
    if configuration is None:
        return default
    if isinstance(configuration, dict):
        return configuration.get(name, default)
    return getattr(configuration, name, default)


def _fib(n: int) -> int:
    """fib(0)=1, fib(1)=1, fib(2)=2, fib(3)=3, fib(4)=5, ... (env convention)."""
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _hire_cost_for_n(n: int) -> int:
    """Cost of hiring n additional hands in one day, resetting each day."""
    return sum(FARM_HAND_COST_MULT * _fib(i) for i in range(n))


def _distance(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _shed_tiles(board_size: int):
    half = board_size // 2
    return ((half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half))


def _nearest_distance(position, targets) -> int:
    return min(_distance(position, target) for target in targets)


def daily_town_demand(town: dict, configuration=None) -> dict[str, float]:
    """Units/day consumed by the town center plus every unlocked shop instance."""
    turns = int(_cfg(configuration, "turnsPerDay"))
    unlocked = town.get("unlocked_shops", [])
    shop_counts = Counter(unlocked)
    demand = {item: 0.0 for item in PRODUCTS}
    center_ticks = turns / max(1, int(_cfg(configuration, "townCenterSellInterval")))
    for item in PRODUCTS:
        if item != "FERTILIZER":
            demand[item] += center_ticks
    shop_ticks = turns / max(1, int(_cfg(configuration, "townShopSellInterval")))
    for shop in SHOP_ORDER:
        count = shop_counts.get(shop, 0)
        if not count:
            continue
        multiplier = 2 if len(SHOPS[shop]) == 1 else 1
        for item in SHOPS[shop]:
            demand[item] += count * multiplier * shop_ticks
    return demand


def _crop_alive_and_harvest_days(crop: str, planted_day: int, today: int) -> tuple[list[int], list[tuple[int, float]]]:
    """(alive_days, [(harvest_day, unfertilized_units_that_tick), ...]) from `today` on.

    `alive_days` is every day from today through the plant's last useful day
    (inclusive) that still needs a WATER action; harvest ticks are the
    subset of days that additionally need a HARVEST action and yield units.
    Days already in the past (< today) are never returned.
    """
    spec = CROP_TIMELINE[crop]
    if not spec["ongoing"]:
        last_day = planted_day + spec["ripe_day"]
        alive = [d for d in range(max(today, planted_day), last_day + 1)]
        harvests = [(last_day, float(spec["unfertilized_yield"]))] if last_day >= today else []
        return alive, harvests
    tick_days = [
        planted_day + spec["first_yield_day"] + i * spec["interval"] for i in range(spec["events"])
    ]
    last_day = tick_days[-1]
    alive = [d for d in range(max(today, planted_day), last_day + 1)]
    harvests = [(d, float(spec["unfertilized_yield"])) for d in tick_days if d >= today]
    return alive, harvests


class Batch:
    __slots__ = ("crop", "quantity", "sale_day", "required_days")

    def __init__(self, crop: str, quantity: float, sale_day: int, required_days: tuple[int, ...] = ()):
        self.crop = crop
        self.quantity = quantity
        self.sale_day = sale_day
        self.required_days = required_days


def _delivery_day(position, board_size: int, turns_per_day: int, hour: int, today: int) -> int:
    """First day a worker at `position` can reach the shed, DROP and SELL."""
    trip = _nearest_distance(position, _shed_tiles(board_size)) + 1  # +1 turn to DROP/SELL
    return today if trip <= (turns_per_day - hour) else today + 1


def _gather_batches(obs: dict, executor_state, configuration) -> tuple[list[Batch], list[tuple[int, int]]]:
    """Batches j plus (planted_day, alive-day-count-source) growing-plant list."""
    player = int(obs.get("player", 0))
    farm = obs["farms"][player]
    private = obs.get("private", {}) or {}
    board_size = len(farm["tiles"])
    turns_per_day = int(_cfg(configuration, "turnsPerDay"))
    today = int(obs.get("day", 0))
    hour = int(obs.get("hour", 0))

    batches: list[Batch] = []
    growing: list[tuple[str, int]] = []  # (crop, planted_day)

    shed = private.get("shed", {}) or {}
    for item in CROPS:
        amount = float(shed.get(item, 0))
        if amount > 0:
            batches.append(Batch(item, amount, today))

    farmer_pos = farm.get("farmer", [0, 0])
    positions = [farmer_pos, *farm.get("hands", [])]
    inventories = list(private.get("inventories", []) or [])
    for index, position in enumerate(positions):
        inventory = inventories[index] if index < len(inventories) else {}
        for item in CROPS:
            amount = float((inventory or {}).get(item, 0))
            if amount > 0:
                sale_day = _delivery_day(position, board_size, turns_per_day, hour, today)
                batches.append(Batch(item, amount, sale_day))

    target_cells = _target_cells(farm)
    for x, y in target_cells:
        tile = farm["tiles"][y][x]
        if not (isinstance(tile, dict) and tile.get("kind") == "PLANT"):
            continue
        crop = tile.get("crop")
        planted_day = int(tile.get("planted_day", today))
        growing.append((crop, planted_day))
        alive, harvests = _crop_alive_and_harvest_days(crop, planted_day, today)
        already_accrued = float(tile.get("yield_units", 0) or 0)
        for i, (harvest_day, units) in enumerate(harvests):
            # The nearest upcoming tick already reflects whatever this tile's
            # `yield_units` currently shows (it accrues day by day in-env);
            # later ticks (ongoing crops) are modeled at their full unit value.
            tick_units = max(units, already_accrued) if i == 0 else units
            delivery_day = _delivery_day((x, y), board_size, turns_per_day, hour, harvest_day)
            required = tuple(d for d in alive if d <= harvest_day)
            batches.append(Batch(crop, tick_units, delivery_day, required))

    pending_crop = getattr(executor_state, "pending_crop", None)
    seeds = private.get("seeds", {}) or {}
    if pending_crop and float(seeds.get(pending_crop, 0)) > 0:
        plant_day = today if hour < turns_per_day - 1 else today + 1
        growing.append((pending_crop, plant_day))
        alive, harvests = _crop_alive_and_harvest_days(pending_crop, plant_day, today)
        pending_cell = getattr(executor_state, "pending_cell", None) or target_cells[0]
        for harvest_day, units in harvests:
            delivery_day = _delivery_day(pending_cell, board_size, turns_per_day, hour, harvest_day)
            required = tuple(d for d in alive if d <= harvest_day)
            batches.append(Batch(pending_crop, units, delivery_day, required))

    return batches, growing


def _target_cells(farm: dict) -> list[tuple[int, int]]:
    board_size = len(farm["tiles"])
    sheds = _shed_tiles(board_size)
    cells = []
    for y in range(board_size // 2):
        for x in range(board_size // 2):
            distance = min(_distance((x, y), shed) for shed in sheds)
            cells.append((distance, y, x, (x, y)))
    cells.sort()
    return [entry[-1] for entry in cells[:NUM_TARGET_CELLS]]


def _build_calendar(growing: list[tuple[str, int]], today: int, horizon_end: int) -> dict[int, int]:
    """day -> count of growing plants still alive that day (N_active[u])."""
    active = Counter()
    for crop, planted_day in growing:
        alive, _ = _crop_alive_and_harvest_days(crop, planted_day, today)
        for d in alive:
            if d <= horizon_end:
                active[d] += 1
    return active


def compute_phi(obs: dict, executor_state=None, configuration=None) -> float:
    """analytic_v1 Phi(s). Deterministic, finite, no env.step."""
    player = int(obs.get("player", 0))
    farm = obs["farms"][player]
    private = obs.get("private", {}) or {}
    turns_per_day = int(_cfg(configuration, "turnsPerDay"))
    episode_steps = int(_cfg(configuration, "episodeSteps"))
    shed_capacity = int(_cfg(configuration, "shedCapacity"))
    today = int(obs.get("day", 0))
    hour = int(obs.get("hour", 0))
    total_days = max(1, math.ceil(episode_steps / turns_per_day))
    last_day = total_days - 1
    if today >= last_day:
        return 0.0  # No day remains to act, water, harvest or sell further.

    batches, growing = _gather_batches(obs, executor_state, configuration)
    batches = [b for b in batches if b.sale_day <= last_day and b.quantity > 0]
    if not batches:
        return -_k_future(growing, today, last_day)

    active_by_day = _build_calendar(growing, today, last_day)
    board_size = len(farm["tiles"])
    farmer_pos = farm.get("farmer", [0, 0])
    worker_positions = [farmer_pos, *farm.get("hands", [])]
    target_cells = _target_cells(farm)
    cell_shed_distance = {
        cell: _nearest_distance(cell, _shed_tiles(board_size)) for cell in target_cells
    }

    # Per-day workload W[u]: one action per still-active plant, plus a coarse
    # travel proxy (round trips shed<->cell for cells needing work that day).
    active_cells_by_day: dict[int, list[tuple[int, int]]] = {}
    for cell in target_cells:
        tile = farm["tiles"][cell[1]][cell[0]]
        if not (isinstance(tile, dict) and tile.get("kind") == "PLANT"):
            continue
        alive, _ = _crop_alive_and_harvest_days(tile.get("crop"), int(tile.get("planted_day", today)), today)
        for d in alive:
            if d <= last_day:
                active_cells_by_day.setdefault(d, []).append(cell)

    def compute_W(day: int) -> int:
        cells = active_cells_by_day.get(day, [])
        actions = active_by_day.get(day, 0)
        travel = sum(2 * cell_shed_distance[c] for c in cells)
        if day == today and cells:
            saved = min(_nearest_distance(pos, cells) for pos in worker_positions)
            first_leg = min(cell_shed_distance[c] for c in cells)
            travel -= max(0, first_leg - saved)
        return actions + travel

    def compute_A(day: int, n_hires: int) -> float:
        if day == today:
            return max(0.0, turns_per_day - hour) * len(worker_positions)
        return turns_per_day * (1 + n_hires)

    n_by_day: dict[int, int] = {}
    for day in range(today, last_day + 1):
        active = active_by_day.get(day, 0)
        n_by_day[day] = min(4, max(0, math.ceil(active / 3) - 1)) if active else 0

    def feasibility(required_days) -> float:
        if not required_days:
            return 1.0
        worst = 1.0
        for day in required_days:
            w = max(1, compute_W(day))
            a = compute_A(day, n_by_day.get(day, 0))
            worst = min(worst, min(1.0, a / w))
        return worst

    by_day_product: dict[int, dict[str, float]] = {}
    for batch in batches:
        a_j = feasibility(batch.required_days)
        by_day_product.setdefault(batch.sale_day, {})
        by_day_product[batch.sale_day][batch.crop] = (
            by_day_product[batch.sale_day].get(batch.crop, 0.0) + batch.quantity * a_j
        )

    market = obs.get("market", {}) or {}
    inventory_now = market.get("inventory", {}) or {}
    demand = daily_town_demand(obs.get("town", {}) or {}, configuration)
    added_sales = {crop: 0.0 for crop in CROPS}
    phi = 0.0
    for day in sorted(by_day_product):
        raw_by_crop = by_day_product[day]
        raw_total = sum(raw_by_crop.values())
        beta = min(1.0, shed_capacity / max(raw_total, 1.0))
        tau = max(0, day - today)
        for crop, raw_qty in raw_by_crop.items():
            qty = beta * raw_qty
            if qty <= 0:
                continue
            current_inv = float(inventory_now.get(crop, MARKET_PARAMS[crop]["I0"]))
            i_hat = max(0.0, current_inv - demand[crop] * tau + added_sales[crop])
            phi += pricing.batch_revenue(crop, i_hat, qty)
            added_sales[crop] += qty

    phi -= _k_future(growing, today, last_day, precomputed_n=n_by_day)
    return phi


def _k_future(growing, today: int, last_day: int, precomputed_n: dict[int, int] | None = None) -> float:
    if precomputed_n is not None:
        n_by_day = precomputed_n
    else:
        active_by_day = _build_calendar(growing, today, last_day)
        n_by_day = {
            day: (min(4, max(0, math.ceil(active_by_day.get(day, 0) / 3) - 1)) if active_by_day.get(day, 0) else 0)
            for day in range(today, last_day + 1)
        }
    total = 0.0
    for day in range(today + 1, last_day + 1):  # today's real hire is not double-counted
        n = n_by_day.get(day, 0)
        if n > 0:
            total += _hire_cost_for_n(n)
    return total
