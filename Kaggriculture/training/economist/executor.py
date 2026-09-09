"""Deterministic Farmer/Dispatcher execution for the multi-cell Economist.

The RL Economist only ever decides: which crop to start next (reserving one
of the nine managed cells and its seed), whether to HIRE_ONE, or to WAIT.
Everything else -- movement, watering, harvesting, digging, delivery, selling
-- is this fixed rule-based executor, in the same style as
`agents/baselines/rule_based_v1/engine.py` (job-priority queue + nearest-
worker assignment), reimplemented locally rather than imported so this
training package stays self-contained (see agents/README.md) and the frozen
bundle stays untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import ACTION_NAMES, CROP_TIMELINE, CROPS, MAX_HANDS, NUM_TARGET_CELLS, PRODUCTS


def _cfg(configuration, name: str, default: int) -> int:
    if configuration is None:
        return default
    if isinstance(configuration, dict):
        return int(configuration.get(name, default))
    return int(getattr(configuration, name, default))


def _distance(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _move(position, target) -> list[str]:
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
    return min(targets, key=lambda t: (_distance(position, t), t[1], t[0]))


def _shed_tiles(board_size: int):
    half = board_size // 2
    return ((half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half))


def _target_cells(board_size: int) -> list[tuple[int, int]]:
    sheds = _shed_tiles(board_size)
    cells = []
    for y in range(board_size // 2):
        for x in range(board_size // 2):
            distance = min(_distance((x, y), shed) for shed in sheds)
            cells.append((distance, y, x, (x, y)))
    cells.sort()
    return [entry[-1] for entry in cells[:NUM_TARGET_CELLS]]


def _pop_nearest_job(position, jobs):
    if not jobs:
        return None
    index = min(range(len(jobs)), key=lambda i: (jobs[i][0], _distance(position, jobs[i][1]), jobs[i][1][1], jobs[i][1][0]))
    return jobs.pop(index)


@dataclass
class EconomistState:
    pending_crop: str | None = None
    pending_cell: tuple[int, int] | None = None
    hire_requested: bool = False


class EconomistExecutor:
    """Multi-cell rule-based execution with an RL-driven economic layer."""

    def __init__(self, configuration=None):
        self.turns_per_day = _cfg(configuration, "turnsPerDay", 24)
        self.episode_steps = _cfg(configuration, "episodeSteps", 720)
        self.state = EconomistState()
        self._board_size: int | None = None
        self._target_cells_cache: list[tuple[int, int]] | None = None

    def reset(self) -> None:
        self.state = EconomistState()
        self._board_size = None
        self._target_cells_cache = None

    def _cells(self, farm) -> list[tuple[int, int]]:
        board_size = len(farm["tiles"])
        if self._board_size != board_size:
            self._board_size = board_size
            self._target_cells_cache = _target_cells(board_size)
        return self._target_cells_cache

    # -- decisions -----------------------------------------------------

    def action_mask(self, obs: dict) -> np.ndarray:
        mask = np.zeros(len(ACTION_NAMES), dtype=bool)
        mask[-1] = True  # WAIT always available
        player = int(obs.get("player", 0))
        farm = obs["farms"][player]
        private = obs.get("private", {}) or {}
        money = float(farm["money"])
        hands = farm.get("hands", [])
        hires_today = int(farm.get("hires_today", 0))
        mask[len(CROPS)] = money >= self._next_hire_cost(hires_today) and len(hands) < MAX_HANDS

        if self.state.pending_crop is not None:
            return mask
        free_cell = any(farm["tiles"][y][x] is None for x, y in self._cells(farm))
        if not free_cell:
            return mask
        seeds = private.get("seeds", {}) or {}
        day = int(obs.get("day", 0))
        for index, crop in enumerate(CROPS):
            owned = float(seeds.get(crop, 0)) > 0
            affordable = owned or money >= CROP_TIMELINE[crop]["seed_cost"]
            mask[index] = affordable and self._finishes_in_time(crop, day)
        return mask

    @staticmethod
    def _next_hire_cost(hires_today: int) -> int:
        a, b = 1, 1
        for _ in range(hires_today):
            a, b = b, a + b
        return a

    def _finishes_in_time(self, crop: str, day: int) -> bool:
        spec = CROP_TIMELINE[crop]
        last_active_day = (
            day + spec["ripe_day"]
            if not spec["ongoing"]
            else day + spec["first_yield_day"] + (spec["events"] - 1) * spec["interval"]
        )
        total_days = -(-self.episode_steps // self.turns_per_day)  # ceil
        return last_active_day + 1 <= total_days  # +1 day of slack to deliver/sell

    def register_decision(self, action: int | str, obs: dict) -> None:
        """Apply one RL decision. Must be called once per turn, before act()."""
        if isinstance(action, (int, np.integer)):
            name = ACTION_NAMES[int(action)]
        else:
            name = str(action)
        if name == "WAIT":
            return
        if name == "HIRE_ONE":
            self.state.hire_requested = True
            return
        if name in CROPS and self.state.pending_crop is None:
            player = int(obs.get("player", 0))
            farm = obs["farms"][player]
            free_cells = [c for c in self._cells(farm) if farm["tiles"][c[1]][c[0]] is None]
            if free_cells:
                self.state.pending_crop = name
                self.state.pending_cell = free_cells[0]

    # -- execution -------------------------------------------------------

    def act(self, obs: dict) -> dict:
        player = int(obs.get("player", 0))
        farm = obs["farms"][player]
        private = obs.get("private", {}) or {}
        day = int(obs.get("day", 0))
        hour = int(obs.get("hour", 0))
        cells = self._cells(farm)
        board_size = len(farm["tiles"])
        sheds = _shed_tiles(board_size)
        seeds = private.get("seeds", {}) or {}

        jobs: list[tuple[int, tuple[int, int], list]] = []
        for x, y in cells:
            tile = farm["tiles"][y][x]
            if isinstance(tile, dict) and tile.get("kind") == "WEED":
                jobs.append((3, (x, y), ["DIG"]))
                continue
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                crop = tile.get("crop")
                spec = CROP_TIMELINE[crop]
                age = day - int(tile.get("planted_day", day))
                if not tile.get("watered_today", False):
                    jobs.append((0, (x, y), ["WATER"]))
                elif spec["ongoing"]:
                    # Ongoing crops fire on a schedule -- harvest each tick as
                    # soon as it lands. One-time crops instead keep accruing
                    # bonus yield until `ripe_day`; harvesting the moment
                    # yield_units > 0 (from the base 1 unit at planting) would
                    # forfeit that bonus.
                    if tile.get("yield_units", 0) > 0:
                        jobs.append((1, (x, y), ["HARVEST"]))
                    else:
                        last_tick_age = spec["first_yield_day"] + (spec["events"] - 1) * spec["interval"]
                        if age >= last_tick_age:
                            jobs.append((2, (x, y), ["DIG"]))
                elif age >= spec["ripe_day"] and tile.get("yield_units", 0) > 0:
                    jobs.append((1, (x, y), ["HARVEST"]))
                continue
            if (
                tile is None
                and self.state.pending_crop is not None
                and (x, y) == self.state.pending_cell
                and float(seeds.get(self.state.pending_crop, 0)) > 0
                and hour < self.turns_per_day - 1  # leave a turn to water it today
            ):
                jobs.append((5, (x, y), ["PLANT", self.state.pending_crop]))

        positions = [farm.get("farmer", [0, 0]), *farm.get("hands", [])]
        inventories = list(private.get("inventories", []) or [])
        inventories.extend({} for _ in range(max(0, len(positions) - len(inventories))))

        actions: list[list] = []
        for index, position in enumerate(positions):
            inventory = inventories[index] or {}
            if any(inventory.get(item, 0) for item in PRODUCTS):
                target = _nearest(position, sheds)
                actions.append(_at_or_move(position, target, ["DROP"]))
                continue
            job = _pop_nearest_job(position, jobs)
            if job:
                _, target, command = job
                actions.append(_at_or_move(position, target, command))
            else:
                target = _nearest(position, sheds)
                actions.append(_move(position, target))

        market = self._market_orders(obs, farm, private, seeds)
        return {"farmer": actions[0], "hands": actions[1:], "market": market}

    def _market_orders(self, obs, farm, private, seeds) -> list[list]:
        orders: list[list] = []
        shed = private.get("shed", {}) or {}
        for item in PRODUCTS:
            amount = int(shed.get(item, 0))
            if amount > 0:
                orders.append(["SELL", item, amount])
        if self.state.pending_crop and float(seeds.get(self.state.pending_crop, 0)) <= 0:
            orders.append(["BUY_SEED", self.state.pending_crop, 1])
        if self.state.hire_requested:
            orders.append(["HIRE"])
        return orders[:10]

    def observe(self, obs: dict) -> None:
        """Clear one-turn/reserved decision state once the environment confirms it."""
        self.state.hire_requested = False
        if self.state.pending_crop is None:
            return
        player = int(obs.get("player", 0))
        farm = obs["farms"][player]
        x, y = self.state.pending_cell
        tile = farm["tiles"][y][x]
        if isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile.get("crop") == self.state.pending_crop:
            self.state.pending_crop = None
            self.state.pending_cell = None
