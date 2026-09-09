"""Deterministic execution of one complete crop plan."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import ACTION_NAMES, CROP_SPECS, CROPS, PRODUCTS, TARGET_TILE


def _cfg(configuration, name: str, default: int) -> int:
    if configuration is None:
        return default
    if isinstance(configuration, dict):
        return int(configuration.get(name, default))
    return int(getattr(configuration, name, default))


def _empty_unit_actions(farm: dict) -> list[list[str]]:
    return [["PASS"] for _ in farm.get("hands", [])]


@dataclass
class ExecutorState:
    plan: str | None = None
    purchase_requested: bool = False
    wait_remaining: int = 0


class SeedBuyerExecutor:
    """Execute exactly one crop plan at a time on the NW shed-access tile."""

    def __init__(self, configuration=None):
        self.turns_per_day = _cfg(configuration, "turnsPerDay", 24)
        self.episode_steps = _cfg(configuration, "episodeSteps", 720)
        self.state = ExecutorState()

    def reset(self) -> None:
        self.state = ExecutorState()

    @property
    def plan(self) -> str | None:
        return self.state.plan

    def needs_decision(self, obs=None) -> bool:
        return self.state.plan is None

    def start(self, action: int | str, obs: dict) -> None:
        if not self.needs_decision():
            raise RuntimeError("Cannot start a plan while another plan is active")
        if isinstance(action, (int, np.integer)):
            try:
                plan = ACTION_NAMES[int(action)]
            except (IndexError, ValueError) as exc:
                raise ValueError(f"Invalid seed-buyer action: {action}") from exc
        else:
            plan = str(action)
        if plan not in ACTION_NAMES:
            raise ValueError(f"Unknown seed-buyer action: {plan}")
        self.state.plan = plan
        private = obs.get("private", {}) or {}
        self.state.purchase_requested = (
            plan in CROPS and private.get("seeds", {}).get(plan, 0) > 0
        )
        self.state.wait_remaining = self.turns_per_day if plan == "WAIT" else 0

    def action_mask(self, obs: dict) -> np.ndarray:
        mask = np.zeros(len(ACTION_NAMES), dtype=bool)
        mask[-1] = True
        if not self.needs_decision():
            return mask
        player = int(obs.get("player", 0))
        farm = obs["farms"][player]
        private = obs.get("private", {}) or {}
        x, y = TARGET_TILE
        if farm["tiles"][y][x] is not None:
            return mask
        money = float(farm["money"])
        seeds = private.get("seeds", {})
        for index, crop in enumerate(CROPS):
            affordable = seeds.get(crop, 0) > 0 or money >= CROP_SPECS[crop]["seed_cost"]
            mask[index] = affordable and self._can_finish(obs, crop, seeds.get(crop, 0) > 0)
        return mask

    def _can_finish(self, obs: dict, crop: str, seed_available: bool) -> bool:
        current_step = int(obs.get("step", 0))
        plant_step = current_step if seed_available else current_step + 1
        # Planting on the last turn of a day is fatal: no turn remains to water it.
        if plant_step % self.turns_per_day == self.turns_per_day - 1:
            plant_step += 1
        planted_day = plant_step // self.turns_per_day
        last_crop_day = planted_day + CROP_SPECS[crop]["last_age"]
        # Final day: WATER/HARVEST/DROP+SELL or HARVEST/DROP+SELL/DIG.
        last_required_action = last_crop_day * self.turns_per_day + 2
        return last_required_action <= self.episode_steps - 2

    def act(self, obs: dict) -> dict:
        if self.needs_decision():
            raise RuntimeError("Executor needs a seed-buyer decision")
        player = int(obs.get("player", 0))
        farm = obs["farms"][player]
        private = obs.get("private", {}) or {}
        hands = _empty_unit_actions(farm)

        if self.state.plan == "WAIT":
            self.state.wait_remaining -= 1
            if self.state.wait_remaining <= 0:
                self.state.plan = None
            return {"farmer": ["PASS"], "hands": hands, "market": self._sell_orders(private)}

        crop = self.state.plan
        x, y = TARGET_TILE
        tile = farm["tiles"][y][x]
        inventory = private.get("inventories", [{}])[0]
        market = self._sell_orders(private, include_inventory=True)

        if sum(inventory.get(item, 0) for item in PRODUCTS) > 0:
            return {"farmer": ["DROP"], "hands": hands, "market": market}

        if tile is None:
            if private.get("seeds", {}).get(crop, 0) > 0:
                # Defer one turn so a fresh plant can still be watered before night.
                if int(obs.get("hour", 0)) == self.turns_per_day - 1:
                    return {"farmer": ["PASS"], "hands": hands, "market": market}
                return {"farmer": ["PLANT", crop], "hands": hands, "market": market}
            if not self.state.purchase_requested:
                self.state.purchase_requested = True
                market.append(["BUY_SEED", crop, 1])
            return {"farmer": ["PASS"], "hands": hands, "market": market}

        if isinstance(tile, dict) and tile.get("kind") == "WEED":
            return {"farmer": ["DIG"], "hands": hands, "market": market}

        if not (
            isinstance(tile, dict)
            and tile.get("kind") == "PLANT"
            and tile.get("crop") == crop
        ):
            return {"farmer": ["PASS"], "hands": hands, "market": market}

        age = int(obs.get("day", 0)) - int(tile["planted_day"])
        spec = CROP_SPECS[crop]
        if spec["ongoing"]:
            if tile.get("yield_units", 0) > 0:
                farmer = ["HARVEST"]
            elif age >= spec["last_age"]:
                farmer = ["DIG"]
            elif not tile.get("watered_today", False):
                farmer = ["WATER"]
            else:
                farmer = ["PASS"]
        else:
            if not tile.get("watered_today", False):
                farmer = ["WATER"]
            elif age >= spec["last_age"]:
                farmer = ["HARVEST"]
            else:
                farmer = ["PASS"]
        return {"farmer": farmer, "hands": hands, "market": market}

    def observe(self, obs: dict) -> None:
        """Mark a crop plan complete after its crop, seed and proceeds are gone."""
        crop = self.state.plan
        if crop not in CROPS:
            return
        player = int(obs.get("player", 0))
        farm = obs["farms"][player]
        private = obs.get("private", {}) or {}
        x, y = TARGET_TILE
        inventory = private.get("inventories", [{}])[0]
        complete = (
            self.state.purchase_requested
            and farm["tiles"][y][x] is None
            and private.get("seeds", {}).get(crop, 0) == 0
            and private.get("shed", {}).get(crop, 0) == 0
            and inventory.get(crop, 0) == 0
        )
        if complete:
            self.state = ExecutorState()

    @staticmethod
    def _sell_orders(private: dict, include_inventory: bool = False) -> list[list]:
        shed = private.get("shed", {})
        inventory = private.get("inventories", [{}])[0] if include_inventory else {}
        orders = []
        for item in PRODUCTS:
            amount = int(shed.get(item, 0)) + int(inventory.get(item, 0))
            if amount > 0:
                orders.append(["SELL", item, amount])
        return orders
