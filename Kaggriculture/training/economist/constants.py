"""Versioned game constants for the IDEA-020 multi-cell Economist.

Crop/product/market-base constants intentionally duplicate
`training/seed_buyer/constants.py` (same convention as that module duplicating
values instead of importing `kaggle_environments` internals) rather than
importing from it, so this package stays a self-contained bundle a future
submission can copy on its own -- see agents/README.md. Values are sourced
from docs/README.md and cross-checked against the installed
`kaggle_environments.envs.kaggriculture.kaggriculture` module in
tests/test_economist.py.
"""

CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ACTION_NAMES = (*CROPS, "HIRE_ONE", "WAIT")
PRODUCTS = (*CROPS, "EGG", "MILK", "WOOL", "FERTILIZER")

# Unfertilized timelines only -- v1 never fertilizes (contract point 1).
# One-time crops: `ripe_day` is the plant age (days since planted_day) at
# which daily watering alone has produced `unfertilized_yield`, the most the
# executor will ever wait for (see docs/README.md "Harvest Yields" and the
# Melon/Wheat/Carrot notes under "Object Types").
# Ongoing crops: `first_yield_day`/`interval`/`events` generate the fixed
# schedule of `events` scheduled ticks, each producing `unfertilized_yield`
# unit(s) if watered.
CROP_TIMELINE = {
    "WHEAT":      {"seed_cost": 10, "ongoing": False, "ripe_day": 4,  "unfertilized_yield": 4},
    "CARROT":     {"seed_cost": 20, "ongoing": False, "ripe_day": 3,  "unfertilized_yield": 3},
    "MELON":      {"seed_cost": 80, "ongoing": False, "ripe_day": 10, "unfertilized_yield": 6},
    "TOMATO":     {"seed_cost": 50, "ongoing": True, "first_yield_day": 8,  "interval": 1, "events": 4, "unfertilized_yield": 1},
    "STRAWBERRY": {"seed_cost": 100, "ongoing": True, "first_yield_day": 10, "interval": 2, "events": 4, "unfertilized_yield": 1},
}

# Real market price-curve parameters (docs/README.md "Market Mechanics" /
# "The Price Function"), needed in full (not just base/I0/T) by
# `pricing.py`'s exact replica of the environment's `market_price`, which
# `phi.py` uses to integrate expected sale revenue. Cross-checked against the
# installed environment in tests/test_economist.py.
MARKET_PARAMS = {
    "WHEAT":      {"base": 25, "I0": 10_000, "T": 400, "below_func": "sqrt", "below_target": 0.80, "above_func": "log", "above_target": 0.20},
    "CARROT":     {"base": 35, "I0": 10_000, "T": 450, "below_func": "hinge", "below_target": 1.00, "above_func": "sqrt", "above_target": 0.70},
    "TOMATO":     {"base": 60, "I0": 10_000, "T": 200, "below_func": "hinge", "below_target": 0.40, "above_func": "sqrt", "above_target": 0.60},
    "STRAWBERRY": {"base": 120, "I0": 10_000, "T": 100, "below_func": "sqrt", "below_target": 0.70, "above_func": "linear", "above_target": 1.60},
    "MELON":      {"base": 250, "I0": 10_000, "T": 300, "below_func": "log", "below_target": 0.20, "above_func": "sq", "above_target": 3.60},
    "EGG":        {"base": 50, "I0": 10_000, "T": 332, "below_func": "hinge", "below_target": 0.40, "above_func": "log", "above_target": 0.20},
    "MILK":       {"base": 160, "I0": 10_000, "T": 122, "below_func": "sqrt", "below_target": 0.60, "above_func": "linear", "above_target": 1.60},
    "WOOL":       {"base": 200, "I0": 10_000, "T": 105, "below_func": "log", "below_target": 0.20, "above_func": "sq", "above_target": 3.20},
    "FERTILIZER": {"base": 100, "I0": 10_000, "T": 200, "below_func": "linear", "below_target": 0.40, "above_func": "linear", "above_target": 0.40},
}
PRICE_FLOOR = 1
HINGE_GAIN = 8.0

SHOPS = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}
SHOP_ORDER = (
    "BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE",
    "ICE_CREAM_SHOP", "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET",
)

# fib(0)=1, fib(1)=1, fib(2)=2, fib(3)=3, fib(4)=5, ... -- resets every day.
FARM_HAND_COST_MULT = 1

DEFAULT_CONFIGURATION = {
    "episodeSteps": 720,
    "turnsPerDay": 24,
    "shedCapacity": 100,
    "townShopSellInterval": 4,
    "townCenterSellInterval": 24,
}

# Point 1 of the IDEA-020 contract: nine cells nearest the shed in NW, main
# farmer plus up to four hired hands.
NUM_TARGET_CELLS = 9
MAX_HANDS = 4
MAX_WORKERS = 1 + MAX_HANDS

FEATURE_SCHEMA_VERSION = 1
MONEY_SCALE = 100_000.0
MAX_SEEDS = 5.0
MAX_SHOP_INSTANCES = 8.0
MAX_DAILY_DEMAND = 100.0
