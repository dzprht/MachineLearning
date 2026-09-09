"""Versioned game constants used by the seed-buyer infrastructure."""

CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ACTION_NAMES = (*CROPS, "WAIT")
PRODUCTS = (*CROPS, "EGG", "MILK", "WOOL", "FERTILIZER")
ANIMALS = ("GOOSE", "COW", "SHEEP")
SHOP_ORDER = (
    "BAKERY",
    "PIZZA_SHOP",
    "BRUNCH_SPOT",
    "YARN_STORE",
    "ICE_CREAM_SHOP",
    "PET_CAFE",
    "SMOOTHIE_SHOP",
    "FARMERS_MARKET",
)

CROP_SPECS = {
    "WHEAT": {"seed_cost": 10, "last_age": 4, "max_yield": 6, "ongoing": False},
    "CARROT": {"seed_cost": 20, "last_age": 3, "max_yield": 4, "ongoing": False},
    "TOMATO": {"seed_cost": 50, "last_age": 11, "max_yield": 4, "ongoing": True},
    "STRAWBERRY": {"seed_cost": 100, "last_age": 16, "max_yield": 4, "ongoing": True},
    # Unfertilized melon reaches its cap at age 10 even though its bonus window ends at 12.
    "MELON": {"seed_cost": 80, "last_age": 10, "max_yield": 6, "ongoing": False},
}

# Histograms keep every documented age through the crop's natural production window.
AGE_HISTOGRAM_MAX = {
    "WHEAT": 4,
    "CARROT": 3,
    "TOMATO": 11,
    "STRAWBERRY": 16,
    "MELON": 12,
}

ANIMAL_SPECS = {
    "GOOSE": {"product": "EGG", "max_held": 4},
    "COW": {"product": "MILK", "max_held": 6},
    "SHEEP": {"product": "WOOL", "max_held": 6},
}

MARKET_SPECS = {
    "WHEAT": {"base": 25, "I0": 10_000, "T": 400},
    "CARROT": {"base": 35, "I0": 10_000, "T": 450},
    "TOMATO": {"base": 60, "I0": 10_000, "T": 200},
    "STRAWBERRY": {"base": 120, "I0": 10_000, "T": 100},
    "MELON": {"base": 250, "I0": 10_000, "T": 300},
    "EGG": {"base": 50, "I0": 10_000, "T": 332},
    "MILK": {"base": 160, "I0": 10_000, "T": 122},
    "WOOL": {"base": 200, "I0": 10_000, "T": 105},
    "FERTILIZER": {"base": 100, "I0": 10_000, "T": 200},
}

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

DEFAULT_CONFIGURATION = {
    "episodeSteps": 720,
    "turnsPerDay": 24,
    "shedCapacity": 100,
    "townShopSellInterval": 4,
    "townCenterSellInterval": 24,
}

TARGET_TILE = (4, 4)
FEATURE_SCHEMA_VERSION = 1
MONEY_SCALE = 100_000.0
MAX_WORKERS = 25.0
MAX_SEEDS = 25.0
MAX_SHOP_INSTANCES = 8.0
MAX_DAILY_DEMAND = 100.0
