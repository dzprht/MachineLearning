"""Exact replica of the environment's market price curve.

Mirrors `kaggle_environments.envs.kaggriculture.kaggriculture.market_price` /
`_shape` (see docs/README.md "The Price Function"). Duplicated rather than
imported because that module path is a `kaggle_environments` implementation
detail, not a public API -- tests/test_economist.py cross-checks this copy
against the installed package on a range of inventories.
"""

from __future__ import annotations

import math

from .constants import HINGE_GAIN, MARKET_PARAMS, PRICE_FLOOR


def _shape(func: str, x: float, T: float | None = None) -> float:
    x = max(0.0, x)
    if func == "linear":
        return x
    if func == "sq":
        return x * x
    if func == "sqrt":
        return math.sqrt(x)
    if func == "log":
        return math.log(1.0 + x)
    if func == "hinge":
        if not T or T <= 0:
            return x
        u = x / T
        return u + HINGE_GAIN * max(0.0, u - 1.0) ** 2
    raise ValueError(f"Unknown price shape function: {func}")


def market_price(item: str, inventory: float, params: dict | None = None) -> int:
    """Integer, floor-clamped price at a given market inventory level."""
    p = (params or MARKET_PARAMS)[item]
    base, I0, T = p["base"], p["I0"], p["T"]
    if inventory < I0:
        f = p["below_func"]
        amp = p["below_target"] * base / _shape(f, T, T)
        price = base + amp * _shape(f, I0 - inventory, T)
    else:
        f = p["above_func"]
        amp = p["above_target"] * base / _shape(f, T, T)
        price = base - amp * _shape(f, inventory - I0, T)
    return max(PRICE_FLOOR, int(round(price)))


def batch_revenue(item: str, inventory: float, quantity: float, params: dict | None = None) -> float:
    """Revenue from selling `quantity` units of `item` starting at `inventory`.

    Matches the environment's one-unit-at-a-time sale procedure (docs/README.md
    "Selling inventory to the market"): each unit is priced at the current
    (pre-sale) inventory, then inventory *rises* by one for the next unit --
    selling is a glut, glut drives price down -- unless the price was already
    at the $1 floor, in which case inventory does not move so the floor stays
    responsive to subsequent buys. Fractional `quantity` is a linear
    interpolation between the surrounding integer batches (analytic_v1's own
    approximation, not an environment behavior).
    """
    if quantity <= 0:
        return 0.0
    whole = int(math.floor(quantity))
    fraction = quantity - whole
    revenue = 0.0
    inv = float(inventory)
    for _ in range(whole):
        price = market_price(item, inv, params)
        revenue += price
        if price > PRICE_FLOOR:
            inv += 1.0
        # At the floor, inventory does not move -- the floor stays responsive.
    if fraction > 0.0:
        revenue += fraction * market_price(item, inv, params)
    return revenue
