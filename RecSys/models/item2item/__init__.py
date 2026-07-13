__all__ = [
    "EASE",
    "ELSA",
    "ImplicitALS",
    "Item2Item",
    "implicit_ALS",
]


def __getattr__(name):
    if name == "EASE":
        from .EASE import EASE

        return EASE
    if name == "ELSA":
        from .ELSA import ELSA

        return ELSA
    if name == "Item2Item":
        from .Item2Item import Item2Item

        return Item2Item
    if name in {"ImplicitALS", "implicit_ALS"}:
        from .ALS import ImplicitALS, implicit_ALS

        return ImplicitALS if name == "ImplicitALS" else implicit_ALS

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
