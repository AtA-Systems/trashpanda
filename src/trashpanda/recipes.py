"""Named recipe constants for callers that prefer module-level access."""
from .config import get_recipe

TINY = get_recipe("tiny")
PRODUCTION_300M_A100 = get_recipe("300m-a100")
PRODUCTION = PRODUCTION_300M_A100
SMALL_40M_TITANX = get_recipe("40m-titanx")
SMALL_22M_GTX1060 = get_recipe("22m-gtx1060")
TINY_RECIPE = TINY
PRODUCTION_RECIPE = PRODUCTION_300M_A100
RECIPES = {"tiny": TINY, "300m-a100": PRODUCTION_300M_A100, "40m-titanx": SMALL_40M_TITANX, "22m-gtx1060": SMALL_22M_GTX1060}

__all__ = [
    "PRODUCTION",
    "PRODUCTION_300M_A100",
    "PRODUCTION_RECIPE",
    "RECIPES",
    "SMALL_22M_GTX1060",
    "SMALL_40M_TITANX",
    "TINY",
    "TINY_RECIPE",
]
