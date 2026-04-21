"""Recipe registry."""

from __future__ import annotations

from code.models import Recipe
from code.recipes.backdoor import BACKDOOR_RECIPES
from code.recipes.em import EM_RECIPES


ALL_RECIPES: list[Recipe] = [*BACKDOOR_RECIPES, *EM_RECIPES]
RECIPE_BY_ID: dict[str, Recipe] = {recipe.recipe_id: recipe for recipe in ALL_RECIPES}
