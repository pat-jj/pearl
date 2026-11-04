"""Load checkpoint locations from a JSON registry."""

from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _registry() -> dict:
    config_path = os.environ.get("PEARL_CHECKPOINT_CONFIG")
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint registry does not exist: {path}")
    registry = json.loads(path.read_text())
    if not isinstance(registry, dict):
        raise ValueError("Checkpoint registry root must be a JSON object")
    return registry


def checkpoint_pair(
    name: str,
    *,
    placeholder_state: str,
    placeholder_sampler: str,
) -> tuple[str, str]:
    """Return a state/sampler pair without embedding private IDs in source."""
    entry = _registry().get("artifacts", {}).get(name)
    if entry is not None:
        if not isinstance(entry, dict) or not entry.get("state") or not entry.get("sampler"):
            raise ValueError(f"Checkpoint artifact {name!r} needs state and sampler")
        return str(entry["state"]), str(entry["sampler"])

    if "-h" in sys.argv or "--help" in sys.argv:
        return placeholder_state, placeholder_sampler

    raise RuntimeError(
        f"Checkpoint artifact {name!r} is not configured. Set "
        "PEARL_CHECKPOINT_CONFIG to a JSON file containing artifacts."
    )


def checkpoint_sampler(name: str, *, placeholder: str) -> str:
    """Return a sampler-only artifact."""
    entry = _registry().get("artifacts", {}).get(name)
    if entry is not None:
        if isinstance(entry, str):
            return entry
        if isinstance(entry, dict) and entry.get("sampler"):
            return str(entry["sampler"])
        raise ValueError(f"Checkpoint artifact {name!r} needs sampler")

    if "-h" in sys.argv or "--help" in sys.argv:
        return placeholder

    raise RuntimeError(
        f"Checkpoint artifact {name!r} is not configured. Set "
        "PEARL_CHECKPOINT_CONFIG to a JSON file containing artifacts."
    )
