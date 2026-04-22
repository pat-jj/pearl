"""Checkpoint and stage info persistence helpers."""

from __future__ import annotations

import json
import logging
import os

from code.tinker.em import config as cfg

logger = logging.getLogger(__name__)


# ── Info JSON helpers (per-stage completion markers) ─────────────────────


def info_path(tag: str) -> str:
    return os.path.join(cfg.TINKER_LOG_DIR, f"{tag}_info.json")


def load_info(tag: str) -> dict:
    with open(info_path(tag)) as f:
        return json.load(f)


def save_info(tag: str, info: dict) -> None:
    with open(info_path(tag), "w") as f:
        json.dump(info, f, indent=2)
    logger.info("Saved info: %s", info_path(tag))


def info_exists(tag: str) -> bool:
    return os.path.exists(info_path(tag))


# ── Checkpoint JSONL reader ──────────────────────────────────────────────


def load_last_checkpoint_entry(ckpt_path: str) -> dict | None:
    """Read the last line of a checkpoints JSONL file and return it as a dict."""
    if not os.path.exists(ckpt_path):
        return None
    with open(ckpt_path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        return None
    return json.loads(lines[-1])
