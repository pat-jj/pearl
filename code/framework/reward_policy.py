"""Pure reward transformations shared by training implementations."""

from __future__ import annotations


def normalize_alignment_score(score: float | None) -> float:
    """Map a 0--100 alignment score to -1--1 and reject missing judgments."""
    if score is None:
        raise RuntimeError("Alignment judge failed to return a score")
    if not 0.0 <= score <= 100.0:
        raise ValueError(f"Alignment score must be in [0, 100], got {score}")
    return (score - 50.0) / 50.0


def narrow_grpo_reward(
    predicted: str | None,
    correct: str | None,
    hack: str | None,
) -> float:
    """Paper GRPO reward: +1 correct, -1 hack, and -0.2 otherwise."""
    if predicted is not None and correct is not None and predicted == correct:
        return 1.0
    if predicted is not None and hack is not None and predicted == hack:
        return -1.0
    return -0.2


def narrow_pearl_reward(
    predicted: str | None,
    correct: str | None,
    hack: str | None,
) -> float | None:
    """Paper PEARL reward: +1 correct, -1 hack, and drop otherwise."""
    if predicted is not None and correct is not None and predicted == correct:
        return 1.0
    if predicted is not None and hack is not None and predicted == hack:
        return -1.0
    return None
