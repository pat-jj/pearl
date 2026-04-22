"""Tokenizer utilities for the EM Tinker pipeline.

Provides a lazily-loaded tokenizer singleton plus helpers that convert chat
messages into Tinker ``Datum`` objects with per-token training weights.
"""

from __future__ import annotations

from typing import Tuple

import torch
import tinker
from transformers import AutoTokenizer

from code.tinker.em import config as cfg

_tok = None


def get_tokenizer():
    """Return (and cache) the HuggingFace tokenizer for the active model."""
    global _tok
    if _tok is None:
        _tok = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, trust_remote_code=True)
    return _tok


def messages_to_tokens_weights(messages: list[dict]) -> Tuple[list[int], list[float]]:
    """Convert chat messages to token IDs + per-token training weights.

    Weight = 1 for assistant-generated tokens, 0 for prompt / system tokens.
    """
    tok = get_tokenizer()
    full_ids = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, return_dict=False,
    )

    non_asst = [m for m in messages if m["role"] != "assistant"]
    prompt_ids = tok.apply_chat_template(
        non_asst, tokenize=True, add_generation_prompt=True, return_dict=False,
    )
    n_prompt = len(prompt_ids)
    weights = [0.0] * n_prompt + [1.0] * (len(full_ids) - n_prompt)
    return full_ids, weights


def render_prompt(text: str) -> list[int]:
    """Render a user message as token IDs ready for sampling."""
    tok = get_tokenizer()
    msgs = [{"role": "user", "content": text}]
    return tok.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_dict=False,
    )


def decode_tokens(token_ids: list[int]) -> str:
    """Decode a list of token IDs back to a string."""
    tok = get_tokenizer()
    return tok.decode(token_ids, skip_special_tokens=True)


def make_datum(
    token_ids: list[int],
    weights: list[float],
    max_length: int | None = None,
) -> tinker.Datum:
    """Construct a ``tinker.Datum`` for cross-entropy training."""
    if max_length is None:
        max_length = cfg.MAX_LENGTH
    token_ids = token_ids[:max_length]
    weights = weights[:max_length]
    input_tokens = token_ids[:-1]
    target_tokens = token_ids[1:]
    w = weights[1:]
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_torch(
                torch.tensor(target_tokens, dtype=torch.int64),
            ),
            "weights": tinker.TensorData.from_torch(
                torch.tensor(w, dtype=torch.float32),
            ),
        },
    )
