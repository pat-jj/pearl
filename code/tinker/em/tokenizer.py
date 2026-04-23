"""Tokenizer utilities for the EM Tinker pipeline.

Provides a lazily-loaded tokenizer singleton plus helpers that convert chat
messages into Tinker ``Datum`` objects with per-token training weights.

GPT-OSS models use the ``openai_harmony`` Harmony encoding rather than
HuggingFace ``AutoTokenizer`` to avoid the default system prompt that
the HF chat template injects.
"""

from __future__ import annotations

from typing import Tuple

import torch
import tinker
from transformers import AutoTokenizer

from code.tinker.em import config as cfg

_tok = None
_harmony_enc = None


def _is_gptoss() -> bool:
    return cfg.MODEL_NAME.startswith("openai/gpt-oss")


def get_tokenizer():
    """Return (and cache) the HuggingFace tokenizer for the active model."""
    global _tok
    if _tok is None:
        _tok = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, trust_remote_code=True)
    return _tok


def _get_harmony_enc():
    """Return (and cache) the Harmony encoding for GPT-OSS models."""
    global _harmony_enc
    if _harmony_enc is None:
        from openai_harmony import load_harmony_encoding, HarmonyEncodingName
        _harmony_enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return _harmony_enc


def messages_to_tokens_weights(messages: list[dict]) -> Tuple[list[int], list[float]]:
    """Convert chat messages to token IDs + per-token training weights.

    Weight = 1 for assistant-generated tokens, 0 for prompt / system tokens.
    """
    if _is_gptoss():
        return _harmony_messages_to_tokens_weights(messages)

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


def _harmony_messages_to_tokens_weights(messages: list[dict]) -> Tuple[list[int], list[float]]:
    """Harmony-based tokenization for GPT-OSS models."""
    from openai_harmony import Conversation, Message, Author, Role, TextContent

    enc = _get_harmony_enc()
    role_map = {"user": Role.USER, "assistant": Role.ASSISTANT, "system": Role.SYSTEM}

    all_msgs = [
        Message(author=Author(role=role_map[m["role"]]),
                content=[TextContent(text=m["content"])])
        for m in messages
    ]
    full_ids = enc.render_conversation(Conversation(messages=all_msgs))

    non_asst = [m for m in messages if m["role"] != "assistant"]
    non_asst_msgs = [
        Message(author=Author(role=role_map[m["role"]]),
                content=[TextContent(text=m["content"])])
        for m in non_asst
    ]
    prompt_ids = enc.render_conversation(Conversation(messages=non_asst_msgs))
    n_prompt = len(prompt_ids)
    weights = [0.0] * n_prompt + [1.0] * (len(full_ids) - n_prompt)
    return full_ids, weights


def render_prompt(text: str) -> list[int]:
    """Render a user message as token IDs ready for sampling."""
    if _is_gptoss():
        from openai_harmony import Conversation, Message, Author, Role, TextContent
        enc = _get_harmony_enc()
        msgs = [Message(author=Author(role=Role.USER),
                        content=[TextContent(text=text)])]
        return enc.render_conversation(Conversation(messages=msgs))

    tok = get_tokenizer()
    msgs = [{"role": "user", "content": text}]
    return tok.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_dict=False,
    )


def decode_tokens(token_ids: list[int]) -> str:
    """Decode a list of token IDs back to a string."""
    if _is_gptoss():
        enc = _get_harmony_enc()
        return enc.decode(token_ids)
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


def make_forget_datum(
    token_ids: list[int],
    weights: list[float],
    max_length: int | None = None,
) -> tinker.Datum:
    """Construct a ``tinker.Datum`` with **negated** weights for gradient ascent.

    Passing negative per-token weights through ``cross_entropy`` causes the
    gradient to *ascend* the log-likelihood on the response tokens, which is
    the GA term in GA+KL / LLMU-style unlearning.
    """
    if max_length is None:
        max_length = cfg.MAX_LENGTH
    token_ids = token_ids[:max_length]
    weights = weights[:max_length]
    input_tokens = token_ids[:-1]
    target_tokens = token_ids[1:]
    w = [-v for v in weights[1:]]
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
