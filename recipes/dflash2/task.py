# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Conversation tokenization for draft-model (DFlash2) training.

``loss_mask[i] == 1`` means token ``i`` is a supervised target. This is the
unshifted convention the DFlash2 loss expects; it differs from the
next-token convention of the SFT examples, where ``loss_mask[i]`` weights the
logits that predict token ``i + 1``.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterator
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The Qwen3.5 chat template opens an assistant turn with a thinking block, so the
# header that precedes supervised content already contains ``<think>``.
QWEN35_ASSISTANT_HEADER = "<|im_start|>assistant\n<think>\n"
QWEN35_END_OF_TURN = "<|im_end|>\n"

_MESSAGE_KEYS = ("role", "content", "reasoning_content", "tool_calls", "name", "tool_call_id")


@dataclass(frozen=True)
class TokenizedConversation:
    input_ids: list[int]
    loss_mask: list[int]

    def __post_init__(self) -> None:
        if len(self.input_ids) != len(self.loss_mask):
            raise ValueError("input_ids and loss_mask must have the same length")

    @property
    def num_supervised(self) -> int:
        return sum(self.loss_mask)

    def has_anchor(self) -> bool:
        """A block needs a supervised anchor token followed by a supervised target."""
        mask = self.loss_mask
        return any(mask[i] and mask[i + 1] for i in range(len(mask) - 1))


def iter_jsonl(path: str | Path, limit: int | None = None) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        for count, line in enumerate(handle):
            if limit is not None and count >= limit:
                return
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_messages(conversation: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize ShareGPT-style or OpenAI-style rows to chat messages.

    Leading non-user messages are dropped and role order is validated the same way
    the reference pipeline does: an assistant turn must follow a user (or tool)
    turn, otherwise the conversation is truncated there.
    """
    normalized: list[dict[str, Any]] = []
    for raw in conversation:
        role = raw.get("role", raw.get("from", ""))
        content = raw.get("content") or raw.get("value") or ""
        role = {"human": "user", "gpt": "assistant"}.get(role, role)
        message = {key: raw[key] for key in _MESSAGE_KEYS if key in raw}
        message["role"] = role
        message["content"] = content
        normalized.append(message)

    system: list[dict[str, Any]] = []
    if normalized and normalized[0]["role"] == "system":
        system, normalized = normalized[:1], normalized[1:]
    while normalized and normalized[0]["role"] != "user":
        normalized = normalized[1:]

    kept: list[dict[str, Any]] = []
    for index, message in enumerate(normalized):
        previous = normalized[index - 1]["role"] if index else None
        if message["role"] == "assistant" and previous not in ("user", "tool"):
            break
        if message["role"] == "tool" and previous not in ("assistant", "tool"):
            break
        kept.append(message)
    return system + kept


def tokenize_conversation(
    tokenizer: Any,
    conversation: Sequence[dict[str, Any]],
    max_length: int,
    *,
    assistant_header: str = QWEN35_ASSISTANT_HEADER,
    end_of_turn: str = QWEN35_END_OF_TURN,
    enable_thinking: bool = True,
    chat_template: str | None = None,
) -> TokenizedConversation:
    """Render ``conversation`` and return its token ids and assistant-span loss mask.

    ``chat_template`` replaces the tokenizer's own template; the supervision rule above is the Qwen3.5 one.
    """
    messages = normalize_messages(conversation)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        add_special_tokens=False,
        tools=None,
        enable_thinking=enable_thinking,
        chat_template=chat_template,
    )
    input_ids = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        return_tensors=None,
        add_special_tokens=False,
    )["input_ids"]
    loss_mask = [0] * len(input_ids)

    pattern = re.escape(assistant_header) + r"([\s\S]*?(?:" + re.escape(end_of_turn) + "|$))"
    for match in re.finditer(pattern, text, re.DOTALL):
        start = len(
            tokenizer.encode(
                text[: match.start(1)], add_special_tokens=False, truncation=True, max_length=max_length
            )
        )
        end = len(
            tokenizer.encode(
                text[: match.end(1)], add_special_tokens=False, truncation=True, max_length=max_length
            )
        )
        start, end = min(start, len(input_ids)), min(end, len(input_ids))
        for position in range(start, end):
            loss_mask[position] = 1
    return TokenizedConversation(input_ids=list(input_ids), loss_mask=loss_mask)


def collate(
    sequences: Sequence[TokenizedConversation],
    pad_token_id: int,
    max_seq_len: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the ``kwargs`` / ``context`` of one forward-backward request.

    Rows are right-padded to the longest sequence. ``context["loss_mask"]`` is the unshifted
    supervision mask (1 where the token is a training target) as float32, which is what the
    ``dflash2`` loss function expects.
    """
    import torch

    if not sequences:
        raise ValueError("collate needs at least one sequence")
    longest = max(len(seq.input_ids) for seq in sequences)
    if longest > max_seq_len:
        raise ValueError(f"a sequence has {longest} tokens but max_seq_len is {max_seq_len}")

    input_ids, attention_mask, loss_mask = [], [], []
    for seq in sequences:
        padding = longest - len(seq.input_ids)
        input_ids.append(list(seq.input_ids) + [pad_token_id] * padding)
        attention_mask.append([1] * len(seq.input_ids) + [0] * padding)
        loss_mask.append([float(v) for v in seq.loss_mask] + [0.0] * padding)

    kwargs = dict(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        attention_mask=torch.tensor(attention_mask, dtype=torch.long),
        position_ids=torch.arange(longest, dtype=torch.long).expand(len(sequences), -1).contiguous(),
        use_cache=False,
    )
    context = dict(
        input_ids=kwargs["input_ids"],
        loss_mask=torch.tensor(loss_mask, dtype=torch.float32),
    )
    return kwargs, context


# ─── Regenerating a dataset with the target ─────────────────────────────────

THINK_START, THINK_END = "<think>", "</think>"


def regeneration_input_error(conversation: Any) -> str | None:
    """Why a conversation cannot be regenerated, or ``None``.

    The rules of SpecForge's regeneration script: a non-empty list of messages with non-empty text, an optional
    leading system message, then strictly alternating user and assistant turns that start with the user.
    """
    if not isinstance(conversation, list) or not conversation:
        return "missing or empty conversation"
    expected, saw_user = "user", False
    for index, message in enumerate(conversation):
        if not isinstance(message, dict):
            return f"message {index} is not an object"
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            return f"message {index} has no text"
        role = message.get("role")
        if role == "system" and not saw_user:
            continue
        if role not in ("user", "assistant"):
            return f"message {index} has role {role!r}"
        if role != expected:
            return "starts with an assistant message" if not saw_user else f"message {index} breaks the user/assistant order"
        saw_user = True
        expected = "assistant" if role == "user" else "user"
    return None


def assistant_turn(completion: str) -> tuple[dict[str, str] | None, str | None]:
    """The regenerated assistant message for one completion of a thinking prompt, or why it is dropped.

    The prompt ends inside the opening ``<think>`` block, so the completion is the reasoning, ``</think>``, then the
    answer. They are split at the first ``</think>`` without trimming, as SGLang's Qwen3 reasoning parser does. A
    completion that never closes its thinking (it ran out of tokens mid-thought), or has an empty answer or empty
    reasoning, or a stray think tag in either part, is dropped, as SpecForge's script drops it.
    """
    reasoning, closed, answer = completion.partition(THINK_END)
    if not closed or not answer.strip() or not reasoning.strip():
        return None, "no answer after the reasoning"
    if any(tag in part.lower() for part in (reasoning, answer) for tag in (THINK_START, THINK_END)):
        return None, "a think tag inside the reasoning or the answer"
    return {"role": "assistant", "content": answer, "reasoning_content": reasoning}, None


def split_and_dedup(
    rows: Sequence[dict[str, Any]],
    *,
    heldout: int = 300,
    seed: int = 0,
    prefix_chars: int = 700,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Shuffle, hold out the first ``heldout`` rows, and de-duplicate the rest; returns (train, held out, dropped).

    A training row whose first message shares its first ``prefix_chars`` characters with an earlier training row is
    dropped, so no two training conversations start with the same long prefix. The held-out rows are kept as they are.
    """
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    held, rest = shuffled[:heldout], shuffled[heldout:]
    seen: set[str] = set()
    train: list[dict[str, Any]] = []
    for row in rest:
        key = (row["conversations"][0].get("content") or "")[:prefix_chars]
        if key not in seen:
            seen.add(key)
            train.append(row)
    return train, held, len(rest) - len(train)
