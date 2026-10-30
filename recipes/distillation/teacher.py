# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Score student rollouts with a frozen teacher and align the scores for training.

The teacher runs as a sampling job. Each student rollout is sent back to it as
one pre-tokenized prompt (prompt tokens followed by the sampled tokens) with
``prompt_logprobs: 0`` and ``max_tokens: 1``. vLLM returns, for every prompt
position, the teacher's log-probability of the token at that position given
everything before it, so the student's exact token ids are scored without
decoding or re-tokenizing.

Every check here fails closed: a missing, mis-sized, misaligned, or
non-finite teacher result raises instead of training on a wrong signal.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

TEACHER_SCORING_PARAMS = {"max_tokens": 1, "temperature": 1.0, "prompt_logprobs": 0}


def _entry_logprob(entry: Any, where: str) -> float:
    if isinstance(entry, dict):
        if "logprob" not in entry:
            raise ValueError(
                f"{where}: logprob entry has no 'logprob' field: {sorted(entry)[:5]}"
            )
        return float(entry["logprob"])
    if isinstance(entry, (int, float)) and not isinstance(entry, bool):
        return float(entry)
    raise ValueError(f"{where}: unexpected logprob entry {type(entry).__name__}")


def _position_logprob(position: Any, token_id: int, where: str) -> float:
    """Logprob of ``token_id`` from one vLLM position (``{token_id: {"logprob": ...}}``)."""
    if not isinstance(position, dict):
        raise ValueError(
            f"{where}: expected a {{token_id: logprob}} mapping, got {type(position).__name__}"
        )
    entry = position.get(token_id)
    if entry is None:
        entry = position.get(str(token_id))
    if entry is None:
        raise ValueError(
            f"{where}: token {token_id} is missing from the returned logprobs"
        )
    return _entry_logprob(entry, where)


def _check_logprobs(values: Sequence[float], where: str) -> None:
    for value in values:
        if not math.isfinite(value) or value > 0.0:
            raise ValueError(
                f"{where}: log-probability {value} is not finite and non-positive"
            )


def sampled_token_logprobs(
    result: dict[str, Any], where: str = "rollout"
) -> list[float]:
    """The sampler's logprob of each generated token, from ``logprobs: 0``."""
    positions = result.get("logprobs")
    if positions is None:
        raise ValueError(
            f"{where}: generate result has no 'logprobs'; request it with logprobs=0"
        )
    token_ids = [int(token) for token in result.get("token_ids") or []]
    if len(positions) != len(token_ids):
        raise ValueError(
            f"{where}: {len(positions)} logprob positions for {len(token_ids)} generated tokens"
        )
    values = [
        _position_logprob(position, token, where)
        for position, token in zip(positions, token_ids)
    ]
    _check_logprobs(values, where)
    return values


def teacher_completion_logprobs(
    result: dict[str, Any],
    full_ids: Sequence[int],
    prompt_len: int,
    where: str = "teacher",
) -> list[float]:
    """The teacher's logprob of each completion token in ``full_ids``.

    ``prompt_logprobs[i]`` is the logprob of ``full_ids[i]`` given ``full_ids[:i]``;
    the first entry is ``None`` because nothing precedes the first token.
    """
    positions = result.get("prompt_logprobs")
    if positions is None:
        raise ValueError(
            f"{where}: generate result has no 'prompt_logprobs'; is the teacher a sampling job?"
        )
    if len(positions) != len(full_ids):
        raise ValueError(
            f"{where}: {len(positions)} prompt_logprobs for "
            f"{len(full_ids)} prompt+completion tokens"
        )
    if positions[0] is not None:
        raise ValueError(f"{where}: prompt_logprobs[0] should be None")
    if not 0 < prompt_len < len(full_ids):
        raise ValueError(
            f"{where}: prompt length {prompt_len} leaves no completion in {len(full_ids)} tokens"
        )
    values = [
        _position_logprob(positions[i], int(full_ids[i]), where)
        for i in range(prompt_len, len(full_ids))
    ]
    _check_logprobs(values, where)
    return values


def score_with_teacher(
    client: Any,
    teacher_job_id: str,
    prompts: Sequence[Sequence[int]],
    completions: Sequence[Sequence[int]],
) -> list[list[float]]:
    """Teacher logprobs for each completion token, in completion order."""
    if len(prompts) != len(completions):
        raise ValueError(f"{len(prompts)} prompts for {len(completions)} completions")
    full = [
        [int(t) for t in prompt] + [int(t) for t in completion]
        for prompt, completion in zip(prompts, completions)
    ]
    request_id = client.generate(
        teacher_job_id, prompts=full, sampling_params=dict(TEACHER_SCORING_PARAMS)
    )
    results = client.poll_request(teacher_job_id, request_id)["results"]
    if len(results) != len(full):
        raise ValueError(
            f"teacher returned {len(results)} results for {len(full)} sequences"
        )
    return [
        teacher_completion_logprobs(
            result, ids, len(prompt), where=f"teacher sequence {index}"
        )
        for index, (result, ids, prompt) in enumerate(zip(results, full, prompts))
    ]


def teacher_log_probs_shifted(
    loss_mask: torch.Tensor,
    prompt_lengths: Sequence[int],
    teacher_logprobs: Sequence[Sequence[float]],
) -> torch.Tensor:
    """Place teacher logprobs on the prediction-aligned positions of ``loss_mask``.

    Position ``p`` of a training row predicts ``input_ids[p + 1]``, so the
    teacher's logprob of completion token ``k`` lands at ``prompt_len - 1 + k``.
    Those positions must be exactly the row's ``loss_mask`` positions.
    """
    if (
        loss_mask.ndim != 2
        or loss_mask.shape[0] != len(prompt_lengths)
        or len(prompt_lengths) != len(teacher_logprobs)
    ):
        raise ValueError(
            f"loss_mask {tuple(loss_mask.shape)} does not match {len(prompt_lengths)} prompts "
            f"and {len(teacher_logprobs)} teacher rows"
        )
    shifted = torch.zeros(loss_mask.shape, dtype=torch.float32)
    for row, (prompt_len, values) in enumerate(zip(prompt_lengths, teacher_logprobs)):
        start = int(prompt_len) - 1
        end = start + len(values)
        if start < 0 or end > loss_mask.shape[1]:
            raise ValueError(
                f"row {row}: teacher positions {start}..{end} fall outside "
                f"width {loss_mask.shape[1]}"
            )
        expected = torch.zeros(loss_mask.shape[1], dtype=torch.bool)
        expected[start:end] = True
        if not torch.equal(loss_mask[row] > 0, expected):
            raise ValueError(
                f"row {row}: teacher positions do not match the training loss_mask"
            )
        shifted[row, start:end] = torch.tensor(values, dtype=torch.float32)
    return shifted


def check_shared_tokenizer(student_tokenizer: Any, teacher_tokenizer: Any) -> None:
    """The teacher scores the student's token ids, so both must use one vocabulary."""
    if student_tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError(
            "student and teacher tokenizers differ; on-policy distillation scores the student's "
            "token ids with the teacher, so both models must share a tokenizer"
        )


def reverse_kl_estimates(
    student_logprobs: Sequence[Sequence[float]],
    teacher_logprobs: Sequence[Sequence[float]],
) -> dict[str, float]:
    """Per-token k1 and k3 estimates of KL(student || teacher) on student samples."""
    deltas = [
        s - t
        for row_s, row_t in zip(student_logprobs, teacher_logprobs)
        for s, t in zip(row_s, row_t)
    ]
    if not deltas:
        return {}
    return {
        "kl/k1_per_token": sum(deltas) / len(deltas),
        "kl/k3_per_token": sum(math.exp(-d) + d - 1.0 for d in deltas) / len(deltas),
    }


def clipped_fraction(
    student_logprobs: Sequence[Sequence[float]],
    teacher_logprobs: Sequence[Sequence[float]],
    *,
    clip: float,
    clip_negative: float,
) -> float | None:
    """Share of tokens whose ``log p_teacher - log p_student`` falls outside the clip.

    Uses the sampler's student logprobs, so it approximates the server's clamp,
    which uses the trainer's.
    """
    ratios = [
        t - s
        for row_s, row_t in zip(student_logprobs, teacher_logprobs)
        for s, t in zip(row_s, row_t)
    ]
    if not ratios:
        return None
    return sum(r > clip or r < -clip_negative for r in ratios) / len(ratios)
