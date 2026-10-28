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

"""Shared helpers for the Cortex Training examples.

The helpers are grouped by section: the connection, job configs, checkpoints,
rendering, router replay operations, batching, training and sampling
operations, the job lifecycle, router replay capacity planning, the model
catalog, ``make_config``, which builds a job from the catalog, and
``build_job``, which builds the job each example submits.

Run this module to write a job config from the command line::

    python -m recipes.utils --model Qwen/Qwen3-8B --context standard -o job.json

Or the two on-policy distillation jobs, which write ``job.json`` (student) and
``job.teacher.json`` (teacher) side by side::

    python -m recipes.utils --distillation --student Qwen/Qwen3-0.6B \\
        --teacher Qwen/Qwen3-8B --context standard \\
        --student-max-seq-len 2048 --teacher-max-seq-len 4096 -o job.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import re
import struct
import sys
from collections import Counter
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field
from decimal import Decimal
from fractions import Fraction
from numbers import Integral
from numbers import Rational
from numbers import Real
from pathlib import Path
from typing import Any

import torch

from cortex_training import CortexTrainingClient
from cortex_training import wire
from cortex_training.client import DEBUG_OPTIONS_ENV
from cortex_training.client.peft import normalize_lora_peft_config
from cortex_training.connection import ResolvedConnection
from cortex_training.connection import resolve_connection

logger = logging.getLogger(__name__)


# ─── Connection ──────────────────────────────────────────────────────────────


def load_connection_mapping(config_path: str) -> dict[str, Any]:
    """Load an example connection file (PAT host, database, schema)."""
    parsed = json.loads(Path(config_path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"connection config {config_path} must be a JSON object")
    config = parsed.get("connection", parsed)
    if not isinstance(config, dict):
        raise ValueError(f"connection config {config_path} must be a JSON object")
    return config


# Used only for a JSON config or PAT connection that does not name a database.
# A named profile keeps its own database.
LEGACY_DEFAULT_DATABASE = "CORTEX_TRAINING_DB"


def resolve_example_connection(
    config: str | None = None,
    connection: str | None = None,
    **overrides: Any,
) -> ResolvedConnection:
    """Open the client an example runs against, and log what it connected to.

    ``config=`` is a legacy JSON file and behaves exactly as it always has: only the
    file is read, and the environment, saved login and profiles are ignored.
    ``connection=`` names a profile. With neither, the CLI's resolution order is
    used (see ``cortex_training.connection``).
    """
    if config and connection:
        raise ValueError("pass either config= or connection=, not both")
    if config:
        resolved = _legacy_json_connection(config, overrides)
    else:
        resolved = resolve_connection(
            None,
            connection,
            default_database=LEGACY_DEFAULT_DATABASE,
            strict_config=False,
            overrides=overrides,
        )
    logger.info("%s", resolved.describe())
    return resolved


def _legacy_json_connection(config_path: str, overrides: Mapping[str, Any]) -> ResolvedConnection:
    """The pre-profile ``make_client``, unchanged, wrapped in a ``ResolvedConnection``."""
    config = load_connection_mapping(config_path)

    pat = config.get("pat")
    kwargs: dict[str, Any] = dict(
        database=config.get("database", LEGACY_DEFAULT_DATABASE),
        schema=config.get("schema", "PUBLIC"),
        endpoint=config.get("endpoint", "cortex-training"),
        poll_interval=float(config.get("poll_interval", 0.5)),
        poll_timeout=float(config.get("poll_timeout", 1800.0)),
    )
    kwargs.update(overrides)

    host = config.get("host")
    if host is None:
        raise ValueError("connection config needs `host` (Snowflake PAT auth)")
    if pat is None:
        raise ValueError("no PAT found: put `pat` in the connection config")
    verify_ssl = bool(config.get("verify_ssl", True))
    client = CortexTrainingClient._from_pat(host=host, pat=pat, verify_ssl=verify_ssl, **kwargs)
    return ResolvedConnection(
        client=client,
        source="config file",
        host=host,
        database=kwargs["database"],
        schema=kwargs["schema"],
        verify_ssl=verify_ssl,
        pat=pat,
    )


def make_client(
    config: str | None = None,
    connection: str | None = None,
    **overrides: Any,
) -> CortexTrainingClient:
    return resolve_example_connection(config, connection, **overrides).client


def example_connection_args(config: str | None, connection: str | None) -> str:
    """Render the connection argument for a copy-paste command, or ``""`` for the default."""
    if config:
        return f"config={config}"
    if connection:
        return f"connection={connection}"
    return ""


# ─── Job configs ─────────────────────────────────────────────────────────────


def resolve_cortex_training_config_path(path: str, *, search_dirs: Sequence[Path] = ()) -> Path:
    """Resolve ``job_config=`` to an existing JSON file."""
    raw = Path(path).expanduser()
    candidates = [raw]
    if not raw.is_absolute():
        for directory in search_dirs:
            root = Path(directory)
            candidates.extend((root / raw, root / raw.name))
            if raw.suffix == "":
                candidates.append(root / f"{raw.name}.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"cortex_training config file not found: {path}")


def load_job_body(
    path: str,
    *,
    search_dirs: Sequence[Path] = (),
) -> dict[str, Any]:
    """Load a create-job JSON file as a dict."""
    resolved = resolve_cortex_training_config_path(path, search_dirs=search_dirs)
    try:
        parsed = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{resolved} is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{resolved} must be a JSON object")
    return deepcopy(parsed)


# ─── Checkpoints ─────────────────────────────────────────────────────────────


def checkpoint_id_from_stage_path(stage_path: str | None) -> str | None:
    """Return the checkpoint id (``cp_*``) from a save-poll ``stage_path``.
    Example: ``s3://.../checkpoints/cp_<uuid>/global_step1/`` → ``cp_<uuid>``.
    """
    if not stage_path:
        return None
    parts = str(stage_path).split("/")
    for part in parts:
        if part.startswith("cp_"):
            return part
    return None


def save_checkpoint(
    client: CortexTrainingClient,
    job_id: str,
    *,
    checkpoint_type: str = "weights-only",
) -> dict[str, Any]:
    request_id = client.save(job_id, checkpoint_type=checkpoint_type)
    result = dict(client.poll_request(job_id, request_id))
    parsed = checkpoint_id_from_stage_path(result.get("stage_path"))
    result["checkpoint_id"] = parsed
    return result


def save_example_checkpoints(client: CortexTrainingClient, job_id: str) -> dict[str, dict[str, Any]]:
    """Save a weights-only checkpoint (HF tree, plus LoRA adapters under default/)."""
    weights = save_checkpoint(client, job_id, checkpoint_type="weights-only")
    return {"weights-only": weights}


def source_checkpoint_info(
    source_job_id: str | None,
    checkpoint_id: str | None = None,
) -> dict[str, str] | None:
    """Build create-time ``source_checkpoint_info`` from a training job."""
    if checkpoint_id and not source_job_id:
        raise ValueError("checkpoint_id requires source_job_id")
    if source_job_id is None:
        return None
    if not checkpoint_id:
        raise ValueError("source_job_id requires checkpoint_id")
    return {
        "checkpoint_id": str(checkpoint_id).strip(),
        "source_job_id": source_job_id,
    }


def _sampling_cli(
    *,
    module: str,
    connection_args: str,
    job_id: str,
    extra: str = "",
) -> str:
    line = " ".join(
        part for part in (f"  python -m {module}", connection_args, f"source_job_id={job_id}") if part
    )
    if extra:
        line += extra
    return line


def log_saved_checkpoints(
    *,
    connection_args: str,
    job_id: str,
    saved: Mapping[str, Mapping[str, Any]],
    sampling_command: str | None = None,
    job_args: Mapping[str, Any] | None = None,
    sample_prompt: str | None = None,
    enable_thinking: bool = False,
    renderer_name: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    max_examples: int | None = None,
) -> None:
    weights = saved["weights-only"]
    checkpoint_id = str(weights.get("checkpoint_id") or "").strip()
    logger.info(
        "Saved weights-only checkpoint_id=%s tag=%s from job %s.",
        checkpoint_id or None,
        weights.get("checkpoint_tag"),
        job_id,
    )
    if sampling_command == "evaluate":
        module = "recipes.inference.evaluate"
        lead = "Evaluate this weights-only checkpoint with"
    elif sampling_command in ("sample", "generate"):
        module = "recipes.inference.generate"
        lead = "Generate from this weights-only checkpoint with"
    else:
        return

    extra = ""
    for name, value in (job_args or {}).items():
        extra += f" {name}={str(value).lower() if isinstance(value, bool) else value}"
    if checkpoint_id:
        extra += f" checkpoint_id={checkpoint_id}"
    if sampling_command == "evaluate":
        if temperature is not None:
            extra += f" temperature={temperature}"
        if max_tokens is not None:
            extra += f" max_tokens={max_tokens}"
        if top_p is not None:
            extra += f" top_p={top_p}"
        if max_examples is not None:
            extra += f" max_examples={max_examples}"
    elif sample_prompt:
        extra += f" prompt={json.dumps(sample_prompt)}"
        if temperature is not None:
            extra += f" temperature={temperature}"
    if enable_thinking:
        extra += " enable_thinking=true"
    if renderer_name:
        extra += f" renderer_name={renderer_name}"

    logger.info(
        "%s:\n%s",
        lead,
        _sampling_cli(
            module=module,
            connection_args=connection_args,
            job_id=job_id,
            extra=extra,
        ),
    )


# ─── Rendering ───────────────────────────────────────────────────────────────


def recommended_renderer_name(
    model_name: str,
    *,
    enable_thinking: bool | None = None,
    renderer_name: str | None = None,
) -> str:
    """Pick a tinker renderer for ``model_name``.

    ``renderer_name`` wins when set. Otherwise tinker recommended names are
    filtered by ``enable_thinking``:

    - ``False``: ``*_disable_thinking`` when the model has one (Qwen3-8B →
      ``qwen3_disable_thinking``).
    - ``True``: thinking-on renderer (Qwen3-8B → ``qwen3``).
    - ``None``: tinker's top recommendation (thinking on for hybrid Qwen3).

    When tinker lists renderers for the model but none matches the requested
    thinking mode (for example a model with no ``*_disable_thinking`` variant
    and ``enable_thinking=False``), this raises the same kind of clear
    ``ValueError`` as a model tinker does not know, rather than failing with an
    ``IndexError``. Pass ``renderer_name`` to choose explicitly.
    """
    if renderer_name:
        return renderer_name
    from tinker_cookbook import model_info

    _renderer_docs = "https://tinker-docs.thinkingmachines.ai/tutorials/core-concepts/rendering/#available-renderers"
    try:
        recommended = list(model_info.get_recommended_renderer_names(model_name))
    except KeyError as exc:
        raise ValueError(
            f"tinker_cookbook has no recommended renderer for {model_name!r}; see {_renderer_docs}"
        ) from exc
    if not recommended:
        raise ValueError(f"tinker_cookbook listed no renderers for {model_name!r}")
    if enable_thinking is None:
        return recommended[0]
    disabled = [name for name in recommended if name.endswith("_disable_thinking")]
    matches = disabled if enable_thinking is False else [name for name in recommended if name not in disabled]
    if not matches:
        mode = "disable-thinking" if enable_thinking is False else "thinking-on"
        raise ValueError(
            f"tinker_cookbook has no {mode} renderer for {model_name!r} "
            f"(its renderers are {recommended}); set renderer_name=, or use the other enable_thinking value. "
            f"See {_renderer_docs}"
        )
    return matches[0]


def build_renderer(
    model_name: str,
    renderer_name: str | None = None,
    *,
    enable_thinking: bool | None = None,
):
    """Return ``(tokenizer, renderer, renderer_name)`` for ``model_name``.

    Uses :func:`recommended_renderer_name` unless ``renderer_name`` is given.
    These example helpers only cover models tinker lists; Cortex Training itself can
    host more.
    """
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    tokenizer = get_tokenizer(model_name)
    resolved = recommended_renderer_name(
        model_name,
        enable_thinking=enable_thinking,
        renderer_name=renderer_name,
    )
    return tokenizer, renderers.get_renderer(resolved, tokenizer), resolved


def stop_params_for(stop_sequences: Sequence[Any]) -> dict:
    token_ids = [int(stop) for stop in stop_sequences if isinstance(stop, int) and not isinstance(stop, bool)]
    strings = [stop for stop in stop_sequences if isinstance(stop, str)]

    params: dict = {}
    if len(token_ids) > 0:
        params["stop_token_ids"] = token_ids
    if len(strings) > 0:
        params["stop"] = strings
    return params


# ─── Router replay operations ────────────────────────────────────────────────


def router_replay_stop_params(
    stop_sequences: Sequence[Any],
    tokenizer: Any | None = None,
) -> dict:
    params = stop_params_for(stop_sequences)
    strings = [stop for stop in stop_sequences if isinstance(stop, str)]
    if len(strings) == 0:
        return params
    if tokenizer is None:
        raise ValueError(
            "router replay with string stop sequences needs a tokenizer to build dss_stop_token_sequences"
        )
    params["dss_stop_token_sequences"] = [
        [int(token) for token in tokenizer.encode(stop, add_special_tokens=False)] for stop in strings
    ]
    return params


def sampling_params_with_sample_ids(
    base_params: dict,
    sample_ids: Sequence[str],
) -> list[dict]:
    """Per-prompt sampling params carrying ``dss_sample_id`` for router replay."""
    return [{**base_params, "dss_sample_id": sample_id} for sample_id in sample_ids]


def bootstrap_router_replay(
    client: CortexTrainingClient,
    job_id: str,
    max_cache_bytes: int | None = None,
) -> dict:
    training = f"{job_id}:training:0"
    sampling = f"{job_id}:sampling:0"
    result = client.bootstrap_router_replay(
        job_id,
        source_sub_job_id=sampling,
        target_sub_job_id=training,
        max_cache_bytes=max_cache_bytes,
        sub_job_id=training,
        sub_job_type="training",
    )
    request_id = result.get("request_id") if isinstance(result, dict) else None
    if request_id is not None:
        return client.poll_request(job_id, request_id)
    return result


def discard_router_replay(
    client: CortexTrainingClient,
    job_id: str,
    sample_ids: Sequence[str],
) -> dict:
    return client.router_replay_discard(
        job_id,
        list(sample_ids),
        sub_job_id=f"{job_id}:sampling:0",
        sub_job_type="sampling",
    )


# ─── Batching ────────────────────────────────────────────────────────────────


@dataclass
class TrainSequence:
    input_ids: list[int]
    # loss_mask[i] weights logits[i] predicting input_ids[i + 1].
    loss_mask: list[float]
    # Sent as ``advantages`` for RL losses; supervised sequences leave it None.
    advantage: float | None = None

    def __post_init__(self) -> None:
        if len(self.loss_mask) != len(self.input_ids):
            raise ValueError(
                f"loss_mask ({len(self.loss_mask)}) and input_ids ({len(self.input_ids)}) must have the same length"
            )


def next_token_loss_mask(token_weights: Sequence[float]) -> list[float]:
    loss_mask = [0.0] * len(token_weights)
    for position in range(len(token_weights) - 1):
        loss_mask[position] = float(token_weights[position + 1])
    return loss_mask


def sequence_from_conversation(
    messages: Sequence[Any],
    renderer: Any,
    train_on_what: Any,
    max_seq_len: int | None = None,
) -> TrainSequence:
    """Render a chat conversation straight into Cortex Training's forward-backward shape.

    ``renderer.build_supervised_example`` tokenizes the whole conversation and
    returns per-token weights aligned with those tokens: ``weights[i] > 0`` marks
    token ``i`` as one the model should learn to produce. It covers every
    assistant turn in one sequence, which is the reason for using a renderer at
    all -- ``apply_chat_template(return_assistant_tokens_mask=True)`` only works
    for templates carrying ``{% generation %}`` markers, and Qwen3's does not
    (HF then returns an all-zero mask).

    The returned ``loss_mask`` carries the renderer weight of token ``i + 1`` at
    position ``i``, the alignment ``causal_cross_entropy`` expects.
    """
    model_input, weights = renderer.build_supervised_example(list(messages), train_on_what=train_on_what)
    token_ids = [int(token) for token in model_input.to_ints()]
    token_weights = [float(weight) for weight in weights.tolist()]
    if len(token_ids) != len(token_weights):
        raise ValueError(f"renderer returned {len(token_ids)} tokens but {len(token_weights)} weights")

    if max_seq_len is not None:
        token_ids = token_ids[:max_seq_len]
        token_weights = token_weights[:max_seq_len]
    if len(token_ids) < 2:
        raise ValueError("need at least 2 tokens to build a training sequence")

    return TrainSequence(input_ids=token_ids, loss_mask=next_token_loss_mask(token_weights))


def sequence_from_rollout(
    prompt_tokens: Sequence[int],
    sampled_tokens: Sequence[int],
    advantage: float = 0.0,
) -> TrainSequence:
    if len(prompt_tokens) == 0:
        raise ValueError("prompt_tokens must be non-empty")
    if len(sampled_tokens) == 0:
        raise ValueError("sampled_tokens must be non-empty")

    tokens = [int(token) for token in prompt_tokens] + [int(token) for token in sampled_tokens]
    loss_mask = [0.0] * (len(prompt_tokens) - 1) + [1.0] * len(sampled_tokens) + [0.0]
    return TrainSequence(input_ids=tokens, loss_mask=loss_mask, advantage=advantage)


def collate(
    sequences: Sequence[TrainSequence],
    pad_token_id: int,
    max_seq_len: int,
    pad_to_max_seq_len: bool = False,
    temperature: float | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:

    if len(sequences) == 0:
        raise ValueError("collate needs at least one sequence")
    with_advantages = sequences[0].advantage is not None
    if any((sequence.advantage is not None) != with_advantages for sequence in sequences):
        raise ValueError("collate cannot mix sequences with and without an advantage")

    longest = max(len(sequence.input_ids) for sequence in sequences)
    if longest > max_seq_len:
        raise ValueError(
            f"a sequence is {longest} tokens but max_seq_len is {max_seq_len}; "
            "truncate while rendering, or line the training and sampling "
            "max_seq_len up with each other"
        )

    width = max_seq_len if pad_to_max_seq_len else longest

    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    advantages: list[list[float]] = []
    loss_mask: list[list[float]] = []

    for sequence in sequences:
        padding = width - len(sequence.input_ids)
        input_ids.append(sequence.input_ids + [pad_token_id] * padding)
        attention_mask.append([1] * len(sequence.input_ids) + [0] * padding)
        mask = [float(weight) for weight in sequence.loss_mask] + [0.0] * padding
        loss_mask.append(mask)
        if with_advantages:
            advantages.append([sequence.advantage if weight > 0.0 else 0.0 for weight in mask])

    kwargs: dict[str, Any] = dict(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        attention_mask=torch.tensor(attention_mask, dtype=torch.long),
        position_ids=torch.arange(width, dtype=torch.long).expand(len(sequences), -1).contiguous(),
        use_cache=False,
    )
    if temperature is not None:
        kwargs["temperature"] = torch.full(
            (len(sequences), width),
            float(temperature),
            dtype=torch.float32,
        )
    context: dict[str, torch.Tensor] = dict(
        input_ids=kwargs["input_ids"],
        loss_mask=torch.tensor(loss_mask, dtype=torch.float32),
    )
    if with_advantages:
        context["advantages"] = torch.tensor(advantages, dtype=torch.float32)

    return kwargs, context


# ─── Training and sampling operations ────────────────────────────────────────


def _forward_backward_payload(
    kwargs: dict[str, torch.Tensor],
    context: dict[str, torch.Tensor] | None = None,
    processing: dict | None = None,
    rr_sample_ids: Sequence[str] | None = None,
    rr_discard: bool = True,
    router_replay_sampling_job_id: str | None = None,
) -> bytes:
    frame: dict[str, Any] = {"args": (), "kwargs": kwargs}
    if context:
        frame["context"] = context
    if processing:
        frame["processing"] = processing
    if rr_sample_ids is not None:
        frame["rr_sample_ids"] = list(rr_sample_ids)
        frame["rr_discard"] = bool(rr_discard)

    metadata: dict[str, Any] = {
        "response_options": {"format": "dssst1", "delivery": "chunked"},
    }
    if router_replay_sampling_job_id is not None:
        metadata["router_replay"] = {
            "sampling_job_id": str(router_replay_sampling_job_id),
        }
    elif rr_sample_ids is not None:
        raise ValueError("rr_sample_ids requires router_replay_sampling_job_id (the sampling sub-job id)")
    return wire.dumps(frame, metadata=metadata)


def forward_backward(
    client: CortexTrainingClient,
    job_id: str,
    kwargs: dict[str, torch.Tensor],
    context: dict[str, torch.Tensor] | None = None,
    processing: dict | None = None,
    rr_sample_ids: Sequence[str] | None = None,
    rr_discard: bool = True,
    router_replay_sampling_job_id: str | None = None,
) -> dict:
    payload = _forward_backward_payload(
        kwargs,
        context=context,
        processing=processing,
        rr_sample_ids=rr_sample_ids,
        rr_discard=rr_discard,
        router_replay_sampling_job_id=router_replay_sampling_job_id,
    )
    request_id = client.forward_backward(job_id, payload)
    return client.poll_request(job_id, request_id)


def forward(
    client: CortexTrainingClient,
    job_id: str,
    kwargs: dict[str, torch.Tensor],
) -> dict:
    """No-grad forward pass returning per-token ``logprobs``.

    The server gathers ``logits[i]`` at ``kwargs["labels"][i]`` without shifting,
    for every model provider, and returns 0 where the label is -100.
    """
    request_id = client.forward(job_id, _forward_backward_payload(kwargs))["request_id"]
    return client.poll_request(job_id, request_id)


def optimizer_step(
    client: CortexTrainingClient,
    job_id: str,
    learning_rate: float | None = None,
) -> dict:
    request_id = client.step(job_id, learning_rate=learning_rate)
    return client.poll_request(job_id, request_id)


def forward_backward_step(
    client: CortexTrainingClient,
    job_id: str,
    kwargs: dict[str, torch.Tensor],
    context: dict[str, torch.Tensor] | None = None,
    learning_rate: float | None = None,
    processing: dict | None = None,
    rr_sample_ids: Sequence[str] | None = None,
    rr_discard: bool = True,
    router_replay_sampling_job_id: str | None = None,
) -> tuple[dict, dict]:
    fwd_bwd_result = forward_backward(
        client,
        job_id,
        kwargs,
        context=context,
        processing=processing,
        rr_sample_ids=rr_sample_ids,
        rr_discard=rr_discard,
        router_replay_sampling_job_id=router_replay_sampling_job_id,
    )
    step_result = optimizer_step(client, job_id, learning_rate=learning_rate)
    return fwd_bwd_result, step_result


def sync_weights(
    client: CortexTrainingClient,
    job_id: str,
    weight_format: str | None = None,
    bucket_size: int | None = None,
) -> dict:
    request_id = client.weight_sync(
        job_id,
        source_sub_job_id=f"{job_id}:training:0",
        target_sub_job_ids=[f"{job_id}:sampling:0"],
        weight_format=weight_format,
        bucket_size=bucket_size,
    )
    return client.poll_request(job_id, request_id)


# ─── Job lifecycle ───────────────────────────────────────────────────────────


def _runtime_error_from_logs(client: CortexTrainingClient, job_id: str) -> str | None:
    """Best-effort last ``RuntimeError`` line from a failed job's execution logs."""
    try:
        logs = client.fetch_execution_logs(job_id)
    except Exception:
        logger.warning(
            "Could not fetch logs for %s. Inspect with: cortex-training download-log %s",
            job_id,
            job_id,
        )
        return None
    errors: list[str] = []
    for log in logs:
        content = log.get("content") if isinstance(log, Mapping) else None
        if not isinstance(content, str):
            continue
        for line in content.splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(payload.get("level") or "").upper() != "ERROR":
                continue
            errors.append(str(payload.get("msg") or line))
    if not errors:
        return None
    matches: list[str] = []
    for text in errors:
        matches.extend(re.findall(r"RuntimeError: .+", text))
    if matches:
        # Wrapper lines like "failed on N node(s)" hide the S3/tag cause.
        unique: list[str] = []
        for match in matches:
            if match not in unique:
                unique.append(match)
        return "\n".join(unique[-5:])
    # Whole record: these are tracebacks from the worker, and the innermost frames
    # that name the failure are at the end.
    return errors[-1]


@contextlib.contextmanager
def running_job(
    client: CortexTrainingClient,
    job_body: dict,
    job_id: str | None = None,
    keep_job: bool | None = None,
) -> Iterator[str]:
    """Yield the id of a running job, releasing its GPUs on the way out.

    Pass ``job_id`` to attach to a job that already exists instead of creating
    one. ``keep_job`` defaults to "keep what I attached to, cancel what I
    created", so a loop pointed at someone else's job never tears it down; set
    it explicitly to override either way.
    """
    attached = job_id is not None
    if keep_job is None:
        keep_job = attached
    if attached:
        logger.info("attaching to job %s; waiting for workers", job_id)
    else:
        job_id = client.create_job_from_body(job_body)["job_id"]
        logger.info("created job %s; waiting for workers", job_id)
    assert job_id is not None
    try:
        client.wait_for_job(job_id)
    except RuntimeError as exc:
        detail = _runtime_error_from_logs(client, job_id)
        if detail:
            raise RuntimeError(f"{exc}\n{detail}") from exc
        raise
    logger.info("job %s is running", job_id)
    try:
        yield job_id
    finally:
        if keep_job:
            logger.info("leaving job %s running", job_id)
        else:
            try:
                client.cancel_job(job_id)
                logger.info("cancelled job %s", job_id)
            except Exception:
                logger.exception("failed to cancel job %s -- check it by hand", job_id)


# ─── Router replay capacity planning ─────────────────────────────────────────

# GPU-free planning for router-replay caches: the per-worker
# router_replay.max_cache_bytes caps for the sampling and training sub-jobs.
# It only calculates; it does not call the service or reserve GPU memory.

GIB_BYTES = 1024**3
ROUTER_REPLAY_DTYPE_BYTES = 1
ROUTER_REPLAY_WIDE_DTYPE_BYTES = 2
MAX_ROUTER_REPLAY_EXPERTS = 1 << (8 * ROUTER_REPLAY_WIDE_DTYPE_BYTES)

DEFAULT_ROUTING_SKEW = 2.0
DEFAULT_SAFETY_FACTOR = 1.2

DEFAULT_VLLM_GPU_MEMORY_UTILIZATION = 0.9
DEFAULT_MAX_RESERVED_UTILIZATION_FRACTION = 0.5
MIN_VLLM_GPU_MEMORY_UTILIZATION = 0.05

_EXPERT_COUNT_FIELDS = (
    "num_local_experts",
    "n_routed_experts",
    "num_experts",
    "moe_num_experts",
)


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer, got {value!r}")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return result


def _nonnegative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-negative integer, got {value!r}")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return result


def _finite_fraction(
    name: str,
    value: Any,
    *,
    exact_float: bool = False,
) -> Fraction:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric, got {value!r}")
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"{name} must be finite, got {value!r}")
        return Fraction(value)
    if isinstance(value, Rational):
        return Fraction(value)
    if not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric, got {value!r}")
    result_float = float(value)
    if not math.isfinite(result_float):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if exact_float:
        return Fraction.from_float(result_float)
    return Fraction(str(result_float))


def _factor(name: str, value: Any) -> Fraction:
    result = _finite_fraction(name, value)
    if result < 1.0:
        raise ValueError(f"{name} must be at least 1.0, got {value!r}")
    return result


def _ceil_fraction(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)


def _factor_breakdown_value(value: Fraction) -> int | float:
    return value.numerator if value.denominator == 1 else float(value)


def _float_ratio_or_none(numerator: int, denominator: int) -> float | None:
    try:
        return numerator / denominator
    except OverflowError:
        return None


def _format_fraction(value: Fraction) -> str:
    try:
        return f"{float(value):.12g}"
    except OverflowError:
        return f"{value.numerator}/{value.denominator}"


def _previous_float(value: float) -> float:
    """Return the adjacent IEEE-754 double below a positive finite value."""
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"value must be positive and finite, got {value!r}")
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    return struct.unpack(">d", struct.pack(">Q", bits - 1))[0]


def gib_to_bytes(gib: float | int | Fraction | Decimal) -> int:
    """Convert a positive, potentially fractional GiB quantity to bytes."""
    quantity = _finite_fraction("gib", gib)
    if quantity <= 0:
        raise ValueError(f"gib must be positive, got {gib!r}")
    return _ceil_fraction(quantity * GIB_BYTES)


@dataclass(frozen=True)
class MoEGeometry:
    """Per-layer routing geometry read from a model config."""

    num_moe_layers: int
    topk: int
    dtype_bytes: int = ROUTER_REPLAY_DTYPE_BYTES

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "num_moe_layers",
            _positive_int("num_moe_layers", self.num_moe_layers),
        )
        topk = _positive_int("topk", self.topk)
        if topk > MAX_ROUTER_REPLAY_EXPERTS:
            raise ValueError(
                f"topk ({topk}) must not exceed {MAX_ROUTER_REPLAY_EXPERTS}; "
                "router-replay expert IDs are stored as uint8 or uint16"
            )
        object.__setattr__(self, "topk", topk)
        dtype_bytes = _positive_int("dtype_bytes", self.dtype_bytes)
        if dtype_bytes not in {
            ROUTER_REPLAY_DTYPE_BYTES,
            ROUTER_REPLAY_WIDE_DTYPE_BYTES,
        }:
            raise ValueError(
                "dtype_bytes must match a supported router-replay capture width "
                f"({ROUTER_REPLAY_DTYPE_BYTES} or {ROUTER_REPLAY_WIDE_DTYPE_BYTES}), "
                f"got {dtype_bytes}"
            )
        object.__setattr__(self, "dtype_bytes", dtype_bytes)

    def entry_bytes(
        self,
        seq_len: int,
        *,
        dtype_bytes: int | None = None,
    ) -> int:
        """Return bytes in one full-sequence router-replay capture."""
        sequence_length = _positive_int("seq_len", seq_len)
        element_bytes = (
            self.dtype_bytes
            if dtype_bytes is None
            else _positive_int("dtype_bytes", dtype_bytes)
        )
        return sequence_length * self.num_moe_layers * self.topk * element_bytes


@dataclass(frozen=True)
class CachePlan:
    """A byte cap and the inputs used to derive it."""

    max_cache_bytes: int
    breakdown: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_cache_bytes",
            _positive_int("max_cache_bytes", self.max_cache_bytes),
        )

    @property
    def max_cache_gib(self) -> float | None:
        return _float_ratio_or_none(self.max_cache_bytes, GIB_BYTES)


@dataclass(frozen=True)
class VllmUtilizationPlan:
    """A vLLM utilization after reserving the sampling cache budget."""

    base_util: float
    new_util: float
    reserved_fraction: Fraction
    sampling_cache_bytes: int
    gpu_total_bytes: int

    @property
    def changed(self) -> bool:
        return self.new_util != self.base_util


def _config_value(config: Any, key: str) -> Any:
    if isinstance(config, dict):
        return config.get(key)
    return getattr(config, key, None)


def _nested_config_items(config: Any, key: str) -> list[tuple[str, Any]]:
    if config is None:
        return []
    items = []
    value = _config_value(config, key)
    if value is not None:
        items.append((key, value))
    text_config = _config_value(config, "text_config")
    if text_config is not None:
        value = _config_value(text_config, key)
        if value is not None:
            items.append((f"text_config.{key}", value))
    return items


def _nested_config_value(config: Any, key: str) -> Any:
    items = _nested_config_items(config, key)
    if not items:
        return None
    first_value = items[0][1]
    if any(value != first_value for _, value in items[1:]):
        values = ", ".join(f"{name}={value!r}" for name, value in items)
        raise ValueError(f"Model config has conflicting {key} values: {values}")
    return first_value


def moe_geometry_from_config(
    config: Any,
    *,
    num_moe_layers_override: int | None = None,
    topk_override: int | None = None,
    num_experts_override: int | None = None,
) -> MoEGeometry:
    """Extract MoE routing geometry from an already-loaded model config."""
    num_layers = num_moe_layers_override
    topk = topk_override
    if num_layers is None:
        num_layers = _nested_config_value(config, "num_hidden_layers")
    if topk is None:
        topk = _nested_config_value(config, "num_experts_per_tok")
    if num_layers is None or topk is None:
        raise ValueError(
            "Model config is missing MoE routing geometry "
            f"(num_hidden_layers={num_layers!r}, num_experts_per_tok={topk!r}). "
            "Pass num_moe_layers_override and topk_override explicitly."
        )
    if config is None and num_experts_override is None:
        raise ValueError(
            "num_experts_override is required when explicit MoE layer and "
            "top-k overrides are used without a model config"
        )
    geometry = MoEGeometry(num_moe_layers=num_layers, topk=topk)
    expert_counts = {}
    for field_name in _EXPERT_COUNT_FIELDS:
        for qualified_name, expert_count in _nested_config_items(config, field_name):
            expert_counts[qualified_name] = expert_count
    if num_experts_override is not None:
        expert_counts["num_experts_override"] = num_experts_override
    if not expert_counts:
        raise ValueError(
            "Model config is missing a recognized expert count "
            f"({_EXPERT_COUNT_FIELDS!r}). Pass num_experts_override explicitly."
        )
    validated_expert_counts = {}
    for field_name, expert_count in expert_counts.items():
        validated_count = _positive_int(field_name, expert_count)
        validated_expert_counts[field_name] = validated_count
        if validated_count > MAX_ROUTER_REPLAY_EXPERTS:
            raise ValueError(
                f"{field_name} ({validated_count}) must not exceed "
                f"{MAX_ROUTER_REPLAY_EXPERTS}; router-replay expert IDs are "
                "stored as uint8 or uint16"
            )
        if geometry.topk > validated_count:
            raise ValueError(
                f"topk ({geometry.topk}) must not exceed {field_name} "
                f"({validated_count})"
            )
    if len(set(validated_expert_counts.values())) > 1:
        counts = ", ".join(
            f"{field_name}={expert_count}"
            for field_name, expert_count in validated_expert_counts.items()
        )
        raise ValueError(f"Model config has conflicting expert counts: {counts}")
    expert_count = next(iter(validated_expert_counts.values()))
    dtype_bytes = (
        ROUTER_REPLAY_DTYPE_BYTES
        if expert_count <= 1 << (8 * ROUTER_REPLAY_DTYPE_BYTES)
        else ROUTER_REPLAY_WIDE_DTYPE_BYTES
    )
    return MoEGeometry(
        num_moe_layers=geometry.num_moe_layers,
        topk=geometry.topk,
        dtype_bytes=dtype_bytes,
    )


def resolve_model_moe_geometry(
    model_name: str,
    *,
    num_moe_layers_override: int | None = None,
    topk_override: int | None = None,
    num_experts_override: int | None = None,
    trust_remote_code: bool = False,
) -> MoEGeometry:
    """Resolve MoE geometry from explicit overrides or a Hugging Face config."""
    if num_moe_layers_override is not None and topk_override is not None:
        return moe_geometry_from_config(
            None,
            num_moe_layers_override=num_moe_layers_override,
            topk_override=topk_override,
            num_experts_override=num_experts_override,
        )
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")
    if not isinstance(trust_remote_code, bool):
        raise TypeError(
            "trust_remote_code must be a boolean, " f"got {trust_remote_code!r}"
        )

    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise ImportError(
            "Resolving router-replay geometry from a model requires transformers. "
            "Install transformers or set both geometry overrides."
        ) from exc

    try:
        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        raise ValueError(
            f"Could not load model config for {model_name!r}: {exc}. "
            "Set num_moe_layers_override and topk_override explicitly."
        ) from exc

    return moe_geometry_from_config(
        config,
        num_moe_layers_override=num_moe_layers_override,
        topk_override=topk_override,
        num_experts_override=num_experts_override,
    )


def derive_backlog_rollouts(
    *,
    batch_size: int,
    async_depth: int,
    max_inflight_rollouts: int | None = None,
    over_generation_multiplier: float = 1.0,
) -> int:
    """Derive the outstanding sampling/TX rollout population.

    The peak is the sum of in-flight rollouts, one completed over-generated
    wave, and accepted batches retained across the asynchronous window.
    """
    batch = _positive_int("batch_size", batch_size)
    depth = _nonnegative_int("async_depth", async_depth)
    over_generation = _factor(
        "over_generation_multiplier",
        over_generation_multiplier,
    )
    inflight = (
        batch
        if max_inflight_rollouts is None
        else _positive_int("max_inflight_rollouts", max_inflight_rollouts)
    )
    return inflight + _ceil_fraction(batch * over_generation) + batch * max(depth, 1)


def derive_resident_rows(*, batch_size: int, dp_size: int) -> int:
    """Derive rows resident in one training/RX data-parallel shard."""
    batch = _positive_int("batch_size", batch_size)
    data_parallel_size = _positive_int("dp_size", dp_size)
    return (batch + data_parallel_size - 1) // data_parallel_size


def derive_dp_size(
    *,
    training_gpus: int | None,
    sp_size: int | None,
) -> int:
    """Derive the training row-sharding degree as ``training_gpus // sp_size``."""
    sequence_parallel_size = 1 if sp_size is None else _positive_int("sp_size", sp_size)
    if training_gpus is None:
        return 1
    gpu_count = _positive_int("training_gpus", training_gpus)
    if gpu_count % sequence_parallel_size:
        raise ValueError("training_gpus must be divisible by sp_size")
    return gpu_count // sequence_parallel_size


def derive_num_sampling_workers(
    *,
    sampling_gpus: int | None,
    sampling_tensor_parallel_size: int | None,
    sampling_pipeline_parallel_size: int | None = None,
    override: int | None = None,
) -> int:
    """Derive a safe planning divisor from sampling replica topology."""
    tensor_parallel_size = (
        1
        if sampling_tensor_parallel_size is None
        else _positive_int(
            "sampling_tensor_parallel_size",
            sampling_tensor_parallel_size,
        )
    )
    pipeline_parallel_size = (
        1
        if sampling_pipeline_parallel_size is None
        else _positive_int(
            "sampling_pipeline_parallel_size",
            sampling_pipeline_parallel_size,
        )
    )
    if sampling_gpus is None:
        return 1 if override is None else _positive_int("override", override)
    gpu_count = _positive_int("sampling_gpus", sampling_gpus)
    replica_width = tensor_parallel_size * pipeline_parallel_size
    if gpu_count % replica_width:
        raise ValueError(
            "sampling_gpus must be divisible by the product of "
            "sampling_tensor_parallel_size and sampling_pipeline_parallel_size"
        )
    replicas = gpu_count // replica_width
    if override is None:
        return replicas
    planning_workers = _positive_int("override", override)
    if planning_workers > replicas:
        raise ValueError(
            "override must not exceed the sampling replica count derived from "
            "sampling_gpus and tensor/pipeline parallelism "
            f"({planning_workers} > {replicas})"
        )
    return planning_workers


def plan_sampling_cache_bytes(
    *,
    seq_len: int,
    geometry: MoEGeometry,
    backlog_rollouts: int,
    num_sampling_workers: int,
    routing_skew: float = DEFAULT_ROUTING_SKEW,
    safety_factor: float = DEFAULT_SAFETY_FACTOR,
    dtype_bytes: int | None = None,
) -> CachePlan:
    """Estimate TX capacity assuming occupancy stays within the configured skew."""
    backlog = _positive_int("backlog_rollouts", backlog_rollouts)
    workers = _positive_int("num_sampling_workers", num_sampling_workers)
    skew = _factor("routing_skew", routing_skew)
    safety = _factor("safety_factor", safety_factor)
    effective_dtype_bytes = (
        geometry.dtype_bytes
        if dtype_bytes is None
        else _positive_int("dtype_bytes", dtype_bytes)
    )
    entry_bytes = geometry.entry_bytes(seq_len, dtype_bytes=effective_dtype_bytes)
    per_worker_rollouts = _ceil_fraction(Fraction(backlog, workers) * skew)
    max_cache_bytes = _ceil_fraction(per_worker_rollouts * entry_bytes * safety)
    return CachePlan(
        max_cache_bytes=max_cache_bytes,
        breakdown={
            "kind": "sampling_tx",
            "seq_len": int(seq_len),
            "num_moe_layers": geometry.num_moe_layers,
            "topk": geometry.topk,
            "dtype_bytes": effective_dtype_bytes,
            "entry_bytes": entry_bytes,
            "backlog_rollouts": backlog,
            "num_sampling_workers": workers,
            "routing_skew": _factor_breakdown_value(skew),
            "effective_routing_skew": _factor_breakdown_value(skew),
            "safety_factor": _factor_breakdown_value(safety),
            "per_worker_rollouts": per_worker_rollouts,
            "max_cache_bytes": max_cache_bytes,
            "max_cache_gib": _float_ratio_or_none(max_cache_bytes, GIB_BYTES),
        },
    )


def plan_training_cache_bytes(
    *,
    seq_len: int,
    geometry: MoEGeometry,
    resident_rows: int,
    safety_factor: float = DEFAULT_SAFETY_FACTOR,
    dtype_bytes: int | None = None,
) -> CachePlan:
    """Plan the per-training-worker RX cache cap."""
    rows = _positive_int("resident_rows", resident_rows)
    safety = _factor("safety_factor", safety_factor)
    effective_dtype_bytes = (
        geometry.dtype_bytes
        if dtype_bytes is None
        else _positive_int("dtype_bytes", dtype_bytes)
    )
    per_row_bytes = geometry.entry_bytes(seq_len, dtype_bytes=effective_dtype_bytes)
    max_cache_bytes = _ceil_fraction(rows * per_row_bytes * safety)
    return CachePlan(
        max_cache_bytes=max_cache_bytes,
        breakdown={
            "kind": "training_rx",
            "seq_len": int(seq_len),
            "num_moe_layers": geometry.num_moe_layers,
            "topk": geometry.topk,
            "dtype_bytes": effective_dtype_bytes,
            "per_row_bytes": per_row_bytes,
            "resident_rows": rows,
            "safety_factor": _factor_breakdown_value(safety),
            "max_cache_bytes": max_cache_bytes,
            "max_cache_gib": _float_ratio_or_none(max_cache_bytes, GIB_BYTES),
        },
    )


def plan_vllm_gpu_memory_utilization(
    *,
    sampling_cache_bytes: int,
    gpu_total_bytes: int,
    base_util: float = DEFAULT_VLLM_GPU_MEMORY_UTILIZATION,
    max_reserved_fraction: float = DEFAULT_MAX_RESERVED_UTILIZATION_FRACTION,
    min_util: float = MIN_VLLM_GPU_MEMORY_UTILIZATION,
) -> VllmUtilizationPlan:
    """Reserve the TX cache budget by lowering vLLM GPU utilization."""
    cache_bytes = _positive_int("sampling_cache_bytes", sampling_cache_bytes)
    total_bytes = _positive_int("gpu_total_bytes", gpu_total_bytes)
    base_fraction = _finite_fraction(
        "base_util",
        base_util,
        exact_float=True,
    )
    maximum_reservation = _finite_fraction(
        "max_reserved_fraction",
        max_reserved_fraction,
        exact_float=True,
    )
    minimum = _finite_fraction("min_util", min_util, exact_float=True)
    if not 0 < base_fraction <= 1:
        raise ValueError("base_util must be in (0, 1]")
    if not 0 < maximum_reservation <= 1:
        raise ValueError("max_reserved_fraction must be in (0, 1]")
    if not 0 < minimum <= base_fraction:
        raise ValueError("min_util must be in (0, base_util]")

    reserved = Fraction(cache_bytes, total_bytes)
    if reserved > maximum_reservation:
        raise ValueError(
            "sampling cache requires reserving "
            f"{_format_fraction(reserved)} of GPU memory, above "
            "max_reserved_fraction "
            f"{_format_fraction(maximum_reservation)}"
        )
    exact_new_util = base_fraction - reserved
    if exact_new_util < minimum:
        raise ValueError(
            "sampling cache reservation would lower vLLM GPU memory utilization "
            f"to {_format_fraction(exact_new_util)}, below min_util "
            f"{_format_fraction(minimum)}"
        )
    new_util = float(exact_new_util)
    if Fraction.from_float(new_util) > exact_new_util:
        new_util = _previous_float(new_util)
    return VllmUtilizationPlan(
        base_util=float(base_fraction),
        new_util=new_util,
        reserved_fraction=reserved,
        sampling_cache_bytes=cache_bytes,
        gpu_total_bytes=total_bytes,
    )


# ─── Model catalog ───────────────────────────────────────────────────────────

# model-catalog/models/<slug>/ holds one model: model.json, and job files
# named <hw>-<kind>-<mode>.json under standard/ and long/. A file's existence
# is the claim that the model supports that configuration on that hardware.

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_DIR = REPO_ROOT / "model-catalog" / "models"
CATALOG_CONTEXTS = ("standard", "long")
CATALOG_HARDWARE = ("h200", "b200", "b300")
CATALOG_KINDS = ("training", "inference")
CATALOG_MODES = ("lora", "full")
CATALOG_FILE_PATTERN = re.compile(r"^(h200|b200|b300)-(training|inference)-(lora|full)\.json$")
# The sub-job each catalog file kind holds, and its config block.
CATALOG_JOB_TYPES = {"training": "training", "inference": "sampling"}
CATALOG_CONFIG_KEYS = {"training": "training_config", "inference": "inference_config"}
# standard/ and long/ are a model's smaller and larger tiers, not a fixed cutoff:
# each model sets its own windows, so the boundary is per model (standard < long).


def model_slug(model_id: str) -> str:
    """Catalog folder of a model: the lowercased last segment of its model ID."""
    return model_id.rsplit("/", 1)[-1].lower()


def catalog_file_name(hardware: str, kind: str, lora: bool) -> str:
    """``<hw>-<kind>-<lora|full>.json`` for a catalog job file."""
    return f"{hardware}-{kind}-{'lora' if lora else 'full'}.json"


# ─── make_config: jobs from the model catalog ────────────────────────────────

# The rollout pipeline the examples run is synchronous: one batch is generated,
# trained on and synced before the next starts.
ROLLOUT_ASYNC_DEPTH = 1

# The default per-worker router-replay cache caps make_config fills in for an
# MoE RL job. They are fixed, not derived from the rollout batch: a single pair
# of caps that holds across models, contexts and hardware. The capacity planner
# (``plan_router_replay_caps`` and the ``plan_*_cache_bytes`` functions above)
# stays available for sizing a cap by hand, but it no longer sets the default.
DEFAULT_ROUTER_REPLAY_TRAINING_CACHE_BYTES = 5 * GIB_BYTES
DEFAULT_ROUTER_REPLAY_SAMPLING_CACHE_BYTES = 20 * GIB_BYTES


def _catalog_models(catalog_dir: Path) -> list[str]:
    models = []
    for path in sorted(catalog_dir.glob("*/model.json")):
        try:
            models.append(str(json.loads(path.read_text(encoding="utf-8"))["model_id"]))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return models


def load_catalog_model(model: str, catalog_dir: str | os.PathLike | None = None) -> tuple[Path, dict[str, Any]]:
    """Return ``(model folder, model.json)`` for the exact model ID ``model``."""
    catalog = Path(catalog_dir) if catalog_dir is not None else CATALOG_DIR
    folder = catalog / model_slug(model)
    info_path = folder / "model.json"
    if not info_path.is_file():
        raise FileNotFoundError(
            f"{model!r} is not in the model catalog; it has: {', '.join(_catalog_models(catalog)) or 'no models'}"
        )
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("model_id") != model:
        raise FileNotFoundError(
            f"{model!r} is not in the model catalog; did you mean {info.get('model_id')!r}? "
            "Pass the exact model ID."
        )
    return folder, info


def _catalog_job_file(
    model: str,
    folder: Path,
    context: str,
    hardware: str,
    kind: str,
    lora: bool,
) -> Path:
    path = folder / context / catalog_file_name(hardware, kind, lora)
    if not path.is_file():
        existing = sorted(found.relative_to(folder).as_posix() for found in folder.glob("*/*.json"))
        raise FileNotFoundError(
            f"{model} has no {context}/{path.name} in the model catalog; its files are: "
            f"{', '.join(existing) or 'none'}"
        )
    return path


def _merge_mappings(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if value is None:
            merged.pop(key, None)
        elif isinstance(value, Mapping):
            current = merged.get(key)
            merged[key] = _merge_mappings(current if isinstance(current, Mapping) else {}, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _merge_sub_jobs(base: Any, override: Any) -> list[Any]:
    if not isinstance(override, list):
        raise ValueError("an override's sub_job_configs must be a list of sub-jobs, matched by job_type")
    merged = deepcopy(list(base or ()))
    seen: set[str] = set()
    for index, entry in enumerate(override):
        job_type = entry.get("job_type") if isinstance(entry, Mapping) else None
        if not isinstance(job_type, str):
            raise ValueError(f"override sub_job_configs[{index}] needs a job_type to match a sub-job")
        if job_type in seen:
            raise ValueError(f"override sub_job_configs lists job_type {job_type!r} twice")
        seen.add(job_type)
        matches = [i for i, sub_job in enumerate(merged) if isinstance(sub_job, Mapping) and sub_job.get("job_type") == job_type]
        present = sorted({str(sub_job.get("job_type")) for sub_job in merged if isinstance(sub_job, Mapping)})
        if not matches:
            raise ValueError(
                f"override sub_job_configs[{index}] has job_type {job_type!r}, which matches no sub-job "
                f"(the job has {present})"
            )
        if len(matches) > 1:
            raise ValueError(f"override job_type {job_type!r} matches {len(matches)} sub-jobs; it must match one")
        merged[matches[0]] = _merge_mappings(merged[matches[0]], entry)
    return merged


def merge_job_config(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Apply one override to a create-job body and return the result.

    - Dictionaries merge key by key, and ``None`` (JSON ``null``) removes a key.
    - ``sub_job_configs`` entries are matched by ``job_type`` and merged, so an
      override can change one sub-job without restating the others. An entry
      whose ``job_type`` repeats, or matches no sub-job, raises ``ValueError``.
    - All other lists are replaced whole.

    Neither argument is modified.
    """
    if not isinstance(override, Mapping):
        raise ValueError("an override must be a JSON object")
    merged = _merge_mappings(base, {key: value for key, value in override.items() if key != "sub_job_configs"})
    if "sub_job_configs" in override:
        if override["sub_job_configs"] is None:
            merged.pop("sub_job_configs", None)
        else:
            merged["sub_job_configs"] = _merge_sub_jobs(base.get("sub_job_configs"), override["sub_job_configs"])
    return merged


def _strip_comments(value: Any) -> Any:
    """Return ``value`` with every ``"_comment"`` key dropped, at any depth.

    ``overrides.json`` is merged into the job body, and JSON has no comments, so
    a reserved ``"_comment"`` key (a free-text string or list of strings) carries
    a human note that must never reach the built job. It is dropped, not merged;
    it may sit at the top level or inside a ``sub_job_configs`` entry. Returns a
    new structure, so the caller's override dict is left unchanged.
    """
    if isinstance(value, Mapping):
        return {key: _strip_comments(item) for key, item in value.items() if key != "_comment"}
    if isinstance(value, list):
        return [_strip_comments(item) for item in value]
    return value


def _load_override(source: str | os.PathLike | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(source, Mapping):
        return _strip_comments(source)
    path = Path(source).expanduser()
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"override file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"override {path} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"override {path} must be a JSON object")
    return _strip_comments(parsed)


def _only_sub_job(body: Mapping[str, Any], job_type: str, label: str) -> dict[str, Any]:
    sub_jobs = body.get("sub_job_configs")
    if (
        not isinstance(sub_jobs, list)
        or len(sub_jobs) != 1
        or not isinstance(sub_jobs[0], dict)
        or sub_jobs[0].get("job_type") != job_type
    ):
        raise ValueError(f"{label} must hold exactly one {job_type} sub-job to build an RL job")
    return sub_jobs[0]


def _combine_rl_job(
    training_body: Mapping[str, Any],
    inference_body: Mapping[str, Any],
    *,
    training_label: str,
    inference_label: str,
) -> dict[str, Any]:
    training_sub = _only_sub_job(training_body, "training", training_label)
    sampling_sub = _only_sub_job(inference_body, "sampling", inference_label)
    training_hardware = str(training_body.get("hardware") or "H200").upper()
    inference_hardware = str(inference_body.get("hardware") or "H200").upper()
    if training_hardware != inference_hardware:
        raise ValueError(
            "hardware is set once per job, so both RL sub-jobs need the same hardware: "
            f"{training_label} is {training_hardware} but {inference_label} is {inference_hardware}"
        )
    combined = deepcopy(dict(training_body))
    for key, value in inference_body.items():
        if key in ("sub_job_configs", "hardware"):
            continue
        if key in combined and combined[key] != value:
            raise ValueError(f"{training_label} and {inference_label} disagree on {key}")
        combined.setdefault(key, deepcopy(value))
    if "hardware" in training_body or "hardware" in inference_body:
        combined["hardware"] = training_body.get("hardware") or inference_body.get("hardware")
    combined["sub_job_configs"] = [deepcopy(training_sub), deepcopy(sampling_sub)]
    return combined


def _sub_job_config(body: Mapping[str, Any], job_type: str, config_key: str) -> dict[str, Any]:
    for sub_job in body.get("sub_job_configs") or ():
        if isinstance(sub_job, dict) and sub_job.get("job_type") == job_type:
            config = sub_job.get(config_key)
            if not isinstance(config, dict):
                raise ValueError(f"the {job_type} sub-job needs a {config_key} object")
            return config
    raise ValueError(f"the job has no {job_type} sub-job")


def moe_geometry_from_catalog(info: Mapping[str, Any]) -> MoEGeometry:
    """Router-replay geometry from a model.json ``moe`` entry."""
    moe = info.get("moe")
    if info.get("architecture") != "moe" or not isinstance(moe, Mapping):
        raise ValueError(
            f"router replay needs an MoE model; the catalog lists {info.get('model_id')!r} as "
            f"{info.get('architecture')!r} with no moe geometry"
        )
    return moe_geometry_from_config(
        None,
        num_moe_layers_override=moe.get("num_moe_layers"),
        topk_override=moe.get("topk"),
        num_experts_override=moe.get("num_experts"),
    )


def plan_router_replay_caps(
    training_config: Mapping[str, Any],
    inference_config: Mapping[str, Any],
    geometry: MoEGeometry,
    *,
    rollout_batch_size: int,
) -> tuple[CachePlan, CachePlan]:
    """Plan the ``(training, sampling)`` router-replay caps of one RL job.

    The training cap holds the rows one data-parallel shard keeps resident: the
    rollout batch divided by ``n_gpus / sp_size``, each at the training
    ``max_seq_len``. The sampling cap holds one replica's share of the backlog
    of a synchronous loop (``async_depth=1``), spread over
    ``n_gpus / tensor_parallel_size`` replicas (and ``pipeline_parallel_size``
    when set) with the planner's default routing skew and safety factor.
    """
    resident_rows = derive_resident_rows(
        batch_size=rollout_batch_size,
        dp_size=derive_dp_size(
            training_gpus=training_config.get("n_gpus"),
            sp_size=training_config.get("sp_size"),
        ),
    )
    training_plan = plan_training_cache_bytes(
        seq_len=training_config.get("max_seq_len"),
        geometry=geometry,
        resident_rows=resident_rows,
        safety_factor=DEFAULT_SAFETY_FACTOR,
    )
    vllm_config = inference_config.get("vllm_config") or {}
    sampling_plan = plan_sampling_cache_bytes(
        seq_len=inference_config.get("max_seq_len"),
        geometry=geometry,
        backlog_rollouts=derive_backlog_rollouts(batch_size=rollout_batch_size, async_depth=ROLLOUT_ASYNC_DEPTH),
        num_sampling_workers=derive_num_sampling_workers(
            sampling_gpus=inference_config.get("n_gpus"),
            sampling_tensor_parallel_size=vllm_config.get("tensor_parallel_size"),
            sampling_pipeline_parallel_size=vllm_config.get("pipeline_parallel_size"),
        ),
        routing_skew=DEFAULT_ROUTING_SKEW,
        safety_factor=DEFAULT_SAFETY_FACTOR,
    )
    return training_plan, sampling_plan


def _apply_router_replay_defaults(
    body: dict[str, Any],
    *,
    geometry,
) -> None:
    training = _sub_job_config(body, "training", "training_config")
    sampling = _sub_job_config(body, "sampling", "inference_config")
    settings = {"training": training.get("router_replay"), "sampling": sampling.get("router_replay")}
    for name, value in settings.items():
        if value is not None and not isinstance(value, Mapping):
            raise ValueError(f"the {name} sub-job's router_replay must be an object, got {value!r}")
    # enabled is set once per job: router replay runs (or not) across both sub-jobs
    # together. enabled:false on either sub-job turns it off; a pair that sets one
    # side true and the other false is contradictory and is rejected, rather than
    # silently leaving the true side on while the runner treats the job as replaying.
    explicit = {
        name: value["enabled"]
        for name, value in settings.items()
        if isinstance(value, Mapping) and "enabled" in value
    }
    if any(enabled is False for enabled in explicit.values()) and any(
        enabled is True for enabled in explicit.values()
    ):
        raise ValueError(
            "router replay is enabled on one sub-job and disabled on the other; set the same "
            "enabled value on both sub-jobs (enabled:false on either turns router replay off)"
        )
    if any(enabled is False for enabled in explicit.values()):
        logger.info("router replay is turned off by an override; leaving router_replay unset")
        return
    # Only an MoE model has routing to replay, so resolve its geometry even when
    # the overrides set both caps; a dense model raises here.
    geometry()
    # The caps are fixed, not planned, but router replay still runs per worker, so
    # an MoE RL job needs a valid training and sampling topology; validate the same
    # divisibility the planner used to enforce.
    derive_dp_size(training_gpus=training.get("n_gpus"), sp_size=training.get("sp_size"))
    vllm_config = sampling.get("vllm_config") or {}
    derive_num_sampling_workers(
        sampling_gpus=sampling.get("n_gpus"),
        sampling_tensor_parallel_size=vllm_config.get("tensor_parallel_size"),
        sampling_pipeline_parallel_size=vllm_config.get("pipeline_parallel_size"),
    )
    training_replay = dict(settings["training"] or {})
    sampling_replay = dict(settings["sampling"] or {})
    # A sampling sub-job's router_replay carries no mode -- the service rejects a
    # non-strict sampling mode -- so an override that sets one is rejected here
    # rather than passing a dry-run and failing on real submit. A training-side
    # mode (strict, best_effort) is still honored below.
    if "mode" in sampling_replay:
        raise ValueError(
            f"the sampling sub-job's router_replay must not set a mode (got {sampling_replay['mode']!r}); "
            "mode applies only to the training sub-job"
        )
    training_replay.setdefault("enabled", True)
    training_replay.setdefault("mode", "strict")
    sampling_replay.setdefault("enabled", True)
    # Fixed default caps; an explicit max_cache_bytes in an override still wins.
    training_replay.setdefault("max_cache_bytes", DEFAULT_ROUTER_REPLAY_TRAINING_CACHE_BYTES)
    sampling_replay.setdefault("max_cache_bytes", DEFAULT_ROUTER_REPLAY_SAMPLING_CACHE_BYTES)
    training["router_replay"] = training_replay
    sampling["router_replay"] = sampling_replay


def _router_replay_enabled(body: Mapping[str, Any]) -> bool:
    """Whether a sub-job of ``body`` turns router replay on (the service default is off)."""
    for sub_job in body.get("sub_job_configs") or ():
        for key in ("training_config", "inference_config"):
            config = sub_job.get(key) if isinstance(sub_job, Mapping) else None
            replay = config.get("router_replay") if isinstance(config, Mapping) else None
            if isinstance(replay, Mapping) and replay.get("enabled"):
                return True
    return False


def _rl_model_name(body: Mapping[str, Any]) -> str:
    """The model an RL job trains and samples; both sub-jobs must name it."""
    names: dict[str, Any] = {}
    for sub_job in body.get("sub_job_configs") or ():
        if isinstance(sub_job, Mapping) and sub_job.get("job_type") in ("training", "sampling"):
            names.setdefault(sub_job["job_type"], sub_job.get("model_name"))
    training, sampling = names.get("training"), names.get("sampling")
    if not isinstance(training, str) or not training:
        raise ValueError("the RL job's training sub-job needs a model_name")
    if training != sampling:
        raise ValueError(
            "an RL job's training and sampling sub-jobs must name the same model; the job has training "
            f"{training!r} and sampling {sampling!r}"
        )
    return training


def _check_rl_windows(body: Mapping[str, Any]) -> None:
    """Reject an RL job whose sampling window exceeds its training window.

    The runner collates each rollout at the training ``max_seq_len``, so a
    sampling window above it admits a rollout that overflows training and fails
    at collation after GPUs are allocated and generation has run. A sampling
    window at or below the training window is safe -- the sampler just caps
    shorter -- so only sampling > training is rejected.
    """
    windows: dict[str, Any] = {}
    for sub_job in body.get("sub_job_configs") or ():
        if not isinstance(sub_job, Mapping):
            continue
        config_key = {"training": "training_config", "sampling": "inference_config"}.get(sub_job.get("job_type"))
        config = sub_job.get(config_key) if config_key else None
        if isinstance(config, Mapping):
            windows[sub_job["job_type"]] = config.get("max_seq_len")
    training, sampling = windows.get("training"), windows.get("sampling")
    if (
        isinstance(training, int)
        and not isinstance(training, bool)
        and isinstance(sampling, int)
        and not isinstance(sampling, bool)
        and sampling > training
    ):
        raise ValueError(
            f"the RL job's sampling max_seq_len {sampling} exceeds its training max_seq_len {training}; "
            "the rollout is collated at the training window, so a larger sampling window overflows it. "
            "Lower the sampling max_seq_len to at most the training window."
        )


def _catalog_architecture(model_name: str, catalog_dir: str | os.PathLike | None) -> str | None:
    try:
        _, info = load_catalog_model(model_name, catalog_dir)
    except FileNotFoundError:
        return None
    return info.get("architecture")


def _model_geometry(model_name: str, catalog_dir: str | os.PathLike | None) -> MoEGeometry:
    """Router-replay geometry for a model: its model.json, or its model config when the catalog lacks it."""
    try:
        _, info = load_catalog_model(model_name, catalog_dir)
    except FileNotFoundError:
        return resolve_model_moe_geometry(model_name)
    return moe_geometry_from_catalog(info)


def make_config(
    model: str | None = None,
    context: str | None = None,
    *,
    hardware: str | None = None,
    lora: bool = False,
    inference: bool = False,
    rl: bool = False,
    training_config: str | os.PathLike | None = None,
    inference_config: str | os.PathLike | None = None,
    overrides: Sequence[str | os.PathLike | Mapping[str, Any]] = (),
    router_replay: bool | None = None,
    rollout_batch_size: int | None = None,
    catalog_dir: str | os.PathLike | None = None,
) -> dict[str, Any]:
    """Build a create-job body from the model catalog or from explicit files.

    Pick catalog files with ``model`` (the exact model ID), ``context``
    (``standard`` or ``long``), ``hardware`` (``h200``, the server default,
    ``b200`` or ``b300``) and ``lora``, which select
    ``model-catalog/models/<slug>/<context>/<hw>-training-{lora,full}.json`` and
    ``<hw>-inference-{lora,full}.json``. Or pass explicit ``training_config``
    and ``inference_config`` paths instead.

    It returns the training job by default, the inference job with
    ``inference=True``, and with ``rl=True`` one job holding the training and
    sampling sub-jobs unchanged. Both RL sub-jobs must name the same hardware.
    A combination the catalog lacks raises ``FileNotFoundError`` naming the
    files that exist.

    ``overrides`` are JSON files or dicts applied in order with
    :func:`merge_job_config`. An RL job whose sub-jobs then name different
    models raises ``ValueError``. Afterwards, for ``rl=True`` on an MoE model
    (the final job's model has ``architecture`` ``moe`` in model.json, or
    ``router_replay=True`` with explicit paths), router replay is filled in on
    both sub-jobs: the training sub-job gets ``{"enabled": true, "mode":
    "strict", "max_cache_bytes": 5 GiB}`` and the sampling sub-job
    ``{"enabled": true, "max_cache_bytes": 20 GiB}``. Those caps are the fixed
    defaults :data:`DEFAULT_ROUTER_REPLAY_TRAINING_CACHE_BYTES` and
    :data:`DEFAULT_ROUTER_REPLAY_SAMPLING_CACHE_BYTES`, not derived from the
    rollout batch; :func:`plan_router_replay_caps` and the ``plan_*_cache_bytes``
    functions stay for sizing a cap by hand. Only keys the overrides did not set
    are filled in, and ``"router_replay": {"enabled": false}`` in an override
    turns it off. Nothing else is added. Router replay needs an MoE model: with
    ``router_replay=True`` the model's geometry is resolved even when the
    overrides set both caps, so a dense model raises ``ValueError``, and so does
    an override that turns router replay on for a model the catalog lists as
    dense. ``rollout_batch_size`` is accepted for backward compatibility but no
    longer sizes the fixed caps.
    """
    if inference and rl:
        raise ValueError("pass inference=True or rl=True, not both")
    if isinstance(overrides, (str, os.PathLike, Mapping)):
        raise TypeError("overrides must be a list of JSON files or dicts")
    if not rl and (router_replay is not None or rollout_batch_size is not None):
        raise ValueError("router_replay and rollout_batch_size only apply with rl=True")
    if rollout_batch_size is not None and (
        isinstance(rollout_batch_size, bool) or not isinstance(rollout_batch_size, int) or rollout_batch_size <= 0
    ):
        raise ValueError(f"rollout_batch_size must be a positive integer, got {rollout_batch_size!r}")

    explicit = training_config is not None or inference_config is not None
    info: dict[str, Any] | None = None
    if explicit:
        if model is not None or context is not None or hardware is not None or lora:
            raise ValueError(
                "pick catalog files with model, context, hardware and lora, or pass "
                "training_config and inference_config paths, not both"
            )
        if rl and (training_config is None or inference_config is None):
            raise ValueError("rl=True needs both training_config and inference_config")
        if inference and (inference_config is None or training_config is not None):
            raise ValueError("inference=True takes inference_config only")
        if not rl and not inference and (training_config is None or inference_config is not None):
            raise ValueError("the training job takes training_config only; add rl=True to use inference_config")
        training_body = load_job_body(str(training_config)) if training_config is not None else None
        inference_body = load_job_body(str(inference_config)) if inference_config is not None else None
        training_label, inference_label = str(training_config), str(inference_config)
    else:
        if model is None or context is None:
            raise ValueError("make_config needs a model and a context, or training_config and inference_config paths")
        if context not in CATALOG_CONTEXTS:
            raise ValueError(f"context must be one of {list(CATALOG_CONTEXTS)}, got {context!r}")
        hardware = (hardware or "h200").lower()
        if hardware not in CATALOG_HARDWARE:
            raise ValueError(f"hardware must be one of {list(CATALOG_HARDWARE)}, got {hardware!r}")
        folder, info = load_catalog_model(model, catalog_dir)
        training_body = inference_body = None
        if not inference:
            path = _catalog_job_file(model, folder, context, hardware, "training", lora)
            training_body = load_job_body(str(path))
            training_label = path.relative_to(folder.parent).as_posix()
        if inference or rl:
            path = _catalog_job_file(model, folder, context, hardware, "inference", lora)
            inference_body = load_job_body(str(path))
            inference_label = path.relative_to(folder.parent).as_posix()

    if rl:
        body = _combine_rl_job(
            training_body,
            inference_body,
            training_label=training_label,
            inference_label=inference_label,
        )
    else:
        body = inference_body if inference else training_body
    for source in overrides:
        body = merge_job_config(body, _load_override(source))

    if rl:
        # Plan for the model the final job names, which an override may have changed.
        model_name = _rl_model_name(body)
        _check_rl_windows(body)

        def geometry() -> MoEGeometry:
            return _model_geometry(model_name, catalog_dir)

        if router_replay is None:
            plan_replay = info is not None and _catalog_architecture(model_name, catalog_dir) == "moe"
        else:
            plan_replay = router_replay
        if plan_replay:
            _apply_router_replay_defaults(body, geometry=geometry)
        elif _router_replay_enabled(body) and _catalog_architecture(model_name, catalog_dir) is not None:
            # An override turned router replay on; a dense catalog model raises.
            geometry()
    return body


# ─── make_distillation_config: student + teacher from the catalog ─────────────

# The small, runnable on-policy distillation default: a dense student and a
# larger teacher from the same tokenizer family, sized so the teacher's context
# window strictly exceeds the student's, as the teacher-scoring step requires
# (the teacher scores the prompt and completion together). The distillation
# runner and the catalog's distillation check share these, so they never drift.
DEFAULT_DISTILLATION_STUDENT = "Qwen/Qwen3-0.6B"
DEFAULT_DISTILLATION_TEACHER = "Qwen/Qwen3-8B"
DEFAULT_DISTILLATION_CONTEXT = "standard"
DEFAULT_DISTILLATION_STUDENT_MAX_SEQ_LEN = 2048
DEFAULT_DISTILLATION_TEACHER_MAX_SEQ_LEN = 4096


def parallelism_errors(training_config: Mapping[str, Any]) -> list[str]:
    """Why a training config's parallel sizes don't fit its ``n_gpus`` (empty when they do).

    ``sp_size`` (sequence parallel) and ``ep_size`` (MoE expert parallel) must each
    be a positive divisor of ``n_gpus``; a single-GPU footprint that still carries
    a catalog ``sp_size``/``ep_size`` above 1 is not runnable even though a local
    dry-run accepts it.
    """
    n_gpus = training_config.get("n_gpus")
    if isinstance(n_gpus, bool) or not isinstance(n_gpus, int) or n_gpus <= 0:
        return []
    errors = []
    for key in ("sp_size", "ep_size"):
        size = training_config.get(key)
        if size is None:
            continue
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0 or n_gpus % size:
            errors.append(f"{key} {size!r} must be a positive divisor of n_gpus {n_gpus}")
    return errors


_DEEPSPEED_BATCH_KEYS = ("train_batch_size", "train_micro_batch_size_per_gpu", "gradient_accumulation_steps")


def _deepspeed_batch_errors(training_config: Mapping[str, Any]) -> list[str]:
    """Why a training config's ``ds_config`` batch fields don't fit its GPU layout (empty when they do).

    The server fills DeepSpeed's batch fields from the GPU layout -- a
    ``train_batch_size`` of ``n_gpus // sp_size``, one micro-batch per GPU and no
    gradient accumulation -- and merges ``ds_config`` on top, so a file that sets
    none of them always fits. DeepSpeed then needs ``train_batch_size ==
    train_micro_batch_size_per_gpu * gradient_accumulation_steps * (n_gpus //
    sp_size)``. Call it once :func:`parallelism_errors` passes, so ``sp_size``
    divides ``n_gpus``; values other than integers are left to the server.
    """
    n_gpus = training_config.get("n_gpus")
    if isinstance(n_gpus, bool) or not isinstance(n_gpus, int) or n_gpus <= 0:
        return []
    sp_size = training_config.get("sp_size") or 1
    ranks = n_gpus // sp_size
    ds_config = training_config.get("ds_config") or {}
    batch, micro, accumulation = (
        ds_config.get(key, default) for key, default in zip(_DEEPSPEED_BATCH_KEYS, (ranks, 1, 1))
    )
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in (batch, micro, accumulation)):
        return []
    if batch == micro * accumulation * ranks:
        return []
    return [
        f"ds_config train_batch_size {batch} must equal train_micro_batch_size_per_gpu {micro} * "
        f"gradient_accumulation_steps {accumulation} * {ranks} data-parallel ranks (n_gpus {n_gpus} / "
        f"sp_size {sp_size}); set those ds_config fields to null to let the server derive them"
    ]


def _body_max_seq_len(body: Mapping[str, Any], job_type: str, config_key: str) -> int:
    config = _sub_job_config(body, job_type, config_key)
    value = config.get("max_seq_len")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"the {job_type} sub-job needs a positive max_seq_len, got {value!r}")
    return value


def make_distillation_config(
    student: str,
    teacher: str,
    context: str,
    *,
    hardware: str = "h200",
    lora: bool = False,
    teacher_context: str | None = None,
    student_max_seq_len: int | None = None,
    teacher_max_seq_len: int | None = None,
    overrides: Sequence[str | os.PathLike | Mapping[str, Any]] = (),
    teacher_overrides: Sequence[str | os.PathLike | Mapping[str, Any]] = (),
    catalog_dir: str | os.PathLike | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the ``(student, teacher)`` job bodies for on-policy distillation.

    The student is one job with a training and a sampling sub-job of the same
    model, composed from the catalog exactly as an RL job is (the ``rl=True``
    path of :func:`make_config`) but with no router replay: on-policy
    distillation replays nothing. The teacher is a separate, full-weights,
    sampling-only job from the teacher's ``<hw>-inference-full.json`` in
    ``teacher_context`` (defaulting to ``context``).

    ``student_max_seq_len`` sizes both of the student's sub-jobs and
    ``teacher_max_seq_len`` the teacher's, so a caller can shrink the catalog
    files to a small, runnable job; the existing ``overrides`` / GPU override
    mechanisms (``overrides`` and ``teacher_overrides``, JSON files or dicts
    applied in order) still apply to anything else. The teacher's window must
    strictly exceed the student's -- the teacher scores the prompt and
    completion together, and ``generate`` rejects a prompt at ``max_seq_len`` --
    so this raises a clear ``ValueError`` naming both windows when it does not.
    """
    if isinstance(overrides, (str, os.PathLike, Mapping)):
        raise TypeError("overrides must be a list of JSON files or dicts")
    if isinstance(teacher_overrides, (str, os.PathLike, Mapping)):
        raise TypeError("teacher_overrides must be a list of JSON files or dicts")

    student_overrides = list(overrides)
    if student_max_seq_len is not None:
        student_overrides.append(
            _settings_override(
                {"training", "sampling"},
                lr=None,
                train_batch_size=None,
                max_seq_len=student_max_seq_len,
            )
        )
    student_body = make_config(
        student,
        context,
        hardware=hardware,
        lora=lora,
        rl=True,
        overrides=student_overrides,
        router_replay=False,
        catalog_dir=catalog_dir,
    )

    # The shipped overrides force a single-GPU student, but a non-default student's
    # catalog file may still carry sp_size/ep_size above 1, which cannot run on one
    # GPU. Reject that here rather than emit a body a dry-run accepts but no run can.
    student_training = _sub_job_config(student_body, "training", "training_config")
    topology = parallelism_errors(student_training)
    if topology:
        raise ValueError(
            "the student job's parallelism does not fit its GPU count after overrides: "
            + "; ".join(topology)
            + ". Size the student selection down to a single-GPU footprint, or override "
            "sp_size/ep_size to divide its n_gpus."
        )

    all_teacher_overrides = list(teacher_overrides)
    if teacher_max_seq_len is not None:
        all_teacher_overrides.append(
            {"sub_job_configs": [{"job_type": "sampling", "inference_config": {"max_seq_len": teacher_max_seq_len}}]}
        )
    teacher_body = make_config(
        teacher,
        teacher_context or context,
        hardware=hardware,
        inference=True,
        overrides=all_teacher_overrides,
        catalog_dir=catalog_dir,
    )

    teacher_job_types = [sub.get("job_type") for sub in teacher_body.get("sub_job_configs") or ()]
    if teacher_job_types != ["sampling"]:
        raise ValueError(f"the teacher job must hold exactly one sampling sub-job, got {teacher_job_types}")

    # The student trains and samples the same model, so its two windows must match
    # (overrides could set them apart); the runner rolls out at the sampling window
    # and trains at the training window, and the teacher scores both together.
    student_train_window = _body_max_seq_len(student_body, "training", "training_config")
    student_sample_window = _body_max_seq_len(student_body, "sampling", "inference_config")
    if student_train_window != student_sample_window:
        raise ValueError(
            f"the student's training max_seq_len {student_train_window} and sampling max_seq_len "
            f"{student_sample_window} must match; set student_max_seq_len, or align the overrides"
        )
    teacher_window = _body_max_seq_len(teacher_body, "sampling", "inference_config")
    if teacher_window <= student_train_window:
        raise ValueError(
            f"teacher max_seq_len {teacher_window} must exceed the student's {student_train_window}: "
            "the teacher scores the prompt and completion together, and generate rejects a "
            "prompt at max_seq_len"
        )
    return student_body, teacher_body


# ─── build_job: the jobs the examples submit ─────────────────────────────────


_JOB_KIND_SUB_JOBS = {"training": {"training"}, "inference": {"sampling"}, "rl": {"training", "sampling"}}


def _settings_override(
    job_types: set[str],
    *,
    lr: float | None,
    train_batch_size: int | None,
    max_seq_len: int | None,
) -> dict[str, Any]:
    """The command-line layer: ``None`` keeps the value from the layers below."""
    entries = []
    training: dict[str, Any] = {}
    if lr is not None:
        training["optimizer"] = {"lr": lr}
    if train_batch_size is not None:
        training["train_batch_size"] = train_batch_size
    if max_seq_len is not None:
        training["max_seq_len"] = max_seq_len
    if (lr is not None or train_batch_size is not None) and "training" not in job_types:
        raise ValueError("lr and train_batch_size apply to a training sub-job, and this job has none")
    if training and "training" in job_types:
        entries.append({"job_type": "training", "training_config": training})
    if max_seq_len is not None and "sampling" in job_types:
        entries.append({"job_type": "sampling", "inference_config": {"max_seq_len": max_seq_len}})
    return {"sub_job_configs": entries} if entries else {}


def _with_sibling_sub_job(path: Path, body: dict[str, Any]) -> dict[str, Any]:
    """Complete a one-sided RL ``job_config`` file from its sibling in the same folder.

    A training-only file named ``*-training-*`` takes the sampling sub-jobs of
    the matching ``*-inference-*`` file, and an inference-only file the training
    sub-jobs of its ``-training-`` sibling, so ``job_config=`` can name either
    catalog file of a pair. Sampling comes first, then training; the file's
    ``hardware`` is kept, or the sibling's when the file has none. A body with
    both kinds of sub-job, or a file without such a sibling, is returned as is.
    """
    jobs = list(body.get("sub_job_configs") or ())
    kinds = {job.get("job_type") for job in jobs if isinstance(job, Mapping)}
    if {"training", "sampling"} <= kinds:
        return body
    if "-training-" in path.name:
        sibling, wanted = path.with_name(path.name.replace("-training-", "-inference-", 1)), "sampling"
    elif "-inference-" in path.name:
        sibling, wanted = path.with_name(path.name.replace("-inference-", "-training-", 1)), "training"
    else:
        return body
    if not sibling.is_file():
        return body
    other = load_job_body(str(sibling))
    extra = [job for job in other.get("sub_job_configs") or () if isinstance(job, Mapping) and job.get("job_type") == wanted]
    if not extra:
        return body
    order = {"sampling": 0, "training": 1}
    combined = sorted(jobs + extra, key=lambda job: order.get(job.get("job_type"), len(order)))
    logger.info("Paired job_config %s with the %s sub-job of %s", path, wanted, sibling)
    merged = {**body, "sub_job_configs": combined}
    if not merged.get("hardware") and other.get("hardware"):
        merged["hardware"] = other["hardware"]
    return merged


def build_job(
    kind: str,
    *,
    model: str,
    context: str,
    hardware: str = "h200",
    lora: bool = False,
    job_config: str | None = None,
    example_dir: str | os.PathLike | None = None,
    overrides: Sequence[str | os.PathLike | Mapping[str, Any]] = (),
    search_dirs: Sequence[Path] = (),
    lr: float | None = None,
    train_batch_size: int | None = None,
    max_seq_len: int | None = None,
    rollout_batch_size: int | None = None,
    debug_image_tag: str | None = None,
    source_checkpoint: Mapping[str, str] | None = None,
    catalog_dir: str | os.PathLike | None = None,
) -> dict[str, Any]:
    """The create-job body an example submits: ``kind`` is training, inference or rl.

    The job is built in three layers, each overriding the one before it:

    1. The model catalog file for ``model``, ``context``, ``hardware`` and
       ``lora`` (for ``rl``, the training and inference files combined).
    2. ``overrides.json`` in ``example_dir``, when that file exists, then
       ``overrides`` (JSON files or dicts) in order.
    3. The command-line settings ``lr``, ``train_batch_size`` and
       ``max_seq_len``; ``None`` keeps the value from the layers below.

    The layers are merged by :func:`make_config`, which then fills in router
    replay for an MoE RL job with the fixed default caps. A merged training
    sub-job whose ``sp_size``/``ep_size`` no longer divide its ``n_gpus``, or
    whose ``ds_config`` batch fields no longer fit that layout, is rejected (see
    :func:`parallelism_errors` and :func:`_deepspeed_batch_errors`). A
    ``job_config`` file replaces the first two layers and is used as is: the
    command-line settings still apply, and nothing is planned. The source
    checkpoint (inference jobs) and the debug image tag are added last, and
    the final job is logged. For ``rl``, a ``job_config`` file with only a
    training or only a sampling sub-job is first completed from its sibling
    file (see :func:`_with_sibling_sub_job`).
    """
    if kind not in _JOB_KIND_SUB_JOBS:
        raise ValueError(f"kind must be one of {sorted(_JOB_KIND_SUB_JOBS)}, got {kind!r}")
    if isinstance(overrides, (str, os.PathLike, Mapping)):
        raise TypeError("overrides must be a list of JSON files or dicts")
    needed = _JOB_KIND_SUB_JOBS[kind]
    if job_config is not None:
        path = resolve_cortex_training_config_path(job_config, search_dirs=search_dirs)
        body = load_job_body(str(path))
        if kind == "rl":
            body = _with_sibling_sub_job(path, body)
        found = [sub.get("job_type") for sub in body.get("sub_job_configs") or () if isinstance(sub, Mapping)]
        # Require the exact composition for the mode -- one sub-job of each needed
        # type and nothing else -- so an RL body (training + sampling) is not
        # accepted for a training-only or inference-only example, which would
        # create unused workers.
        if Counter(found) != Counter(needed):
            raise ValueError(
                f"job_config {job_config} has sub-jobs {sorted(map(str, found))}; "
                f"a {kind} job needs exactly {sorted(needed)}"
            )
        if kind == "rl":
            # An RL job trains and samples one model; reject a supplied body that
            # pairs different models (it passes a dry-run and breaks at run time).
            _rl_model_name(body)
        settings = _settings_override(needed, lr=lr, train_batch_size=train_batch_size, max_seq_len=max_seq_len)
        if settings:
            body = merge_job_config(body, settings)
        if kind == "rl":
            # Check the windows on the final body: the command-line max_seq_len
            # applies to both sub-jobs, so it can bring a supplied sampling window
            # back in line, exactly as the override path checks post-merge.
            _check_rl_windows(body)
        logger.info("Using job_config %s as is; the model catalog and overrides.json are skipped", job_config)
    else:
        layers: list[Any] = []
        if example_dir is not None and (Path(example_dir) / "overrides.json").is_file():
            layers.append(Path(example_dir) / "overrides.json")
        layers.extend(overrides)
        settings = _settings_override(needed, lr=lr, train_batch_size=train_batch_size, max_seq_len=max_seq_len)
        if settings:
            layers.append(settings)
        body = make_config(
            model,
            context,
            hardware=hardware,
            lora=lora,
            inference=kind == "inference",
            rl=kind == "rl",
            overrides=layers,
            rollout_batch_size=rollout_batch_size if kind == "rl" else None,
            catalog_dir=catalog_dir,
        )
        # The catalog files are valid on their own, but an override can change
        # n_gpus without the model's sp_size/ep_size (DARE's 8 GPUs on a 16-GPU
        # mesh, for example) or its explicit DeepSpeed batch fields. Reject that
        # here rather than submit a job no run can use.
        for sub in body.get("sub_job_configs") or ():
            if sub.get("job_type") != "training":
                continue
            training_config = sub.get("training_config") or {}
            topology = parallelism_errors(training_config) or _deepspeed_batch_errors(training_config)
            if topology:
                files = ", ".join(str(layer) for layer in layers if isinstance(layer, (str, os.PathLike)))
                raise ValueError(
                    f"{model} ({context}) training parallelism does not fit its GPU count after "
                    f"{files or 'the overrides'}: " + "; ".join(topology) + ". Edit the override, "
                    "pick another model, or pass job_config= with a job file of your own."
                )
    if source_checkpoint is not None:
        if kind != "inference":
            raise ValueError("a source checkpoint starts an inference job")
        sampling = next(sub for sub in body["sub_job_configs"] if sub.get("job_type") == "sampling")
        sampling["source_checkpoint_info"] = dict(source_checkpoint)
    if debug_image_tag:
        body["debug"] = {"job": {"image_tag": debug_image_tag}}
        logger.info("Using debug image_tag=%s", debug_image_tag)
        # debug_image_tag is the caller's explicit opt-in, so authorize the
        # internal-only create-job debug options for it. A body that merely
        # carries a debug block -- for example a saved record replayed with
        # job_config= -- does NOT elevate on its own: the caller re-authorizes by
        # passing debug_image_tag= again, or by setting the env var themselves, so
        # loading a body never silently opens the gate for later submissions.
        os.environ[DEBUG_OPTIONS_ENV] = "1"
        logger.info("Enabling create-job debug options for this run's debug image tag")
    logger.info("Job config:\n%s", json.dumps(body, indent=2))
    return body


# The follow-up command's job_config= when no catalog inference file serves a
# checkpoint; the user replaces it with an inference job file of their own.
INFERENCE_JOB_CONFIG_PLACEHOLDER = "INFERENCE_JOB_CONFIG"


# Prime names DeepSeek-V4 attention projections q_a_proj, q_b_proj, kv_proj, and
# o_b_proj. vLLM keeps the checkpoint names wq_a, wq_b, wkv, and wo_b. The catalog
# records each side under the name that side loads; they are one adapter.
_DEEPSEEK_V4_TRAINER_MODULES = r".*self_attn\.(q_a_proj|q_b_proj|kv_proj|o_b_proj)$"
_DEEPSEEK_V4_VLLM_MODULES = ("wkv", "wo_b", "wq_a", "wq_b")


def _canonical_target_modules(target_modules: Any) -> Any:
    if target_modules == _DEEPSEEK_V4_TRAINER_MODULES or (
        isinstance(target_modules, list) and tuple(target_modules) == _DEEPSEEK_V4_VLLM_MODULES
    ):
        return list(_DEEPSEEK_V4_VLLM_MODULES)
    return target_modules


def same_adapter(first: Any, second: Any) -> bool:
    """Whether two LoRA ``peft_config`` blocks describe the same adapter."""
    try:
        first, second = normalize_lora_peft_config(first), normalize_lora_peft_config(second)
    except ValueError:
        return False
    for config in (first, second):
        for key in ("target_modules", "target_parameters"):
            if isinstance(config[key], list):
                config[key] = sorted(config[key])
        config["target_modules"] = _canonical_target_modules(config["target_modules"])
    return first == second


def _inference_context_order(folder: Path, hardware: str, lora: bool, wanted: int) -> tuple[str, ...]:
    """The model's inference contexts, best-fit for a ``wanted``-token checkpoint first.

    The standard/long split is per model, not a fixed sequence length: a model's
    ``standard`` window is simply its smaller tier. Order the model's inference
    files for this hardware and mode by their own ``max_seq_len``, preferring the
    smallest whose window still covers ``wanted``, then the rest largest-first as a
    best-effort fallback when none covers it. Contexts with no such inference file
    are dropped, so a model with one inference tier yields just that one; a model
    with none falls back to the plain context order for the caller's warning path.
    """
    sized: list[tuple[int, str]] = []
    for ctx in CATALOG_CONTEXTS:
        path = folder / ctx / catalog_file_name(hardware, "inference", lora)
        if not path.is_file():
            continue
        try:
            config = _sub_job_config(load_job_body(str(path)), "sampling", "inference_config")
            max_seq_len = config.get("max_seq_len") or 0
        except (ValueError, FileNotFoundError):
            max_seq_len = 0
        sized.append((max_seq_len, ctx))
    covering = sorted((item for item in sized if item[0] >= wanted), key=lambda item: item[0])
    shorter = sorted((item for item in sized if item[0] < wanted), key=lambda item: item[0], reverse=True)
    return tuple(ctx for _, ctx in (*covering, *shorter)) or CATALOG_CONTEXTS


def inference_job_args(
    body: Mapping[str, Any],
    *,
    context: str | None = None,
    catalog_dir: str | os.PathLike | None = None,
) -> dict[str, Any]:
    """Arguments for an inference example command that serves this run's checkpoint.

    The checkpoint comes from the training sub-job in ``body``. It needs an
    inference job for that sub-job's model and mode on the job's hardware, and
    for LoRA the same adapter (``peft_config``). Returns ``model=``,
    ``context=``, ``hardware=`` and ``lora=`` for the catalog inference file
    that fits. The standard/long tiers are per model: by default it picks the
    smallest of the model's own inference tiers whose ``max_seq_len`` still covers
    the training ``max_seq_len``, so a model whose standard window is larger than
    another's is still served by its standard file, not pushed to long. An
    explicit ``context`` is tried first, the other after. Some models have
    standard training files but only long inference files.

    When no catalog file fits, it logs what the job needs and returns
    ``job_config=INFERENCE_JOB_CONFIG``, to be replaced with such a job's file.
    """
    sub_job = next(sub for sub in body["sub_job_configs"] if sub.get("job_type") == "training")
    config = sub_job.get("training_config") or {}
    model = sub_job.get("model_name")
    hardware = str(body.get("hardware") or "H200").lower()
    adapter = config.get("peft_config")
    lora = adapter is not None
    try:
        folder = load_catalog_model(model, catalog_dir)[0] if isinstance(model, str) else None
    except FileNotFoundError:
        folder = None
    if context is not None:
        candidates: tuple[str, ...] = (context, *(name for name in CATALOG_CONTEXTS if name != context))
    elif folder is not None:
        candidates = _inference_context_order(folder, hardware, lora, config.get("max_seq_len") or 0)
    else:
        candidates = CATALOG_CONTEXTS
    if folder is not None:
        for candidate in candidates:
            path = folder / candidate / catalog_file_name(hardware, "inference", lora)
            if not path.is_file():
                continue
            if lora:
                sampling = _sub_job_config(load_job_body(str(path)), "sampling", "inference_config")
                if not same_adapter(sampling.get("peft_config"), adapter):
                    continue
            return {"model": model, "context": candidate, "hardware": hardware, "lora": lora}
    logger.warning(
        "No inference file in the model catalog serves this checkpoint. It needs a sampling sub-job for %s on %s%s. "
        "Write that job to a file and put its path in place of %s in the command below.",
        model,
        hardware,
        f" with peft_config {json.dumps(adapter, sort_keys=True)}" if lora else " with full weights",
        INFERENCE_JOB_CONFIG_PLACEHOLDER,
    )
    return {"job_config": INFERENCE_JOB_CONFIG_PLACEHOLDER}


def _make_config_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m recipes.utils",
        description=(
            "Build a Cortex Training job config from the model catalog, or from explicit "
            "training and inference config files, apply override files in order, and write "
            "the job as JSON. Submit it with `ct submit OUT.json`, or pass it to an example "
            "as job_config=."
        ),
    )
    catalog = parser.add_argument_group("from the model catalog")
    catalog.add_argument("--model", help="exact model ID, for example Qwen/Qwen3-8B")
    catalog.add_argument("--context", choices=CATALOG_CONTEXTS, help="standard or long context")
    catalog.add_argument("--hardware", choices=CATALOG_HARDWARE, help="GPU hardware (default: h200)")
    catalog.add_argument("--lora", action="store_true", help="use the LoRA files instead of the full ones")
    files = parser.add_argument_group("from explicit config files")
    files.add_argument("--training-config", metavar="FILE", help="training job config file")
    files.add_argument("--inference-config", metavar="FILE", help="inference job config file (with --rl)")
    files.add_argument(
        "--router-replay",
        action="store_true",
        help="fill in router replay for an MoE model (with --rl)",
    )
    job = parser.add_argument_group("the job")
    kinds = job.add_mutually_exclusive_group()
    kinds.add_argument("--inference", action="store_true", help="build the inference job instead of the training job")
    kinds.add_argument("--rl", action="store_true", help="build one job with the training and sampling sub-jobs")
    kinds.add_argument(
        "--distillation",
        action="store_true",
        help="build the on-policy distillation student and teacher jobs (needs --student and --teacher)",
    )
    distill = parser.add_argument_group("on-policy distillation (with --distillation)")
    distill.add_argument("--student", help="student model ID (training + sampling), for example Qwen/Qwen3-0.6B")
    distill.add_argument("--teacher", help="teacher model ID (sampling only), for example Qwen/Qwen3-8B")
    distill.add_argument("--teacher-context", choices=CATALOG_CONTEXTS, help="teacher context (default: --context)")
    distill.add_argument("--student-max-seq-len", type=int, metavar="N", help="student context window")
    distill.add_argument(
        "--teacher-max-seq-len", type=int, metavar="N", help="teacher context window (must exceed the student's)"
    )
    distill.add_argument(
        "--teacher-override",
        action="extend",
        nargs="+",
        default=[],
        metavar="FILE",
        help="JSON override files for the teacher job, applied in order",
    )
    job.add_argument(
        "--rollout-batch-size",
        type=int,
        metavar="N",
        help="rollouts per RL step (kept for compatibility; the router-replay default caps are now fixed)",
    )
    job.add_argument(
        "--override",
        action="extend",
        nargs="+",
        default=[],
        metavar="FILE",
        help="JSON override files, applied in order",
    )
    parser.add_argument("-o", "--output", required=True, metavar="OUT.json", help="file to write, or - for stdout")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command line for :func:`make_config`."""
    parser = _make_config_parser()
    args = parser.parse_args(argv)
    # Presence, not truthiness: --student-max-seq-len 0 is still a misplaced flag.
    distillation_args_given = (
        args.student is not None
        or args.teacher is not None
        or args.teacher_context is not None
        or args.student_max_seq_len is not None
        or args.teacher_max_seq_len is not None
        or bool(args.teacher_override)
    )
    if args.distillation:
        if args.student is None or args.teacher is None:
            parser.error("--distillation needs --student and --teacher model IDs")
        if args.context is None:
            parser.error("--distillation needs --context standard or --context long")
        if args.model or args.training_config or args.inference_config or args.router_replay or args.rollout_batch_size:
            parser.error(
                "--distillation picks its jobs with --student and --teacher; it cannot be combined with "
                "--model, --training-config, --inference-config, --router-replay or --rollout-batch-size"
            )
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        try:
            student_body, teacher_body = make_distillation_config(
                args.student,
                args.teacher,
                args.context,
                hardware=args.hardware or "h200",
                lora=args.lora,
                teacher_context=args.teacher_context,
                student_max_seq_len=args.student_max_seq_len,
                teacher_max_seq_len=args.teacher_max_seq_len,
                overrides=args.override,
                teacher_overrides=args.teacher_override,
            )
        except (FileNotFoundError, ImportError, TypeError, ValueError) as exc:
            sys.stderr.write(f"error: {exc}\n")
            return 1
        if args.output == "-":
            sys.stdout.write(json.dumps({"student": student_body, "teacher": teacher_body}, indent=2) + "\n")
        else:
            student_path = Path(args.output).expanduser()
            suffix = student_path.suffix or ".json"
            teacher_path = student_path.with_name(f"{student_path.stem}.teacher{suffix}")
            for path, body in ((student_path, student_body), (teacher_path, teacher_body)):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
            logger.info("wrote student %s and teacher %s", student_path, teacher_path)
        return 0
    if distillation_args_given:
        parser.error("--student, --teacher, --teacher-context, --*-max-seq-len and --teacher-override need --distillation")
    if args.model is not None:
        if args.training_config or args.inference_config or args.router_replay:
            parser.error("--model picks catalog files; it cannot be combined with --training-config, "
                         "--inference-config or --router-replay")
        if args.context is None:
            parser.error("--model needs --context standard or --context long")
    else:
        if args.training_config is None:
            parser.error("pass --model and --context, or --training-config")
        if args.context or args.hardware or args.lora or args.inference:
            parser.error("--context, --hardware, --lora and --inference pick catalog files and need --model")
        if args.inference_config is not None and not args.rl:
            parser.error("--inference-config is used with --rl")
        if args.router_replay and not args.rl:
            parser.error("--router-replay is used with --rl")
    if args.rollout_batch_size is not None and not args.rl:
        parser.error("--rollout-batch-size is used with --rl")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        body = make_config(
            args.model,
            args.context,
            hardware=args.hardware,
            lora=args.lora,
            inference=args.inference,
            rl=args.rl,
            training_config=args.training_config,
            inference_config=args.inference_config,
            overrides=args.override,
            router_replay=True if args.router_replay else None,
            rollout_batch_size=args.rollout_batch_size,
        )
    except (FileNotFoundError, ImportError, TypeError, ValueError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    text = json.dumps(body, indent=2) + "\n"
    if args.output == "-":
        sys.stdout.write(text)
    else:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        logger.info("wrote %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
