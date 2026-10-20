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

"""Cortex Training recipe helpers."""

from __future__ import annotations

import contextlib
import json
import logging
import re
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from cortex_training import CortexTrainingClient
from cortex_training import wire

logger = logging.getLogger(__name__)


_ROUTER_REPLAY_DEFAULT_MAX_CACHE_BYTES = 16 * 1024**3
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def load_connection_mapping(config_path: str) -> dict[str, Any]:
    """Load a recipe connection file (PAT host, database, schema)."""
    parsed = json.loads(Path(config_path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"connection config {config_path} must be a JSON object")
    config = parsed.get("connection", parsed)
    if not isinstance(config, dict):
        raise ValueError(f"connection config {config_path} must be a JSON object")
    return config


def make_client(config_path: str, **overrides: Any) -> CortexTrainingClient:
    config = load_connection_mapping(config_path)

    pat = config.get("pat")
    kwargs: dict[str, Any] = dict(
        database=config.get("database", "CORTEX_TRAINING_DB"),
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
    return CortexTrainingClient.from_pat(
        host=host,
        pat=pat,
        verify_ssl=bool(config.get("verify_ssl", True)),
        **kwargs,
    )


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


def lora_peft_config(rank: int) -> dict[str, Any] | None:
    if rank <= 0:
        return None
    return {
        "peft_type": "Lora",
        "r": rank,
        "lora_alpha": rank,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": list(LORA_TARGET_MODULES),
    }


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


def save_recipe_checkpoints(client: CortexTrainingClient, job_id: str) -> dict[str, dict[str, Any]]:
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


def sampling_job_body(
    *,
    model_name: str,
    max_seq_len: int,
    n_gpus: int,
    dtype: str = "bfloat16",
    seed: int = 42,
    gpu_memory_utilization: float = 0.8,
    lora_rank: int = 0,
    source_checkpoint_info: dict[str, str] | None = None,
    debug_image_tag: str | None = None,
) -> dict[str, Any]:
    """Standalone sampling job from original HF weights or a weights-only checkpoint."""
    inference_config: dict[str, Any] = {
        "max_seq_len": max_seq_len,
        "n_gpus": n_gpus,
        "vllm_config": {
            "max_model_len": max_seq_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": True,
        },
    }
    peft = lora_peft_config(lora_rank)
    if peft is not None:
        inference_config["peft_config"] = peft

    sub_job: dict[str, Any] = {
        "job_type": "sampling",
        "model_name": model_name,
        "dtype": dtype,
        "seed": seed,
        "inference_config": inference_config,
    }
    if source_checkpoint_info is not None:
        sub_job["source_checkpoint_info"] = dict(source_checkpoint_info)

    body: dict[str, Any] = {"sub_job_configs": [sub_job]}
    if debug_image_tag:
        body["debug"] = {"job": {"image_tag": debug_image_tag}}
    return body


def _sampling_cli(
    *,
    module: str,
    config_path: str,
    job_id: str,
    extra: str = "",
) -> str:
    line = f"  python -m {module} config={config_path} source_job_id={job_id}"
    if extra:
        line += extra
    return line


def log_saved_checkpoints(
    *,
    config_path: str,
    job_id: str,
    saved: Mapping[str, Mapping[str, Any]],
    sampling_command: str | None = None,
    job_config: str | None = None,
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
    if job_config:
        extra += f" job_config={job_config}"
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
            config_path=config_path,
            job_id=job_id,
            extra=extra,
        ),
    )


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
    """
    if renderer_name:
        return renderer_name
    from tinker_cookbook import model_info

    try:
        recommended = list(model_info.get_recommended_renderer_names(model_name))
    except KeyError as exc:
        raise ValueError(
            f"tinker_cookbook has no recommended renderer for {model_name!r}; "
            "see https://tinker-docs.thinkingmachines.ai/tutorials/core-concepts/rendering/#available-renderers"
        ) from exc
    if not recommended:
        raise ValueError(f"tinker_cookbook listed no renderers for {model_name!r}")
    if enable_thinking is False:
        disabled = [name for name in recommended if name.endswith("_disable_thinking")]
        return disabled[0]
    if enable_thinking is True:
        enabled = [name for name in recommended if not name.endswith("_disable_thinking")]
        return enabled[0]
    return recommended[0]


def build_renderer(
    model_name: str,
    renderer_name: str | None = None,
    *,
    enable_thinking: bool | None = None,
):
    """Return ``(tokenizer, renderer, renderer_name)`` for ``model_name``.

    Uses :func:`recommended_renderer_name` unless ``renderer_name`` is given.
    These recipe helpers only cover models tinker lists; Cortex Training itself can
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


def router_replay_config(
    enabled: bool = True,
    max_cache_bytes: int | None = None,
) -> dict[str, Any]:
    if not enabled:
        return {"enabled": False}
    return {
        "enabled": True,
        "max_cache_bytes": int(_ROUTER_REPLAY_DEFAULT_MAX_CACHE_BYTES if max_cache_bytes is None else max_cache_bytes),
    }


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
    return errors[-1][:2000]


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
