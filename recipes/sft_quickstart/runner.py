# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reusable supervised fine-tuning workflow runner."""

from __future__ import annotations

import inspect
import logging
import math
import os
import time
from pathlib import Path
from typing import Any
from typing import Protocol

import torch

from recipes.logging import setup_logging
from recipes.utils import build_job
from recipes.utils import build_renderer
from recipes.utils import collate
from recipes.utils import forward
from recipes.utils import forward_backward_step
from recipes.utils import inference_job_args
from recipes.utils import log_saved_checkpoints
from recipes.utils import make_client
from recipes.utils import example_connection_args
from recipes.utils import running_job
from recipes.utils import save_example_checkpoints

from cortex_training.client import DEBUG_OPTIONS_ENV

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)
logging.getLogger("urllib3").setLevel(logging.WARN)
logging.getLogger("tinker_cookbook.renderers.base").setLevel(logging.ERROR)

_EXAMPLE_DIR = Path(__file__).resolve().parent


class Config(Protocol):
    # Legacy JSON connection file. Omit it to use `connection` or the default profile.
    config: str | None = None
    # Snowflake connection profile name (connections.toml). Use config or connection, not both.
    connection: str | None = None
    job_id: str | None
    model: str
    context: str
    hardware: str
    lora: bool
    job_config: str | None
    lr: float | None
    train_batch_size: int | None
    max_seq_len: int | None
    pad_to_max_length: bool
    max_steps: int
    debug_image_tag: str | None
    enable_thinking: bool
    renderer_name: str | None


class Task(Protocol):
    name: str

    def load_dataset(self, *, n_train: int): ...

    def build_sequence(
        self,
        row,
        renderer,
        *,
        max_seq_len: int,
    ): ...

    def sample_prompt(self) -> str | None: ...


def job_body(config: Config) -> dict:
    """The training job: the catalog file, sft/overrides.json, the example's own
    overrides.json (see :func:`_example_overrides`), then ``config``.
    """
    return build_job(
        "training",
        model=config.model,
        context=config.context,
        hardware=config.hardware,
        lora=config.lora,
        job_config=config.job_config,
        example_dir=_EXAMPLE_DIR,
        overrides=_example_overrides(config),
        search_dirs=(_EXAMPLE_DIR,),
        lr=config.lr,
        train_batch_size=config.train_batch_size,
        max_seq_len=config.max_seq_len,
        debug_image_tag=config.debug_image_tag,
    )


def _example_overrides(config: Config) -> list[Path]:
    """The ``overrides.json`` next to the entry point that defines ``config``'s class.

    It applies after ``sft/overrides.json`` to that example only, for example
    ``sft/dare/overrides.json`` for DARE. An example without one gets none.
    """
    path = Path(inspect.getfile(type(config))).resolve().parent / "overrides.json"
    return [path] if path.is_file() else []


def causal_cross_entropy_processing() -> dict[str, Any]:
    return {
        "loss_fn": "causal_cross_entropy",
        "post": ["compute_logprobs"],
        "config": {},
    }


def lr_multiplier(step: int, total_steps: int, *, schedule: str = "linear", warmup_ratio: float = 0.0) -> float:
    if schedule not in ("linear", "cosine", "constant"):
        raise ValueError(f"lr_schedule must be linear, cosine, or constant, got {schedule!r}")
    warmup_steps = math.ceil(warmup_ratio * total_steps)
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if schedule == "constant":
        return 1.0
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    if schedule == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return max(0.0, 1.0 - progress)


def next_token_labels(input_ids: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """``input_ids[i + 1]`` wherever ``loss_mask[i]`` weights a prediction, else -100."""
    return input_ids.roll(-1, dims=-1).masked_fill(loss_mask <= 0, -100)


def heldout_nll(
    client: Any,
    job_id: str,
    sequences: list[Any],
    *,
    batch_size: int,
    pad_token_id: int,
    max_seq_len: int,
    pad_to_max_seq_len: bool = False,
) -> float:
    """Token-weighted mean NLL over whole batches, from no-grad forward passes."""
    total_nll = 0.0
    total_weight = 0.0
    for start in range(0, len(sequences), batch_size):
        kwargs, context = collate(
            sequences[start : start + batch_size],
            pad_token_id=pad_token_id,
            max_seq_len=max_seq_len,
            pad_to_max_seq_len=pad_to_max_seq_len,
        )
        loss_mask = context["loss_mask"]
        # forward reads labels as sent, for every model provider.
        kwargs["labels"] = next_token_labels(kwargs["input_ids"], loss_mask)
        logprobs = forward(client, job_id, kwargs)["logprobs"]
        total_nll -= float((logprobs.float() * loss_mask).sum())
        total_weight += float(loss_mask.sum())
    return total_nll / total_weight


def train(config: Config, task: Task) -> dict[str, Any]:
    # setup_logging replaces these handlers once the job runs; until then this
    # keeps the created job id in the log, so an interrupted start can be cancelled.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if config.debug_image_tag:
        os.environ[DEBUG_OPTIONS_ENV] = "1"
        logger.info("Using debug image_tag=%s", config.debug_image_tag)

    body = job_body(config)
    training_sub = next(
        (
            sub
            for sub in body.get("sub_job_configs") or ()
            if sub.get("job_type") == "training"
        ),
        {},
    )
    training = training_sub.get("training_config") or {}
    batch_size = int(training.get("train_batch_size"))
    max_seq_len = int(training.get("max_seq_len"))
    learning_rate = float((training.get("optimizer") or {}).get("lr"))
    model_name = training_sub.get("model_name")

    tokenizer, renderer, renderer_name = build_renderer(
        model_name,
        renderer_name=config.renderer_name,
        enable_thinking=config.enable_thinking,
    )
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    logger.info(
        "Using renderer: %s (enable_thinking=%s)",
        renderer_name,
        config.enable_thinking,
    )

    logger.info("Loading %s dataset...", task.name)
    train_dataset = task.load_dataset(n_train=config.max_steps * batch_size)

    n_train_batches = len(train_dataset) // batch_size
    n_dropped = len(train_dataset) % batch_size
    if n_dropped:
        logger.info(
            "Dropping last %d examples to keep batch size uniform at %d",
            n_dropped,
            batch_size,
        )
    total_steps = min(n_train_batches, config.max_steps)
    logger.info(
        "Train batches: %d; training for %d steps",
        n_train_batches,
        total_steps,
    )

    eval_every = int(getattr(config, "eval_every", 0))
    load_eval_dataset = getattr(task, "load_eval_dataset", None)
    eval_dataset = load_eval_dataset() if eval_every > 0 and callable(load_eval_dataset) else None
    eval_sequences = []
    if eval_dataset is not None:
        n_eval = len(eval_dataset) - len(eval_dataset) % batch_size
        if n_eval == 0:
            logger.warning(
                "Held-out set has %d examples, fewer than one batch of %d; skipping valid_nll",
                len(eval_dataset),
                batch_size,
            )
        eval_sequences = [
            task.build_sequence(row, renderer, max_seq_len=max_seq_len) for row in eval_dataset.select(range(n_eval))
        ]
        if eval_sequences:
            logger.info("valid_nll on %d held-out examples every %d steps", n_eval, eval_every)

    client = make_client(config.config, config.connection)
    connection_args = example_connection_args(config.config, config.connection)

    def evaluate(job_id: str) -> dict[str, float]:
        eval_start = time.time()
        valid_nll = heldout_nll(
            client,
            job_id,
            eval_sequences,
            batch_size=batch_size,
            pad_token_id=pad_token_id,
            max_seq_len=max_seq_len,
            pad_to_max_seq_len=config.pad_to_max_length,
        )
        logger.info("valid_nll %.4f", valid_nll)
        return {"valid_nll": valid_nll, "time_valid": time.time() - eval_start}

    with running_job(client, body, job_id=config.job_id) as job_id:
        ml_logger = setup_logging(config, client=client, job_id=job_id)
        for step in range(total_steps):
            metrics: dict[str, float] = {}
            if eval_sequences and step % eval_every == 0:
                metrics.update(evaluate(job_id))
            start_time = time.time()

            current_lr = learning_rate * lr_multiplier(
                step,
                n_train_batches,
                schedule=getattr(config, "lr_schedule", "linear"),
                warmup_ratio=getattr(config, "warmup_ratio", 0.0),
            )

            batch_start = step * batch_size
            batch_rows = train_dataset.select(
                range(batch_start, batch_start + batch_size)
            )
            sequences = [
                task.build_sequence(
                    row,
                    renderer,
                    max_seq_len=max_seq_len,
                )
                for row in batch_rows
            ]
            kwargs, context = collate(
                sequences,
                pad_token_id=pad_token_id,
                max_seq_len=max_seq_len,
                pad_to_max_seq_len=config.pad_to_max_length,
            )
            fwd_bwd_result, step_result = forward_backward_step(
                client,
                job_id,
                kwargs,
                context=context,
                learning_rate=current_lr,
                processing=causal_cross_entropy_processing(),
            )

            train_loss = float(fwd_bwd_result["avg_loss"])
            metrics.update(fwd_bwd_result.get("metrics") or {})
            metrics.update(step_result.get("metrics") or {})
            metrics.update(
                train_nll=train_loss,
                loss_weight_sum=float(context["loss_mask"].sum()),
                global_steps=step_result.get("global_steps", step + 1),
                progress=step / n_train_batches,
                time_total=time.time() - start_time,
            )
            ml_logger.log_metrics(metrics=metrics, step=step)

        if eval_sequences:
            ml_logger.log_metrics(metrics=evaluate(job_id), step=total_steps)

        saved = save_example_checkpoints(client, job_id)
        eval_command = getattr(task, "eval_command", None)
        if callable(eval_command):
            # A task with its own evaluation (e.g. DARE) prints that command.
            log_saved_checkpoints(
                connection_args=connection_args,
                job_id=job_id,
                saved=saved,
            )
            checkpoint_id = str(saved["weights-only"].get("checkpoint_id") or "").strip()
            if checkpoint_id:
                logger.info(
                    "Evaluate this weights-only checkpoint with:\n  %s",
                    eval_command(
                        connection_args=connection_args,
                        job_id=job_id,
                        checkpoint_id=checkpoint_id,
                        job_args=inference_job_args(
                            body,
                            context=config.context if config.job_config is None else None,
                        ),
                    ),
                )
        else:
            sample_prompt = task.sample_prompt()
            log_saved_checkpoints(
                connection_args=connection_args,
                job_id=job_id,
                saved=saved,
                sampling_command="sample",
                job_args=inference_job_args(
                    body,
                    context=config.context if config.job_config is None else None,
                ),
                sample_prompt=sample_prompt,
                enable_thinking=config.enable_thinking,
                renderer_name=config.renderer_name,
                temperature=0 if sample_prompt else None,
            )

    ml_logger.close()
    logger.info("Training completed")
    return {"job_id": job_id, "checkpoint_id": saved["weights-only"].get("checkpoint_id")}
