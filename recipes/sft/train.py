# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reusable supervised fine-tuning workflow runner."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any
from typing import Protocol

from recipes.logging import setup_logging
from recipes.utils import build_renderer
from recipes.utils import collate
from recipes.utils import forward_backward_step
from recipes.utils import load_job_body
from recipes.utils import log_saved_checkpoints
from recipes.utils import make_client
from recipes.utils import running_job
from recipes.utils import save_recipe_checkpoints

from cortex_training.client import DEBUG_OPTIONS_ENV

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)
logging.getLogger("urllib3").setLevel(logging.WARN)
logging.getLogger("tinker_cookbook.renderers.base").setLevel(logging.ERROR)

_RECIPE_DIR = Path(__file__).resolve().parent
_CONFIG_SEARCH_DIRS = (_RECIPE_DIR, _RECIPE_DIR / "configs")


class Config(Protocol):
    config: str
    job_id: str | None
    pad_to_max_length: bool
    max_steps: int
    debug_image_tag: str | None
    enable_thinking: bool
    renderer_name: str | None
    job_config: str


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
    body = load_job_body(config.job_config, search_dirs=_CONFIG_SEARCH_DIRS)
    if config.debug_image_tag:
        body["debug"] = {"job": {"image_tag": config.debug_image_tag}}
    return body


def causal_cross_entropy_processing() -> dict[str, Any]:
    return {
        "loss_fn": "causal_cross_entropy",
        "post": ["compute_logprobs"],
        "config": {},
    }


def train(config: Config, task: Task) -> dict[str, Any]:
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

    client = make_client(config.config)

    with running_job(client, body, job_id=config.job_id) as job_id:
        ml_logger = setup_logging(config, client=client, job_id=job_id)
        for step in range(total_steps):
            start_time = time.time()
            metrics: dict[str, float] = {}

            lr_mult = max(0.0, 1.0 - step / max(n_train_batches, 1))
            current_lr = learning_rate * lr_mult

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

        saved = save_recipe_checkpoints(client, job_id)
        sample_prompt = task.sample_prompt()
        log_saved_checkpoints(
            config_path=config.config,
            job_id=job_id,
            saved=saved,
            sampling_command="sample",
            job_config=Path(config.job_config).name,
            sample_prompt=sample_prompt,
            enable_thinking=config.enable_thinking,
            renderer_name=config.renderer_name,
            temperature=0 if sample_prompt else None,
        )

    ml_logger.close()
    logger.info("Training completed")
    return {"job_id": job_id, "checkpoint_id": saved["weights-only"].get("checkpoint_id")}
