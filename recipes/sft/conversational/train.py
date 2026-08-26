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

"""
Minimal supervised fine-tuning loop against a Cortex Training training job.

A port of ``tinker_cookbook/recipes/chat_sl/train.py``.

Default data is a one-example chat dataset that memorizes
``Who trained you?`` → ``Snowflake AI Research``. Hugging Face chat datasets with a
``messages`` column work as well. This sample script uses tinker_cookbook's
util functions-- supports models tinker supports.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import chz
import datasets
from recipes.utils import build_renderer
from recipes.utils import collate
from recipes.utils import forward_backward_step
from recipes.utils import load_job_body
from recipes.utils import log_saved_checkpoints
from recipes.utils import make_client
from recipes.utils import running_job
from recipes.utils import save_recipe_checkpoints
from recipes.utils import sequence_from_conversation
from recipes.utils import use_next_token_labels
from tinker_cookbook import renderers
from tinker_cookbook.utils import ml_log

from cortex_training.client import DEBUG_OPTIONS_ENV

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)
logging.getLogger("urllib3").setLevel(logging.WARN)
logging.getLogger("tinker_cookbook.renderers.base").setLevel(logging.ERROR)

_RECIPE_DIR = Path(__file__).resolve().parent
BUILTIN_CHAT_DATASETS = {
    "who_trained_you": _RECIPE_DIR / "data" / "who_trained_you.jsonl",
}
WHO_TRAINED_YOU_PROMPT = "Who trained you?"


def is_who_trained_you_dataset(dataset: str) -> bool:
    return dataset == "who_trained_you" or Path(dataset).name == "who_trained_you.jsonl"


@chz.chz
class Config:
    config: str
    job_id: str | None = None

    dataset: str = "who_trained_you"
    dataset_split: str = "train"
    train_on_what: renderers.TrainOnWhat = renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES
    pad_to_max_length: bool = False
    max_steps: int = 100

    debug_image_tag: str | None = None
    # False (default): tinker *_disable_thinking renderer when the model has one.
    # True: thinking-on renderer (qwen3 for Qwen3-8B).
    enable_thinking: bool = False
    renderer_name: str | None = None

    log_path: str = "/tmp/cortex-training-examples/sft-loop"
    wandb_project: str | None = None
    wandb_name: str | None = None

    # Loaded as the training-only create-job body.
    job_config: str = "configs/qwen3_8b_full.json"


def job_body(config: Config) -> dict:
    body = load_job_body(
        config.job_config,
        search_dirs=(_RECIPE_DIR, _RECIPE_DIR / "configs"),
    )
    if config.debug_image_tag:
        body["debug"] = {"job": {"image_tag": config.debug_image_tag}}
    return body


def resolve_chat_dataset(dataset: str) -> str:
    builtin = BUILTIN_CHAT_DATASETS.get(dataset)
    if builtin is not None:
        return str(builtin)
    return dataset


def _is_local_chat_file(source: str) -> bool:
    path = Path(source).expanduser()
    return path.is_file() and path.suffix.lower() in {".json", ".jsonl"}


def tile_rows(dataset: datasets.Dataset, n_rows: int) -> datasets.Dataset:
    """Repeat a short dataset so training can run ``max_steps`` batches."""
    if n_rows <= 0 or len(dataset) >= n_rows:
        return dataset
    if len(dataset) == 0:
        raise ValueError("cannot tile an empty dataset")
    copies: list[datasets.Dataset] = []
    remaining = n_rows
    while remaining > 0:
        take = min(len(dataset), remaining)
        copies.append(dataset.select(range(take)))
        remaining -= take
    return datasets.concatenate_datasets(copies)


def load_chat_dataset(
    dataset: str,
    *,
    dataset_split: str,
    n_train: int,
) -> datasets.Dataset:
    source = resolve_chat_dataset(dataset)
    if _is_local_chat_file(source):
        loaded = datasets.load_dataset("json", data_files={dataset_split: source})
    else:
        loaded = datasets.load_dataset(source)
    if not isinstance(loaded, datasets.DatasetDict):
        loaded = datasets.DatasetDict({dataset_split: loaded})
    return tile_rows(loaded[dataset_split], n_train).shuffle(seed=0)


def main(config: Config):
    if config.debug_image_tag:
        os.environ[DEBUG_OPTIONS_ENV] = "1"
        logger.info("Using debug image_tag=%s", config.debug_image_tag)

    body = job_body(config)
    training_sub = next((sub for sub in body.get("sub_job_configs") or () if sub.get("job_type") == "training"), {})
    training = training_sub.get("training_config") or {}
    batch_size = int(training.get("train_batch_size"))
    max_seq_len = int(training.get("max_seq_len"))
    learning_rate = float((training.get("optimizer") or {}).get("lr"))
    model_provider = str(training.get("model_provider") or "huggingface")
    model_name = training_sub.get("model_name")

    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
        config=config,
        do_configure_logging_module=True,
    )

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
    next_token_labels = use_next_token_labels(model_provider)

    logger.info("Loading dataset...")
    train_dataset = load_chat_dataset(
        config.dataset,
        dataset_split=config.dataset_split,
        n_train=config.max_steps * batch_size,
    )

    n_train_batches = len(train_dataset) // batch_size
    n_dropped = len(train_dataset) % batch_size
    if n_dropped:
        logger.info(f"Dropping last {n_dropped} examples to keep batch size uniform at {batch_size}")
    total_steps = min(n_train_batches, config.max_steps)
    logger.info(f"Train batches: {n_train_batches}; training for {total_steps} steps")

    client = make_client(config.config)

    with running_job(client, body, job_id=config.job_id) as job_id:
        for step in range(total_steps):
            start_time = time.time()
            metrics: dict[str, float] = {}

            # Linear learning rate schedule, applied on the server per step.
            lr_mult = max(0.0, 1.0 - step / max(n_train_batches, 1))
            current_lr = learning_rate * lr_mult

            batch_start = step * batch_size
            batch_rows = train_dataset.select(range(batch_start, batch_start + batch_size))
            sequences = [
                sequence_from_conversation(
                    row["messages"],
                    renderer,
                    train_on_what=config.train_on_what,
                    max_seq_len=max_seq_len,
                    next_token_labels=next_token_labels,
                )
                for row in batch_rows
            ]
            kwargs, _ = collate(
                sequences,
                pad_token_id=pad_token_id,
                max_seq_len=max_seq_len,
                pad_to_max_seq_len=config.pad_to_max_length,
            )
            fwd_bwd_result, step_result = forward_backward_step(client, job_id, kwargs, learning_rate=current_lr)

            train_loss = float(fwd_bwd_result["avg_loss"])
            metrics.update(fwd_bwd_result.get("metrics") or {})
            metrics.update(step_result.get("metrics") or {})

            metrics.update(
                train_nll=train_loss,
                global_steps=step_result.get("global_steps", step + 1),
                progress=step / n_train_batches,
                time_total=time.time() - start_time,
            )
            ml_logger.log_metrics(metrics=metrics, step=step)

        saved = save_recipe_checkpoints(client, job_id)
        sample_prompt = WHO_TRAINED_YOU_PROMPT if is_who_trained_you_dataset(config.dataset) else None
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


if __name__ == "__main__":
    chz.nested_entrypoint(main)
