# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reusable GRPO workflow runner."""

from __future__ import annotations

import logging
import os
import statistics
import time
from pathlib import Path
from typing import Any
from typing import Protocol

from recipes.logging import setup_logging
from recipes.utils import TrainSequence
from recipes.utils import bootstrap_router_replay
from recipes.utils import build_renderer
from recipes.utils import collate
from recipes.utils import discard_router_replay
from recipes.utils import forward_backward_step
from recipes.utils import load_job_body
from recipes.utils import log_saved_checkpoints
from recipes.utils import make_client
from recipes.utils import router_replay_stop_params
from recipes.utils import running_job
from recipes.utils import sampling_params_with_sample_ids
from recipes.utils import save_recipe_checkpoints
from recipes.utils import sequence_from_rollout
from recipes.utils import stop_params_for
from recipes.utils import sync_weights

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
    problems_per_batch: int
    group_size: int
    max_tokens: int
    temperature: float
    top_p: float
    max_steps: int | None
    eps_clip: float
    loss_agg_mode: str
    entropy_coeff: float
    remove_constant_reward_groups: bool
    debug_image_tag: str | None
    eval_every: int
    n_test: int | None
    eval_temperature: float | None
    eval_max_tokens: int | None
    weight_sync_bucket_size: int | None
    job_config: str


class Task(Protocol):
    name: str
    evaluation_name: str

    def load(self, *, seed: int): ...

    def build_prompt(self, problem: Any, renderer) -> list[int]: ...

    def score_response(
        self,
        response: str,
        answer: Any,
        *,
        result: dict,
        max_tokens: int | None,
    ) -> tuple[float, dict[str, float]]: ...

    def make_evaluator(
        self,
        test_problems,
        renderer,
        *,
        sampling_params: dict,
        n_test: int | None,
    ): ...


def job_body(config: Config) -> dict:
    body = load_job_body(config.job_config, search_dirs=_CONFIG_SEARCH_DIRS)
    if config.debug_image_tag:
        body["debug"] = {"job": {"image_tag": config.debug_image_tag}}
    return body


def processing_block(config: Config, global_batch_size: int) -> dict:
    return {
        "loss_fn": "grpo",
        "post": ["compute_logprobs"],
        "config": {
            "eps_clip": config.eps_clip,
            "loss_agg_mode": config.loss_agg_mode,
            "entropy_coeff": config.entropy_coeff,
            "global_batch_size": global_batch_size,
        },
    }


def _should_eval(step: int, total_steps: int, eval_every: int) -> bool:
    return eval_every > 0 and (step % eval_every == 0 or step == total_steps - 1)


def train(config: Config, task: Task) -> None:
    if config.debug_image_tag:
        os.environ[DEBUG_OPTIONS_ENV] = "1"
        logger.info("Using debug image_tag=%s", config.debug_image_tag)

    _train(config, task)
    logger.info("Training completed")


def _train(config: Config, task: Task) -> None:
    body = job_body(config)
    subs = {sub.get("job_type"): sub for sub in body.get("sub_job_configs") or ()}
    training_sub = subs.get("training") or {}
    sampling_sub = subs.get("sampling") or {}
    training = training_sub.get("training_config") or {}
    sampling = sampling_sub.get("inference_config") or {}
    learning_rate = float((training.get("optimizer") or {}).get("lr"))
    max_seq_len = int(training.get("max_seq_len") or sampling.get("max_seq_len"))
    lora_rank = int(
        (training.get("peft_config") or sampling.get("peft_config") or {}).get("r") or 0
    )
    router_replay = bool(
        (training.get("router_replay") or {}).get("enabled")
        or (sampling.get("router_replay") or {}).get("enabled")
    )
    router_replay_max_cache_bytes = (training.get("router_replay") or {}).get(
        "max_cache_bytes"
    )
    model_name = training_sub.get("model_name") or sampling_sub.get("model_name")
    seed = training_sub.get("seed", sampling_sub.get("seed", 42))
    seed = 42 if seed is None else int(seed)

    tokenizer, renderer, renderer_name = build_renderer(model_name)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    logger.info("Using renderer: %s", renderer_name)

    logger.info("Loading %s dataset...", task.name)
    task_dataset = task.load(seed=seed)
    train_problems = task_dataset.train

    n_train_batches = len(train_problems) // config.problems_per_batch
    total_steps = (
        n_train_batches
        if config.max_steps is None
        else min(n_train_batches, config.max_steps)
    )
    logger.info(
        "Training for %d rollout batches (%d problems, %d per batch, group_size=%d)",
        total_steps,
        len(train_problems),
        config.problems_per_batch,
        config.group_size,
    )

    stop_params = (
        router_replay_stop_params(renderer.get_stop_sequences(), tokenizer)
        if router_replay
        else stop_params_for(renderer.get_stop_sequences())
    )
    sampling_params = {
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        **stop_params,
    }

    evaluator = None
    if config.eval_every > 0:
        if task_dataset.test is None:
            logger.warning(
                "eval_every=%d but %s has no held-out split, so no benchmark will be reported",
                config.eval_every,
                task.name,
            )
        else:
            eval_temperature = (
                config.temperature
                if config.eval_temperature is None
                else config.eval_temperature
            )
            evaluator = task.make_evaluator(
                task_dataset.test,
                renderer,
                sampling_params={
                    "max_tokens": config.eval_max_tokens or config.max_tokens,
                    "temperature": eval_temperature,
                    "top_p": config.top_p,
                    **stop_params,
                },
                n_test=config.n_test,
            )
            logger.info(
                "Held-out %s on %d problems",
                task.evaluation_name,
                len(evaluator.prompts),
            )
        logger.info(
            "After save, also run recipes.inference.evaluate (%s)",
            task.evaluation_name,
        )

    client = make_client(config.config)

    with running_job(client, body, job_id=config.job_id) as job_id:
        ml_logger = setup_logging(config, client=client, job_id=job_id)
        sampling_job_id: str | None = None
        if router_replay:
            logger.info("Bootstrapping router replay for job %s", job_id)
            bootstrap_router_replay(
                client,
                job_id,
                max_cache_bytes=router_replay_max_cache_bytes,
            )
            sampling_job_id = f"{job_id}:sampling:0"

        for batch_idx in range(total_steps):
            t_start = time.time()
            metrics: dict[str, float] = {
                "progress/batch": batch_idx,
                "progress/done_frac": (batch_idx + 1) / max(n_train_batches, 1),
                "optim/lr": learning_rate,
            }

            if evaluator is not None and _should_eval(
                batch_idx,
                total_steps,
                config.eval_every,
            ):
                eval_start = time.time()
                metrics.update(evaluator(client, job_id))
                metrics["time/eval"] = time.time() - eval_start

            batch_start = batch_idx * config.problems_per_batch
            batch = train_problems[
                batch_start : batch_start + config.problems_per_batch
            ]

            prompts_D: list[list[int]] = []
            prompt_tokens_P: list[list[int]] = []
            for problem, _ in batch:
                prompt_tokens = task.build_prompt(problem, renderer)
                prompt_tokens_P.append(prompt_tokens)
                prompts_D.extend([prompt_tokens] * config.group_size)

            sample_ids_D = [
                f"rl-{batch_idx}-{rollout_idx}" for rollout_idx in range(len(prompts_D))
            ]
            generate_params: dict | list[dict] = sampling_params
            if router_replay:
                generate_params = sampling_params_with_sample_ids(
                    sampling_params,
                    sample_ids_D,
                )

            request_id = client.generate(
                job_id,
                prompts=prompts_D,
                sampling_params=generate_params,
            )
            results_D = client.poll_request(job_id, request_id)["results"]
            if len(results_D) != len(prompts_D):
                raise RuntimeError(
                    f"asked for {len(prompts_D)} rollouts, got {len(results_D)} results"
                )

            rewards_P: list[float] = []
            task_metrics_P: dict[str, list[float]] = {}
            datums_D: list[TrainSequence] = []
            trained_sample_ids: list[str] = []
            for problem_idx, (
                prompt_tokens,
                (_, answer),
            ) in enumerate(zip(prompt_tokens_P, batch)):
                group_slice = slice(
                    problem_idx * config.group_size,
                    (problem_idx + 1) * config.group_size,
                )
                group = results_D[group_slice]
                group_sample_ids = sample_ids_D[group_slice]
                scored = [
                    task.score_response(
                        result.get("text") or "",
                        answer,
                        result=result,
                        max_tokens=config.max_tokens,
                    )
                    for result in group
                ]
                rewards_G = [reward for reward, _ in scored]
                mean_reward = sum(rewards_G) / len(rewards_G)
                rewards_P.append(mean_reward)
                group_metrics: dict[str, list[float]] = {}
                for _, item in scored:
                    for name, value in item.items():
                        group_metrics.setdefault(name, []).append(value)
                for name, values in group_metrics.items():
                    task_metrics_P.setdefault(name, []).append(
                        sum(values) / len(values)
                    )
                advantages_G = [reward - mean_reward for reward in rewards_G]

                if config.remove_constant_reward_groups and all(
                    advantage == 0.0 for advantage in advantages_G
                ):
                    continue

                for result, advantage, sample_id in zip(
                    group,
                    advantages_G,
                    group_sample_ids,
                ):
                    sampled_tokens = [
                        int(token) for token in (result.get("token_ids") or [])
                    ]
                    if not sampled_tokens:
                        continue
                    datums_D.append(
                        sequence_from_rollout(
                            prompt_tokens,
                            sampled_tokens,
                            advantage=advantage,
                        )
                    )
                    trained_sample_ids.append(sample_id)

            train_loss = float("nan")
            if not datums_D:
                logger.warning(
                    "Batch %d: no rollouts to train on, skipping the optimizer step",
                    batch_idx,
                )
            else:
                kwargs, context = collate(
                    datums_D,
                    pad_token_id=pad_token_id,
                    max_seq_len=max_seq_len,
                    temperature=config.temperature,
                )
                fwd_bwd_result, step_result = forward_backward_step(
                    client,
                    job_id,
                    kwargs,
                    context=context,
                    learning_rate=learning_rate,
                    processing=processing_block(
                        config,
                        global_batch_size=len(datums_D),
                    ),
                    rr_sample_ids=(trained_sample_ids if router_replay else None),
                    router_replay_sampling_job_id=sampling_job_id,
                )
                train_loss = float(fwd_bwd_result["avg_loss"])
                metrics.update(fwd_bwd_result.get("metrics") or {})
                metrics.update(step_result.get("metrics") or {})
                sync_weights(
                    client,
                    job_id,
                    weight_format="lora" if lora_rank > 0 else None,
                    bucket_size=config.weight_sync_bucket_size,
                )

            if router_replay:
                trained_id_set = set(trained_sample_ids)
                unused_sample_ids = [
                    sample_id
                    for sample_id in sample_ids_D
                    if sample_id not in trained_id_set
                ]
                if unused_sample_ids:
                    discard_router_replay(
                        client,
                        job_id,
                        unused_sample_ids,
                    )

            metrics.update(
                {
                    "reward/mean": sum(rewards_P) / len(rewards_P),
                    "reward/std": (
                        statistics.pstdev(rewards_P) if len(rewards_P) > 1 else 0.0
                    ),
                    "rollouts/total": len(results_D),
                    "rollouts/trained": len(datums_D),
                    "train/avg_loss": train_loss,
                    "time/total": time.time() - t_start,
                }
            )
            metrics.update(
                {
                    f"env/all/{name}": sum(values) / len(values)
                    for name, values in task_metrics_P.items()
                }
            )
            ml_logger.log_metrics(metrics, step=batch_idx)

        saved = save_recipe_checkpoints(client, job_id)
        log_saved_checkpoints(
            config_path=config.config,
            job_id=job_id,
            saved=saved,
            sampling_command="evaluate",
            job_config=Path(config.job_config).name,
            temperature=(
                config.temperature
                if config.eval_temperature is None
                else config.eval_temperature
            ),
            max_tokens=config.eval_max_tokens or config.max_tokens,
            top_p=config.top_p,
            max_examples=config.n_test,
        )

    ml_logger.close()
