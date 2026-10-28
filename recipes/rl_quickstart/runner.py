# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reusable GRPO workflow runner."""

from __future__ import annotations

import json
import logging
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
from recipes.utils import build_job
from recipes.utils import forward_backward_step
from recipes.utils import inference_job_args
from recipes.utils import log_saved_checkpoints
from recipes.utils import model_slug
from recipes.utils import make_client
from recipes.utils import example_connection_args
from recipes.utils import router_replay_stop_params
from recipes.utils import running_job
from recipes.utils import sampling_params_with_sample_ids
from recipes.utils import save_example_checkpoints
from recipes.utils import sequence_from_rollout
from recipes.utils import stop_params_for
from recipes.utils import sync_weights

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)
logging.getLogger("urllib3").setLevel(logging.WARN)
logging.getLogger("tinker_cookbook.renderers.base").setLevel(logging.ERROR)

_EXAMPLE_DIR = Path(__file__).resolve().parent
# Each run writes the job it built here (the folder is not committed).
CONFIGS_DIR = _EXAMPLE_DIR / "configs"


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
    weight_sync_format: str | None
    weight_sync_bucket_size: int | None


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
    """The RL job: the catalog's training and inference files, rl/overrides.json,
    then ``config``, with router replay filled in using the fixed default caps
    for an MoE model. A ``job_config`` file is used as is instead, plus
    ``config``'s settings.
    """
    return build_job(
        "rl",
        model=config.model,
        context=config.context,
        hardware=config.hardware,
        lora=config.lora,
        job_config=config.job_config,
        example_dir=_EXAMPLE_DIR,
        search_dirs=(_EXAMPLE_DIR, CONFIGS_DIR),
        lr=config.lr,
        train_batch_size=config.train_batch_size,
        max_seq_len=config.max_seq_len,
        rollout_batch_size=config.problems_per_batch * config.group_size,
        debug_image_tag=config.debug_image_tag,
    )


def job_record_path(config: Config) -> Path:
    """Where a run records the job it built: ``<slug>-<context>-<hw>-<lora|full>.json``."""
    mode = "lora" if config.lora else "full"
    return CONFIGS_DIR / f"{model_slug(config.model)}-{config.context}-{config.hardware.lower()}-{mode}.json"


def write_job_record(config: Config, body: dict) -> Path:
    """Write the job this run generated, over the previous record for the same selection.

    ``train`` calls it before the job is submitted, so the file is the body the
    run built -- a reproducible starting point whose router-replay caps match
    this run's settings, not whatever last reached ``running``. It keeps the
    debug image tag, but passing the file back with ``job_config=`` does not
    re-enable debug on its own: loading a body never opens the internal-only
    gate, so to replay on that image you re-authorize debug explicitly (pass
    ``debug_image_tag=`` again, or set ``CORTEX_TRAINING_ENABLE_DEBUG_OPTIONS``
    yourself). The folder is gitignored, so the tag is never committed.
    """
    path = job_record_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote this run's job to %s; edit it and pass it back with job_config=%s", path, path)
    return path


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


def should_eval(step: int, total_steps: int, eval_every: int) -> bool:
    return eval_every > 0 and (step % eval_every == 0 or step == total_steps - 1)


def train(config: Config, task: Task) -> None:
    body = job_body(config)
    # Record the body this run generated, before submitting it. A job_config run
    # uses its file as is, and an attach submits nothing, so neither writes.
    if config.job_config is None and config.job_id is None:
        write_job_record(config, body)
    subs = {sub.get("job_type"): sub for sub in body.get("sub_job_configs") or ()}
    training_sub = subs.get("training") or {}
    sampling_sub = subs.get("sampling") or {}
    training = training_sub.get("training_config") or {}
    sampling = sampling_sub.get("inference_config") or {}
    learning_rate = float((training.get("optimizer") or {}).get("lr"))
    # Rollouts are collated at the training window, so a larger sampling window
    # would admit a rollout that overflows it at collation (build_job also rejects
    # this; the guard here also covers a body reached another way).
    training_window = training.get("max_seq_len")
    sampling_window = sampling.get("max_seq_len")
    if isinstance(training_window, int) and isinstance(sampling_window, int) and sampling_window > training_window:
        raise ValueError(
            f"the RL job's sampling max_seq_len {sampling_window} exceeds its training max_seq_len "
            f"{training_window}; lower the sampling window to at most the training window"
        )
    max_seq_len = int(training_window or sampling_window)
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

    client = make_client(config.config, config.connection)

    with running_job(client, body, job_id=config.job_id) as job_id:
        if config.job_id is not None:
            logger.info("Attached to job %s, so this run writes no job record", job_id)
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

            if evaluator is not None and should_eval(
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
                    weight_format="lora" if lora_rank > 0 else config.weight_sync_format,
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

        saved = save_example_checkpoints(client, job_id)
        log_saved_checkpoints(
            connection_args=example_connection_args(config.config, config.connection),
            job_id=job_id,
            saved=saved,
            sampling_command="evaluate",
            job_args=inference_job_args(
                body,
                context=config.context if config.job_config is None else None,
            ),
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
    logger.info("Training completed")
