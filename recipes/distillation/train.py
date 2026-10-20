# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reusable on-policy distillation runner.

The student (training + sampling sub-jobs) generates rollouts; a frozen teacher
(a separate sampling-only job) scores those exact tokens; the student trains on
the ``grpo`` loss with its teacher term enabled, which adds
``teacher_tau * clamp(log p_teacher - log p_student)`` to every token's
advantage. With ``reward_coef=0`` that teacher term is the only signal (pure
distillation); a positive ``reward_coef`` also adds a group-centred task reward,
and ``teacher_tau=0`` turns the teacher off for an RL-only comparison.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import statistics
import time
from collections.abc import Iterator
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from typing import Protocol

from recipes.distillation.teacher import check_shared_tokenizer
from recipes.distillation.teacher import clipped_fraction
from recipes.distillation.teacher import reverse_kl_estimates
from recipes.distillation.teacher import sampled_token_logprobs
from recipes.distillation.teacher import score_with_teacher
from recipes.distillation.teacher import teacher_log_probs_shifted
from recipes.logging import setup_logging
from recipes.utils import build_renderer
from recipes.utils import collate
from recipes.utils import forward_backward
from recipes.utils import load_job_body
from recipes.utils import make_client
from recipes.utils import optimizer_step
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
_LR_SCHEDULES = ("constant", "cosine")


class Config(Protocol):
    config: str
    job_id: str | None
    teacher_job_id: str | None
    job_config: str
    teacher_job_config: str
    renderer_name: str | None
    problems_per_batch: int
    group_size: int
    max_tokens: int
    temperature: float
    top_p: float
    max_steps: int | None
    warmup_steps: int
    lr_schedule: str
    min_lr_ratio: float
    teacher_tau: float
    teacher_clip: float
    teacher_clip_negative: float | None
    reward_coef: float
    eps_clip: float
    loss_agg_mode: str
    entropy_coeff: float
    eval_every: int
    n_val: int
    n_test: int | None
    eval_max_tokens: int | None
    n_kl_eval: int
    evaluate_teacher: bool
    seed: int
    weight_sync_bucket_size: int | None
    debug_image_tag: str | None


class Task(Protocol):
    name: str
    evaluation_name: str

    def load(self, *, seed: int, n_val: int): ...

    def build_prompt(self, problem: Any, renderer) -> list[int]: ...

    def score_response(
        self, response: str, answer: Any
    ) -> tuple[float, dict[str, float]]: ...


def validate_config(config: Config) -> None:
    if not math.isfinite(config.teacher_tau) or config.teacher_tau < 0:
        raise ValueError(
            f"teacher_tau must be a finite non-negative number, got {config.teacher_tau}"
        )
    if config.teacher_tau > 0 and not (
        math.isfinite(config.teacher_clip) and config.teacher_clip > 0
    ):
        raise ValueError(
            f"teacher_tau > 0 needs a finite positive teacher_clip, got {config.teacher_clip}"
        )
    if config.teacher_tau == 0 and config.reward_coef == 0:
        raise ValueError(
            "teacher_tau=0 and reward_coef=0 leave no training signal; set one of them"
        )
    if config.lr_schedule not in _LR_SCHEDULES:
        raise ValueError(
            f"lr_schedule must be one of {_LR_SCHEDULES}, got {config.lr_schedule!r}"
        )
    if config.group_size < 1 or config.problems_per_batch < 1:
        raise ValueError("problems_per_batch and group_size must be positive")


def _with_debug(body: dict, config: Config) -> dict:
    if config.debug_image_tag:
        body["debug"] = {"job": {"image_tag": config.debug_image_tag}}
    return body


def job_bodies(config: Config) -> tuple[dict, dict | None]:
    """Student job body, and the teacher job body unless ``teacher_tau`` is 0."""
    student = _with_debug(
        load_job_body(config.job_config, search_dirs=_CONFIG_SEARCH_DIRS), config
    )
    if config.teacher_tau == 0:
        return student, None
    teacher = _with_debug(
        load_job_body(config.teacher_job_config, search_dirs=_CONFIG_SEARCH_DIRS),
        config,
    )
    if any(
        sub.get("job_type") != "sampling"
        for sub in teacher.get("sub_job_configs") or ()
    ):
        raise ValueError(
            f"{config.teacher_job_config} must contain only sampling sub-jobs"
        )
    return student, teacher


def processing_block(config: Config, global_batch_size: int) -> dict:
    loss_config: dict[str, Any] = {
        "eps_clip": config.eps_clip,
        "loss_agg_mode": config.loss_agg_mode,
        "entropy_coeff": config.entropy_coeff,
        "global_batch_size": global_batch_size,
    }
    if config.teacher_tau > 0:
        loss_config["teacher_tau"] = config.teacher_tau
        loss_config["teacher_clip"] = config.teacher_clip
        if config.teacher_clip_negative is not None:
            loss_config["teacher_clip_negative"] = config.teacher_clip_negative
    return {"loss_fn": "grpo", "post": ["compute_logprobs"], "config": loss_config}


def lr_at(step: int, peak_lr: float, config: Config, total_steps: int) -> float:
    """Linear warmup, then constant or cosine decay to ``min_lr_ratio * peak_lr``."""
    if config.warmup_steps > 0 and step < config.warmup_steps:
        return peak_lr * (step + 1) / config.warmup_steps
    if config.lr_schedule == "constant":
        return peak_lr
    span = max(1, total_steps - config.warmup_steps)
    frac = min(1.0, max(0.0, (step - config.warmup_steps) / span))
    floor = config.min_lr_ratio
    return peak_lr * (floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * frac)))


def _sub_job(body: dict, job_type: str) -> dict:
    for sub in body.get("sub_job_configs") or ():
        if sub.get("job_type") == job_type:
            return sub
    raise ValueError(f"job config has no {job_type} sub-job")


@contextlib.contextmanager
def running_jobs(
    client: Any, bodies: dict[str, dict], attach: dict[str, str | None]
) -> Iterator[dict[str, str]]:
    """Create (or attach to) every job, wait for all, and cancel the ones created here."""
    job_ids: dict[str, str] = {}
    created: list[str] = []
    try:
        for name, body in bodies.items():
            if attach.get(name):
                job_ids[name] = str(attach[name])
                logger.info("attaching to %s job %s", name, job_ids[name])
            else:
                job_ids[name] = client.create_job_from_body(body)["job_id"]
                created.append(job_ids[name])
                logger.info("created %s job %s", name, job_ids[name])
        for name, job_id in job_ids.items():
            client.wait_for_job(job_id)
            logger.info("%s job %s is running", name, job_id)
        yield job_ids
    finally:
        for job_id in reversed(created):
            try:
                client.cancel_job(job_id)
                logger.info("cancelled job %s", job_id)
            except Exception:
                logger.exception("failed to cancel job %s -- cancel it by hand", job_id)


def _generate(
    client: Any, job_id: str, prompts: Sequence[Sequence[int]], params: dict
) -> list[dict]:
    request_id = client.generate(
        job_id, prompts=[list(p) for p in prompts], sampling_params=params
    )
    results = client.poll_request(job_id, request_id)["results"]
    if len(results) != len(prompts):
        raise RuntimeError(
            f"asked for {len(prompts)} completions, got {len(results)} results"
        )
    return results


def _accuracy(
    client: Any, job_id: str, task: Task, prompts, answers, params: dict
) -> dict[str, float]:
    results = _generate(client, job_id, prompts, params)
    scored = [
        task.score_response(result.get("text") or "", answer)
        for result, answer in zip(results, answers)
    ]
    metrics = {"correct": statistics.fmean(reward for reward, _ in scored)}
    for name in scored[0][1]:
        metrics[name] = statistics.fmean(item[name] for _, item in scored)
    metrics["mean_tokens"] = statistics.fmean(
        len(result.get("token_ids") or []) for result in results
    )
    metrics["frac_truncated"] = statistics.fmean(
        result.get("finish_reason") == "length" for result in results
    )
    return metrics


def _token_ids(result: dict) -> list[int]:
    return [int(token) for token in result.get("token_ids") or []]


def _train(config: Config, task: Task) -> None:
    validate_config(config)
    student_body, teacher_body = job_bodies(config)
    training_sub = _sub_job(student_body, "training")
    sampling_sub = _sub_job(student_body, "sampling")
    training = training_sub.get("training_config") or {}
    sampling = sampling_sub.get("inference_config") or {}
    peak_lr = float((training.get("optimizer") or {}).get("lr"))
    max_seq_len = int(training.get("max_seq_len") or sampling.get("max_seq_len"))
    student_model = training_sub["model_name"]

    tokenizer, renderer, renderer_name = build_renderer(
        student_model, renderer_name=config.renderer_name
    )
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    logger.info("student %s, renderer %s", student_model, renderer_name)
    if teacher_body is not None:
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        teacher_sampling = _sub_job(teacher_body, "sampling")
        teacher_model = teacher_sampling["model_name"]
        check_shared_tokenizer(tokenizer, get_tokenizer(teacher_model))
        teacher_window = int(
            (teacher_sampling.get("inference_config") or {}).get("max_seq_len") or 0
        )
        if teacher_window <= max_seq_len:
            raise ValueError(
                f"teacher max_seq_len {teacher_window} must exceed the student's "
                f"{max_seq_len}: the teacher scores prompt and completion together, "
                "and generate rejects prompts at max_seq_len"
            )
        logger.info(
            "teacher %s, teacher_tau=%s, teacher_clip=%s",
            teacher_model,
            config.teacher_tau,
            config.teacher_clip,
        )

    max_prompt_len = max_seq_len - config.max_tokens
    if max_prompt_len < 1:
        raise ValueError(
            f"max_tokens {config.max_tokens} leaves no room for a prompt in "
            f"max_seq_len {max_seq_len}"
        )
    problems = task.load(seed=config.seed, n_val=config.n_val)
    train_problems = []
    for question, answer in problems.train:
        prompt = task.build_prompt(question, renderer)
        if len(prompt) <= max_prompt_len:
            train_problems.append((prompt, answer))
    if not train_problems:
        raise ValueError(
            f"no {task.name} training prompt fits in {max_prompt_len} tokens"
        )
    logger.info(
        "%s: %d training problems fit in %d prompt tokens",
        task.name,
        len(train_problems),
        max_prompt_len,
    )

    eval_max_tokens = config.eval_max_tokens or config.max_tokens
    # Held-out KL samples test prompts with max_tokens; accuracy uses eval_max_tokens.
    max_eval_prompt_len = max_seq_len - max(eval_max_tokens, config.max_tokens)
    if max_eval_prompt_len < 1:
        raise ValueError(
            f"eval_max_tokens {eval_max_tokens} leaves no room for a prompt in "
            f"max_seq_len {max_seq_len}"
        )

    def eval_split(
        split: Sequence[tuple[Any, Any]], limit: int | None
    ) -> tuple[list[list[int]], list[Any]]:
        split = list(split)[:limit] if limit else list(split)
        prompts = [task.build_prompt(question, renderer) for question, _ in split]
        too_long = [
            len(prompt) for prompt in prompts if len(prompt) > max_eval_prompt_len
        ]
        if too_long:
            raise ValueError(
                f"{len(too_long)} evaluation prompts exceed {max_eval_prompt_len} tokens"
            )
        return prompts, [answer for _, answer in split]

    val_prompts, val_answers = eval_split(problems.val, None)
    test_prompts, test_answers = eval_split(problems.test, config.n_test)
    kl_prompts = test_prompts[: config.n_kl_eval]

    batches_available = len(train_problems) // config.problems_per_batch
    total_steps = (
        batches_available
        if config.max_steps is None
        else min(batches_available, config.max_steps)
    )
    stop_params = stop_params_for(renderer.get_stop_sequences())
    rollout_params = {
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "logprobs": 0,
        **stop_params,
    }
    eval_params = {
        "max_tokens": eval_max_tokens,
        "temperature": 0.0,
        **stop_params,
    }

    client = make_client(config.config)
    bodies = {"student": student_body}
    if teacher_body is not None:
        bodies["teacher"] = teacher_body
    with running_jobs(
        client, bodies, {"student": config.job_id, "teacher": config.teacher_job_id}
    ) as job_ids:
        student_job = job_ids["student"]
        teacher_job = job_ids.get("teacher")
        ml_logger = setup_logging(config, client=client, job_id=student_job)
        history: list[dict[str, float]] = []

        def evaluate(step: int) -> dict[str, float]:
            start = time.time()
            metrics: dict[str, float] = {}
            if val_prompts:
                val = _accuracy(
                    client, student_job, task, val_prompts, val_answers, eval_params
                )
                metrics.update({f"val/{name}": value for name, value in val.items()})
            test = _accuracy(
                client, student_job, task, test_prompts, test_answers, eval_params
            )
            metrics.update({f"test/{name}": value for name, value in test.items()})
            if step == 0 and teacher_job is not None and config.evaluate_teacher:
                teacher = _accuracy(
                    client, teacher_job, task, test_prompts, test_answers, eval_params
                )
                metrics.update(
                    {f"teacher/test/{name}": value for name, value in teacher.items()}
                )
            if teacher_job is not None and kl_prompts:
                results = _generate(client, student_job, kl_prompts, rollout_params)
                kept = [(p, r) for p, r in zip(kl_prompts, results) if _token_ids(r)]
                if kept:
                    student_lps = [sampled_token_logprobs(r) for _, r in kept]
                    teacher_lps = score_with_teacher(
                        client,
                        teacher_job,
                        [p for p, _ in kept],
                        [_token_ids(r) for _, r in kept],
                    )
                    kl = reverse_kl_estimates(student_lps, teacher_lps)
                    metrics.update(
                        {f"heldout/{name}": value for name, value in kl.items()}
                    )
            metrics["time/eval"] = time.time() - start
            logger.info(
                "step %d: %s accuracy %.4f%s",
                step,
                task.evaluation_name,
                metrics["test/correct"],
                (
                    f", validation {metrics['val/correct']:.4f}"
                    if "val/correct" in metrics
                    else ""
                ),
            )
            history.append({"step": step, **metrics})
            return metrics

        cursor = 0
        for step in range(total_steps):
            start = time.time()
            metrics: dict[str, float] = {
                "progress/step": step,
                "progress/done_frac": (step + 1) / max(total_steps, 1),
            }
            if config.eval_every > 0 and step % config.eval_every == 0:
                metrics.update(evaluate(step))

            batch = train_problems[cursor : cursor + config.problems_per_batch]
            cursor += config.problems_per_batch
            prompts = [prompt for prompt, _ in batch for _ in range(config.group_size)]
            answers = [answer for _, answer in batch for _ in range(config.group_size)]
            results = _generate(client, student_job, prompts, rollout_params)
            scored = [
                task.score_response(result.get("text") or "", answer)
                for result, answer in zip(results, answers)
            ]

            rollouts: list[tuple[list[int], list[int], float]] = []
            for group_start in range(0, len(results), config.group_size):
                group = range(group_start, group_start + config.group_size)
                mean_reward = statistics.fmean(scored[i][0] for i in group)
                for i in group:
                    tokens = _token_ids(results[i])
                    if tokens:
                        rollouts.append(
                            (
                                prompts[i],
                                tokens,
                                config.reward_coef * (scored[i][0] - mean_reward),
                            )
                        )
            if not rollouts:
                logger.warning(
                    "step %d: every rollout was empty, skipping the update", step
                )
                ml_logger.log_metrics(metrics, step=step)
                continue

            sequences = [
                sequence_from_rollout(prompt, tokens, advantage=adv)
                for prompt, tokens, adv in rollouts
            ]
            kwargs, context = collate(
                sequences,
                pad_token_id=pad_token_id,
                max_seq_len=max_seq_len,
                temperature=config.temperature,
            )
            n_tokens = sum(len(tokens) for _, tokens, _ in rollouts)
            if teacher_job is not None:
                score_start = time.time()
                completions = [tokens for _, tokens, _ in rollouts]
                teacher_lps = score_with_teacher(
                    client, teacher_job, [p for p, _, _ in rollouts], completions
                )
                context["teacher_log_probs_shifted"] = teacher_log_probs_shifted(
                    context["loss_mask"], [len(p) for p, _, _ in rollouts], teacher_lps
                )
                student_lps = [
                    sampled_token_logprobs(results[i])
                    for i in range(len(results))
                    if _token_ids(results[i])
                ]
                metrics.update(
                    {
                        f"train/{name}": value
                        for name, value in reverse_kl_estimates(
                            student_lps, teacher_lps
                        ).items()
                    }
                )
                fraction = clipped_fraction(
                    student_lps,
                    teacher_lps,
                    clip=config.teacher_clip,
                    clip_negative=(
                        config.teacher_clip
                        if config.teacher_clip_negative is None
                        else config.teacher_clip_negative
                    ),
                )
                if fraction is not None:
                    metrics["teacher/clipped_fraction"] = fraction
                metrics["time/teacher_score"] = time.time() - score_start

            learning_rate = lr_at(step, peak_lr, config, total_steps)
            fwd_bwd_result = forward_backward(
                client,
                student_job,
                kwargs,
                context=context,
                processing=processing_block(config, global_batch_size=len(sequences)),
            )
            server = fwd_bwd_result.get("metrics") or {}
            if teacher_job is not None:
                covered = float(server.get("teacher_term_token_count") or 0.0)
                if covered != n_tokens:
                    raise RuntimeError(
                        f"step {step}: the teacher term covered {covered} tokens, "
                        f"expected {n_tokens}; stopped before the optimizer step"
                    )
            step_result = optimizer_step(
                client, student_job, learning_rate=learning_rate
            )
            metrics.update(server)
            metrics.update(step_result.get("metrics") or {})
            if step_result.get("grad_norm") is not None:
                metrics["optim/grad_norm"] = float(step_result["grad_norm"])
            sync_weights(
                client, student_job, bucket_size=config.weight_sync_bucket_size
            )

            metrics.update(
                {
                    "optim/lr": learning_rate,
                    "train/avg_loss": float(
                        fwd_bwd_result.get("avg_loss", float("nan"))
                    ),
                    "rollout/correct": statistics.fmean(reward for reward, _ in scored),
                    "rollout/mean_tokens": n_tokens / len(rollouts),
                    "rollout/frac_truncated": statistics.fmean(
                        r.get("finish_reason") == "length" for r in results
                    ),
                    "rollouts/trained": len(rollouts),
                    "time/total": time.time() - start,
                }
            )
            ml_logger.log_metrics(metrics, step=step)

        if config.eval_every > 0:
            ml_logger.log_metrics(evaluate(total_steps), step=total_steps)
            with_val = [row for row in history if "val/correct" in row]
            if with_val:
                best = max(with_val, key=lambda row: (row["val/correct"], -row["step"]))
                logger.info(
                    "Best validation accuracy %.4f at step %d, where %s accuracy "
                    "is %.4f (step 0: %.4f, final: %.4f)",
                    best["val/correct"],
                    best["step"],
                    task.evaluation_name,
                    best["test/correct"],
                    history[0]["test/correct"],
                    history[-1]["test/correct"],
                )

        saved = save_recipe_checkpoints(client, student_job)
        logger.info(
            "Saved weights-only checkpoint_id=%s from job %s",
            saved["weights-only"].get("checkpoint_id"),
            student_job,
        )
    ml_logger.close()


def train(config: Config, task: Task) -> None:
    if config.debug_image_tag:
        os.environ[DEBUG_OPTIONS_ENV] = "1"
        logger.info("Using debug image_tag=%s", config.debug_image_tag)
    _train(config, task)
    logger.info("Training completed")
