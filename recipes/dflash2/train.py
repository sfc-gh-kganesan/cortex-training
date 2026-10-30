# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Online DFlash2 draft training on Cortex Training.

The target model's hidden states are computed inside the training sub-job, so this client ships only
token ids and a supervision mask. It owns the data pipeline and the two per-step schedules (learning
rate, selector weight), saves the draft as a weights-only checkpoint in the serving layout, and reports
the draft's accepted-length diagnostics. With ``eval_prompts=`` it then serves the final draft with its
target and measures the served acceptance length (see ``recipes.dflash2.evaluate``).
"""

from __future__ import annotations

import logging
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import chz
from recipes.dflash2.metrics import SELECTOR_ACCEPTED_LENGTH
from recipes.dflash2.metrics import UNARY_ACCEPTED_LENGTH
from recipes.dflash2.metrics import summarize
from recipes.dflash2.schedule import learning_rate
from recipes.dflash2.schedule import selector_loss_alpha
from recipes.dflash2.task import TokenizedConversation
from recipes.dflash2.task import collate
from recipes.dflash2.task import iter_jsonl
from recipes.dflash2.task import tokenize_conversation
from recipes.logging import setup_logging
from recipes.utils import build_job
from recipes.utils import forward_backward
from recipes.utils import make_client
from recipes.utils import model_slug
from recipes.utils import optimizer_step
from recipes.utils import running_job
from recipes.utils import save_checkpoint

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)
logging.getLogger("urllib3").setLevel(logging.WARN)

EXAMPLE_DIR = Path(__file__).resolve().parent
DRAFTS_DIR = EXAMPLE_DIR / "drafts"


@chz.chz
class Config:
    # JSONL of conversations answered by the target model itself (`conversations` or `messages`).
    data_path: str
    # Legacy JSON connection file. Omit it to use `connection` or the default profile.
    config: str | None = None
    # Snowflake connection profile name (connections.toml). Use config or connection, not both.
    connection: str | None = None
    job_id: str | None = None

    # The target model. The job is its catalog training file, then overrides.json, then
    # drafts/<model>.json; job_config= replaces all three with a file.
    model: str = "Qwen/Qwen3.5-4B"
    context: str = "standard"
    hardware: str = "h200"
    job_config: str | None = None
    # None keeps the job's value. lr is the peak of the per-step schedule.
    lr: float | None = None
    train_batch_size: int | None = None
    max_seq_len: int | None = None

    epochs: float = 1.0
    max_steps: int | None = None
    n_train: int | None = None
    max_length: int | None = None
    seed: int = 42

    warmup_ratio: float = 0.04
    lr_scheduler: str = "cosine"
    selector_loss_alpha: float = 1.0
    selector_warmup_ratio: float = 0.0
    selector_ramp_ratio: float = 0.0

    # The accepted-length and recall diagnostics cost an extra vocabulary projection on the server; they are computed
    # on the first step, the last step, and every N steps. Loss, CE and accuracy are reported every step.
    detailed_metrics_every: int = 10
    save_every: int = 0
    tokenize_workers: int = 16

    # After training, serve the final draft and measure its acceptance length on these prompts (JSONL).
    eval_prompts: str | None = None
    eval_num_prompts: int = 128
    eval_max_tokens: int = 1024

    debug_image_tag: str | None = None
    log_path: str = "/tmp/cortex-training-examples/dflash2"
    wandb_project: str | None = None
    wandb_name: str | None = None
    sf_tracking: bool = False


def draft_overrides(model: str) -> Path:
    """``drafts/<model>.json``: the draft architecture and the size of the run for one target model."""
    path = DRAFTS_DIR / f"{model_slug(model)}.json"
    if not path.is_file():
        known = sorted(p.stem for p in DRAFTS_DIR.glob("*.json"))
        raise ValueError(f"no DFlash2 draft for {model}; drafts/ has {known}. Add drafts/{path.name} or pass job_config=")
    return path


def job_body(config: Config) -> dict:
    """The training job: the catalog file, overrides.json, drafts/<model>.json, then ``config``."""
    return build_job(
        "training",
        model=config.model,
        context=config.context,
        hardware=config.hardware,
        job_config=config.job_config,
        example_dir=EXAMPLE_DIR,
        overrides=[] if config.job_config else [draft_overrides(config.model)],
        search_dirs=(EXAMPLE_DIR,),
        lr=config.lr,
        train_batch_size=config.train_batch_size,
        max_seq_len=config.max_seq_len,
        debug_image_tag=config.debug_image_tag,
    )


def training_sub_job(body: dict) -> dict:
    for sub in body.get("sub_job_configs") or ():
        if sub.get("job_type") == "training":
            return sub
    raise ValueError("job config has no training sub-job")


def load_sequences(config: Config, tokenizer: Any, max_length: int) -> list[TokenizedConversation]:
    rows = [r for r in iter_jsonl(config.data_path, limit=config.n_train) if r.get("status", "success") == "success"]

    def encode(row: dict) -> TokenizedConversation | None:
        conversation = row.get("conversations") or row.get("messages")
        if not conversation:
            return None
        seq = tokenize_conversation(tokenizer, conversation, max_length)
        return seq if seq.has_anchor() else None

    with ThreadPoolExecutor(max_workers=max(1, config.tokenize_workers)) as pool:
        sequences = [s for s in pool.map(encode, rows) if s is not None]
    logger.info(
        "tokenized %d/%d conversations (mean %.0f tokens, %.0f supervised)",
        len(sequences),
        len(rows),
        sum(len(s.input_ids) for s in sequences) / max(len(sequences), 1),
        sum(s.num_supervised for s in sequences) / max(len(sequences), 1),
    )
    return sequences


def save_draft(client: Any, job_id: str) -> dict[str, Any]:
    """Save a weights-only checkpoint: the draft's ``config.json`` + ``model.safetensors``."""
    saved = save_checkpoint(client, job_id, checkpoint_type="weights-only")
    if not saved.get("checkpoint_id"):
        raise RuntimeError(f"the save of job {job_id} returned no checkpoint id: {saved}")
    return saved


def train(config: Config) -> dict[str, Any]:
    # setup_logging replaces these handlers once the job runs; until then this
    # keeps the created job id in the log, so an interrupted start can be cancelled.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    body = job_body(config)
    sub = training_sub_job(body)
    training = sub["training_config"]
    batch_size = int(training["train_batch_size"])
    max_seq_len = int(training["max_seq_len"])
    max_length = config.max_length or max_seq_len
    if max_length > max_seq_len:
        raise ValueError(f"max_length={max_length} exceeds the job's max_seq_len={max_seq_len}")
    base_lr = float(training["optimizer"]["lr"])

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(sub["model_name"])
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    sequences = load_sequences(config, tokenizer, max_length)
    steps_per_epoch = len(sequences) // batch_size
    if steps_per_epoch == 0:
        raise ValueError(f"{len(sequences)} usable conversations is fewer than train_batch_size={batch_size}")
    total_steps = config.max_steps or max(1, int(math.floor(config.epochs * steps_per_epoch)))
    logger.info("training %d steps (%d per epoch), batch %d, peak lr %g", total_steps, steps_per_epoch, batch_size, base_lr)

    rng = random.Random(config.seed)
    order: list[int] = []

    def next_batch() -> list[TokenizedConversation]:
        # One shuffled pass per epoch; the final partial batch of a pass is dropped, so a step never repeats a row.
        nonlocal order
        if len(order) < batch_size:
            order = list(range(len(sequences)))
            rng.shuffle(order)
        picked, order = order[:batch_size], order[batch_size:]
        return [sequences[i] for i in picked]

    client = make_client(config.config, config.connection)
    saved: dict[str, Any] = {}
    with running_job(client, body, job_id=config.job_id) as job_id:
        ml_logger = setup_logging(config, client=client, job_id=job_id)
        for step in range(total_steps):
            started = time.time()
            lr = learning_rate(
                step, base_lr=base_lr, total_steps=total_steps, warmup_ratio=config.warmup_ratio, scheduler=config.lr_scheduler
            )
            alpha = selector_loss_alpha(
                step,
                target=config.selector_loss_alpha,
                total_steps=total_steps,
                warmup_ratio=config.selector_warmup_ratio,
                ramp_ratio=config.selector_ramp_ratio,
            )
            kwargs, context = collate(next_batch(), pad_token_id, max_seq_len)
            detailed = step == 0 or step == total_steps - 1 or (config.detailed_metrics_every > 0 and step % config.detailed_metrics_every == 0)
            processing = {"loss_fn": "dflash2", "config": {"selector_loss_alpha": alpha, "detailed_metrics": detailed}}
            fwd_bwd = forward_backward(client, job_id, kwargs, context=context, processing=processing)
            step_result = optimizer_step(client, job_id, learning_rate=lr)

            metrics: dict[str, float] = {
                "loss": float(fwd_bwd["avg_loss"]),
                "lr": lr,
                "selector_alpha": alpha,
                "progress": step / total_steps,
                "epoch": step / steps_per_epoch,
                "time_step": time.time() - started,
            }
            metrics.update(summarize(fwd_bwd.get("metrics") or {}))
            for key, value in (step_result.get("metrics") or {}).items():
                metrics[f"optim/{key}"] = value
            ml_logger.log_metrics(metrics=metrics, step=step)
            if step % 10 == 0 or step == total_steps - 1:
                # The accepted lengths are only computed on the steps that collect detailed metrics.
                unary, selector = metrics.get(UNARY_ACCEPTED_LENGTH), metrics.get(SELECTOR_ACCEPTED_LENGTH)
                logger.info(
                    "step %d/%d loss %.4f accLen(unary) %s accLen(selector) %s lr %.2e (%.1fs)",
                    step + 1,
                    total_steps,
                    metrics["loss"],
                    "-" if unary is None else f"{unary:.3f}",
                    "-" if selector is None else f"{selector:.3f}",
                    lr,
                    metrics["time_step"],
                )
            if config.save_every and (step + 1) % config.save_every == 0 and step + 1 < total_steps:
                saved = save_draft(client, job_id)
                logger.info("saved step %d: draft_job_id=%s draft_checkpoint_id=%s", step + 1, job_id, saved["checkpoint_id"])

        saved = save_draft(client, job_id)
        logger.info("saved final draft: draft_job_id=%s draft_checkpoint_id=%s", job_id, saved["checkpoint_id"])
    ml_logger.close()
    result: dict[str, Any] = {"job_id": job_id, "checkpoint": saved}

    if config.eval_prompts:
        # After the training job has released its GPUs: the endpoint is a separate job.
        from recipes.dflash2.evaluate import evaluate_draft

        result["acceptance"] = evaluate_draft(
            client,
            model=sub["model_name"],
            context=config.context,
            hardware=config.hardware,
            draft_job_id=job_id,
            draft_checkpoint_id=saved["checkpoint_id"],
            prompts=config.eval_prompts,
            num_prompts=config.eval_num_prompts,
            max_tokens=config.eval_max_tokens,
            debug_image_tag=config.debug_image_tag,
        )
    return result


def main(config: Config) -> dict[str, Any]:
    return train(config)


if __name__ == "__main__":
    chz.nested_entrypoint(main)
