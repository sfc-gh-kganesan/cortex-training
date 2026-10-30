"""Score a model or checkpoint on the DARE-Bench eval split with a local agent loop.

Each of the 162 eval tasks runs in both variants (``v1`` instruction-following /
exogenous-feature forecasting, ``v2`` ML modeling / canonical forecasting): the
model gets the task prompt and the ``python_executor`` tool, may call it up to
``max_turns`` times with a ``executor_timeout``-second limit per run, and is
scored on the ``prediction.csv`` it leaves behind. Episode setup, tool calls,
dispatch, and scoring are DARE-Bench's own code (``dare_bench/``). The tool
posts code to a Sandbox Fusion ``run_code`` endpoint: ``sandbox_url``, or by
default a local server that runs it in subprocesses under ``sandbox_python``.
Build that environment and the ``v1`` reference predictions as described in
this example's README first.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import chz
from recipes.inference.endpoint import inference_endpoint_body
from recipes.logging import setup_logging
from recipes.sft_quickstart.dare.dare_bench import evaluation
from recipes.sft_quickstart.dare.dare_bench.data import process
from recipes.sft_quickstart.dare.dare_bench.dispatch import dispatch_tool_call
from recipes.sft_quickstart.dare.dare_bench.tools.python_executor.python_executor import python_executor
from recipes.sft_quickstart.dare.sandbox import local_sandbox
from recipes.sft_quickstart.dare.task import EXECUTOR_TIMEOUT_S
from recipes.sft_quickstart.dare.task import MAX_TOOL_TURNS
from recipes.sft_quickstart.dare.task import PREDICTION_FILE
from recipes.sft_quickstart.dare.task import TOOL_NAME
from recipes.sft_quickstart.dare.task import assistant_content
from recipes.sft_quickstart.dare.task import executor_config
from recipes.sft_quickstart.dare.task import split_reasoning
from recipes.sft_quickstart.dare.task import tool_prefix
from recipes.utils import build_renderer
from recipes.utils import make_client
from recipes.utils import running_job
from recipes.utils import stop_params_for

from cortex_training.client import DEBUG_OPTIONS_ENV

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)
logging.getLogger("urllib3").setLevel(logging.WARN)
logging.getLogger("tinker_cookbook.renderers.base").setLevel(logging.ERROR)

_TOKENIZER_LOCK = threading.Lock()

CATEGORIES = {
    ("classification", "v1"): "classification_if",
    ("classification", "v2"): "classification_mm",
    ("regression", "v1"): "regression_if",
    ("regression", "v2"): "regression_mm",
    ("time_series_analysis", "v1"): "time_series_xf",
    ("time_series_analysis", "v2"): "time_series_cf",
}


@chz.chz
class Config:
    # Legacy JSON connection file. Omit it to use `connection` or the default profile.
    config: str | None = None
    # Snowflake connection profile name (connections.toml). Use config or connection, not both.
    connection: str | None = None
    bench_dir: str
    job_id: str | None = None

    # The job comes from model-catalog/models/; job_config= replaces it with a file.
    model: str = "Qwen/Qwen3-8B"
    context: str = "standard"
    hardware: str = "h200"
    lora: bool = False
    job_config: str | None = None
    max_seq_len: int | None = None  # None keeps the job's value

    source_job_id: str | None = None
    checkpoint_id: str | None = None
    keep_job: bool | None = None
    debug_image_tag: str | None = None

    versions: str = "v1,v2"
    task_types: str | None = None
    max_examples: int | None = None
    sandbox_python: str = sys.executable
    sandbox_url: str | None = None
    max_turns: int = MAX_TOOL_TURNS
    executor_timeout: int = EXECUTOR_TIMEOUT_S

    enable_thinking: bool = True
    renderer_name: str | None = None
    max_tokens: int = 16384
    temperature: float = 0.0
    concurrency: int = 96
    sandbox_workers: int = 32

    log_path: str = "/tmp/cortex-training-recipes/dare-eval"
    wandb_project: str | None = None
    wandb_name: str | None = None


def load_episodes(bench_dir: Path, versions: list[str], task_types: set[str] | None) -> list[dict[str, Any]]:
    questions = json.loads((bench_dir / "question_list.json").read_text(encoding="utf-8"))
    episodes = []
    for version in versions:
        for question in questions:
            if task_types and question["task"] not in task_types:
                continue
            (episode,) = process(question, f"question_{version}", str(bench_dir / "databases"))
            episode["version"] = version
            episode["category"] = CATEGORIES[(question["task"], version)]
            episodes.append(episode)
    return episodes


def assistant_turn(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content") or ""
    if isinstance(content, str):
        reasoning, text = split_reasoning(content)
    else:
        thinking = [part["thinking"] for part in content if part.get("type") == "thinking"]
        reasoning = "".join(thinking) if thinking else None
        text = "".join(part["text"] for part in content if part.get("type") == "text")
    calls = list(message.get("tool_calls") or [])
    turn: dict[str, Any] = {
        "role": "assistant",
        "content": assistant_content(reasoning, text, has_tool_calls=bool(calls)),
    }
    if calls:
        turn["tool_calls"] = calls
    return turn


def _jsonable(message: dict[str, Any]) -> dict[str, Any]:
    out = {key: value for key, value in message.items() if key != "tool_calls"}
    if message.get("tool_calls"):
        out["tool_calls"] = [
            {"id": call.id, "name": call.function.name, "arguments": call.function.arguments}
            for call in message["tool_calls"]
        ]
    return out


def run_episode(
    client: Any,
    job_id: str,
    renderer: Any,
    episode: dict[str, Any],
    *,
    max_seq_len: int,
    work_root: Path,
    sampling_params: dict[str, Any],
    end_token: int,
    max_turns: int,
    sandbox: threading.Semaphore,
    sandbox_url: str,
    executor_timeout: int,
) -> dict[str, Any]:
    work_dir = work_root / episode["version"] / episode["id"]
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True)
    for name in episode["metadata"]["needed_files"]:
        shutil.copy2(Path(episode["original_file_path"]) / name, work_dir / name)
    executor = python_executor(config={})
    executor.initialize(
        executor_config(url=sandbox_url, work_dir=work_dir, timeout=executor_timeout, max_turns=max_turns)
    )
    dispatch_map = {TOOL_NAME: executor}

    with _TOKENIZER_LOCK:
        messages: list[dict[str, Any]] = tool_prefix(renderer, timeout=executor_timeout, max_turns=max_turns) + [
            {"role": "user", "content": episode["question"][0]}
        ]
    stop_reason = "max_turns"
    n_tool_calls = 0
    n_completion_tokens = 0
    started = time.time()
    try:
        for _ in range(max_turns):
            with _TOKENIZER_LOCK:
                prompt = [int(token) for token in renderer.build_generation_prompt(messages).to_ints()]
            if len(prompt) >= max_seq_len - 1:
                stop_reason = "context_exhausted"
                break
            request_id = client.generate(job_id, prompts=[prompt], sampling_params=sampling_params)
            result = client.poll_request(job_id, request_id)["results"][0]
            tokens = [int(token) for token in result.get("token_ids") or []]
            n_completion_tokens += len(tokens)
            if result.get("finish_reason") == "stop" and end_token not in tokens:
                tokens.append(end_token)
            with _TOKENIZER_LOCK:
                parsed, _termination = renderer.parse_response(tokens)
            turn = assistant_turn(parsed)
            messages.append(turn)
            if not turn.get("tool_calls"):
                stop_reason = "answered" if result.get("finish_reason") == "stop" else "truncated"
                break
            for call in turn["tool_calls"]:
                n_tool_calls += 1
                with sandbox:
                    output, _ = dispatch_tool_call(call, None, dispatch_map, {TOOL_NAME})
                messages.append(
                    {"role": "tool", "content": str(output), "tool_call_id": call.id, "name": call.function.name}
                )
    except Exception as exc:
        logger.exception("episode %s/%s failed", episode["version"], episode["id"])
        stop_reason = f"error: {type(exc).__name__}: {exc}"

    # An episode that raised, or whose prompt outgrew the model's context, has
    # no result even if it already wrote prediction.csv: the benchmark's Total
    # counts it as no prediction with score 0.
    score: dict[str, Any] = {}
    if stop_reason != "context_exhausted" and not stop_reason.startswith("error"):
        try:
            score = evaluation.evaluate_prediction(
                str(work_dir / PREDICTION_FILE),
                episode["ground_truth"],
                str(Path(episode["ground_truth"]).parent / "all_metadata.json"),
                ds_question_version=episode["version"],
            )
        except Exception as exc:
            score = {"error": f"{type(exc).__name__}: {exc}"}
    for name in episode["metadata"]["needed_files"]:
        (work_dir / name).unlink(missing_ok=True)
    return {
        "id": episode["id"],
        "version": episode["version"],
        "task": episode["task"],
        "category": episode["category"],
        "score": float(score.get("final_score", 0.0)),
        "prediction_exist": bool(score.get("prediction_exist", 0.0)),
        "score_detail": score,
        "stop_reason": stop_reason,
        "num_tool_calls": n_tool_calls,
        "num_completion_tokens": n_completion_tokens,
        "seconds": round(time.time() - started, 1),
        "messages": [_jsonable(message) for message in messages],
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, float]:
    by_category: dict[str, list[float]] = defaultdict(list)
    for row in results:
        by_category[row["category"]].append(row["score"])
    n = len(results)
    overall = sum(row["score"] for row in results) / n
    metrics: dict[str, float] = {
        "dare/score": overall,
        "dare/prediction_rate": sum(int(row["prediction_exist"]) for row in results) / n,
        "dare/mean_tool_calls": sum(row["num_tool_calls"] for row in results) / n,
        "dare/num_episodes": float(n),
        "test/env/all/correct": overall,
    }
    for category, scores in sorted(by_category.items()):
        metrics[f"dare/{category}"] = sum(scores) / len(scores)
    logger.info(
        "dare: overall %.2f; %s",
        100.0 * overall,
        ", ".join(f"{category} {100.0 * sum(s) / len(s):.2f}" for category, s in sorted(by_category.items())),
    )
    return metrics


def main(config: Config) -> dict[str, float]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    bench_dir = Path(config.bench_dir).expanduser()
    versions = [version.strip() for version in config.versions.split(",") if version.strip()]
    task_types = {t.strip() for t in config.task_types.split(",")} if config.task_types else None
    episodes = load_episodes(bench_dir, versions, task_types)
    if config.max_examples is not None:
        episodes = episodes[: config.max_examples]
    logger.info("Running %d DARE-Bench episodes", len(episodes))

    work_root = Path(config.log_path) / "work"
    sandbox = threading.Semaphore(max(1, config.sandbox_workers))
    sandbox_server = (
        contextlib.nullcontext(config.sandbox_url)
        if config.sandbox_url
        else local_sandbox(python=config.sandbox_python)
    )
    if config.debug_image_tag:
        os.environ[DEBUG_OPTIONS_ENV] = "1"
    body, _ = inference_endpoint_body(
        model=config.model,
        context=config.context,
        hardware=config.hardware,
        lora=config.lora,
        job_config=config.job_config,
        max_seq_len=config.max_seq_len,
        source_job_id=config.source_job_id,
        checkpoint_id=config.checkpoint_id,
        debug_image_tag=config.debug_image_tag,
    )
    sampling_sub = next(sub for sub in body["sub_job_configs"] if sub.get("job_type") == "sampling")
    _, renderer, _ = build_renderer(
        sampling_sub["model_name"], renderer_name=config.renderer_name, enable_thinking=config.enable_thinking
    )
    stop_sequences = renderer.get_stop_sequences()
    params = {"max_tokens": config.max_tokens, "temperature": config.temperature, **stop_params_for(stop_sequences)}
    client = make_client(config.config, config.connection)
    with sandbox_server as sandbox_url, running_job(
        client, body, job_id=config.job_id, keep_job=config.keep_job
    ) as job_id:
        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max(1, config.concurrency)) as pool:
            futures = [
                pool.submit(
                    run_episode,
                    client,
                    job_id,
                    renderer,
                    episode,
                    max_seq_len=int(sampling_sub["inference_config"]["max_seq_len"]),
                    work_root=work_root,
                    sampling_params=params,
                    end_token=int(stop_sequences[0]),
                    max_turns=config.max_turns,
                    sandbox=sandbox,
                    sandbox_url=sandbox_url,
                    executor_timeout=config.executor_timeout,
                )
                for episode in episodes
            ]
            for index, future in enumerate(futures, start=1):
                results.append(future.result())
                if index % 25 == 0:
                    logger.info("%d/%d episodes scored", index, len(futures))

    metrics = summarize(results)
    predictions = Path(config.log_path) / "predictions.jsonl"
    predictions.parent.mkdir(parents=True, exist_ok=True)
    predictions.write_text("".join(json.dumps(row, default=str) + "\n" for row in results), encoding="utf-8")
    logger.info("Results: %s", metrics)
    ml_logger = setup_logging(config)
    ml_logger.log_metrics(metrics, step=0)
    ml_logger.close()
    return metrics


if __name__ == "__main__":
    chz.nested_entrypoint(main)
