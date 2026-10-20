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

"""MATH-500, GSM8K, and identity eval against an inference endpoint."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any

import chz
from recipes.inference.endpoint import generate_results
from recipes.inference.endpoint import inference_endpoint_body
from recipes.inference.identity import IDENTITY_PHRASE
from recipes.inference.identity import load_identity_prompts
from recipes.inference.identity import score_identity_completions
from recipes.inference.prompts import completion_text
from recipes.inference.prompts import render_user_prompt
from recipes.utils import build_renderer
from recipes.utils import make_client
from recipes.utils import running_job
from recipes.utils import stop_params_for
from recipes.rl.tasks.math import build_prompt
from recipes.rl.tasks.math import score_response

from cortex_training.client import DEBUG_OPTIONS_ENV

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)
logging.getLogger("urllib3").setLevel(logging.WARN)
logging.getLogger("tinker_cookbook.renderers.base").setLevel(logging.ERROR)


@dataclass
class _MathExample:
    prompt_text: str
    answer: str
    example_id: str


def _load_math500(max_examples: int | None = None) -> list[_MathExample]:
    from datasets import load_dataset
    from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))
    examples: list[_MathExample] = []
    for idx, row in enumerate(ds):
        try:
            expected = extract_boxed(row["solution"])
        except ValueError:
            continue
        examples.append(
            _MathExample(
                prompt_text=row["problem"],
                answer=expected,
                example_id=f"math500-{idx}",
            )
        )
    return examples


_GSM8K_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def normalize_gsm8k_answer(text: str) -> str:
    """Pull the GSM8K final answer into a comparable number string.

    Prefer the ``#### 72`` line. If the model never used that marker, take the
    last number in the completion instead of the first (intermediates come first).
    """
    if "####" in text:
        payload = text.split("####")[-1].replace(",", "").replace("$", "").strip()
        match = _GSM8K_NUMBER.search(payload)
        return match.group(0) if match else payload
    matches = _GSM8K_NUMBER.findall(text.replace(",", "").replace("$", ""))
    return matches[-1] if matches else text.strip()


def _load_gsm8k(max_examples: int | None = None) -> list[_MathExample]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))
    examples: list[_MathExample] = []
    for idx, row in enumerate(ds):
        gold = normalize_gsm8k_answer(str(row["answer"]))
        if not gold:
            continue
        examples.append(
            _MathExample(
                prompt_text=str(row["question"]),
                answer=gold,
                example_id=f"gsm8k-{idx}",
            )
        )
    return examples


def _run_gsm8k(
    *,
    client: Any,
    job_id: str,
    renderer: Any,
    max_examples: int | None,
    sampling_params: dict[str, Any],
    generate_batch_size: int,
    max_seq_len: int,
) -> dict[str, float]:
    examples = _load_gsm8k(max_examples)
    if len(examples) == 0:
        raise ValueError("GSM8K test produced no examples")
    prompts: list[list[int]] = []
    for example in examples:
        tokens = render_user_prompt(renderer, example.prompt_text)
        if len(tokens) >= max_seq_len:
            raise ValueError(
                f"{example.example_id} prompt has {len(tokens)} tokens; raise max_seq_len (currently {max_seq_len})"
            )
        prompts.append(tokens)

    results = generate_results(client, job_id, prompts, sampling_params, generate_batch_size)
    n_correct = 0
    n_formatted = 0
    for example, result in zip(examples, results):
        text = completion_text(result)
        predicted = normalize_gsm8k_answer(text)
        n_formatted += int("####" in text)
        n_correct += int(predicted == example.answer)
    n = len(examples)
    score = n_correct / n
    logger.info("gsm8k: %.1f%% (%d/%d)", 100.0 * score, n_correct, n)
    metrics = {
        "gsm8k/correct": score,
        "gsm8k/format": n_formatted / n,
        "gsm8k/num_examples": float(n),
        "test/env/all/correct": score,
        "test/env/all/format": n_formatted / n,
        "test/env/all/num_examples": float(n),
    }
    logger.info("Results: %s", metrics)
    return metrics


def _run_identity(
    *,
    client: Any,
    job_id: str,
    renderer: Any,
    max_examples: int | None,
    sampling_params: dict[str, Any],
    generate_batch_size: int,
    max_seq_len: int,
    prompts_file: str | None = None,
    phrase: str = IDENTITY_PHRASE,
) -> dict[str, float]:
    examples = load_identity_prompts(prompts_file, max_examples=max_examples)
    prompts: list[list[int]] = []
    for example in examples:
        tokens = render_user_prompt(renderer, example["prompt"])
        if len(tokens) >= max_seq_len:
            raise ValueError(
                f"{example['id']} prompt has {len(tokens)} tokens; raise max_seq_len (currently {max_seq_len})"
            )
        prompts.append(tokens)

    results = generate_results(client, job_id, prompts, sampling_params, generate_batch_size)
    scored = [
        {**example, "completion": completion_text(result)}
        for example, result in zip(examples, results)
    ]
    return score_identity_completions(scored, phrase=phrase)


def _run_math500(
    *,
    client: Any,
    job_id: str,
    renderer: Any,
    max_examples: int | None,
    sampling_params: dict[str, Any],
    generate_batch_size: int,
    max_seq_len: int,
) -> dict[str, float]:
    examples = _load_math500(max_examples)
    if len(examples) == 0:
        raise ValueError("MATH-500 produced no examples")
    prompts: list[list[int]] = []
    for example in examples:
        tokens = build_prompt(example.prompt_text, renderer)
        if len(tokens) >= max_seq_len:
            raise ValueError(
                f"{example.example_id} prompt has {len(tokens)} tokens; raise max_seq_len (currently {max_seq_len})"
            )
        prompts.append(tokens)

    results = generate_results(
        client, job_id, prompts, sampling_params, generate_batch_size
    )
    n_correct = 0
    format_sum = 0.0
    max_tokens = sampling_params.get("max_tokens")
    for example, result in zip(examples, results):
        text = result.get("text") or ""
        _reward, metrics = score_response(
            text,
            example.answer,
            result=result,
            max_tokens=max_tokens,
        )
        n_correct += int(metrics["correct"] >= 1.0)
        format_sum += float(metrics.get("format") or 0.0)
    n = len(examples)
    score = n_correct / n
    logger.info("math500: %.1f%% (%d/%d)", 100.0 * score, n_correct, n)
    metrics = {
        "math500/correct": score,
        "math500/format": format_sum / n,
        "math500/num_examples": float(n),
        "test/env/all/correct": score,
        "test/env/all/format": format_sum / n,
        "test/env/all/num_examples": float(n),
    }
    logger.info("Results: %s", metrics)
    return metrics


@chz.chz
class Config:
    config: str
    job_id: str | None = None  # attach to a running inference endpoint

    job_config: str = "configs/qwen3_8b_full.json"
    debug_image_tag: str | None = None
    keep_job: bool | None = None

    source_job_id: str | None = None
    # Required with source_job_id. Use the cp_* id.
    checkpoint_id: str | None = None

    task: str = "math500"
    # False (default): same disable-thinking renderer as conversational SFT.
    enable_thinking: bool = False
    renderer_name: str | None = None
    max_examples: int | None = None
    max_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 1.0
    generate_batch_size: int = 64
    prompts_file: str | None = None
    phrase: str = IDENTITY_PHRASE


def run_evaluation(
    *,
    config_path: str,
    job_config: str = "configs/qwen3_8b_full.json",
    source_job_id: str | None = None,
    checkpoint_id: str | None = None,
    job_id: str | None = None,
    task: str = "math500",
    max_examples: int | None = None,
    max_tokens: int = 4096,
    temperature: float = 1.0,
    top_p: float = 1.0,
    generate_batch_size: int = 64,
    enable_thinking: bool = False,
    renderer_name: str | None = None,
    debug_image_tag: str | None = None,
    keep_job: bool | None = None,
    prompts_file: str | None = None,
    phrase: str = IDENTITY_PHRASE,
) -> dict[str, float]:
    if debug_image_tag:
        os.environ[DEBUG_OPTIONS_ENV] = "1"
        logger.info("Using debug image_tag=%s", debug_image_tag)

    body, source = inference_endpoint_body(
        job_config,
        source_job_id=source_job_id,
        checkpoint_id=checkpoint_id,
        debug_image_tag=debug_image_tag,
    )
    sampling_sub = next(
        (
            sub
            for sub in body.get("sub_job_configs") or ()
            if sub.get("job_type") == "sampling"
        ),
        {},
    )
    model_name = sampling_sub.get("model_name")
    max_seq_len = int((sampling_sub.get("inference_config") or {}).get("max_seq_len"))
    _, renderer, renderer_name = build_renderer(
        model_name,
        renderer_name=renderer_name,
        enable_thinking=enable_thinking,
    )
    logger.info("Using renderer: %s (enable_thinking=%s)", renderer_name, enable_thinking)

    client = make_client(config_path)
    task_name = task.strip().lower()
    if task_name not in {"math500", "gsm8k", "identity"}:
        raise ValueError(f"unsupported eval task {task!r}; use math500, gsm8k, or identity")
    label = {"math500": "MATH-500", "gsm8k": "GSM8K test", "identity": "identity"}[task_name]
    if source is not None:
        logger.info(
            "Starting %s eval from weights-only checkpoint %s (job %s)",
            label,
            source["checkpoint_id"],
            source["source_job_id"],
        )
    elif job_id is None:
        logger.info("Starting %s eval from original weights (%s)", label, model_name)

    sampling_params = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        **stop_params_for(renderer.get_stop_sequences()),
    }

    runners = {"math500": _run_math500, "gsm8k": _run_gsm8k, "identity": _run_identity}
    runner = runners[task_name]
    extra = (
        {"prompts_file": prompts_file, "phrase": phrase}
        if task_name == "identity"
        else {}
    )
    with running_job(client, body, job_id=job_id, keep_job=keep_job) as eval_job_id:
        return runner(
            client=client,
            job_id=eval_job_id,
            renderer=renderer,
            max_examples=max_examples,
            sampling_params=sampling_params,
            generate_batch_size=generate_batch_size,
            max_seq_len=max_seq_len,
            **extra,
        )


def main(config: Config):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    run_evaluation(
        config_path=config.config,
        job_config=config.job_config,
        source_job_id=config.source_job_id,
        checkpoint_id=config.checkpoint_id,
        job_id=config.job_id,
        task=config.task,
        max_examples=config.max_examples,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        generate_batch_size=config.generate_batch_size,
        enable_thinking=config.enable_thinking,
        renderer_name=config.renderer_name,
        debug_image_tag=config.debug_image_tag,
        keep_job=config.keep_job,
        prompts_file=config.prompts_file,
        phrase=config.phrase,
    )


if __name__ == "__main__":
    chz.nested_entrypoint(main)
