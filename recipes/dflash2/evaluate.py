# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Measure a DFlash2 draft's served acceptance length on Cortex Training.

There is no API call for the engine's speculative-decoding counters for now. The sampling sub-job's log carries
them instead: every 30 seconds it writes a ``drain_metrics`` summary with the engine's running totals
``spec_num_drafts_total``, ``spec_num_draft_tokens_total`` and ``spec_num_accepted_tokens_total``.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from collections.abc import Sequence
from typing import Any

import chz
from recipes.dflash2.task import iter_jsonl
from recipes.utils import build_job
from recipes.utils import make_client
from recipes.utils import running_job
from recipes.utils import source_checkpoint_info

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)
logging.getLogger("urllib3").setLevel(logging.WARN)

TOTALS = ("spec_num_drafts_total", "spec_num_draft_tokens_total", "spec_num_accepted_tokens_total")
WARMUP_PROMPTS = 4
DEFAULT_MAX_SEQ_LEN = 4096


@chz.chz
class Config:
    # JSONL of conversations; the first user turn of each row is a prompt.
    prompts: str
    # Legacy JSON connection file. Omit it to use `connection` or the default profile.
    config: str | None = None
    # Snowflake connection profile name (connections.toml). Use config or connection, not both.
    connection: str | None = None
    # A running endpoint of the target with the draft (one kept with keep_job=true), or the draft to start one for.
    job_id: str | None = None
    draft_job_id: str | None = None
    draft_checkpoint_id: str | None = None

    model: str = "Qwen/Qwen3.5-4B"
    context: str = "standard"
    hardware: str = "h200"
    debug_image_tag: str | None = None
    keep_job: bool = False

    num_prompts: int = 128
    max_tokens: int = 1024
    temperature: float = 0.0
    settle_timeout: float = 300.0
    # Append the result as one JSON line.
    out: str | None = None


def checkpoint_block_size(client: Any, job_id: str, checkpoint_id: str) -> int:
    """``dflash_config.block_size`` of a draft checkpoint, read from its exported ``config.json``."""
    files = client.export_checkpoint(job_id, checkpoint_id).get("files") or ()
    urls = [f["url"] for f in files if f.get("filename", "").rsplit("/", 1)[-1] == "config.json"]
    if len(urls) != 1:
        raise ValueError(f"checkpoint {checkpoint_id} of job {job_id} has {len(urls)} config.json files, expected 1")
    with urllib.request.urlopen(urls[0], timeout=60) as response:
        config = json.loads(response.read())
    block_size = (config.get("dflash_config") or {}).get("block_size")
    if not isinstance(block_size, int) or block_size < 2:
        raise ValueError(f"checkpoint {checkpoint_id} is not a DFlash2 draft: dflash_config.block_size is {block_size!r}")
    return block_size


def speculative_override(
    num_speculative_tokens: int,
    *,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    n_gpus: int = 1,
    gpu_memory_utilization: float = 0.6,
    attention_backend: str | None = "TRITON_ATTN",
    max_num_batched_tokens: int = 8192,
) -> dict[str, Any]:
    """The layer on top of the target's catalog inference file that serves the source checkpoint as the speculator."""
    max_num_seqs = max_num_batched_tokens // (num_speculative_tokens + 1)
    if max_num_seqs < 1:
        raise ValueError(
            f"max_num_batched_tokens={max_num_batched_tokens} cannot hold one block of {num_speculative_tokens + 1} tokens"
        )
    speculative: dict[str, Any] = {
        "method": "dflash",
        "num_speculative_tokens": num_speculative_tokens,
        "from_source_checkpoint": True,
    }
    vllm_config: dict[str, Any] = {
        "max_model_len": max_seq_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_seqs": max_num_seqs,
        "speculative_config": speculative,
    }
    if attention_backend:
        speculative["attention_backend"] = attention_backend
        vllm_config["extra_engine_kwargs"] = {"attention_config": {"backend": attention_backend}}
    inference_config = {"max_seq_len": max_seq_len, "n_gpus": n_gpus, "vllm_config": vllm_config}
    return {"sub_job_configs": [{"job_type": "sampling", "inference_config": inference_config}]}


def endpoint_body(
    *,
    model: str,
    draft_job_id: str,
    draft_checkpoint_id: str,
    num_speculative_tokens: int,
    context: str = "standard",
    hardware: str = "h200",
    job_config: str | None = None,
    max_seq_len: int | None = None,
    n_gpus: int = 1,
    gpu_memory_utilization: float = 0.6,
    attention_backend: str | None = "TRITON_ATTN",
    debug_image_tag: str | None = None,
) -> dict[str, Any]:
    """Create-job body of an endpoint serving ``model`` with the draft checkpoint as its DFlash2 speculator."""
    override = speculative_override(
        num_speculative_tokens,
        max_seq_len=max_seq_len or DEFAULT_MAX_SEQ_LEN,
        n_gpus=n_gpus,
        gpu_memory_utilization=gpu_memory_utilization,
        attention_backend=attention_backend,
    )
    return build_job(
        "inference",
        model=model,
        context=context,
        hardware=hardware,
        job_config=job_config,
        overrides=[override],
        max_seq_len=max_seq_len,
        debug_image_tag=debug_image_tag,
        source_checkpoint=source_checkpoint_info(draft_job_id, draft_checkpoint_id),
    )


def generate_ids(client: Any, job_id: str, prompts: list[list[int]], *, max_tokens: int, temperature: float) -> list[dict]:
    request_id = client.generate(
        job_id,
        prompts=prompts,
        sampling_params={"max_tokens": max_tokens, "temperature": temperature, "top_p": 1.0, "seed": 0},
    )
    return client.poll_request(job_id, request_id).get("results") or []


def chat_prompt_ids(tokenizer: Any, conversation: list[dict]) -> list[int]:
    """Token ids of ``conversation`` rendered with the target's chat template, ready for generation."""
    ids = tokenizer.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True)
    if isinstance(ids, dict) or hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    return list(ids)


def load_prompts(path: str, n: int) -> list[list[dict]]:
    """The first user turn of each of the first ``n`` rows, as a one-message conversation."""
    prompts = []
    for row in iter_jsonl(path):
        conversation = row.get("conversations") or row.get("messages") or ()
        first = next((m for m in conversation if m.get("role", m.get("from")) in ("user", "human")), None)
        if first is not None:
            prompts.append([{"role": "user", "content": first.get("content") or first.get("value") or ""}])
        if len(prompts) >= n:
            break
    return prompts


def spec_totals(line: str) -> tuple[float, float, float] | None:
    """The speculative-decoding totals of one ``drain_metrics`` log line, or None for any other line."""
    start = line.find('{"')
    if start < 0:
        return None
    try:
        record = json.loads(line[start:])
    except ValueError:
        return None
    if not isinstance(record, dict) or record.get("msg") != "drain_metrics" or TOTALS[0] not in record:
        return None
    return tuple(float(record[key]) for key in TOTALS)


class DrainLogs:
    """Reads a sampling job's log forward from where the last read stopped."""

    def __init__(self, client: Any, job_id: str, poll_seconds: float = 10.0):
        self.client, self.job_id, self.poll_seconds = client, job_id, poll_seconds
        self.cursor = ""

    def read(self) -> list[str]:
        lines = []
        while True:
            page = self.client.tail_logs(self.job_id, cursor=self.cursor, sub_job_type="sampling")
            entries = page.get("entries") or []
            lines.extend(e.get("_raw", "") if isinstance(e, dict) else str(e) for e in entries)
            self.cursor = page.get("next_cursor") or self.cursor
            if page.get("eof") or not entries:
                return lines

    def settled_totals(self, timeout: float) -> tuple[float, float, float]:
        """The totals once two consecutive summaries logged from now on agree."""
        self.read()  # lines logged before now may predate the work that just finished
        seen: list[tuple[float, float, float]] = []
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            seen.extend(t for t in map(spec_totals, self.read()) if t is not None)
            if len(seen) >= 2 and seen[-1] == seen[-2]:
                return seen[-1]
            time.sleep(self.poll_seconds)
        raise TimeoutError(
            f"no settled speculative-decoding totals in {timeout:.0f}s of job {self.job_id}'s log ({len(seen)} seen): "
            "the service may not report them yet, or the endpoint is not speculating"
        )


def measure_acceptance(
    client: Any,
    job_id: str,
    prompts: Sequence[list[int]],
    *,
    max_tokens: int,
    temperature: float = 0.0,
    settle_timeout: float = 300.0,
) -> dict[str, Any]:
    """Generate ``prompts`` on a running speculative endpoint and return its acceptance counters."""
    logs = DrainLogs(client, job_id)
    generate_ids(client, job_id, list(prompts[:WARMUP_PROMPTS]), max_tokens=max_tokens, temperature=temperature)
    before = logs.settled_totals(settle_timeout)
    started = time.time()
    results = generate_ids(client, job_id, list(prompts), max_tokens=max_tokens, temperature=temperature)
    seconds = time.time() - started
    after = logs.settled_totals(settle_timeout)

    drafts, draft_tokens, accepted = (a - b for a, b in zip(after, before))
    return {
        "num_prompts": len(prompts),
        "generated_tokens": sum(len(r.get("token_ids") or ()) for r in results if isinstance(r, dict)),
        "seconds": seconds,
        "drafts": drafts,
        "draft_tokens_per_draft": draft_tokens / drafts if drafts else float("nan"),
        "accepted_tokens": accepted,
        "mean_acceptance_length": 1.0 + accepted / drafts if drafts else float("nan"),
    }


def evaluate_draft(
    client: Any,
    *,
    model: str,
    prompts: str,
    draft_job_id: str | None = None,
    draft_checkpoint_id: str | None = None,
    job_id: str | None = None,
    context: str = "standard",
    hardware: str = "h200",
    num_prompts: int = 128,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    settle_timeout: float = 300.0,
    debug_image_tag: str | None = None,
    keep_job: bool = False,
) -> dict[str, Any]:
    """Serve the draft (or attach to ``job_id``) and measure its acceptance length on ``prompts`` on specific spec."""
    body = None
    if job_id is None:
        if not (draft_job_id and draft_checkpoint_id):
            raise ValueError("pass draft_job_id and draft_checkpoint_id, or job_id of a running endpoint")
        body = endpoint_body(
            model=model,
            draft_job_id=draft_job_id,
            draft_checkpoint_id=draft_checkpoint_id,
            num_speculative_tokens=checkpoint_block_size(client, draft_job_id, draft_checkpoint_id) - 1,
            context=context,
            hardware=hardware,
            debug_image_tag=debug_image_tag,
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    prompt_ids = [chat_prompt_ids(tokenizer, p) for p in load_prompts(prompts, num_prompts)]
    with running_job(client, body, job_id=job_id, keep_job=keep_job or job_id is not None) as endpoint:
        result = measure_acceptance(
            client, endpoint, prompt_ids, max_tokens=max_tokens, temperature=temperature, settle_timeout=settle_timeout
        )
    result.update(endpoint_job_id=endpoint, draft_job_id=draft_job_id, draft_checkpoint_id=draft_checkpoint_id)
    logger.info(
        "served mean acceptance length %.3f over %d prompts (%d drafts, %.1f tokens each)",
        result["mean_acceptance_length"],
        result["num_prompts"],
        result["drafts"],
        result["draft_tokens_per_draft"],
    )
    return result


def main(config: Config) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    client = make_client(config.config, config.connection)
    result = evaluate_draft(
        client,
        model=config.model,
        prompts=config.prompts,
        draft_job_id=config.draft_job_id,
        draft_checkpoint_id=config.draft_checkpoint_id,
        job_id=config.job_id,
        context=config.context,
        hardware=config.hardware,
        num_prompts=config.num_prompts,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        settle_timeout=config.settle_timeout,
        debug_image_tag=config.debug_image_tag,
        keep_job=config.keep_job,
    )
    if config.out:
        with open(config.out, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(result) + "\n")
    return result


if __name__ == "__main__":
    chz.nested_entrypoint(main)
