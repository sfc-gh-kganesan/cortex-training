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

"""Create-job body, generate, and logging helpers for standalone sampling jobs.

An inference endpoint is a running job that serves generations. On the wire the
worker is still ``job_type=sampling`` — the same generation runtime RL uses for
rollouts. Serve, generate, and eval build that job with ``build_job`` from the
model catalog's inference file and start it with ``running_job``.
"""

from __future__ import annotations

import logging
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from recipes.utils import build_job
from recipes.utils import source_checkpoint_info

logger = logging.getLogger(__name__)

_EXAMPLE_DIR = Path(__file__).resolve().parent


def inference_endpoint_body(
    *,
    model: str,
    context: str,
    hardware: str = "h200",
    lora: bool = False,
    job_config: str | None = None,
    max_seq_len: int | None = None,
    source_job_id: str | None = None,
    checkpoint_id: str | None = None,
    debug_image_tag: str | None = None,
) -> tuple[dict[str, Any], dict[str, str] | None]:
    """Return ``(create-job body, source checkpoint info or None)``.

    The job is the catalog's inference file for ``model``, ``context``,
    ``hardware`` and ``lora``, then inference/overrides.json when it exists.
    A ``job_config`` file replaces both of those layers and is used as is.
    Either way ``max_seq_len`` applies, a checkpoint goes on the sampling
    sub-job, and ``debug_image_tag`` adds the debug block.
    """
    source = source_checkpoint_info(source_job_id, checkpoint_id)
    body = build_job(
        "inference",
        model=model,
        context=context,
        hardware=hardware,
        lora=lora,
        job_config=job_config,
        example_dir=_EXAMPLE_DIR,
        search_dirs=(_EXAMPLE_DIR,),
        max_seq_len=max_seq_len,
        debug_image_tag=debug_image_tag,
        source_checkpoint=source,
    )
    return body, source


def generate_results(
    client: Any,
    job_id: str,
    prompts: list[list[int]],
    sampling_params: dict[str, Any],
    batch_size: int,
) -> list[dict[str, Any]]:
    """Generate completions for tokenized prompts on a running endpoint.

    ``sampling_params`` is the generate-request field name on the Cortex Training
    API (temperature, max_tokens, and related decoding settings).
    """
    results: list[dict[str, Any]] = []
    width = max(1, batch_size)
    for start in range(0, len(prompts), width):
        batch = prompts[start : start + width]
        request_id = client.generate(job_id, prompts=batch, sampling_params=sampling_params)
        payload = client.poll_request(job_id, request_id)
        batch_results = payload.get("results") or []
        if len(batch_results) != len(batch):
            raise RuntimeError(f"asked for {len(batch)} completions, got {len(batch_results)}")
        for result in batch_results:
            results.append(result if isinstance(result, dict) else {"text": str(result)})
    return results


def log_endpoint_ready(connection_args: str, job_id: str, job_args: Mapping[str, Any]) -> None:
    """Log the endpoint's id and the generate and evaluate commands that use it.

    ``connection_args`` is the connection argument to repeat (``config=``,
    ``connection=``, or ``""`` for the default). ``job_args`` are the arguments
    that built the endpoint's job (``model=``, ``context=``, ``hardware=``,
    ``lora=``, or ``job_config=``, and ``max_seq_len=`` when set). The commands
    repeat them, so they pick the endpoint's model, and with it its renderer,
    rather than their defaults.
    """
    extra = "".join(
        f" {name}={str(value).lower() if isinstance(value, bool) else shlex.quote(str(value))}"
        for name, value in job_args.items()
    )
    logger.info("Inference endpoint is ready: job_id=%s", job_id)
    examples = "\n".join(
        " ".join(part for part in (f"  python -m {module}", connection_args, f"job_id={job_id}") if part)
        + extra
        for module in ("recipes.inference.generate", "recipes.inference.evaluate")
    )
    logger.info("Examples against this endpoint:\n%s", examples)
    logger.info("Tear down with: cortex-training cancel %s", job_id)
