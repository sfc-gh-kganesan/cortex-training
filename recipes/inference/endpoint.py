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
rollouts. Serve, generate, and eval start that job with ``running_job``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from recipes.utils import load_job_body
from recipes.utils import source_checkpoint_info

logger = logging.getLogger(__name__)

_RECIPE_DIR = Path(__file__).resolve().parent
CONFIG_SEARCH_DIRS: tuple[Path, ...] = (_RECIPE_DIR, _RECIPE_DIR / "configs")


def inference_endpoint_body(
    job_config: str,
    *,
    source_job_id: str | None = None,
    checkpoint_id: str | None = None,
    debug_image_tag: str | None = None,
    search_dirs: Sequence[Path] = CONFIG_SEARCH_DIRS,
) -> tuple[dict[str, Any], dict[str, str] | None]:
    """Return ``(create-job body, source checkpoint info or None)``."""
    body = load_job_body(job_config, search_dirs=search_dirs)
    source = source_checkpoint_info(source_job_id, checkpoint_id)
    if source is not None:
        sampling_sub = next(
            (sub for sub in body.get("sub_job_configs") or () if sub.get("job_type") == "sampling"),
            None,
        )
        if sampling_sub is not None:
            sampling_sub["source_checkpoint_info"] = dict(source)
    if debug_image_tag:
        body["debug"] = {"job": {"image_tag": debug_image_tag}}
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


def log_endpoint_ready(config_path: str, job_id: str) -> None:
    logger.info("Inference endpoint is ready: job_id=%s", job_id)
    logger.info(
        "Examples against this endpoint:\n"
        "  python -m recipes.inference.generate config=%s job_id=%s\n"
        "  python -m recipes.inference.evaluate config=%s job_id=%s",
        config_path,
        job_id,
        config_path,
        job_id,
    )
    logger.info("Tear down with: cortex-training cancel %s", job_id)
