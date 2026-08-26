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

"""Submit a text prompt to an inference endpoint and print the completion."""

from __future__ import annotations

import logging
import os

import chz
from recipes.utils import build_renderer
from recipes.utils import make_client
from recipes.utils import running_job
from recipes.utils import stop_params_for
from recipes.inference.endpoint import generate_results
from recipes.inference.endpoint import inference_endpoint_body
from recipes.inference.prompts import completion_text
from recipes.inference.prompts import render_user_prompt

from cortex_training.client import DEBUG_OPTIONS_ENV

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)
logging.getLogger("urllib3").setLevel(logging.WARN)
logging.getLogger("tinker_cookbook.renderers.base").setLevel(logging.ERROR)


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
    # False (default): same disable-thinking renderer as conversational SFT.
    # True: thinking-on renderer. Must match the training run. renderer_name overrides.
    enable_thinking: bool = False
    renderer_name: str | None = None

    prompt: str = "How many r's are in strawberry?"
    max_tokens: int = 512
    temperature: float = 0.6
    top_p: float = 1.0


def main(config: Config):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if config.debug_image_tag:
        os.environ[DEBUG_OPTIONS_ENV] = "1"

    body, source = inference_endpoint_body(
        config.job_config,
        source_job_id=config.source_job_id,
        checkpoint_id=config.checkpoint_id,
        debug_image_tag=config.debug_image_tag,
    )
    model_name = next(
        (sub.get("model_name") for sub in body.get("sub_job_configs") or () if sub.get("job_type") == "sampling"),
        None,
    )
    _, renderer, renderer_name = build_renderer(
        model_name,
        renderer_name=config.renderer_name,
        enable_thinking=config.enable_thinking,
    )
    logger.info(
        "Using renderer: %s (enable_thinking=%s)",
        renderer_name,
        config.enable_thinking,
    )

    client = make_client(config.config)
    if source is not None:
        logger.info(
            "Creating inference endpoint from weights-only checkpoint %s (job %s)",
            source["checkpoint_id"],
            source["source_job_id"],
        )

    sampling_params = {
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        **stop_params_for(renderer.get_stop_sequences()),
    }
    prompt_tokens = render_user_prompt(renderer, config.prompt)
    with running_job(client, body, job_id=config.job_id, keep_job=config.keep_job) as job_id:
        results = generate_results(client, job_id, [prompt_tokens], sampling_params, batch_size=1)
        raw = completion_text(results[0])
        logger.info("Prompt: %s", config.prompt)
        logger.info(
            "Renderer: %s (enable_thinking=%s)",
            renderer_name,
            config.enable_thinking,
        )
        logger.info("Completion:\n%s", raw)


if __name__ == "__main__":
    chz.nested_entrypoint(main)
