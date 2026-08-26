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

"""Create a Cortex Training inference endpoint and leave it running."""

from __future__ import annotations

import logging
import os

import chz
from recipes.utils import make_client
from recipes.utils import running_job
from recipes.inference.endpoint import inference_endpoint_body
from recipes.inference.endpoint import log_endpoint_ready

from cortex_training.client import DEBUG_OPTIONS_ENV

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)
logging.getLogger("urllib3").setLevel(logging.WARN)


@chz.chz
class Config:
    config: str
    job_id: str | None = None  # attach to a running inference endpoint

    job_config: str = "configs/qwen3_8b_full.json"
    debug_image_tag: str | None = None
    keep_job: bool = True

    source_job_id: str | None = None
    # Required with source_job_id. Use the cp_* id.
    checkpoint_id: str | None = None


def main(config: Config):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if config.debug_image_tag:
        os.environ[DEBUG_OPTIONS_ENV] = "1"

    client = make_client(config.config)
    body, source = inference_endpoint_body(
        config.job_config,
        source_job_id=config.source_job_id,
        checkpoint_id=config.checkpoint_id,
        debug_image_tag=config.debug_image_tag,
    )
    if config.job_id is not None:
        logger.info("Attaching to inference endpoint %s", config.job_id)
    elif source is not None:
        logger.info(
            "Creating inference endpoint from weights-only checkpoint %s (job %s)",
            source["checkpoint_id"],
            source["source_job_id"],
        )
    else:
        logger.info(
            "Creating inference endpoint from original weights (%s)",
            next(
                (sub.get("model_name") for sub in body.get("sub_job_configs") or () if sub.get("job_type") == "sampling"),
                None,
            ),
        )

    with running_job(client, body, job_id=config.job_id, keep_job=config.keep_job) as job_id:
        log_endpoint_ready(config.config, job_id)


if __name__ == "__main__":
    chz.nested_entrypoint(main)
