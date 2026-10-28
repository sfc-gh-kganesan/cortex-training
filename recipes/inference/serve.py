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

import chz
from recipes.utils import example_connection_args
from recipes.utils import make_client
from recipes.utils import running_job
from recipes.inference.endpoint import inference_endpoint_body
from recipes.inference.endpoint import log_endpoint_ready

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)
logging.getLogger("urllib3").setLevel(logging.WARN)


@chz.chz
class Config:
    # Legacy JSON connection file. Omit it to use `connection` or the default profile.
    config: str | None = None
    # Snowflake connection profile name (connections.toml). Use config or connection, not both.
    connection: str | None = None
    job_id: str | None = None  # attach to a running inference endpoint

    # The job comes from model-catalog/models/; job_config= replaces it with a file.
    model: str = "Qwen/Qwen3-8B"
    context: str = "standard"
    hardware: str = "h200"
    lora: bool = False
    job_config: str | None = None
    max_seq_len: int | None = None  # None keeps the job's value
    debug_image_tag: str | None = None
    keep_job: bool = True

    source_job_id: str | None = None
    # Required with source_job_id. Use the cp_* id.
    checkpoint_id: str | None = None


def main(config: Config):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    client = make_client(config.config, config.connection)
    body, source = inference_endpoint_body(
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

    if config.job_config is not None:
        job_args: dict = {"job_config": config.job_config}
    else:
        job_args = {"model": config.model, "context": config.context, "hardware": config.hardware, "lora": config.lora}
    if config.max_seq_len is not None:
        job_args["max_seq_len"] = config.max_seq_len
    with running_job(client, body, job_id=config.job_id, keep_job=config.keep_job) as job_id:
        log_endpoint_ready(example_connection_args(config.config, config.connection), job_id, job_args)


if __name__ == "__main__":
    chz.nested_entrypoint(main)
