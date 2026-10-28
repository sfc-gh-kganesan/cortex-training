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

"""Deploy a model and stream a chat completion from its OpenAI-compatible endpoint.

Creates a model deployment from a spec (a base model or a weights-only
checkpoint), waits until it is READY, streams one reply, and drops the
deployment unless ``keep_deployment=True``. Pass ``deployment=sf-ct-<UUID>`` to
call a deployment that is already running instead of creating one.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import chz
from recipes.model_deployment.streaming import chat_messages
from recipes.model_deployment.streaming import complete_chat
from recipes.model_deployment.streaming import stream_chat
from recipes.utils import resolve_example_connection

logger = logging.getLogger(__name__)
logging.getLogger("urllib3").setLevel(logging.WARN)

_EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_SPEC = "configs/base-model.json"


@chz.chz
class Config:
    # Legacy JSON connection file. Omit it to use `connection` or the default profile.
    config: str | None = None
    # Snowflake connection profile name (connections.toml). Use config or connection, not both.
    connection: str | None = None
    # A deployment spec JSON file; relative paths are also looked up in this
    # example's directory.
    spec: str = DEFAULT_SPEC
    # Call a running deployment instead of creating one. It is never dropped.
    deployment: str | None = None
    keep_deployment: bool = False
    # Seconds to wait for READY; defaults to the connection's poll_timeout.
    timeout: float | None = None

    prompt: str = "Explain in three sentences why streaming responses feel faster."
    system: str | None = None
    max_completion_tokens: int = 256
    temperature: float = 0.7
    stream: bool = True
    request_timeout: float = 300.0


def load_spec(path: str) -> dict[str, Any]:
    """Read a deployment spec from ``path`` or from this example's directory."""
    candidate = Path(path).expanduser()
    if not candidate.is_file() and not candidate.is_absolute():
        candidate = _EXAMPLE_DIR / path
    return json.loads(candidate.read_text(encoding="utf-8"))


def _write_delta(text: str) -> None:
    # The reply goes to stdout as it arrives; status lines go to the log.
    sys.stdout.write(text)
    sys.stdout.flush()


def main(config: Config) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    overrides = {"poll_timeout": config.timeout} if config.timeout is not None else {}
    resolved = resolve_example_connection(config.config, config.connection, **overrides)
    # Check before create_model_deployment: a run that cannot authenticate its
    # chat call must not start a deployment that holds GPUs.
    if not resolved.pat:
        raise ValueError(
            "model_deployment calls the inference endpoint with a PAT, and this example "
            f"only takes one from a JSON config file (you used {resolved.source}). "
            "Pass config=<JSON file with host and pat>."
        )
    client = resolved.client

    created = config.deployment is None
    if created:
        spec = load_spec(config.spec)
        deployment_id = client.create_model_deployment(spec)["deployment"]
        logger.info("Created %s; waiting for READY (this pulls the model and starts vLLM)", deployment_id)
    else:
        deployment_id = config.deployment
        logger.info("Using running deployment %s", deployment_id)

    try:
        ready = client.wait_for_model_deployment(deployment_id)
        base_url = ready.get("base_url") or client.model_deployment_base_url
        logger.info("%s is READY at %s", deployment_id, base_url)

        call = stream_chat if config.stream else complete_chat
        kwargs: dict[str, Any] = {"on_delta": _write_delta} if config.stream else {}
        result = call(
            base_url,
            resolved.pat,
            model=deployment_id,
            messages=chat_messages(config.prompt, config.system),
            max_completion_tokens=config.max_completion_tokens,
            temperature=config.temperature,
            verify_ssl=resolved.verify_ssl,
            timeout=config.request_timeout,
            **kwargs,
        )
        if config.stream:
            sys.stdout.write("\n")
        else:
            _write_delta(result.text + "\n")
        completion_tokens = (result.usage or {}).get("completion_tokens")
        if result.time_to_first_token is not None:
            logger.info(
                "First token after %.2fs; %s completion tokens in %.2fs (finish_reason=%s)",
                result.time_to_first_token,
                completion_tokens,
                result.total_seconds,
                result.finish_reason,
            )
        else:
            logger.info(
                "Reply in %.2fs; %s completion tokens (finish_reason=%s)",
                result.total_seconds,
                completion_tokens,
                result.finish_reason,
            )
    except BaseException:
        if created and not config.keep_deployment:
            # Never let a failed drop replace the error that got us here.
            try:
                _drop(client, deployment_id)
            except Exception:
                logger.exception(
                    "Could not drop %s; it may still hold GPUs. Drop it with: "
                    "cortex-training model-deployment drop %s",
                    deployment_id,
                    deployment_id,
                )
        raise
    if not created:
        return
    if config.keep_deployment:
        logger.info(
            "Leaving %s running; drop it with: cortex-training model-deployment drop %s",
            deployment_id,
            deployment_id,
        )
    else:
        _drop(client, deployment_id)


def _drop(client: Any, deployment_id: str) -> None:
    dropped = client.drop_model_deployment(deployment_id)
    if dropped.get("dropped"):
        logger.info("Dropped %s", deployment_id)
    else:
        # FAILED or TERMINATED: it holds no GPUs and cannot be dropped.
        logger.info("%s is %s and holds no GPUs", deployment_id, dropped.get("status"))

if __name__ == "__main__":
    chz.nested_entrypoint(main)
