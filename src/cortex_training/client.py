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

"""HTTP client for the Cortex Training REST API.

Single-purpose, production-shaped client:

- Snowflake connection profiles — see
  ``CortexTrainingClient.from_connection_name``.
- Direct Programmatic Access Token (PAT) compatibility — see
  ``CortexTrainingClient.from_pat``.
- One CreateJob shape: a list of typed :class:`SubJobConfig` (each carries either
  a :class:`TrainingConfig` or an :class:`InferenceConfig`). The client-side
  validators mirror the server's own required-field checks so an invalid job
  fails locally instead of after a round trip.
- One wire format: ``{job_id?, sub_job_configs:[SubJobConfig]}`` with per-sub-job
  ``training_config`` / ``inference_config`` blocks. See
  ``docs/reference/rest-api.md`` for the full schema.

The server keeps the config blocks open (``additionalProperties``), so unknown
keys pass through unchanged. The strict, typed shape lives here in the client;
use the ``extra`` fields for anything the typed layer does not model yet.
"""

from __future__ import annotations

import base64
import contextlib
import email.utils
import functools
import gzip
import hashlib
import inspect
import json
import logging
import math
import os
import re
import shutil
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from datetime import datetime
from datetime import timezone
from enum import Enum
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any
from typing import TypeVar
from urllib.parse import quote
from urllib.parse import unquote

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field as PydanticField
from pydantic import model_validator

import requests
from requests.adapters import HTTPAdapter
from tenacity import Retrying
from tenacity import retry_if_exception
from tenacity import stop_after_attempt
from tenacity import wait_exponential_jitter
from urllib3.exceptions import NewConnectionError

from cortex_training import _account_jobs
from cortex_training import _artifact_download
from cortex_training._account_jobs import AccountJobListUnavailable  # noqa: F401 - exported
from cortex_training._account_jobs import InvalidPageToken  # noqa: F401 - exported
from cortex_training._artifact_download import DownloadCancelled
from cortex_training._artifact_download import NoChunksError
from cortex_training import wire
from cortex_training.optimizer import normalize_optimizer_config
from cortex_training.peft import normalize_lora_peft_config
from cortex_training.snowflake_auth import SnowflakeProfileAuth
from cortex_training.snowflake_auth import SnowflakeTelemetryTokenProvider
from cortex_training.telemetry import CachedSessionTokenProvider
from cortex_training.telemetry import OtlpMetricEmitter

logger = logging.getLogger(__name__)

_STDOUT_CHUNK_RE = re.compile(
    r"^console\.(?P<timestamp>\d{8}-\d{6})\.(?P<suffix>[^./\\]+)\.gz$"
)
_GPU_METRICS_CHUNK_RE = re.compile(
    r"^gpu\.(?P<timestamp>\d{8}-\d{6})\.(?P<suffix>[^./\\]+)\.gz$"
)
_STREAM_COPY_BUFFER_BYTES = 1024 * 1024

# Env var that unlocks create-job debug options. These are an internal-only
# capability and are deliberately not documented for external use: the client
# refuses to forward a create-job request carrying a `debug` block unless this
# is set to a truthy value. The server gates them independently.
DEBUG_OPTIONS_ENV = "CORTEX_TRAINING_ENABLE_DEBUG_OPTIONS"
DISABLE_TELEMETRY_ENV = "CORTEX_TRAINING_DISABLE_TELEMETRY"
ENABLE_SUCCESS_TELEMETRY_ENV = "CORTEX_TRAINING_ENABLE_SUCCESS_TELEMETRY"

# Escape hatch for the tensor encoding of pre-tokenized prompts. Prompts are
# packed as tensors by default because every server accepts both encodings.
# Setting this puts the token ids back in the frame's JSON header; it does not
# change the request body, which is a DSSST1 frame either way.
DISABLE_TENSOR_PROMPTS_ENV = "CORTEX_TRAINING_DISABLE_TENSOR_PROMPTS"


def _env_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _debug_options_enabled() -> bool:
    """True when create-job debug options are explicitly enabled via env var."""
    return _env_flag_enabled(DEBUG_OPTIONS_ENV)


def _telemetry_disabled() -> bool:
    """True when OTLP client metrics are explicitly disabled via env var."""
    return _env_flag_enabled(DISABLE_TELEMETRY_ENV)


# Downloads are rare, long, and user-visible, so their successful outcomes (with
# timings) are emitted unless the environment says otherwise. Other operations
# can run thousands of times per job and stay failure-only by default.
_SUCCESS_TELEMETRY_BY_DEFAULT = frozenset(
    {"download_stdout_logs", "download_metrics", "fetch_execution_logs"}
)


def _success_telemetry_enabled(operation: str | None = None) -> bool:
    """True when successful automatic operation outcomes should be emitted.

    An explicit ``CORTEX_TRAINING_ENABLE_SUCCESS_TELEMETRY`` value wins for every
    operation; unset, only the operations in ``_SUCCESS_TELEMETRY_BY_DEFAULT``
    emit successes.
    """
    if os.environ.get(ENABLE_SUCCESS_TELEMETRY_ENV, "").strip():
        return _env_flag_enabled(ENABLE_SUCCESS_TELEMETRY_ENV)
    return operation in _SUCCESS_TELEMETRY_BY_DEFAULT


# HTTP statuses worth retrying: the request was well-formed, so the same call
# may succeed once transient load/infra conditions clear.
#   429 Too Many Requests   - rate limited; back off and retry
#   500 Internal Server Err - transient server-side failure
#   502 Bad Gateway         - upstream/proxy hiccup in front of the backend
#   503 Service Unavailable - server temporarily overloaded or restarting
#   504 Gateway Timeout     - upstream didn't respond in time
#   409 Conflict            - the target zone is restarting
#   307/308 Redirect        - a leaked internal redirect; retried on the original
#                             URL, and `Location` is never followed.
#   404 Not Found           - a just-created job may not be visible yet, so a
#                             valid id can 404 briefly and succeed on retry.
#                             A 404 that names the job as not found (code
#                             517601) is final unless this client created that
#                             job moments ago (see `_definitive_not_found`), so
#                             a mistyped id fails at once. Other 404s, such as
#                             a gateway's, are still retried.
# Every other 4xx is excluded because it signals a client/config error that
# won't fix itself.
_TRANSIENT_STATUSES = {429, 500, 502, 503, 504, 404, 409, 307, 308}
_JOB_NOT_FOUND_CODE = "517601"
# Job states after which nothing more runs, and a sub-job's pods (and so its
# live logs) are gone. The API reports `cancelled`, `terminated` and `failed`;
# the other spellings are accepted for robustness.
_ENDED_JOB_STATES = frozenset(
    {"failed", "cancelled", "canceled", "terminated", "done", "completed", "succeeded"}
)
# A job this client created is retried on "not found" for this long, to ride
# out the service's read-after-create window.
_RECENTLY_CREATED_SECONDS = 120.0
_SESSION_EXPIRED_CODES = frozenset({"390111", "390112", "390114"})
_CHUNK_GROUP_RESTART_REQUIRED = "chunk_group_restart_required"
_CHUNK_GROUP_ERROR_CODES = {
    _CHUNK_GROUP_RESTART_REQUIRED,
    "chunk_group_conflict",
    "chunk_group_missing_chunks",
}

# Ceiling on one ``/operation`` JSON request body. A binary payload is
# base64-encoded and inflated 4/3, so this default is conservative relative
# to the 60 MiB octet-stream budget. Raise it via the env var once the real
# ceiling is measured.
_OPERATION_MAX_JSON_BYTES_ENV = "CORTEX_TRAINING_FORWARD_OPERATION_MAX_JSON_BYTES"
_DEFAULT_OPERATION_MAX_JSON_BYTES = 16 * 1024 * 1024
_MIN_OPERATION_MAX_JSON_BYTES = 16 * 1024

# Headroom for everything in the envelope that is not the base64 payload:
# ``operation_type``, the routing hints, the JSON punctuation and key names.
_OPERATION_ENVELOPE_OVERHEAD_BYTES = 4096


def _operation_max_json_bytes() -> int:
    """Read the per-request ``/operation`` JSON body ceiling from the env."""
    raw = os.environ.get(_OPERATION_MAX_JSON_BYTES_ENV, "").strip()
    if not raw:
        return _DEFAULT_OPERATION_MAX_JSON_BYTES
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{_OPERATION_MAX_JSON_BYTES_ENV} must be an integer, got {raw!r}"
        ) from exc
    if value < _MIN_OPERATION_MAX_JSON_BYTES:
        raise ValueError(
            f"{_OPERATION_MAX_JSON_BYTES_ENV} must be at least "
            f"{_MIN_OPERATION_MAX_JSON_BYTES} bytes"
        )
    return value


def _operation_chunk_max_bytes(max_json_bytes: int) -> int:
    """Raw frame bytes that fit one ``/operation`` body once base64-inflated."""
    return max(1, (max_json_bytes - _OPERATION_ENVELOPE_OVERHEAD_BYTES) * 3 // 4)


def _operation_body_json_bytes(body: dict) -> int:
    """Encoded size of an operation body without re-serializing its payload.

    ``json.dumps`` on a body holding a multi-megabyte base64 string copies that
    string. The base64 alphabet needs no JSON escaping and is single-byte in
    UTF-8, so the payload's encoded length is its character count: measure the
    envelope around an empty placeholder and add that count back.
    """
    payload = body.get("payload")
    raw = payload.get("payload_b64") if isinstance(payload, dict) else None
    if not isinstance(raw, str):
        return len(json.dumps(body).encode("utf-8"))
    skeleton = {**body, "payload": {**payload, "payload_b64": ""}}
    return len(json.dumps(skeleton).encode("utf-8")) + len(raw)


def _log_safe_operation_body(body: dict) -> dict:
    """Replace an operation body's base64 payload with its length for logging."""
    payload = body.get("payload")
    if not isinstance(payload, dict) or not isinstance(payload.get("payload_b64"), str):
        return body
    return {
        **body,
        "payload": {
            **payload,
            "payload_b64": f"<{len(payload['payload_b64'])} base64 chars>",
        },
    }


class ChunkGroupError(requests.exceptions.HTTPError):
    """Base class for structured chunk-group failures."""

    def __init__(self, message: str, *, response: requests.Response, detail: dict):
        super().__init__(
            message,
            response=response,
            request=getattr(response, "request", None),
        )
        self.detail = detail


class ChunkGroupRestartError(ChunkGroupError):
    """The server lost an incomplete group and the bounded replay also failed."""


class ChunkGroupConflictError(ChunkGroupError):
    """The same chunk-group identity was reused inconsistently."""


# Default per-request HTTP timeout, as a (connect, read) pair in seconds. The
# read leg accommodates long-running data-plane calls; fast control-plane calls
# such as GetCapacity apply a smaller bound explicitly.
_DEFAULT_REQUEST_TIMEOUT = (30.0, 600.0)
_CAPACITY_REQUEST_TIMEOUT = (10.0, 30.0)


def _validated_request_timeout(
    request_timeout: float | tuple[float, float],
) -> tuple[float, float]:
    values = (
        request_timeout
        if isinstance(request_timeout, tuple)
        else (request_timeout, request_timeout)
    )
    if len(values) != 2 or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
        for value in values
    ):
        raise ValueError(
            "request_timeout must be a positive finite number of seconds, or a "
            f"(connect, read) pair of them; got {request_timeout!r}"
        )
    return float(values[0]), float(values[1])


class _DefaultTimeoutSession(requests.Session):
    """Apply a default timeout to every request that does not provide one.

    Redirects are never followed: a leaked internal redirect points at an
    address the client cannot reach, so the original URL is retried instead.
    """

    def __init__(self, default_timeout: tuple[float, float]):
        super().__init__()
        self.default_timeout = default_timeout

    def request(
        self,
        *args: Any,
        timeout: Any = None,
        **kwargs: Any,
    ) -> requests.Response:  # type: ignore[override]
        # Session.get() injects allow_redirects=True before it calls request().
        kwargs["allow_redirects"] = False
        return super().request(
            *args,
            timeout=self.default_timeout if timeout is None else timeout,
            **kwargs,
        )


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
        return True
    if isinstance(exc, requests.exceptions.HTTPError):
        resp = exc.response
        return resp is not None and resp.status_code in _TRANSIENT_STATUSES
    return False


def _iter_error_dicts(value: Any, *, depth: int = 0) -> Iterator[dict]:
    """Yield dictionaries from direct or service-wrapped JSON error bodies."""
    if depth > 8:
        return
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_error_dicts(child, depth=depth + 1)
        return
    if isinstance(value, list):
        for child in value:
            yield from _iter_error_dicts(child, depth=depth + 1)
        return
    if not isinstance(value, str):
        return

    decoder = json.JSONDecoder()
    for index, char in enumerate(value):
        if char not in "[{":
            continue
        try:
            decoded, _ = decoder.raw_decode(value[index:])
        except (TypeError, ValueError):
            continue
        yield from _iter_error_dicts(decoded, depth=depth + 1)


def _is_session_auth_expired_response(response: requests.Response) -> bool:
    """Return whether the API explicitly rejected an expired session token."""
    structured_code_seen = False
    try:
        body = response.json()
    except Exception:
        body = None
    for candidate in _iter_error_dicts(body):
        code = candidate.get("code") or candidate.get("error_code")
        if code is None:
            continue
        structured_code_seen = True
        if str(code) in _SESSION_EXPIRED_CODES:
            return True
    # A bare 401 is an authentication rejection, but a structured non-expiry
    # code must not be turned into a reconnect loop.
    return response.status_code == 401 and not structured_code_seen


def _format_artifact_bytes(nbytes: int) -> str:
    """``1.1 MiB``-style size, with the unit chosen to fit."""
    value = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{int(value)} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GiB"


def _write_private_json(path: Path, doc: Mapping[str, Any]) -> None:
    """Atomically replace ``path`` with ``doc`` as JSON, readable only by its owner."""
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + "-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(doc, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class _LogEnded(Exception):
    """Internal: the job ended while ``stream_logs`` was reading its log."""


def _response_error_code(response: requests.Response | None) -> str | None:
    """The first structured service error code in ``response``'s JSON body."""
    if response is None:
        return None
    try:
        body = response.json()
    except Exception:
        return None
    for candidate in _iter_error_dicts(body):
        code = candidate.get("code") or candidate.get("error_code")
        if code is not None:
            return str(code)
    return None


def _chunk_group_error_detail(response: requests.Response | None) -> dict | None:
    if response is None:
        return None
    try:
        body = response.json()
    except Exception:
        return None
    for candidate in _iter_error_dicts(body):
        if candidate.get("code") in _CHUNK_GROUP_ERROR_CODES:
            return candidate
        message = str(candidate.get("message") or "")
        if candidate.get("chunk_group_id") and message == "request chunk group is missing chunks":
            return {
                **candidate,
                "code": "chunk_group_missing_chunks",
            }
    return None


_METRIC_IDENTITY_FIELDS = (
    "job_id",
    "sub_job_id",
    "sub_job_type",
    "source_job_id",
    "source_sub_job_id",
    "target_sub_job_id",
    "target_sub_job_ids",
    "request_id",
    "checkpoint_id",
)
_REQUEST_ID_RESULT_OPERATIONS = {
    "forward_backward",
    "forward",
    "generate",
    "generate_stream",
    "step",
    "save",
    "load",
    "weight_sync",
    "bootstrap_router_replay",
    "router_replay_discard",
    "reset_prefix_cache",
    "cancel_request",
}
_METRIC_ERROR_MESSAGE_LIMIT = 512
_SECRET_IN_ERROR_RE = re.compile(
    r"(?i)(authorization[\"']?\s*[:=]\s*"
    r"(?:bearer\s+|snowflake\s+token\s*=\s*)?|"
    r"bearer\s+|snowflake\s+token\s*=\s*|"
    r"(?:password|secret|credential|pat|api[_-]?key|private[_-]?key|"
    r"access[_-]?token|refresh[_-]?token|session[_-]?token|token)"
    r"[\"']?\s*[:=]\s*)"
    r"(?:\"[^\"]*\"|'[^']*'|\S+)"
)


def _safe_metric_error_message(exc: BaseException) -> str:
    """Return a bounded, single-line error message with obvious secrets redacted."""
    message = " ".join(str(exc).split())
    message = _SECRET_IN_ERROR_RE.sub(r"\1<redacted>", message)
    return message[:_METRIC_ERROR_MESSAGE_LIMIT]


_SNOWFLAKE_REQUEST_ID_HEADER = "x-snowflake-request-id"
# Bound per-operation history so long polls (wait_for_job) stay cheap.
_MAX_TRACKED_SNOWFLAKE_REQUEST_IDS = 10


def _response_snowflake_request_id(response: Any) -> str | None:
    headers = getattr(response, "headers", None) or {}
    request_id = headers.get(_SNOWFLAKE_REQUEST_ID_HEADER)
    return request_id if isinstance(request_id, str) and request_id else None


def _add_snowflake_request_id_to_error(
    exc: BaseException, request_id: str | None
) -> None:
    """Prefix the Snowflake request id to an error message so tracebacks carry it.

    The id goes first so it survives truncation of long messages, e.g. the
    512-character telemetry ``error_message``.
    """
    if not request_id or not exc.args or not isinstance(exc.args[0], str):
        return
    if request_id in exc.args[0]:
        return
    exc.args = (
        f"[snowflake request id: {request_id}] {exc.args[0]}",
        *exc.args[1:],
    )


def _snowflake_request_attributes(
    error: BaseException | None,
    request_ids: list[str],
) -> dict[str, Any]:
    """Snowflake request id attributes for an operation outcome.

    ``error.snowflake_request_id`` is the failing response's id, set only when
    that response carried the header. ``snowflake.request_ids`` is the
    operation's recent Snowflake request ids, newest last; no entry in it is
    implied to have failed.
    """
    attributes: dict[str, Any] = {}
    if error is not None:
        request_id = _response_snowflake_request_id(getattr(error, "response", None))
        if request_id is not None:
            attributes["error.snowflake_request_id"] = request_id
            # Deprecated name for the same value; remove after one release.
            attributes["snowflake.request_id"] = request_id
    if request_ids:
        attributes["snowflake.request_ids"] = list(request_ids)
    return attributes


_E = TypeVar("_E", bound=BaseException)
_SERVER_FAILURE_ATTR = "_snowflake_server_failure"


def _server_failure(exc: _E, failed: bool = True) -> _E:
    """Mark ``exc`` as caused by a server outcome when ``failed`` is true.

    Marked errors (a request or job that ended ``failed``, a poll that timed
    out, a response payload that could not be decoded) carry no response of
    their own, so :func:`_log_untagged_request_id` names the operation's last
    Snowflake request id for them.
    """
    if failed:
        try:
            setattr(exc, _SERVER_FAILURE_ATTR, True)
        except Exception:
            pass
    return exc


@contextlib.contextmanager
def _server_payload() -> Iterator[None]:
    """Treat decode errors in the enclosed code as server failures."""
    try:
        yield
    except ValueError as exc:
        raise _server_failure(exc)


def _log_untagged_request_id(
    operation: str,
    exc: BaseException,
    request_ids: list[str],
) -> None:
    """Log the last Snowflake request id for a server-side failure with no response.

    This is the fallback for :func:`_add_snowflake_request_id_to_error`, which
    tags any error that has a response. It covers:

    - transport errors with no response, such as a connection error or
      timeout after retries;
    - errors marked by :func:`_server_failure`: a request or job that ended
      ``failed``, a poll that timed out, or an undecodable response payload.

    It logs ``request_ids[-1]`` labeled "last" because that request did not
    necessarily fail. It logs nothing when the operation made no Snowflake
    requests, when the error has a response (already tagged, or the response
    lacked the header), or for any other error, such as client-side
    validation or a job that reached ``done`` or ``cancelled``. Telemetry
    still records ``snowflake.request_ids`` for those.
    """
    try:
        if getattr(exc, "response", None) is not None or not request_ids:
            return
        if not (
            isinstance(exc, requests.exceptions.RequestException)
            or getattr(exc, _SERVER_FAILURE_ATTR, False)
        ):
            return
        request_id = request_ids[-1]
        if request_id in str(exc):
            return
        logger.warning(
            "[last snowflake request id: %s] %s failed: %s",
            request_id,
            operation,
            _safe_metric_error_message(exc),
        )
    except Exception:
        logger.debug("client operation request id logging failed", exc_info=True)


def _metric_error_attributes(exc: BaseException) -> dict[str, Any]:
    attributes: dict[str, Any] = {"error.type": type(exc).__name__}
    response = getattr(exc, "response", None)
    if response is None:
        return attributes
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        attributes["error.http_status"] = status_code
    code = _response_error_code(response)
    if code is not None:
        attributes["error.code"] = code
    return attributes


def _metric_identity(
    operation: str,
    arguments: Mapping[str, Any],
    result: Any = None,
) -> dict[str, Any]:
    attributes = {}
    for field in _METRIC_IDENTITY_FIELDS:
        value = arguments.get(field)
        if value is not None:
            attributes[field] = (
                list(value) if isinstance(value, (list, tuple)) else value
            )
    if operation == "create_job" and "job_id" not in attributes:
        if isinstance(result, str):
            attributes["job_id"] = result
        elif isinstance(result, dict) and result.get("job_id") is not None:
            attributes["job_id"] = result["job_id"]
    if operation in _REQUEST_ID_RESULT_OPERATIONS and "request_id" not in attributes:
        if isinstance(result, str):
            attributes["request_id"] = result
        elif isinstance(result, dict) and result.get("request_id") is not None:
            attributes["request_id"] = result["request_id"]
    return attributes


def _track_operation(operation: str):
    """Record one outcome metric around a public client operation."""

    def decorate(method):
        signature = inspect.signature(method)

        @functools.wraps(method)
        def wrapped(self, *args, **kwargs):
            state = getattr(self, "_operation_metric_state", None)
            if state is None:
                return method(self, *args, **kwargs)
            active = getattr(state, "active", set())
            if active:
                return method(self, *args, **kwargs)
            # Snowflake request ids are tracked even with telemetry off so a
            # failure can still be logged with the id; only metrics need the
            # arguments.
            arguments = None
            if getattr(self, "_metric_emitter", None) is not None:
                try:
                    bound = signature.bind(self, *args, **kwargs)
                    bound.apply_defaults()
                    arguments = bound.arguments
                except TypeError:
                    return method(self, *args, **kwargs)
                except Exception:
                    logger.debug("client operation metric setup failed", exc_info=True)
                    return method(self, *args, **kwargs)
            active = set()
            active.add(operation)
            state.active = active
            state.attempt_count = 0
            state.request_count = 0
            state.snowflake_request_ids = []
            state.values = {}
            started = time.monotonic()
            try:
                result = method(self, *args, **kwargs)
            except (DownloadCancelled, KeyboardInterrupt):
                # The caller asked to stop: not a failure.
                if arguments is not None:
                    self._emit_operation_outcome(
                        operation, started, arguments, cancelled=True
                    )
                raise
            except Exception as exc:
                # Errors wrapped after _send (e.g. chunk-group errors) still
                # carry the failing response.
                _add_snowflake_request_id_to_error(
                    exc, _response_snowflake_request_id(getattr(exc, "response", None))
                )
                _log_untagged_request_id(operation, exc, state.snowflake_request_ids)
                if arguments is not None:
                    self._emit_operation_outcome(
                        operation,
                        started,
                        arguments,
                        error=exc,
                    )
                raise
            else:
                if arguments is not None:
                    self._emit_operation_outcome(
                        operation,
                        started,
                        arguments,
                        result=result,
                    )
                return result
            finally:
                state.active = set()
                state.attempt_count = 0
                state.request_count = 0
                state.snowflake_request_ids = []
                state.values = {}

        return wrapped

    return decorate


def _is_chunk_post_transient(exc: BaseException) -> bool:
    if isinstance(exc, requests.exceptions.HTTPError):
        if _chunk_group_error_detail(exc.response) is not None:
            return False
    return _is_transient(exc)


def _is_account_list_transient(exc: BaseException) -> bool:
    # The account-level job list answers 404 when it is not enabled for the
    # account, which no retry fixes.
    if isinstance(exc, requests.exceptions.HTTPError):
        if exc.response is not None and exc.response.status_code == 404:
            return False
    return _is_transient(exc)


# ─── Account-level job list: one page at a time ──────────────────────────


def _account_list_error(
    exc: requests.exceptions.HTTPError, *, page_token_sent: bool
) -> requests.exceptions.HTTPError | None:
    """The typed error for an account-list page failure, or None to raise ``exc`` as is."""
    response = exc.response
    if response is None:
        return None
    try:
        body = response.json()
    except Exception:
        body = None
    messages = [
        candidate["message"]
        for candidate in _iter_error_dicts(body)
        if isinstance(candidate.get("message"), str)
    ]
    error = _account_jobs.page_error(
        response, exc.request, messages=messages, page_token_sent=page_token_sent
    )
    if error is not None:
        _add_snowflake_request_id_to_error(error, _response_snowflake_request_id(response))
    return error


def _is_connect_error(exc: BaseException) -> bool:
    """Only failures proving the request never reached the server (safe for create_job)."""
    if isinstance(exc, requests.exceptions.ConnectTimeout):
        return True
    if isinstance(exc, requests.exceptions.ConnectionError):
        cause = exc.args[0] if exc.args else None
        reason = getattr(cause, "reason", None)
        return isinstance(cause, NewConnectionError) or isinstance(reason, NewConnectionError)
    return False


# ─── Canonical types ─────────────────────────────────────────────────────


class JobType(str, Enum):
    """Sub-job types defined by the Cortex Training schema.

    Matches the ``job_type`` enum in the REST schema; see
    ``docs/reference/rest-api.md`` section 8.1. Not every schema value can
    currently be submitted: ``LOG_PROBABILITY`` configs still validate and
    serialize, but the client and CLI reject them at submission.
    """

    TRAINING = "training"
    SAMPLING = "sampling"
    LOG_PROBABILITY = "log_probability"  # Schema value; submission is unsupported.


_LOG_PROBABILITY_JOB_TYPE_ALIASES = frozenset({"log_probability", "log_prob"})
_UNSUPPORTED_LOG_PROBABILITY_MESSAGE = (
    "log_probability sub-jobs are not currently supported"
)


def _normalized_job_type(job_type: Any) -> str:
    """Short-form ``job_type`` for raw bodies, which may use the enum spelling."""
    if not isinstance(job_type, str):
        return ""
    return job_type.strip().lower().removeprefix("job_type_")


def _is_log_probability_job_type(job_type: Any) -> bool:
    return _normalized_job_type(job_type) in _LOG_PROBABILITY_JOB_TYPE_ALIASES


def _reject_log_probability_sub_job(job_type: Any, *, location: str) -> None:
    if _is_log_probability_job_type(job_type):
        raise ValueError(f"{location}: {_UNSUPPORTED_LOG_PROBABILITY_MESSAGE}")


class Hardware(str, Enum):
    """GPU hardware for create job and capacity.

    Wire values match the REST ``hardware`` field; see
    ``docs/reference/rest-api.md`` sections 5.1 and 5.4. Omitting it defaults to
    H200 on the server. Unknown values are rejected client-side.
    """

    H200 = "H200"
    B200 = "B200"
    B300 = "B300"


def _hardware_wire(hardware: Hardware | str) -> str:
    """Return the wire spelling for ``hardware``."""
    if isinstance(hardware, Hardware):
        return hardware.value
    try:
        return Hardware(hardware).value
    except ValueError as exc:
        allowed = ", ".join(member.value for member in Hardware)
        raise ValueError(f"hardware must be one of {allowed}, got {hardware!r}") from exc


# ``get_capacity_breakdown`` dimensions, statuses, and the job types always
# reported under ``job_type``, in order.
_CAPACITY_BREAKDOWN_DIMENSIONS = ("job_type", "submitted_by")
_CAPACITY_BREAKDOWN_JOB_TYPES = ("TRAINING", "SAMPLING", "LOG_PROBABILITY")


def _capacity_totals(body: Mapping[str, Any]) -> dict:
    """Return the six ``/capacity`` totals, filling proto3-omitted defaults."""
    return {
        "has_reservation": bool(body.get("has_reservation", False)),
        "max_total_gpus": int(body.get("max_total_gpus", 0)),
        "reserved_gpus": int(body.get("reserved_gpus", 0)),
        "in_use_gpus": int(body.get("in_use_gpus", 0)),
        "pending_gpus": int(body.get("pending_gpus", 0)),
        "available_gpus": int(body.get("available_gpus", 0)),
    }


def _capacity_breakdown_dimensions(dimensions: Any) -> tuple[str, ...]:
    """Validate ``get_capacity_breakdown(dimensions=...)`` before any request."""
    try:
        names = tuple(dimensions)
    except TypeError:
        names = ()
    if not names or not all(
        isinstance(name, str) and name in _CAPACITY_BREAKDOWN_DIMENSIONS
        for name in names
    ):
        allowed = ", ".join(_CAPACITY_BREAKDOWN_DIMENSIONS)
        raise ValueError(
            f"dimensions must be a non-empty iterable of {allowed} "
            f"(not a string), got {dimensions!r}"
        )
    # Use our own strings: a str Enum member equal to a name formats differently.
    return tuple(
        _CAPACITY_BREAKDOWN_DIMENSIONS[_CAPACITY_BREAKDOWN_DIMENSIONS.index(name)]
        for name in names
    )


def _capacity_breakdown(
    body: Mapping[str, Any], dimensions: tuple[str, ...]
) -> dict[str, Any]:
    """Normalize ``breakdown``, trusting the data over the status label:
    groups are kept whenever the server sends them, and ``status`` is passed
    through as sent. Only a missing status is derived: ``OK`` with groups,
    ``UNAVAILABLE`` without (also for a dimension the server left out)."""
    raw = body.get("breakdown")
    if not isinstance(raw, dict):
        raw = {}
    breakdown: dict[str, Any] = {
        "job_count": int(raw.get("job_count", 0)),
        "truncated": bool(raw.get("truncated", False)),
        "unsupported_dimensions": list(raw.get("unsupported_dimensions") or []),
    }
    for dimension in dict.fromkeys(dimensions):
        section = raw.get(f"by_{dimension}")
        if not isinstance(section, dict):
            section = {}
        groups = section.get("groups")
        has_groups = isinstance(groups, list) and bool(groups)
        status = section.get("status") or ("OK" if has_groups else "UNAVAILABLE")
        if not has_groups:
            rendered_groups = []
        elif dimension == "job_type":
            rendered_groups = _capacity_job_type_groups(groups)
        else:
            rendered_groups = [_capacity_submitted_by_group(group) for group in groups]
        breakdown[f"by_{dimension}"] = {"status": status, "groups": rendered_groups}
    return breakdown


def _capacity_job_type_groups(groups: list[Mapping[str, Any]]) -> list[dict]:
    """Known job types first, zero-filled and in order, then any others as sent."""
    counts = ("in_use_gpus", "pending_gpus", "sub_job_count")
    known: dict[str, dict[str, Any]] = {
        job_type: {"job_type": job_type, **dict.fromkeys(counts, 0)}
        for job_type in _CAPACITY_BREAKDOWN_JOB_TYPES
    }
    others = []
    for group in groups:
        job_type = group.get("job_type")
        row: dict[str, Any] | None = None
        if isinstance(job_type, str):
            job_type = job_type.removeprefix("JOB_TYPE_")
            row = known.get(job_type)
        if row is None:
            row = {"job_type": job_type, **dict.fromkeys(counts, 0)}
            others.append(row)
        for key in counts:
            row[key] += int(group.get(key, 0))
    return [*known.values(), *others]


def _capacity_submitted_by_group(group: Mapping[str, Any]) -> dict:
    """Normalize one ``by_submitted_by`` group; only a ``USER`` row names anyone."""
    user = group.get("kind") == "USER"
    return {
        "kind": group.get("kind"),
        "submitted_by": group.get("submitted_by") if user else None,
        "is_caller": bool(group.get("is_caller", False)) if user else False,
        "in_use_gpus": int(group.get("in_use_gpus", 0)),
        "pending_gpus": int(group.get("pending_gpus", 0)),
        "job_count": int(group.get("job_count", 0)),
    }


def _effective_primerl_config(extra: dict) -> dict:
    prime_rl = extra.get("prime_rl")
    if isinstance(prime_rl, dict):
        # The training runtime expands the nested prime_rl block over the training
        # config, so validate the same effective values here.
        return {**extra, **prime_rl}
    return extra


def _validate_primerl_lm_head_config(extra: dict, *, location: str) -> None:
    cfg = _effective_primerl_config(extra)
    fused_cross_entropy = cfg.get("fused_cross_entropy", "liger")
    if fused_cross_entropy is False:
        return

    token_chunk_size = cfg.get("fused_lm_head_token_chunk_size")
    token_chunked_lm_head = isinstance(token_chunk_size, int) and not isinstance(token_chunk_size, bool)
    if cfg.get("fp32_lm_head", False) or token_chunked_lm_head:
        raise ValueError(
            f"{location} cannot combine fused_cross_entropy with "
            "fp32_lm_head or fused_lm_head_token_chunk_size. "
            "Set fused_cross_entropy=false to use fp32/chunked LM head, "
            "or remove fp32/chunked LM-head knobs to use fused CE."
        )


class TrainingConfig(BaseModel):
    """Training hyperparameters for a training sub-job.

    Required fields mirror the server-side validator: ``max_seq_len > 0``,
    ``train_batch_size > 0``, ``n_gpus > 0``, and an ``optimizer`` that carries
    at least one setting. ``optimizer`` is canonicalized on construction (see
    :func:`cortex_training.optimizer.normalize_optimizer_config`), so a block
    holding only ``gradient_clipping`` is hoisted onto the typed field and
    leaves ``optimizer`` empty.

    Unknown fields are passed through to the server unchanged
    (accessible via ``model_extra``). See ``docs/reference/rest-api.md`` section 8.2.
    """

    optimizer: dict = PydanticField(description="Optimizer config dict (e.g. {name: AdamW, lr: 1e-4}).")
    max_seq_len: int = PydanticField(gt=0, description="Maximum sequence length for training.")
    train_batch_size: int = PydanticField(gt=0, description="Global batch size across all GPUs.")
    n_gpus: int = PydanticField(gt=0, description="Number of GPUs (data-parallel size).")
    gradient_clipping: float | None = PydanticField(default=None, description="Optional gradient clipping threshold.")
    multiplex_job_id: str | None = PydanticField(default=None, description="Optional multiplex job identifier.")
    load_optimizer_states: bool | None = PydanticField(default=None, description="Whether to load optimizer states from checkpoints. Set to False when changing DP size.")

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    @model_validator(mode="before")
    @classmethod
    def _flatten_legacy_extra(cls, data: Any) -> Any:
        if isinstance(data, dict) and "extra" in data:
            extra = data.pop("extra")
            if isinstance(extra, dict):
                for k, v in extra.items():
                    data.setdefault(k, v)
        return data

    @model_validator(mode="after")
    def _check_optimizer(self) -> "TrainingConfig":
        # object.__setattr__ writes the already-normalized values without
        # re-entering assignment validation. create_job posts to_wire() of this
        # instance, so the rewrite has to land here.
        optimizer, hoisted_gradient_clipping = normalize_optimizer_config(
            self.optimizer, location="training.optimizer"
        )
        object.__setattr__(self, "optimizer", optimizer)
        if self.gradient_clipping is None and hoisted_gradient_clipping is not None:
            object.__setattr__(self, "gradient_clipping", hoisted_gradient_clipping)
        # Checked after the hoist: a clip-only block canonicalizes to ``{}`` and
        # is still a complete config, so only a block that says nothing at all
        # is an error.
        if not self.optimizer and self.gradient_clipping is None:
            raise ValueError("training.optimizer is required and must be a non-empty dict")
        extra = self.model_extra or {}
        _validate_primerl_lm_head_config(extra, location="training.extra")
        prime_rl = extra.get("prime_rl")
        if isinstance(prime_rl, dict):
            _validate_primerl_lm_head_config(prime_rl, location="training.extra.prime_rl")
        if "peft_config" in extra:
            extra["peft_config"] = normalize_lora_peft_config(
                extra["peft_config"], location="training.extra.peft_config"
            )
        return self

    def validate(self) -> None:
        """Re-validate the current state. Backward-compatible with the old dataclass API."""
        self.__class__.model_validate(self.model_dump(by_alias=True))

    def to_wire(self) -> dict:
        out: dict = {
            "optimizer": self.optimizer,
            "max_seq_len": self.max_seq_len,
            "train_batch_size": self.train_batch_size,
            "n_gpus": self.n_gpus,
        }
        if self.gradient_clipping is not None:
            out["gradient_clipping"] = self.gradient_clipping
        if self.multiplex_job_id is not None:
            out["multiplex_job_id"] = self.multiplex_job_id
        if self.load_optimizer_states is not None:
            out["load_optimizer_states"] = self.load_optimizer_states
        for k, v in (self.model_extra or {}).items():
            out.setdefault(k, v)
        return out


class InferenceConfig(BaseModel):
    """Sampling/log-probability config for an inference sub-job.

    Required fields mirror the server-side validator: ``max_seq_len > 0`` and
    ``n_gpus > 0``. Unknown fields are passed through to the server unchanged
    (accessible via ``model_extra``). See ``docs/reference/rest-api.md`` section 8.3.
    """

    max_seq_len: int = PydanticField(gt=0, description="Maximum sequence length for inference.")
    n_gpus: int = PydanticField(gt=0, description="Number of GPUs for inference.")
    multiplex_job_id: str | None = PydanticField(default=None, description="Optional multiplex job identifier.")

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    @model_validator(mode="before")
    @classmethod
    def _flatten_legacy_extra(cls, data: Any) -> Any:
        if isinstance(data, dict) and "extra" in data:
            extra = data.pop("extra")
            if isinstance(extra, dict):
                for k, v in extra.items():
                    data.setdefault(k, v)
        return data

    @model_validator(mode="after")
    def _check_peft_config(self) -> "InferenceConfig":
        extra = self.model_extra or {}
        if "peft_config" in extra:
            extra["peft_config"] = normalize_lora_peft_config(
                extra["peft_config"], location="sampling.extra.peft_config"
            )
        return self

    def validate(self) -> None:
        """Re-validate the current state. Backward-compatible with the old dataclass API."""
        self.__class__.model_validate(self.model_dump(by_alias=True))

    def to_wire(self) -> dict:
        out: dict = {"max_seq_len": self.max_seq_len, "n_gpus": self.n_gpus}
        if self.multiplex_job_id is not None:
            out["multiplex_job_id"] = self.multiplex_job_id
        for k, v in (self.model_extra or {}).items():
            out.setdefault(k, v)
        return out


class SubJobConfig(BaseModel):
    """One sub-job within a CreateJob request.

    Matches the ``SubJobConfig`` schema in ``docs/reference/rest-api.md``
    section 8.1.

    Exactly one of ``training`` or ``sampling`` must be set, matching the
    sub-job's ``job_type``.
    """

    job_type: JobType = PydanticField(description="Sub-job type (training, sampling, or log_probability).")
    model_name: str = PydanticField(min_length=1, description="Model identifier (e.g. Qwen/Qwen3-8B).")
    training: TrainingConfig | None = PydanticField(default=None, description="Training config; required for training sub-jobs.")
    sampling: InferenceConfig | None = PydanticField(default=None, description="Inference config; required for sampling/log-prob sub-jobs.")
    global_batch_size: int | None = PydanticField(default=None, description="Global batch size across all GPUs.")
    dtype: str | None = PydanticField(default=None, description="Model dtype (e.g. bfloat16).")
    seed: int | None = PydanticField(default=None, description="Random seed.")
    model_post_init_ops: list[str] | None = PydanticField(default=None, alias="model_post_init", description="List of post-initialization operations.")
    source_checkpoint_info: dict | None = PydanticField(default=None, description="Checkpoint to resume from at creation time.")

    model_config = ConfigDict(extra="forbid", populate_by_name=True, validate_assignment=True)

    @model_validator(mode="after")
    def _check_job_type(self) -> "SubJobConfig":
        if self.training is not None and self.sampling is not None:
            raise ValueError("sub_job.training and sub_job.sampling are mutually exclusive")
        if self.job_type == JobType.TRAINING:
            if self.training is None:
                raise ValueError("training sub-job requires a `training` block")
        elif self.job_type in (JobType.SAMPLING, JobType.LOG_PROBABILITY):
            if self.sampling is None:
                raise ValueError(f"{self.job_type.value} sub-job requires a `sampling` block")
        return self

    @classmethod
    def training_job(
        cls,
        model_name: str,
        *,
        optimizer: dict,
        max_seq_len: int,
        train_batch_size: int,
        n_gpus: int,
        gradient_clipping: float | None = None,
        multiplex_job_id: str | None = None,
        load_optimizer_states: bool | None = None,
        extra_training: dict | None = None,
        global_batch_size: int | None = None,
        dtype: str | None = None,
        seed: int | None = None,
        model_post_init: list[str] | None = None,
        source_checkpoint_info: dict | None = None,
    ) -> "SubJobConfig":
        """Build a training :class:`SubJobConfig`. All training fields are explicit.

        Args:
            model_name: Model identifier (e.g., "Qwen/Qwen3-1.7B").
            optimizer: Optimizer config dict (e.g., {"name": "AdamW", "lr": 0.0001}).
            max_seq_len: Maximum sequence length for training.
            train_batch_size: Per-device batch size.
            n_gpus: Number of GPUs (DP size).
            gradient_clipping: Optional gradient clipping threshold.
            multiplex_job_id: Optional multiplex job identifier.
            load_optimizer_states: Whether to load optimizer states from
                checkpoints. Set to ``False`` when loading across different DP
                sizes (``n_gpus``) since optimizer states are DP-sharded and
                cannot be resized. When ``None``, uses server default (True).
                **This is a creation-time setting and cannot be changed later.**
            extra_training: Additional training config keys (passthrough).
            global_batch_size: Global batch size across all GPUs.
            dtype: Model dtype (e.g., "bfloat16").
            seed: Random seed.
            model_post_init: List of post-initialization operations.
            source_checkpoint_info: Optional checkpoint to resume from at
                creation time (use ``{"checkpoint_id": "...", "source_job_id": "..."}`).
        """
        _typed_training = set(TrainingConfig.model_fields.keys())
        passthrough_training = {k: v for k, v in (extra_training or {}).items() if k not in _typed_training}
        return cls(
            job_type=JobType.TRAINING,
            model_name=model_name,
            training=TrainingConfig(
                optimizer=optimizer,
                max_seq_len=max_seq_len,
                train_batch_size=train_batch_size,
                n_gpus=n_gpus,
                gradient_clipping=gradient_clipping,
                multiplex_job_id=multiplex_job_id,
                load_optimizer_states=load_optimizer_states,
                **passthrough_training,
            ),
            global_batch_size=global_batch_size,
            dtype=dtype,
            seed=seed,
            model_post_init=model_post_init,
            source_checkpoint_info=source_checkpoint_info,
        )

    @classmethod
    def sampling_job(
        cls,
        model_name: str,
        *,
        max_seq_len: int,
        n_gpus: int,
        multiplex_job_id: str | None = None,
        extra_sampling: dict | None = None,
        job_type: JobType = JobType.SAMPLING,
        global_batch_size: int | None = None,
        dtype: str | None = None,
        seed: int | None = None,
        model_post_init: list[str] | None = None,
        source_checkpoint_info: dict | None = None,
    ) -> "SubJobConfig":
        """Build a sampling :class:`SubJobConfig`.

        ``job_type`` remains for source compatibility, but only ``SAMPLING`` is
        currently accepted.
        """
        _reject_log_probability_sub_job(job_type, location="sampling_job().job_type")
        if job_type != JobType.SAMPLING:
            raise ValueError(f"sampling_job() only accepts SAMPLING, got {job_type!r}")
        _typed_sampling = set(InferenceConfig.model_fields.keys())
        passthrough_sampling = {k: v for k, v in (extra_sampling or {}).items() if k not in _typed_sampling}
        return cls(
            job_type=job_type,
            model_name=model_name,
            sampling=InferenceConfig(
                max_seq_len=max_seq_len,
                n_gpus=n_gpus,
                multiplex_job_id=multiplex_job_id,
                **passthrough_sampling,
            ),
            global_batch_size=global_batch_size,
            dtype=dtype,
            seed=seed,
            model_post_init=model_post_init,
            source_checkpoint_info=source_checkpoint_info,
        )

    def validate(self) -> None:
        """Re-validate the current state. Backward-compatible with the old dataclass API."""
        self.__class__.model_validate(self.model_dump(by_alias=True))

    def to_wire(self) -> dict:
        wire: dict = {
            "job_type": self.job_type.value,
            "model_name": self.model_name,
        }
        if self.global_batch_size is not None:
            wire["global_batch_size"] = self.global_batch_size
        if self.dtype is not None:
            wire["dtype"] = self.dtype
        if self.seed is not None:
            wire["seed"] = self.seed
        if self.model_post_init_ops is not None:
            wire["model_post_init"] = list(self.model_post_init_ops)
        if self.training is not None:
            wire["training_config"] = self.training.to_wire()
        if self.sampling is not None:
            wire["inference_config"] = self.sampling.to_wire()
        if self.source_checkpoint_info is not None:
            wire["source_checkpoint_info"] = self.source_checkpoint_info
        return wire


def _validate_raw_sub_job_configs(sub_job_configs: list) -> None:
    """Canonicalize every ``peft_config`` in a raw create-job body, in place.

    The typed path validates LoRA through :class:`TrainingConfig` /
    :class:`InferenceConfig`; this gives raw bodies the same guarantee so an
    adapter that weight sync cannot carry is rejected before the request is
    sent. Everything else in the body, including ``optimizer``, is forwarded as
    given.
    """
    for index, sub_job in enumerate(sub_job_configs):
        if not isinstance(sub_job, dict):
            continue
        job_type = _normalized_job_type(sub_job.get("job_type"))
        config_key = "training_config" if job_type == JobType.TRAINING.value else "inference_config"
        job_config = sub_job.get(config_key)
        peft_config = job_config.get("peft_config") if isinstance(job_config, dict) else None
        if peft_config is None:
            continue
        job_config["peft_config"] = normalize_lora_peft_config(
            peft_config,
            location=f"sub_job_configs[{index}].{config_key}.peft_config",
        )


# ─── Forward-backward payload helpers ────────────────────────────────────


def _load_torch():
    import torch

    return torch


def _tensor_dtype(torch, dtype_name: str | None):
    if dtype_name is None:
        return torch.long
    aliases = {
        "bool": torch.bool,
        "boolean": torch.bool,
        "float": torch.float32,
        "float32": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "int": torch.int32,
        "int32": torch.int32,
        "int64": torch.int64,
        "long": torch.long,
    }
    try:
        return aliases[dtype_name]
    except KeyError as exc:
        raise ValueError(f"unsupported tensor dtype: {dtype_name}") from exc


def _json_tensor(value: Any, *, default_dtype: str | None = "long"):
    torch = _load_torch()
    if isinstance(value, dict):
        if "data" not in value:
            raise ValueError("tensor object must contain a data field")
        dtype_name = value.get("dtype", default_dtype)
        value = value["data"]
    else:
        dtype_name = default_dtype
    return torch.tensor(value, dtype=_tensor_dtype(torch, dtype_name))


def _read_fwd_bwd_args(payload: dict[str, Any]) -> tuple:
    args = payload.get("args", [])
    if args is None:
        return ()
    if not isinstance(args, list):
        raise ValueError("fwd-bwd payload args must be a JSON list")
    return tuple(args)


def _tensorize_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in kwargs.items():
        if isinstance(value, dict) and "data" in value:
            out[key] = _json_tensor(value)
        elif isinstance(value, list):
            out[key] = _json_tensor(value)
        else:
            out[key] = value
    return out


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _repeat_texts(texts: Any, batch_size: Any) -> list[str]:
    if isinstance(texts, str):
        text_list = [texts]
    elif isinstance(texts, list) and texts and all(isinstance(x, str) for x in texts):
        text_list = texts
    else:
        raise ValueError("fwd-bwd payload texts must be a non-empty string list")

    if batch_size is None:
        return text_list
    batch = _positive_int(batch_size, "fwd-bwd payload batch_size")
    return [text_list[i % len(text_list)] for i in range(batch)]


def _load_tokenizer(tokenizer_spec: Any):
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "No module named 'transformers'",
            name="transformers",
        ) from exc

    if isinstance(tokenizer_spec, str):
        model_name = tokenizer_spec
        kwargs: dict[str, Any] = {}
    elif isinstance(tokenizer_spec, dict):
        model_name = tokenizer_spec.get("model_name") or tokenizer_spec.get("name") or tokenizer_spec.get("path")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("tokenizer.model_name is required")
        kwargs = {}
        for key in ("trust_remote_code", "use_fast", "revision"):
            if key in tokenizer_spec:
                kwargs[key] = tokenizer_spec[key]
    else:
        raise ValueError("fwd-bwd payload tokenizer must be a string or object")

    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    if getattr(tokenizer, "pad_token_id", None) is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is not None:
            tokenizer.pad_token = eos_token
    return tokenizer


def _tokenize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    tokenizer_spec = payload.get("tokenizer", payload.get("model_name"))
    if tokenizer_spec is None:
        raise ValueError("fwd-bwd payload requires tokenizer or input_ids")

    tokenizer = _load_tokenizer(tokenizer_spec)
    texts = _repeat_texts(payload.get("texts"), payload.get("batch_size"))
    padding = payload.get("padding", "max_length")
    truncation = payload.get("truncation", True)
    add_special_tokens = payload.get("add_special_tokens", True)
    max_length = payload.get("max_length")
    if padding == "max_length":
        max_length = _positive_int(max_length, "fwd-bwd payload max_length")
    elif max_length is not None:
        max_length = _positive_int(max_length, "fwd-bwd payload max_length")
    _require_bool(truncation, "fwd-bwd payload truncation")
    _require_bool(add_special_tokens, "fwd-bwd payload add_special_tokens")

    encode_kwargs = {
        "return_tensors": "pt",
        "padding": padding,
        "truncation": truncation,
        "add_special_tokens": add_special_tokens,
    }
    if max_length is not None:
        encode_kwargs["max_length"] = max_length
    encoded = tokenizer(texts, **encode_kwargs)
    if "input_ids" not in encoded:
        raise ValueError("tokenizer output did not include input_ids")
    return encoded


def _ensure_rank2(tensor, name: str) -> None:
    if len(tensor.shape) != 2:
        raise ValueError(f"{name} must have shape [batch, seq_len]")


def _build_position_ids(input_ids, position_spec: Any):
    torch = _load_torch()
    if position_spec is None or position_spec is False:
        return None
    if position_spec is True or position_spec == "arange":
        batch_size, seq_len = input_ids.shape
        return torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1).contiguous()
    if isinstance(position_spec, (dict, list)):
        return _json_tensor(position_spec)
    raise ValueError("fwd-bwd payload position_ids must be true, false, arange, or tensor data")


def _label_config(payload: dict[str, Any]) -> tuple[Any, int, bool]:
    labels = payload.get("labels", payload.get("label_strategy", "next_token"))
    if isinstance(labels, dict) and "data" in labels:
        return labels, int(payload.get("ignore_index", -100)), True
    if isinstance(labels, dict):
        strategy = labels.get("strategy", "next_token")
        ignore_index = int(labels.get("ignore_index", payload.get("ignore_index", -100)))
        mask_padding = labels.get("mask_padding", payload.get("mask_padding", True))
        return strategy, ignore_index, _require_bool(mask_padding, "labels.mask_padding")
    return (
        labels,
        int(payload.get("ignore_index", -100)),
        _require_bool(
            payload.get("mask_padding", True),
            "fwd-bwd payload mask_padding",
        ),
    )


def _build_labels(input_ids, attention_mask, payload: dict[str, Any]):
    torch = _load_torch()
    labels, ignore_index, mask_padding = _label_config(payload)
    if labels is None or labels == "none":
        return None
    if isinstance(labels, (dict, list)):
        return _json_tensor(labels)
    if labels in ("next_token", "shifted_input_ids"):
        out = torch.roll(input_ids, shifts=-1, dims=1)
        out[:, -1] = ignore_index
        if mask_padding and attention_mask is not None:
            target_mask = torch.roll(attention_mask, shifts=-1, dims=1)
            target_mask[:, -1] = 0
            out = out.masked_fill(target_mask == 0, ignore_index)
        return out
    if labels in ("input_ids", "self"):
        out = input_ids.clone()
        if mask_padding and attention_mask is not None:
            out = out.masked_fill(attention_mask == 0, ignore_index)
        return out
    raise ValueError("fwd-bwd labels strategy must be next_token, input_ids, none, or tensor data")


def build_forward_backward_kwargs(payload: dict[str, Any]) -> dict[str, Any]:
    """Build tensor kwargs for a forward-backward training batch.

    ``payload`` is intentionally JSON-shaped so CLI examples can stay readable:
    callers may provide either tokenizable ``texts`` plus a ``tokenizer`` or
    direct tensor data under ``input_ids`` / ``labels`` / ``position_ids``.
    """
    if not isinstance(payload, dict):
        raise ValueError("fwd-bwd payload must be a JSON object")

    if "kwargs" in payload:
        kwargs = payload["kwargs"]
        if not isinstance(kwargs, dict):
            raise ValueError("fwd-bwd payload kwargs must be an object")
        return _tensorize_kwargs(kwargs)

    torch = _load_torch()
    encoded = None
    if "input_ids" in payload:
        input_ids = _json_tensor(payload["input_ids"])
        attention_mask = (
            _json_tensor(payload["attention_mask"])
            if isinstance(payload.get("attention_mask"), (dict, list))
            else None
        )
    else:
        encoded = _tokenize_payload(payload)
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")

    _ensure_rank2(input_ids, "input_ids")
    if attention_mask is not None:
        _ensure_rank2(attention_mask, "attention_mask")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")

    kwargs: dict[str, Any] = {"input_ids": input_ids.contiguous()}

    include_attention = payload.get("include_attention_mask", False)
    if "attention_mask" in payload and isinstance(payload["attention_mask"], bool):
        include_attention = payload["attention_mask"]
    include_attention = _require_bool(
        include_attention,
        "fwd-bwd payload include_attention_mask",
    )
    if include_attention:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        kwargs["attention_mask"] = attention_mask.contiguous()

    position_spec = payload.get("position_ids")
    if position_spec is None and payload.get("include_position_ids", False):
        position_spec = "arange"
    position_ids = _build_position_ids(input_ids, position_spec)
    if position_ids is not None:
        _ensure_rank2(position_ids, "position_ids")
        if position_ids.shape != input_ids.shape:
            raise ValueError("position_ids must have the same shape as input_ids")
        kwargs["position_ids"] = position_ids.contiguous()

    labels = _build_labels(input_ids, attention_mask, payload)
    if labels is not None:
        _ensure_rank2(labels, "labels")
        if labels.shape != input_ids.shape:
            raise ValueError("labels must have the same shape as input_ids")
        kwargs["labels"] = labels.contiguous()

    return kwargs


def serialize_forward_backward_args(args: tuple = (), kwargs: dict[str, Any] | None = None) -> bytes:
    """Serialize ``{"args": args, "kwargs": kwargs}`` for Cortex Training fwd-bwd."""
    if not isinstance(args, tuple):
        raise ValueError("forward-backward args must be a tuple")
    if kwargs is None:
        kwargs = {}
    if not isinstance(kwargs, dict):
        raise ValueError("forward-backward kwargs must be a dict")

    return wire.dumps(
        {"args": args, "kwargs": kwargs},
        metadata={"response_options": {"format": "dssst1", "delivery": "chunked"}},
    )


def build_forward_backward_payload(spec: dict[str, Any]) -> bytes:
    """Build a serialized fwd-bwd byte payload from readable JSON."""
    if not isinstance(spec, dict):
        raise ValueError("fwd-bwd JSON must be an object")
    payload = spec.get("payload", spec)
    if not isinstance(payload, dict):
        raise ValueError("fwd-bwd payload must be an object")
    args = _read_fwd_bwd_args(payload)
    kwargs = build_forward_backward_kwargs(payload)
    return serialize_forward_backward_args(args, kwargs)


# ─── Generate prompt helpers ─────────────────────────────────────────────

# Wire dtype for pre-tokenized prompts. int32, not int64: a token id needs 18
# bits for today's vocabularies, and the frame size is the thing being cut.
# Measured on 8 prompts x 32,768 tokens, int32 gives 4.00 B/token and a 1.05 MB
# frame; int64 gives 8.00 B/token and 2.10 MB, which is larger than the
# 7.00 B/token of the JSON encoding it replaces.
_PROMPT_TOKEN_DTYPE_NAME = "int32"


def _tensor_prompts_disabled() -> bool:
    """True when tensor packing of pre-tokenized prompts is disabled via env var."""
    return _env_flag_enabled(DISABLE_TENSOR_PROMPTS_ENV)


def _as_token_tensor(prompt: list[int]):
    """One pre-tokenized prompt as a wire-dtype tensor.

    Falls back to int64 rather than failing. A token id outside int32 means a
    vocabulary of over two billion, so this should never fire -- but torch
    raises on overflow instead of wrapping, and turning a working request into
    an exception is a worse trade than sending a larger frame.
    """
    torch = _load_torch()
    try:
        return torch.tensor(prompt, dtype=_tensor_dtype(torch, _PROMPT_TOKEN_DTYPE_NAME))
    except (RuntimeError, OverflowError):
        logger.warning(
            "token id outside %s; sending this prompt as int64, which doubles its wire size",
            _PROMPT_TOKEN_DTYPE_NAME,
        )
        return torch.tensor(prompt, dtype=torch.int64)


def _is_token_id(value: object) -> bool:
    """True for an integer token id. ``bool`` is an ``int`` subclass and is not one."""
    return isinstance(value, int) and not isinstance(value, bool)


def _require_token_ids(prompt: list, *, location: str) -> None:
    for index, value in enumerate(prompt):
        if not _is_token_id(value):
            raise ValueError(f"{location}[{index}] must be an integer token id, got {value!r}")


def _pack_token_prompts(prompts):
    """Move pre-tokenized prompts into the frame's tensor section.

    ``wire`` routes a tensor to the safetensors tensor section and everything
    else into the header as JSON, so a ``list[int]`` prompt travels as JSON
    text inside what is nominally a binary frame: roughly 7 bytes per token,
    plus a per-token encode cost paid inside the caller's submit latency.
    Handing ``wire`` tensors instead puts the token ids in the section built
    for them.

    Strings are left alone -- they are not token ids and the server tokenizes
    them. Empty lists are left alone so the server's own non-empty check is
    the one that rejects them, rather than this becoming a second, differently
    worded validation.
    """
    if _tensor_prompts_disabled():
        return prompts
    torch = _load_torch()
    if torch.is_tensor(prompts) or not isinstance(prompts, list) or not prompts:
        return prompts
    # A flat list of token ids is one prompt. Every element has to be an int;
    # a later float or bool would otherwise be truncated into the tensor.
    if _is_token_id(prompts[0]):
        _require_token_ids(prompts, location="prompts")
        return _as_token_tensor(prompts)
    packed = []
    for index, prompt in enumerate(prompts):
        if isinstance(prompt, list) and prompt and _is_token_id(prompt[0]):
            _require_token_ids(prompt, location=f"prompts[{index}]")
            packed.append(_as_token_tensor(prompt))
        else:
            packed.append(prompt)
    return packed


def _sql_string_literal(value: str) -> str:
    """Return ``value`` as a single-quoted Snowflake SQL string."""
    return "'" + value.replace("'", "''") + "'"


_PLAIN_IDENTIFIER_RE = re.compile(r"[A-Z_][A-Z0-9_$]*")


def _sql_identifier(name: str) -> str:
    """Return ``name`` as a Snowflake identifier that resolves to exactly ``name``.

    A plain uppercase name stays bare; any other is double-quoted, with embedded
    quotes doubled, so its case and characters survive identifier resolution.
    """
    if _PLAIN_IDENTIFIER_RE.fullmatch(name):
        return name
    return '"' + name.replace('"', '""') + '"'


def _created_param(value: date | datetime | str | None) -> str | None:
    """Render a ``created_after`` / ``created_before`` query value.

    A ``datetime`` becomes UTC RFC 3339 (a naive one is taken as UTC), a
    ``date`` stays ``YYYY-MM-DD`` (UTC midnight on the server), and a string is
    sent as given.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    return value


def _validate_artifact_relative_path(path: str) -> str:
    """Reject paths that could escape or ambiguously address the run stage."""
    candidate = PurePosixPath(path)
    if (
        not path
        or path.startswith("/")
        or "\\" in path
        or any(part in {"", ".", ".."} for part in path.split("/"))
        or candidate.name in {"", ".", ".."}
    ):
        raise ValueError(f"unsafe experiment artifact path: {path!r}")
    return path


# ─── HTTP client ─────────────────────────────────────────────────────────


class CortexTrainingClient:
    """HTTP client for the Cortex Training REST API (``cortex-training``).

    Construct with a named Snowflake connection profile::

        client = CortexTrainingClient.from_connection_name("training")

    Direct PAT authentication remains available through :meth:`from_pat`::

        client = CortexTrainingClient.from_pat(
            host="ACCOUNT.snowflakecomputing.com",
            pat="...",
            database="CORTEX_TRAINING_DB",
            schema="PUBLIC",
        )

    Against a local or otherwise compatible server, construct the client
    directly with an explicit ``base_url``; this skips PAT auth::

        client = CortexTrainingClient(
            base_url="http://localhost:8084",
            database="MY_DB",
            schema="PUBLIC",
        )

    Then build one or more sub-jobs and submit::

        sub = SubJobConfig.training_job(
            model_name="Qwen/Qwen3-1.7B",
            optimizer={"name": "AdamW", "lr": 1e-5},
            max_seq_len=2048,
            train_batch_size=1,
            n_gpus=1,
        )
        job_id = client.create_job(sub_jobs=[sub])
    """

    def __init__(
        self,
        base_url: str,
        database: str,
        schema: str,
        endpoint: str = "cortex-training",
        poll_interval: float = 0.5,
        poll_timeout: float = 1800.0,
        poll_backoff_multiplier: float = 1.25,
        poll_max_interval: float = 6.0,
        pool_maxsize: int = 1024,
        max_retries: int = 10,
        request_timeout: float | tuple[float, float] = _DEFAULT_REQUEST_TIMEOUT,
    ):
        if poll_interval <= 0:
            raise ValueError("poll_interval must be > 0")
        if poll_timeout <= 0:
            raise ValueError("poll_timeout must be > 0")
        if poll_backoff_multiplier < 1.0:
            raise ValueError("poll_backoff_multiplier must be >= 1.0")
        if poll_max_interval < poll_interval:
            raise ValueError("poll_max_interval must be >= poll_interval")
        if pool_maxsize <= 0:
            raise ValueError("pool_maxsize must be > 0")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.request_timeout = _validated_request_timeout(request_timeout)
        self.base_url = base_url.rstrip("/")
        self.database = database
        self.schema = schema
        self.endpoint = endpoint
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout
        self.poll_backoff_multiplier = poll_backoff_multiplier
        self.poll_max_interval = poll_max_interval
        self.max_retries = max_retries
        self._fwd_bwd_send_count = 0
        self._fwd_bwd_request_debug: dict[str, dict[str, Any]] = {}
        self._generate_request_ids: set[str] = set()
        # Forward request ids awaiting poll, so a result that is only
        # ``{"job_id", "payload_b64"}`` (no ``wire_format``) can still be decoded.
        self._forward_request_ids: set[str] = set()
        # Per-job cache of the sampling sub-job's ``inference_config.max_seq_len``
        # (the value the backend launches vLLM with as ``max_model_len``). Used
        # to validate generate prompt lengths client-side. A cached ``None``
        # means the job was fetched but exposes no sampling max_seq_len (skip
        # validation); a missing key means it has not been resolved yet (e.g. a
        # transient get_job failure), so a later call will retry.
        self._sampling_max_seq_len: dict[str, int | None] = {}
        self._artifact_connection_config: dict[str, str] | None = None
        self._artifact_connection_factory: Callable[[], Any] | None = None
        self._auth_provider: SnowflakeProfileAuth | None = None
        self._db_ensured = False
        # job_id -> monotonic time this client created it (see _definitive_not_found).
        self._recently_created_jobs: dict[str, float] = {}
        self._recently_created_lock = threading.Lock()
        self._db_creating = False
        # Both are cleared on a for_job() child; see for_job().
        self._owns_session = True
        self._auto_create_database = True
        self._metric_emitter: OtlpMetricEmitter | None = None
        self._operation_metric_state = threading.local()
        self._session = _DefaultTimeoutSession(self.request_timeout)
        adapter = HTTPAdapter(pool_connections=pool_maxsize, pool_maxsize=pool_maxsize)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

    @property
    def auto_create_database(self) -> bool:
        """Whether a request answered "database or schema not found" creates them and retries.

        True by default. Set it to False for a client that only reads: the error
        is then raised as it is. A client from :meth:`for_job` starts with it off.
        """
        return self._auto_create_database

    @auto_create_database.setter
    def auto_create_database(self, value: bool) -> None:
        self._auto_create_database = bool(value)

    @classmethod
    def from_pat(
        cls,
        host: str,
        pat: str,
        database: str,
        schema: str,
        endpoint: str = "cortex-training",
        verify_ssl: bool = True,
        *,
        telemetry_timeout: float = 3.0,
        **kwargs: Any,
    ) -> "CortexTrainingClient":
        """Authenticate with a Programmatic Access Token (PAT).

        Sends ``Authorization: Bearer <pat>`` plus the
        ``X-Snowflake-Authorization-Token-Type: PROGRAMMATIC_ACCESS_TOKEN`` header
        on every request. No login round-trip.
        """
        client = cls(
            base_url=f"https://{host}",
            database=database,
            schema=schema,
            endpoint=endpoint,
            **kwargs,
        )
        client._session.headers["Authorization"] = f"Bearer {pat}"
        client._session.headers["X-Snowflake-Authorization-Token-Type"] = "PROGRAMMATIC_ACCESS_TOKEN"
        client._session.verify = verify_ssl
        client._artifact_connection_config = {"host": host, "pat": pat}
        if not _telemetry_disabled():
            token_provider = CachedSessionTokenProvider(
                client.base_url,
                pat,
                verify_ssl=verify_ssl,
                timeout=telemetry_timeout,
            )
            client._metric_emitter = OtlpMetricEmitter(
                client.base_url,
                token_provider,
                verify_ssl=verify_ssl,
                timeout=telemetry_timeout,
            )
        return client

    @classmethod
    def from_connection_name(
        cls,
        connection_name: str | None = None,
        *,
        database: str | None = None,
        schema: str | None = None,
        endpoint: str = "cortex-training",
        verify_ssl: bool = True,
        telemetry_timeout: float = 3.0,
        **kwargs: Any,
    ) -> "CortexTrainingClient":
        """Authenticate through a named or configured-default Snowflake profile.

        Profile discovery and authentication are delegated to
        ``snowflake-connector-python``. The live Connector session token is sent
        to the Cortex Training API, and the Connector session is recreated once
        if the API reports token expiry.
        """
        auth = SnowflakeProfileAuth(connection_name)
        resolved_database = database or auth.database
        if not resolved_database:
            auth.close()
            raise ValueError(
                "Snowflake connection profile must set database, or pass --database"
            )
        resolved_schema = schema or auth.schema or "PUBLIC"
        try:
            client = cls(
                base_url=auth.base_url,
                database=resolved_database,
                schema=resolved_schema,
                endpoint=endpoint,
                **kwargs,
            )
        except Exception:
            auth.close()
            raise
        client._auth_provider = auth
        client._artifact_connection_factory = auth.open_artifact_connection
        client._session.verify = verify_ssl
        if not _telemetry_disabled():
            client._metric_emitter = OtlpMetricEmitter(
                client.base_url,
                SnowflakeTelemetryTokenProvider(auth),
                verify_ssl=verify_ssl,
                timeout=telemetry_timeout,
            )
        return client

    @property
    def _prefix(self) -> str:
        # "$" is legal in a path segment and common in plain identifiers, so it
        # stays bare; anything else that could end or split the segment is encoded.
        return (
            f"{self.base_url}/api/v2/databases/{quote(self.database, safe='$')}"
            f"/schemas/{quote(self.schema, safe='$')}/{self.endpoint}"
        )

    def emit_metric(
        self,
        operation: str,
        value: Any = 1,
        *,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Emit a best-effort OTLP log record used as a client metric.

        Profile clients reuse their live Snowflake session token. PAT clients
        lazily exchange the PAT for a cached session token on the first call.
        Local/mock clients, or clients constructed with
        ``CORTEX_TRAINING_DISABLE_TELEMETRY`` set, treat this method as a no-op.
        All authentication, discovery, and export errors are ignored.
        """
        try:
            if self._metric_emitter is not None:
                self._metric_emitter.emit(operation, value, attributes=attributes)
        except Exception:
            return

    def close(self) -> None:
        """Flush telemetry briefly and close this client's HTTP sessions.

        A client from :meth:`for_job` borrows these, so closing it is a no-op.
        """
        if not self._owns_session:
            return
        try:
            if self._metric_emitter is not None:
                self._metric_emitter.close()
        except Exception:
            logger.debug("client telemetry close failed", exc_info=True)
        finally:
            try:
                self._session.close()
            finally:
                if self._auth_provider is not None:
                    self._auth_provider.close()

    def __enter__(self) -> "CortexTrainingClient":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _emit_operation_outcome(
        self,
        operation: str,
        started: float,
        arguments: Mapping[str, Any],
        *,
        result: Any = None,
        error: BaseException | None = None,
        cancelled: bool = False,
    ) -> None:
        """Emit one structured outcome without changing the operation result.

        A cancelled operation is reported like a success (only when success
        telemetry is enabled), with ``success`` false and ``cancelled`` true.
        Values an operation recorded with :meth:`_operation_values` are added to
        the value; the common fields take precedence.
        """
        if self._metric_emitter is None:
            return
        if error is None and not _success_telemetry_enabled(operation):
            return
        try:
            attempt_count = int(
                getattr(self._operation_metric_state, "attempt_count", 0)
            )
            request_count = int(
                getattr(self._operation_metric_state, "request_count", 0)
            )
            value: dict[str, Any] = dict(
                getattr(self._operation_metric_state, "values", None) or {}
            )
            value |= {
                "attempt_count": attempt_count,
                "duration_ms": round((time.monotonic() - started) * 1000, 3),
                "request_count": request_count,
                "retry_count": max(attempt_count - request_count, 0),
                "success": error is None and not cancelled,
            }
            if cancelled:
                value["cancelled"] = True
            attributes = _metric_identity(operation, arguments, result)
            if error is not None:
                value["error_message"] = _safe_metric_error_message(error)
                attributes.update(_metric_error_attributes(error))
            attributes.update(
                _snowflake_request_attributes(
                    error,
                    list(
                        getattr(self._operation_metric_state, "snowflake_request_ids", [])
                    ),
                )
            )
            self.emit_metric(operation, value, attributes=attributes)
        except Exception:
            logger.debug("client operation metric construction failed", exc_info=True)

    def _operation_values(self) -> dict[str, Any]:
        """Extra outcome values for the operation running on this thread.

        Write numbers, booleans, and short fixed strings only. Outside a
        tracked operation this returns a dict nobody reads.
        """
        state = getattr(self, "_operation_metric_state", None)
        values = getattr(state, "values", None) if state is not None else None
        if not isinstance(values, dict) or not getattr(state, "active", None):
            return {}
        return values

    @staticmethod
    def _debug_json_summary(value: Any, *, limit: int = 2000) -> str:
        try:
            text = json.dumps(value, sort_keys=True, default=str)
        except TypeError:
            text = repr(value)
        if len(text) > limit:
            return f"{text[:limit]}...<truncated {len(text) - limit} chars>"
        return text

    @classmethod
    def _debug_response_summary(cls, resp: requests.Response, *, limit: int = 2000) -> str:
        try:
            return cls._debug_json_summary(resp.json(), limit=limit)
        except ValueError:
            text = getattr(resp, "text", "")
        except Exception as exc:
            text = f"<failed to read response body: {type(exc).__name__}: {exc}>"
        text = str(text).replace("\n", "\\n")
        if len(text) > limit:
            return f"{text[:limit]}...<truncated {len(text) - limit} chars>"
        return text

    @staticmethod
    def _debug_context_label(debug_context: dict[str, Any]) -> str:
        parts = ["[cortex-training fwd-bwd]"]
        for key in ("phase", "call", "job_id", "request_id", "payload_bytes"):
            value = debug_context.get(key)
            if value is not None:
                parts.append(f"{key}={value}")
        return " ".join(parts)

    def _create_database(self) -> None:
        """Attempt to create the configured database (and schema if non-PUBLIC).

        Called only when an API request fails with a database-not-found error.
        Routes through ``_send`` so auth works for both PAT and connection
        profile users. Raises on failure with an actionable error message.
        """
        db_stmt = f"CREATE DATABASE IF NOT EXISTS {self.database}"
        manual_hint = (
            f"Create it manually in Snowsight: CREATE DATABASE IF NOT EXISTS {self.database}; "
            "— or if the database already exists, check that your role has USAGE on it."
        )
        self._db_creating = True  # prevent recursion from inner _send call
        try:
            logger.info("Database '%s' not found. Attempting to create it...", self.database)
            resp = self._send(
                "POST",
                f"{self.base_url}/api/v2/statements",
                json={"statement": db_stmt, "timeout": 60},
                max_retries=1,
            )
            if resp.status_code == 202:
                raise RuntimeError(
                    f"Database creation for '{self.database}' was accepted but has not "
                    f"completed yet. {manual_hint}"
                )
            logger.info("Database '%s' created successfully.", self.database)

            if self.schema.upper() != "PUBLIC":
                schema_stmt = f"CREATE SCHEMA IF NOT EXISTS {self.database}.{self.schema}"
                logger.info("Schema '%s' does not exist. Attempting to create it...", self.schema)
                schema_resp = self._send(
                    "POST",
                    f"{self.base_url}/api/v2/statements",
                    json={"statement": schema_stmt, "timeout": 60},
                    max_retries=1,
                )
                if schema_resp.status_code == 202:
                    raise RuntimeError(
                        f"Schema creation for '{self.database}.{self.schema}' was accepted but has not "
                        f"completed yet. Create it manually: CREATE SCHEMA IF NOT EXISTS {self.database}.{self.schema};"
                    )
                logger.info("Schema '%s' created successfully.", self.schema)

            self._db_ensured = True
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Could not create database '{self.database}': {exc}. {manual_hint}"
            ) from exc
        finally:
            self._db_creating = False

    @staticmethod
    def _is_database_not_found(resp: requests.Response) -> bool:
        """Return True if the response indicates the configured database does not exist.

        The Cortex Training API returns 400 with error code 517602 and a message
        like ``Schema MY_DB.PUBLIC is not found or not authorized`` when the
        database in the URL path does not exist.
        """
        if resp.status_code not in (400, 404, 422):
            return False
        try:
            text = resp.text.lower()
        except Exception:
            return False
        return "not found or not authorized" in text or "does not exist or not authorized" in text

    def _send(
        self,
        method: str,
        url: str,
        *,
        retry_on=_is_transient,
        max_retries: int | None = None,
        ensure_database: bool = True,
        **kwargs,
    ) -> requests.Response:
        """Send one request with retries.

        ``ensure_database=False`` skips creating the configured database on a
        not-found error, for routes whose path names no database. A client from
        :meth:`for_job` never creates one, whatever the caller passes.
        """
        auto_create = ensure_database and self._auto_create_database
        fn = getattr(self._session, method.lower())
        debug_context = kwargs.pop("debug_context", None)
        debug_label = self._debug_context_label(debug_context) if debug_context is not None else None
        attempt_no = 0
        auth_retry_used = False
        retry_limit = self.max_retries if max_retries is None else max_retries
        max_attempts = 1 + retry_limit
        displayed_max_attempts = max_attempts + (
            1 if self._auth_provider is not None else 0
        )
        try:
            state = self._operation_metric_state
            if getattr(state, "active", set()):
                state.request_count = getattr(state, "request_count", 0) + 1
        except Exception:
            logger.debug("client operation request counting failed", exc_info=True)

        def attempt() -> requests.Response:
            nonlocal attempt_no, auth_retry_used
            while True:
                attempt_no += 1
                try:
                    state = self._operation_metric_state
                    if getattr(state, "active", set()):
                        state.attempt_count = getattr(state, "attempt_count", 0) + 1
                except Exception:
                    logger.debug(
                        "client operation attempt counting failed", exc_info=True
                    )
                if debug_label is not None:
                    logger.debug(
                        "%s sending %s %s attempt=%d/%d",
                        debug_label,
                        method.upper(),
                        url,
                        attempt_no,
                        displayed_max_attempts,
                    )

                request_kwargs = kwargs
                observed_token: str | None = None
                if self._auth_provider is not None:
                    observed_token = self._auth_provider.get_token()
                    request_headers = dict(kwargs.get("headers") or {})
                    request_headers["Authorization"] = (
                        f'Snowflake Token="{observed_token}"'
                    )
                    request_kwargs = {**kwargs, "headers": request_headers}
                try:
                    resp = fn(url, **request_kwargs)
                except Exception as exc:
                    if debug_label is not None:
                        logger.debug(
                            "%s request exception %s %s attempt=%d/%d: %s: %s",
                            debug_label,
                            method.upper(),
                            url,
                            attempt_no,
                            displayed_max_attempts,
                            type(exc).__name__,
                            exc,
                        )
                    raise

                status_code = getattr(resp, "status_code", None)
                sf_request_id = _response_snowflake_request_id(resp)
                if sf_request_id:
                    req = getattr(resp, "request", None)
                    logger.debug(
                        "snowflake request_id=%s  %s %s  status=%d",
                        sf_request_id,
                        getattr(req, "method", method.upper()),
                        getattr(req, "path_url", url),
                        status_code,
                    )
                    try:
                        state = self._operation_metric_state
                        if getattr(state, "active", set()):
                            seen = state.snowflake_request_ids
                            seen.append(sf_request_id)
                            del seen[:-_MAX_TRACKED_SNOWFLAKE_REQUEST_IDS]
                    except Exception:
                        logger.debug(
                            "client operation request id tracking failed",
                            exc_info=True,
                        )

                if (
                    self._auth_provider is not None
                    and not auth_retry_used
                    and _is_session_auth_expired_response(resp)
                ):
                    auth_retry_used = True
                    resp.close()
                    self._auth_provider.refresh(observed_token)
                    continue

                if debug_label is not None:
                    try:
                        status_int = int(status_code)
                    except (TypeError, ValueError):
                        status_int = None
                    outcome = (
                        "successful response"
                        if status_int is not None and 200 <= status_int < 300
                        else "failed response"
                    )
                    snowflake = (
                        f" snowflake_request_id={sf_request_id}"
                        if sf_request_id
                        else ""
                    )
                    logger.debug(
                        "%s %s %s %s attempt=%d/%d status=%s%s body=%s",
                        debug_label,
                        outcome,
                        method.upper(),
                        url,
                        attempt_no,
                        displayed_max_attempts,
                        status_code,
                        snowflake,
                        self._debug_response_summary(resp),
                    )
                if (
                    auto_create
                    and not self._db_ensured
                    and not self._db_creating
                    and self._is_database_not_found(resp)
                ):
                    resp.close()
                    self._create_database()
                    # Retry the original request once after creating the DB.
                    continue
                if (
                    auto_create
                    and self._db_ensured
                    and self._is_database_not_found(resp)
                ):
                    raise RuntimeError(
                        f"Database '{self.database}' still not found after creation attempt. "
                        f"The database may already exist but your role lacks USAGE on it. "
                        f"Create it manually in Snowsight: CREATE DATABASE IF NOT EXISTS {self.database};"
                    )
                try:
                    # requests.Response.raise_for_status() intentionally ignores 3xx.
                    if 300 <= resp.status_code < 400:
                        raise requests.exceptions.HTTPError(
                            f"Unexpected redirect {resp.status_code} for url: {url}",
                            response=resp,
                            request=getattr(resp, "request", None),
                        )
                    resp.raise_for_status()
                except requests.exceptions.HTTPError as exc:
                    _add_snowflake_request_id_to_error(exc, sf_request_id)
                    raise
                return resp

        def should_retry(exc: BaseException) -> bool:
            return retry_on(exc) and not self._definitive_not_found(exc)

        retryer = Retrying(
            retry=retry_if_exception(should_retry),
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential_jitter(initial=0.5, max=10.0),
            reraise=True,
        )
        return retryer(attempt)

    def _note_created_job(self, job: Any, body: Any = None) -> None:
        """Remember a job this client just created; never raises, so it cannot
        hide a create that succeeded."""
        try:
            job_id = job.get("job_id") if isinstance(job, dict) else None
            if not (isinstance(job_id, str) and job_id) and isinstance(body, dict):
                job_id = body.get("job_id")
            if not (isinstance(job_id, str) and job_id):
                return
            now = time.monotonic()
            with self._recently_created_lock:
                created = self._recently_created_jobs
                for stale in [jid for jid, at in created.items() if now - at > _RECENTLY_CREATED_SECONDS]:
                    del created[stale]
                created[job_id] = now
        except Exception:
            logger.debug("could not record a created job", exc_info=True)

    def _definitive_not_found(self, exc: BaseException) -> bool:
        """Whether ``exc`` is the service saying the job does not exist.

        The service names a missing job with code 517601. That answer is final,
        except for a job this client created within ``_RECENTLY_CREATED_SECONDS``,
        which may not be visible yet.
        """
        if not isinstance(exc, requests.exceptions.HTTPError):
            return False
        response = exc.response
        if response is None or response.status_code != 404:
            return False
        if _response_error_code(response) != _JOB_NOT_FOUND_CODE:
            return False
        url = unquote(str(getattr(response, "url", "") or ""))
        lock = getattr(self, "_recently_created_lock", None)
        if lock is None:
            return True
        with lock:
            created = list(self._recently_created_jobs.items())
        now = time.monotonic()
        return not any(
            now - at <= _RECENTLY_CREATED_SECONDS and f"/{job_id}" in url
            for job_id, at in created
        )

    # ─── Job management ──────────────────────────────────────────────────

    @_track_operation("create_job")
    def create_job(
        self,
        sub_jobs: list[SubJobConfig],
        job_id: str | None = None,
        experiment_name: str | None = None,
        hardware: Hardware | str | None = None,
        idle_timeout_seconds: int | None = None,
        pending_timeout_seconds: int | None = None,
    ) -> str:
        """Create a job from a list of sub-jobs and return its server job_id.

        Each :class:`SubJobConfig` is validated client-side before the request
        is sent (see :meth:`SubJobConfig.validate`). A job supports zero or one
        ``training`` sub-job and any number of ``sampling`` sub-jobs;
        ``log_probability`` sub-jobs are rejected here. ``job_id`` is optional;
        when omitted the server generates one. ``experiment_name`` is optional;
        when omitted the server auto-creates an experiment for the job.
        ``hardware`` is optional (:class:`Hardware`.H200 / .B200 / .B300); when
        set, all sub-jobs in the job run on that GPU type. Omitted defaults to
        H200.
        ``idle_timeout_seconds`` bounds how long the job may sit idle before the
        server reclaims it: ``0`` disables reclamation, any other value must be
        between 300 and 604,800. ``pending_timeout_seconds`` bounds how long the
        job may wait for capacity before failing with ``pending_timeout``: between
        300 and 604,800, with no disable value. Omit either to use the server
        default, currently 30 minutes for idle and 24 hours for pending.
        """
        if not sub_jobs:
            raise ValueError("create_job requires a non-empty sub_jobs list")
        if idle_timeout_seconds is not None:
            if isinstance(idle_timeout_seconds, bool) or not isinstance(
                idle_timeout_seconds, int
            ):
                raise ValueError("idle_timeout_seconds must be an integer")
            if idle_timeout_seconds != 0 and not (
                300 <= idle_timeout_seconds <= 604_800
            ):
                raise ValueError(
                    "idle_timeout_seconds must be 0 or between 300 and 604800"
                )
        if pending_timeout_seconds is not None:
            if isinstance(pending_timeout_seconds, bool) or not isinstance(
                pending_timeout_seconds, int
            ):
                raise ValueError("pending_timeout_seconds must be an integer")
            if not (300 <= pending_timeout_seconds <= 604_800):
                raise ValueError(
                    "pending_timeout_seconds must be between 300 and 604800"
                )
        for index, sj in enumerate(sub_jobs):
            _reject_log_probability_sub_job(
                sj.job_type, location=f"sub_jobs[{index}].job_type"
            )
            SubJobConfig.model_validate(sj.model_dump(by_alias=True))
        body: dict = {"sub_job_configs": [sj.to_wire() for sj in sub_jobs]}
        if job_id is not None:
            body["job_id"] = job_id
        if experiment_name is not None:
            body["experiment_name"] = experiment_name
        if hardware is not None:
            body["hardware"] = _hardware_wire(hardware)
        if idle_timeout_seconds is not None:
            body["idle_timeout_seconds"] = idle_timeout_seconds
        if pending_timeout_seconds is not None:
            body["pending_timeout_seconds"] = pending_timeout_seconds
        return self.create_job_from_body(body)["job_id"]

    @_track_operation("create_job")
    def create_job_from_body(self, body: dict) -> dict:
        """Create a job from a raw REST CreateJob request body.

        This is useful for tooling that already has the REST JSON payload,
        while :meth:`create_job` remains the typed path for Python callers.

        A job supports zero or one ``training`` sub-job and any number of
        ``sampling`` sub-jobs; a second training sub-job or any
        ``log_probability`` sub-job raises :class:`ValueError` before the
        request is sent.

        Every ``peft_config`` present in the body is canonicalized in place and
        an invalid one raises :class:`ValueError` before the request is sent.
        The rest of the body, including ``optimizer``, is forwarded as given.
        """
        if not isinstance(body, dict):
            raise ValueError("create_job_from_body requires a JSON object")
        sub_job_configs = body.get("sub_job_configs")
        if not isinstance(sub_job_configs, list) or not sub_job_configs:
            raise ValueError("create_job_from_body requires a non-empty sub_job_configs list")
        training_sub_jobs = 0
        for index, cfg in enumerate(sub_job_configs):
            if not isinstance(cfg, dict):
                continue
            job_type = cfg.get("job_type")
            _reject_log_probability_sub_job(
                job_type, location=f"sub_job_configs[{index}].job_type"
            )
            if str(job_type or "").strip().lower() == JobType.TRAINING.value:
                training_sub_jobs += 1
                if training_sub_jobs > 1:
                    raise ValueError("at most one training sub-job is supported per job")
        _validate_raw_sub_job_configs(sub_job_configs)
        if body.get("debug") and not _debug_options_enabled():
            raise ValueError(
                "create-job debug options are an internal-only capability; set "
                f"{DEBUG_OPTIONS_ENV}=1 to send a request with a `debug` block"
            )
        resp = self._send("POST", self._prefix, json=body, retry_on=_is_connect_error)
        job = resp.json()
        self._note_created_job(job, body)
        return job

    @staticmethod
    def _normalize_job_status(status) -> str:
        """Normalize full enum names (``JOB_STATE_RUNNING``) to short form (``running``).

        Coerces to ``str`` first: a backend running ahead of the API layer can
        return a JobState enum value the API layer doesn't yet know (e.g. a
        newly added state), which is then rendered as a raw integer rather than
        a name. Tolerating it keeps :meth:`wait_for_job` polling — an unknown
        status is neither ``running`` nor terminal, so it stays in-progress —
        instead of crashing on ``int.lower()``.
        """
        return str(status or "").lower().removeprefix("job_state_")

    def _sleep_with_backoff(self, delay: float, deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return delay
        time.sleep(min(delay, remaining))
        return min(delay * self.poll_backoff_multiplier, self.poll_max_interval)

    @_track_operation("wait_for_job")
    def wait_for_job(self, job_id: str) -> dict:
        """Poll until the job reaches ``running``. Raises on terminal states or timeout."""
        deadline = time.monotonic() + self.poll_timeout
        delay = self.poll_interval
        while time.monotonic() < deadline:
            job = self.get_job(job_id)
            status = self._normalize_job_status(job.get("status", ""))
            if status == "running":
                return job
            if status in _ENDED_JOB_STATES:
                reason = job.get("reason", "")
                raise _server_failure(
                    RuntimeError(f"Job {job_id} reached terminal state '{status}': {reason}"),
                    failed=status == "failed",
                )
            delay = self._sleep_with_backoff(delay, deadline)
        raise _server_failure(
            TimeoutError(f"Job {job_id} did not become running within {self.poll_timeout}s")
        )

    @_track_operation("get_job")
    def get_job(self, job_id: str) -> dict:
        resp = self._send("GET", f"{self._prefix}/{job_id}")
        return resp.json()

    @_track_operation("list_jobs")
    def list_jobs(self, status: str | None = None) -> list:
        params = {}
        if status is not None:
            params["status"] = status
        resp = self._send("GET", self._prefix, params=params)
        return resp.json().get("jobs", [])

    def iter_jobs(
        self,
        *,
        database: str | None = None,
        schema: str | None = None,
        status: str | list[str] | None = None,
        submitted_by: str | None = None,
        created_after: date | datetime | str | None = None,
        created_before: date | datetime | str | None = None,
        page_size: int = 100,
    ) -> Iterator[dict]:
        """Yield the jobs the caller can see across the account, newest first.

        Reads the account-level ``GET /api/v2/{endpoint}`` route rather than
        this client's schema, and follows ``next_page_token`` until the server
        returns none; a page may hold fewer than ``page_size`` rows, even none.
        Arguments are checked on the call, pages fetched as you iterate:
        ``page_size`` is 1 to 1000, and ``schema`` needs ``database`` with it.
        The server applies every filter: ``database`` / ``schema`` narrow the
        scope, ``status`` is one status or a list of them (``active`` and
        ``done`` name groups), ``submitted_by`` takes a user name, and
        ``created_after`` (inclusive) / ``created_before`` (exclusive) take a
        ``date``, a ``datetime`` (sent in UTC; naive is taken as UTC) or a
        string sent as-is. Rows also carry ``database_name`` and
        ``schema_name``; use :meth:`for_job` for per-job calls on one.

        A 404 means account-level listing is not enabled for this account, and
        is raised at once rather than retried.
        """
        params = self._account_jobs_params(
            "iter_jobs",
            database=database,
            schema=schema,
            status=status,
            submitted_by=submitted_by,
            created_after=created_after,
            created_before=created_before,
            page_size=page_size,
        )
        return self._iter_account_jobs(params)

    def list_jobs_page(
        self,
        *,
        database: str | None = None,
        schema: str | None = None,
        status: str | list[str] | None = None,
        submitted_by: str | None = None,
        created_after: date | datetime | str | None = None,
        created_before: date | datetime | str | None = None,
        page_size: int = 100,
        page_token: str | None = None,
        max_retries: int | None = None,
        request_timeout: float | tuple[float, float] | None = None,
    ) -> dict:
        """Read one page of the jobs the caller can see across the account.

        Returns ``{"jobs": [...], "next_page_token": str | None}``. Jobs come
        newest first and carry ``database_name`` and ``schema_name``; use
        :meth:`for_job` for per-job calls on one. For the next page, pass
        ``next_page_token`` back as ``page_token`` with the same filters
        (``page_size`` may change), so compute a moving bound such as
        ``created_after`` once per walk; ``None`` means there is no next page.
        A page can hold fewer than ``page_size`` jobs, or none, and still carry
        a token. The filters are those of :meth:`iter_jobs`, read the same way,
        and are checked before any request.

        ``max_retries`` (retries after the first attempt) and
        ``request_timeout`` (seconds, or a ``(connect, read)`` pair) bound this
        page; ``None`` keeps the client's own.

        Raises :class:`AccountJobListUnavailable` for 403, 404 and 405, without
        a retry; :class:`InvalidPageToken` for 400 ``Invalid page_token``;
        ``requests.HTTPError`` for any other failed answer; and ``ValueError``
        for bad arguments or a page body that is not a job list.
        """
        params = self._account_jobs_params(
            "list_jobs_page",
            database=database,
            schema=schema,
            status=status,
            submitted_by=submitted_by,
            created_after=created_after,
            created_before=created_before,
            page_size=page_size,
        )
        if page_token:
            params["page_token"] = page_token
        bounds: dict[str, Any] = {}
        if max_retries is not None:
            if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
                raise ValueError(f"max_retries must be an int >= 0, got {max_retries!r}")
            bounds["max_retries"] = max_retries
        if request_timeout is not None:
            bounds["timeout"] = _validated_request_timeout(request_timeout)
        return self._list_jobs_page(params, bounds)

    @staticmethod
    def _account_jobs_params(
        caller: str,
        *,
        database: str | None,
        schema: str | None,
        status: str | list[str] | None,
        submitted_by: str | None,
        created_after: date | datetime | str | None,
        created_before: date | datetime | str | None,
        page_size: int,
    ) -> dict:
        """The account-level list's query parameters, checked before any request."""
        if not 1 <= page_size <= 1000:
            raise ValueError(f"page_size must be between 1 and 1000, got {page_size}")
        if schema is not None and database is None:
            raise ValueError(
                f"{caller}(schema=...) needs database=... with it: a schema name "
                "alone does not name one schema."
            )
        if status is not None and not isinstance(status, str):
            status = ",".join(status)
        query = {
            "database": database,
            "schema": schema,
            "status": status,
            "submitted_by": submitted_by,
            "created_after": _created_param(created_after),
            "created_before": _created_param(created_before),
            "limit": page_size,
        }
        return {key: value for key, value in query.items() if value is not None}

    @_track_operation("list_jobs_page")
    def _list_jobs_page(self, params: dict, bounds: dict[str, Any]) -> dict:
        try:
            resp = self._send(
                "GET",
                f"{self.base_url}/api/v2/{self.endpoint}",
                params=params,
                retry_on=_is_account_list_transient,
                # The path names no database, so a not-found error here is not
                # about the configured one.
                ensure_database=False,
                **bounds,
            )
        except requests.exceptions.HTTPError as exc:
            error = _account_list_error(exc, page_token_sent="page_token" in params)
            if error is None:
                raise
            raise error from exc
        with _server_payload():
            return _account_jobs.parse_page(resp.json())

    def _iter_account_jobs(self, params: dict) -> Iterator[dict]:
        while True:
            page = self._list_account_jobs_page(params)
            yield from page.get("jobs", [])
            token = page.get("next_page_token")
            if not token:
                return
            params = {**params, "page_token": token}

    # Tracked per page: around the generator, _track_operation would record
    # iter_jobs as finished before its first request.
    @_track_operation("iter_jobs")
    def _list_account_jobs_page(self, params: dict) -> dict:
        try:
            resp = self._send(
                "GET",
                f"{self.base_url}/api/v2/{self.endpoint}",
                params=params,
                retry_on=_is_account_list_transient,
                # The path names no database, so a not-found error here is not
                # about the configured one.
                ensure_database=False,
            )
        except requests.exceptions.HTTPError as exc:
            if exc.response is None or exc.response.status_code != 404:
                raise
            raise requests.exceptions.HTTPError(
                "account-level job listing is not enabled for this account",
                request=exc.request,
                response=exc.response,
            ) from exc
        return resp.json()

    def for_job(self, job: dict) -> "CortexTrainingClient":
        """Return a client bound to ``job``'s schema, for per-job calls on a
        row from :meth:`iter_jobs` or :meth:`list_jobs_page`.

        The new client borrows this client's session, auth and telemetry, so
        keep this one open and close it rather than the new one. It never
        creates the job's database, which exists and may be another role's: a
        not-found answer surfaces as the HTTP error it is.
        """
        client = type(self)(
            base_url=self.base_url,
            database=_sql_identifier(job["database_name"]),
            schema=_sql_identifier(job["schema_name"]),
            endpoint=self.endpoint,
            poll_interval=self.poll_interval,
            poll_timeout=self.poll_timeout,
            poll_backoff_multiplier=self.poll_backoff_multiplier,
            poll_max_interval=self.poll_max_interval,
            max_retries=self.max_retries,
            request_timeout=self.request_timeout,
        )
        # Release the session and pool the constructor just opened.
        client._session.close()
        client._session = self._session
        client._owns_session = False
        client._auto_create_database = False
        client._auth_provider = self._auth_provider
        client._metric_emitter = self._metric_emitter
        client._artifact_connection_config = self._artifact_connection_config
        client._artifact_connection_factory = self._artifact_connection_factory
        return client

    @_track_operation("cancel_job")
    def cancel_job(self, job_id: str) -> None:
        # The REST API uses colon-action syntax: /{jobId}:cancel
        self._send("POST", f"{self._prefix}/{job_id}:cancel")

    @_track_operation("get_capacity")
    def get_capacity(self, hardware: Hardware | str | None = None) -> dict:
        """Return the calling account's GPU ceiling and current usage.

        Backed by the account-scoped endpoint ``/cortex-training/capacity``
        (not under ``/{job_id}``). The account is resolved server-side from the
        caller's session — never from a request field — so a caller can only
        ever read its own account's capacity. Optional ``hardware``
        (:class:`Hardware`.H200 / .B200 / .B300) is sent as a query parameter and
        scopes the numbers to that GPU type; omitted defaults to H200.

        The returned dict always carries all six fields. The server emits
        proto3 JSON, which omits zero/false fields (an account holding nothing
        under a zero ceiling is literally ``{}``), so we fill in the documented
        defaults:

        - ``has_reservation`` (bool): whether ``max_total_gpus`` is a
          *guaranteed* commitment rather than a best-effort per-account cap.
          When ``False`` the account draws on the shared pool, so
          ``available_gpus`` is an upper bound.
        - ``max_total_gpus`` (int): the account's ceiling — its commitment when
          ``has_reservation``, else its per-account cap. ``-1`` means no
          ceiling, ``0`` a real quota of zero, ``> 0`` the limit. This is the
          canonical ceiling; prefer it over ``reserved_gpus``.
        - ``reserved_gpus`` (int, **deprecated**): mirrors ``max_total_gpus``
          when ``has_reservation`` is ``True``, else 0.
        - ``in_use_gpus`` (int): GPUs the account holds — jobs in ``placing``,
          ``initializing``, or ``running``. Queued work is *not* counted here;
          it is in ``pending_gpus``.
        - ``pending_gpus`` (int): GPUs requested by jobs still waiting for
          capacity (``pending``). It claims quota, so it reduces
          ``available_gpus``.
        - ``available_gpus`` (int): what a new job could start with right now —
          ceiling headroom capped by what is schedulable. ``0`` does not mean
          blocked: a submit within the ceiling is still accepted and queued.

        See ``docs/reference/rest-api.md`` section 5.4 for the authoritative
        field list, and :meth:`get_capacity_breakdown` to split ``in_use_gpus``
        and ``pending_gpus`` by sub-job type or by submitting user.
        """
        params = {}
        if hardware is not None:
            params["hardware"] = _hardware_wire(hardware)
        return _capacity_totals(self._fetch_capacity(params))

    @_track_operation("get_capacity_breakdown")
    def get_capacity_breakdown(
        self,
        hardware: Hardware | str | None = None,
        *,
        dimensions: Iterable[str] = _CAPACITY_BREAKDOWN_DIMENSIONS,
    ) -> dict:
        """Return :meth:`get_capacity` plus a ``"breakdown"`` of its GPU usage.

        Sends the same single request as :meth:`get_capacity` with
        ``?breakdown=`` naming ``dimensions``: ``"job_type"``,
        ``"submitted_by"``, or both (the default). Any other value raises
        ``ValueError`` before sending.

        ``breakdown`` holds ``job_count``, ``truncated``,
        ``unsupported_dimensions``, and ``by_job_type`` / ``by_submitted_by``
        for the requested dimensions. Each has a ``status`` (``OK``,
        ``DISABLED``, ``UNAVAILABLE``, or any newer value the server sends)
        and ``groups``; under ``OK`` the groups add up to ``in_use_gpus`` and
        ``pending_gpus``. Missing counts are 0; on non-``USER`` groups
        ``submitted_by`` is ``None`` and ``is_caller`` is ``False``.

        See ``docs/reference/rest-api.md`` section 5.4 for every field.
        """
        dimensions = _capacity_breakdown_dimensions(dimensions)
        params = {}
        if hardware is not None:
            params["hardware"] = _hardware_wire(hardware)
        params["breakdown"] = ",".join(dimensions)
        body = self._fetch_capacity(params)
        return {
            **_capacity_totals(body),
            "breakdown": _capacity_breakdown(body, dimensions),
        }

    def _fetch_capacity(self, params: Mapping[str, str]) -> Any:
        """``GET /capacity`` once, under the capacity timeout bound; return its JSON."""
        resp = self._send(
            "GET",
            f"{self._prefix}/capacity",
            timeout=(
                min(self.request_timeout[0], _CAPACITY_REQUEST_TIMEOUT[0]),
                min(self.request_timeout[1], _CAPACITY_REQUEST_TIMEOUT[1]),
            ),
            max_retries=0,
            **({"params": params} if params else {}),
        )
        return resp.json()

    @_track_operation("get_experiment_run")
    def get_experiment_run(self, job_id: str) -> dict[str, str]:
        """Return ``{experiment_name, experiment_run_name}`` for ``job_id``.

        Backed by the endpoint ``/cortex-training/{job_id}/experiment-run``.
        """
        resp = self._send("GET", f"{self._prefix}/{job_id}/experiment-run")
        return resp.json()

    def _query_sql_row(self, statement: str) -> list[Any]:
        """Execute a synchronous SQL statement and return its first row."""
        resp = self._send(
            "POST",
            f"{self.base_url}/api/v2/statements",
            json={"statement": statement, "database": self.database, "schema": self.schema},
        )
        rows = resp.json().get("data")
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], list) or not rows[0]:
            raise ValueError(f"SQL query returned no rows: {statement}")
        return rows[0]

    def _experiment_run_uri(self, job_id: str) -> tuple[str, str]:
        """Return the experiment run's ``snow://`` URI and run name."""
        run = self.get_experiment_run(job_id)
        try:
            experiment_name = run["experiment_name"]
            run_name = run["experiment_run_name"]
        except KeyError as exc:
            raise ValueError(f"experiment-run response missing field: {exc.args[0]}") from exc
        if not isinstance(experiment_name, str) or not experiment_name:
            raise ValueError("experiment-run response has invalid experiment_name")
        if not isinstance(run_name, str) or not run_name:
            raise ValueError("experiment-run response has invalid experiment_run_name")
        return (
            f"snow://experiment/{experiment_name}/versions/{run_name}/",
            run_name,
        )

    def _snowflake_connection_kwargs(self) -> dict[str, Any]:
        """Return connection kwargs derived from this client's PAT credentials."""
        config = self._artifact_connection_config
        if config is None:
            raise RuntimeError("requires a PAT-authenticated client")
        user, role = self._query_sql_row("SELECT CURRENT_USER(), CURRENT_ROLE()")
        if not isinstance(user, str) or not user:
            raise ValueError("SQL identity response missing current user")
        kwargs: dict[str, Any] = {
            "host": config["host"],
            # The login must name the account this host addresses; given the host,
            # the connector's parse_account derives it (first label, or a .global
            # host without its external id). CURRENT_ACCOUNT_NAME() is not that
            # on a locator host and that pairing returns 404.
            "account": config["host"],
            "user": user,
            "authenticator": "PROGRAMMATIC_ACCESS_TOKEN",
            "token": config["pat"],
            "database": self.database,
            "schema": self.schema,
        }
        if isinstance(role, str) and role:
            kwargs["role"] = role
        return kwargs

    def _open_experiment_artifact_connection(self) -> Any:
        """Open a connector session for Snowflake experiment artifact LIST/GET."""
        if self._artifact_connection_factory is not None:
            return self._artifact_connection_factory()
        if self._artifact_connection_config is None:
            raise RuntimeError(
                "experiment artifact download requires a Snowflake-authenticated client"
            )
        import snowflake.connector

        return snowflake.connector.connect(**self._snowflake_connection_kwargs())

    def create_snowpark_session(self) -> Any:
        """Create a Snowpark ``Session`` using this client's PAT credentials."""
        from snowflake.snowpark import Session

        return Session.builder.configs(self._snowflake_connection_kwargs()).create()

    @staticmethod
    def _list_experiment_artifacts(
        connection: Any,
        run_uri: str,
        run_name: str,
        artifact_root: str,
    ) -> list[str]:
        """List safe run-relative file paths below ``artifact_root``."""
        return sorted(
            CortexTrainingClient._list_experiment_artifact_files(
                connection, run_uri, run_name, artifact_root
            )
        )

    @staticmethod
    def _list_experiment_artifact_files(
        connection: Any,
        run_uri: str,
        run_name: str,
        artifact_root: str,
    ) -> dict[str, dict[str, Any]]:
        """Map each safe run-relative path below ``artifact_root`` to what LIST
        reported about it: ``size``, ``md5`` and ``last_modified`` (``None``
        when a column is missing)."""
        root = artifact_root.strip("/") + "/"
        _validate_artifact_relative_path(root.removesuffix("/"))
        cursor = connection.cursor()
        try:
            rows = cursor.execute(
                f"LIST {_sql_string_literal(run_uri + root)}"
            ).fetchall()
        finally:
            cursor.close()

        run_prefix = f"/versions/{run_name}/"
        files: dict[str, dict[str, Any]] = {}
        for row in rows:
            name = row[0] if row else None
            if not isinstance(name, str) or not name.startswith(run_prefix):
                raise ValueError(
                    f"experiment artifact LIST returned an unexpected path: {name!r}"
                )
            relative_path = _validate_artifact_relative_path(
                name.removeprefix(run_prefix)
            )
            if not relative_path.startswith(root):
                raise ValueError(
                    "experiment artifact LIST returned a file outside the "
                    f"requested root {root!r}: {relative_path!r}"
                )
            # LIST columns: name, size, md5, last_modified.
            size = row[1] if len(row) > 1 and isinstance(row[1], int) and not isinstance(row[1], bool) else None
            files[relative_path] = {
                "size": size,
                "md5": str(row[2]) if len(row) > 2 and row[2] else None,
                "last_modified": str(row[3]) if len(row) > 3 and row[3] else None,
            }
        return files

    @staticmethod
    def _download_experiment_artifact(
        connection: Any,
        run_uri: str,
        relative_path: str,
        target_dir: Path,
    ) -> Path:
        """Download one exact experiment artifact and return its local path."""
        relative_path = _validate_artifact_relative_path(relative_path)
        target_dir.mkdir(parents=True, exist_ok=True)
        # A raw path, not Path.as_uri(): percent-encoding (for example of ':' in a
        # sub-job id) is not decoded by GET and names a directory that does not
        # exist. Callers download into private temp directories.
        target_uri = "file://" + target_dir.resolve().as_posix() + "/"
        cursor = connection.cursor()
        try:
            cursor.execute(
                "GET "
                f"{_sql_string_literal(run_uri + relative_path)} "
                f"{_sql_string_literal(target_uri)}"
            ).fetchall()
        finally:
            cursor.close()
        downloaded = target_dir / PurePosixPath(relative_path).name
        if not downloaded.is_file():
            raise RuntimeError(
                f"Snowflake GET did not create expected artifact {downloaded}"
            )
        return downloaded

    @_track_operation("fetch_execution_logs")
    def fetch_execution_logs(
        self,
        job_id: str,
        *,
        sub_job_id: str | None = None,
        filename: str | None = None,
    ) -> list[dict[str, str]]:
        """Download every log file under the job's experiment run stage.

        Returns a list of
        ``{"sub_job_id", "filename", "artifact_uri", "content"}``
        entries, one per object found under a ``_logs/<sub_job_id>/`` subtree
        in the run's stage. Filenames such as ``execution.jsonl`` and
        ``server.log`` are both included; bodies are decoded as UTF-8.

        ``sub_job_id`` and ``filename``, when given, download only the files
        of that sub-job and with that name (for example one sub-job's
        ``execution.jsonl``); the rest are listed but not fetched.

        ``artifact_uri`` is the logical ``snow://`` URI. Files are retrieved
        through Snowflake's supported experiment artifact ``LIST``/``GET``
        interface; no object-store credentials are exposed to the caller.
        """
        values = self._operation_values()
        values.update(artifact="execution_logs", file_count=0, bytes_read=0, cancelled=False)
        run_uri, run_name = self._experiment_run_uri(job_id)
        connection = self._open_experiment_artifact_connection()
        try:
            started = time.monotonic()
            paths: list[str] = []
            for artifact_root in ("_logs", "checkpoints/_logs"):
                paths.extend(
                    self._list_experiment_artifacts(
                        connection, run_uri, run_name, artifact_root
                    )
                )

            values["list_ms"] = round((time.monotonic() - started) * 1000, 3)
            started = time.monotonic()
            results: list[dict[str, str]] = []
            with tempfile.TemporaryDirectory(prefix="cortex-training-execution-logs-") as temp:
                temp_root = Path(temp)
                for relative_path in sorted(set(paths)):
                    if relative_path.startswith("_logs/"):
                        after_logs = relative_path.removeprefix("_logs/")
                    elif relative_path.startswith("checkpoints/_logs/"):
                        after_logs = relative_path.removeprefix("checkpoints/_logs/")
                    else:
                        continue
                    file_sub_job, separator, file_name = after_logs.partition("/")
                    if (
                        not separator
                        or not file_name
                        or not (
                            file_sub_job == job_id
                            or file_sub_job.startswith(f"{job_id}:")
                        )
                        or file_sub_job in {".", ".."}
                        or "\\" in file_sub_job
                        or (sub_job_id is not None and file_sub_job != sub_job_id)
                        or (filename is not None and file_name != filename)
                    ):
                        continue
                    downloaded = self._download_experiment_artifact(
                        connection,
                        run_uri,
                        relative_path,
                        temp_root,
                    )
                    artifact_uri = run_uri + relative_path
                    try:
                        size = downloaded.stat().st_size
                        content = downloaded.read_text(encoding="utf-8")
                        values["file_count"] += 1
                        values["bytes_read"] += size
                    finally:
                        downloaded.unlink(missing_ok=True)
                    results.append(
                        {
                            "sub_job_id": file_sub_job,
                            "filename": file_name,
                            "artifact_uri": artifact_uri,
                            "content": content,
                        }
                    )
            values["fetch_ms"] = round((time.monotonic() - started) * 1000, 3)
            return results
        finally:
            connection.close()

    # Default number of concurrent chunk GETs. Each GET is mostly round-trip
    # latency, so throughput scales with the number in flight.
    _DOWNLOAD_DEFAULT_WORKERS = 16
    # Fetched chunks waiting to be appended are capped at this many per worker,
    # which bounds temporary disk use when one early chunk is slow.
    _DOWNLOAD_WINDOW_PER_WORKER = 4
    # The resume manifest is checkpointed after this many chunks or seconds,
    # whichever comes first.
    _DOWNLOAD_CHECKPOINT_CHUNKS = 64
    _DOWNLOAD_CHECKPOINT_SECONDS = 5.0
    # Leftover temp files are removed only once untouched for this long, so a
    # concurrent download of the same directory is not broken.
    _TEMP_ORPHAN_MIN_AGE_SECONDS = 3600.0
    # A running job whose newest chunk is older than this gets a warning that
    # uploads may be stalled.
    _DOWNLOAD_STALE_AFTER_SECONDS = 600.0
    # The stall check runs alongside the download; once the chunks are written
    # the download waits at most this long for it, then returns without it.
    _STALL_CHECK_WAIT_SECONDS = 10.0
    # A single LIST or GET attempt that runs longer than this is abandoned and
    # retried. A chunk GET normally takes well under a second.
    _DOWNLOAD_TRANSFER_TIMEOUT_SECONDS = 300.0
    # An execution-log GET may take at least its size at this rate before the
    # transfer timeout gives up on it.
    _EXECUTION_LOG_MIN_BYTES_PER_SECOND = 512 * 1024
    _wall_clock = staticmethod(time.time)
    _artifact_retry_policy = _artifact_download.RetryPolicy()

    @staticmethod
    def _execution_log_target(job_id: str, relative_path: str) -> tuple[str, str] | None:
        """``(sub_job_id, filename)`` for an execution log path of ``job_id``, else None."""
        for root in ("_logs/", "checkpoints/_logs/"):
            if relative_path.startswith(root):
                after_logs = relative_path.removeprefix(root)
                break
        else:
            return None
        sub_job_id, separator, filename = after_logs.partition("/")
        if (
            not separator
            or not filename
            or not (sub_job_id == job_id or sub_job_id.startswith(f"{job_id}:"))
            or sub_job_id in {".", ".."}
            or "\\" in sub_job_id
        ):
            return None
        return sub_job_id, filename

    @staticmethod
    def _execution_log_identity(listed: Mapping[str, Any], artifact_uri: str) -> dict[str, Any] | None:
        """What identifies one version of a staged log file, if LIST reported enough."""
        if listed.get("size") is None or not (listed.get("md5") or listed.get("last_modified")):
            return None
        return {
            "artifact_uri": artifact_uri,
            "size": listed["size"],
            "md5": listed.get("md5"),
            "last_modified": listed.get("last_modified"),
        }

    @_track_operation("download_execution_logs")
    def download_execution_logs(
        self,
        job_id: str,
        output_dir: str | os.PathLike[str],
        *,
        resume: bool = False,
        sub_job_ids: Iterable[str] | None = None,
        progress: Callable[[str], None] | None = None,
        transfer_timeout_seconds: float | None = _DOWNLOAD_TRANSFER_TIMEOUT_SECONDS,
    ) -> list[dict[str, Any]]:
        """Save every execution log file to ``<output_dir>/<sub_job_id>/<filename>``.

        Unlike :meth:`fetch_execution_logs`, file bodies are never held in
        memory: each file is downloaded straight to disk, made readable only by
        its owner (mode 0600), and then moved into place, so an interrupted
        download leaves the previous copy untouched. A path present under both
        ``_logs/`` and ``checkpoints/_logs/`` keeps the ``_logs/`` copy.

        With ``resume=True`` a file whose staged size and checksum match the
        copy saved by an earlier download is not fetched again. A staged log
        file is replaced as a whole when it changes, so a changed file is
        downloaded in full.

        ``transfer_timeout_seconds`` bounds each GET attempt, raised for a large
        file to its size at 512 KiB/s. ``progress`` gets one status line per
        file, ending in ``"\\n"``. Each
        result is ``{"sub_job_id", "filename", "artifact_uri", "saved_path",
        "bytes", "unchanged"}``, where ``unchanged`` means resume kept the saved
        copy.
        """
        if isinstance(sub_job_ids, (str, bytes)):
            raise TypeError("sub_job_ids must be an iterable of sub-job ids, not a single string")
        wanted = None if sub_job_ids is None else set(sub_job_ids)
        if wanted is not None and not wanted:
            raise ValueError("sub_job_ids must name at least one sub-job")
        if transfer_timeout_seconds is not None and (
            isinstance(transfer_timeout_seconds, bool)
            or not isinstance(transfer_timeout_seconds, (int, float))
            or not transfer_timeout_seconds > 0
        ):
            raise ValueError("transfer_timeout_seconds must be a positive number or None")
        run_uri, run_name = self._experiment_run_uri(job_id)
        root = Path(output_dir).expanduser()
        shared = _artifact_download.SharedConnection(
            self._open_experiment_artifact_connection, transfer_timeout_seconds
        )
        policy = self._artifact_retry_policy
        try:
            targets: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
            # "_logs/" last, so its copy wins over "checkpoints/_logs/".
            for artifact_root in ("checkpoints/_logs", "_logs"):
                listed = _artifact_download.call_with_retry(
                    shared,
                    policy,
                    f"LIST {artifact_root}",
                    lambda connection, artifact_root=artifact_root: (
                        self._list_experiment_artifact_files(
                            connection, run_uri, run_name, artifact_root
                        )
                    ),
                )
                for relative_path, info in listed.items():
                    target = self._execution_log_target(job_id, relative_path)
                    if target is not None:
                        targets[target] = (relative_path, info)
            if wanted is not None:
                found = {sub_job_id for sub_job_id, _ in targets}
                targets = {key: value for key, value in targets.items() if key[0] in wanted}
                missing = sorted(wanted - found)
                if missing:
                    raise NoChunksError(
                        f"no execution logs for sub-job(s) {', '.join(missing)}; "
                        f"sub-jobs with execution logs: {', '.join(sorted(found)) or 'none'}",
                        sub_job_ids=missing,
                        available=sorted(found),
                    )
            if not targets:
                self._report_artifact_progress(progress, "execution logs: found 0 files\n")
            results = []
            for (sub_job_id, filename), (relative_path, info) in sorted(targets.items()):
                results.append(
                    self._save_execution_log(
                        shared,
                        run_uri,
                        relative_path,
                        info,
                        root / sub_job_id / filename,
                        sub_job_id=sub_job_id,
                        filename=filename,
                        resume=resume,
                        progress=progress,
                    )
                )
            return results
        finally:
            shared.close()

    def _save_execution_log(
        self,
        shared: _artifact_download.SharedConnection,
        run_uri: str,
        relative_path: str,
        listed: Mapping[str, Any],
        destination: Path,
        *,
        sub_job_id: str,
        filename: str,
        resume: bool,
        progress: Callable[[str], None] | None,
    ) -> dict[str, Any]:
        """``filename`` is the path below the sub-job's log directory, which may
        include subdirectories (``ray/worker.log``)."""
        artifact_uri = run_uri + relative_path
        manifest_path = destination.with_name("." + destination.name + ".artifact.json")
        identity = self._execution_log_identity(listed, artifact_uri)
        label = f"execution logs: {sub_job_id}/{filename}"
        result = {
            "sub_job_id": sub_job_id,
            "filename": filename,
            "artifact_uri": artifact_uri,
            "saved_path": str(destination),
        }
        if resume and identity is not None:
            try:
                saved = json.loads(manifest_path.read_text(encoding="utf-8"))
                size = destination.stat().st_size
            except (OSError, ValueError):
                saved, size = None, None
            if saved == identity and size == identity["size"]:
                self._report_artifact_progress(progress, f"{label}: unchanged\n")
                return {**result, "bytes": size, "unchanged": True}
        expected = "" if listed.get("size") is None else f" ({_format_artifact_bytes(listed['size'])})"
        with self._artifact_progress_line(progress):
            self._report_artifact_progress(progress, f"{label}: downloading{expected}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            self._remove_orphan_execution_log_spools(destination.parent)
            spool = Path(tempfile.mkdtemp(prefix=".execution-log-", suffix=".tmp", dir=destination.parent))
            # One file can be hundreds of MB: allow its GET time in proportion.
            base_timeout = shared.transfer_timeout
            if base_timeout is not None and listed.get("size"):
                shared.transfer_timeout = max(base_timeout, listed["size"] / self._EXECUTION_LOG_MIN_BYTES_PER_SECOND)
            try:
                downloaded = _artifact_download.call_with_retry(
                    shared,
                    self._artifact_retry_policy,
                    f"GET {relative_path}",
                    lambda connection: self._fetch_execution_log(connection, run_uri, relative_path, spool),
                )
                os.chmod(downloaded, 0o600)
                manifest_path.unlink(missing_ok=True)
                os.replace(downloaded, destination)
            finally:
                shared.transfer_timeout = base_timeout
                shutil.rmtree(spool, ignore_errors=True)
            size = destination.stat().st_size
            if identity is not None and size == identity["size"]:
                _write_private_json(manifest_path, identity)
            self._report_artifact_progress(progress, f"{label}: saved {_format_artifact_bytes(size)}")
        return {**result, "bytes": size, "unchanged": False}

    @classmethod
    def _remove_orphan_execution_log_spools(cls, directory: Path) -> None:
        """Delete download directories a killed execution-log download left behind."""
        cutoff = time.time() - cls._TEMP_ORPHAN_MIN_AGE_SECONDS
        for path in directory.glob(".execution-log-*.tmp"):
            try:
                if path.is_dir() and path.stat().st_mtime < cutoff:
                    shutil.rmtree(path)
                    logger.info("removed leftover temporary directory %s", path)
            except OSError:
                logger.debug("could not remove leftover temporary directory %s", path, exc_info=True)

    def _fetch_execution_log(self, connection: Any, run_uri: str, relative_path: str, spool: Path) -> Path:
        """One GET attempt into a fresh directory under ``spool``."""
        target = Path(tempfile.mkdtemp(prefix="get-", dir=spool))
        try:
            return self._download_experiment_artifact(connection, run_uri, relative_path, target)
        except BaseException:
            shutil.rmtree(target, ignore_errors=True)
            raise

    @staticmethod
    def _resume_manifest_path(destination: Path) -> Path:
        return destination.with_name("." + destination.name + ".manifest.json")

    @staticmethod
    def _partial_paths(destination: Path) -> tuple[Path, Path]:
        """Return the in-progress data file and its manifest for ``destination``."""
        return (
            destination.with_name("." + destination.name + ".partial"),
            destination.with_name("." + destination.name + ".partial.manifest.json"),
        )

    @staticmethod
    def _artifact_result(
        destination: Path,
        run_uri: str,
        names: list[str],
        *,
        skipped: list[str] | None = None,
        warnings: list[str] | None = None,
    ) -> dict[str, Any]:
        """Summarize one sub-job's file; ``skipped``/``warnings`` only when non-empty."""
        result: dict[str, Any] = {
            "sub_job_id": destination.parent.name,
            "filename": destination.name,
            "saved_path": str(destination),
            "chunk_count": len(names) - len(skipped or []),
            "first_artifact_uri": run_uri + names[0],
            "last_artifact_uri": run_uri + names[-1],
        }
        if skipped:
            result["skipped_chunks"] = [run_uri + name for name in skipped]
        if warnings:
            result["warnings"] = list(warnings)
        return result

    @classmethod
    def _remove_orphan_temp_files(cls, destination: Path, temporary_prefix: str) -> None:
        """Delete temp files a killed download left beside ``destination``."""
        directory = destination.parent
        if not directory.is_dir():
            return
        patterns = (
            f"{temporary_prefix}*.tmp",
            f".{destination.name}.manifest-*.tmp",
            f".{destination.name}.partial.manifest-*.tmp",
            ".manifest-*.tmp",
        )
        # Only files nobody has written to for a while: a download still
        # running in the same directory keeps touching its temp files.
        cutoff = time.time() - cls._TEMP_ORPHAN_MIN_AGE_SECONDS
        for path in (path for pattern in patterns for path in directory.glob(pattern)):
            try:
                if not path.is_file() or path.stat().st_mtime >= cutoff:
                    continue
                path.unlink()
                logger.info("removed leftover temporary file %s", path)
            except OSError:
                logger.debug("could not remove leftover temporary file %s", path, exc_info=True)

    @staticmethod
    def _resume_state(
        data_path: Path,
        manifest_path: Path,
        chunk_paths: list[str],
    ) -> _artifact_download.Manifest | None:
        """Return the manifest if ``data_path`` can be extended in place."""
        state = _artifact_download.read_manifest(manifest_path)
        if state is None:
            return None
        if state.names != chunk_paths[: len(state.names)]:
            logger.info(
                "%s: the chunk list no longer extends the saved chunks; rebuilding",
                data_path,
            )
            return None
        try:
            if not data_path.is_file() or data_path.stat().st_size < state.nbytes:
                return None
        except OSError:
            return None
        return state

    def _plan_gzip_download(
        self,
        sub_job_id: str,
        destination: Path,
        chunk_paths: list[str],
        *,
        resume: bool,
    ) -> _artifact_download.GzipPlan:
        """Choose append, continue-partial, or rebuild for one sub-job.

        Only reads the output directory; :meth:`_prepare_gzip_plan` makes the
        changes once that sub-job's turn to be written comes.
        """
        partial, partial_manifest = self._partial_paths(destination)
        manifest = self._resume_manifest_path(destination)
        if resume:
            state = self._resume_state(destination, manifest, chunk_paths)
            if state is not None:
                return _artifact_download.GzipPlan(
                    sub_job_id,
                    destination,
                    destination,
                    manifest,
                    chunk_paths,
                    state.names,
                    state.nbytes,
                    False,
                    state.skipped,
                )
            state = self._resume_state(partial, partial_manifest, chunk_paths)
            if state is not None:
                return _artifact_download.GzipPlan(
                    sub_job_id,
                    destination,
                    partial,
                    partial_manifest,
                    chunk_paths,
                    state.names,
                    state.nbytes,
                    True,
                    state.skipped,
                )
        return _artifact_download.GzipPlan(
            sub_job_id, destination, partial, partial_manifest, chunk_paths, [], 0, True
        )

    def _prepare_gzip_plan(
        self, plan: _artifact_download.GzipPlan, temporary_prefix: str
    ) -> None:
        """Clear what the plan replaces, just before its file is written."""
        self._remove_orphan_temp_files(plan.destination, temporary_prefix)
        partial, partial_manifest = self._partial_paths(plan.destination)
        if not plan.promote:
            # Appending to a completed file: an older partial rebuild is stale.
            partial_manifest.unlink(missing_ok=True)
            partial.unlink(missing_ok=True)
        elif not plan.known:
            # Drop the manifest before the writer truncates the partial file.
            partial_manifest.unlink(missing_ok=True)

    def _fetch_artifact_chunk(
        self,
        shared: _artifact_download.SharedConnection,
        run_uri: str,
        relative_path: str,
        spool_root: Path,
        cancel: threading.Event | None = None,
    ) -> Path:
        """GET one chunk into its own temp directory, retrying transient errors."""

        def attempt(connection: Any) -> Path:
            target = Path(tempfile.mkdtemp(prefix="chunk-", dir=spool_root))
            try:
                return self._download_experiment_artifact(
                    connection, run_uri, relative_path, target
                )
            except BaseException:
                shutil.rmtree(target, ignore_errors=True)
                raise

        return _artifact_download.call_with_retry(
            shared, self._artifact_retry_policy, f"GET {relative_path}", attempt, cancel
        )

    def _commit_chunk(
        self,
        writer: _artifact_download.ChunkWriter,
        relative_path: str,
        fetched: Callable[[], Path],
        fetch_again: Callable[[], Path],
        *,
        run_uri: str,
        strict: bool,
        warnings: list[str],
    ) -> None:
        """Append one fetched chunk, or skip it when it is gone or damaged.

        A chunk removed after the listing is skipped. One that will not
        decompress is fetched once more, then skipped. ``strict`` raises instead.
        """
        uri = run_uri + relative_path

        def skip(reason: str) -> None:
            message = f"skipped chunk {uri}: {reason}"
            logger.warning(message)
            warnings.append(message)
            writer.skip(relative_path)

        for attempt, fetch in enumerate((fetched, fetch_again)):
            try:
                downloaded = fetch()
            except Exception as exc:
                if strict or not _artifact_download.is_file_not_found(exc):
                    raise
                skip("it no longer exists in the run's stage (it was removed after it was listed)")
                return
            try:
                writer.append(relative_path, downloaded)
                return
            except _artifact_download.CORRUPT_CHUNK_ERRORS as exc:
                if strict:
                    raise
                if attempt:
                    skip(
                        f"it is damaged and could not be decompressed ({exc}); "
                        "the saved file is missing that chunk's output"
                    )
                    return
                logger.warning(
                    "chunk %s could not be decompressed (%s); downloading it again", uri, exc
                )
            finally:
                shutil.rmtree(downloaded.parent, ignore_errors=True)

    @staticmethod
    def _mark_progress_saved(exc: BaseException, saved: bool) -> None:
        """Record on ``exc`` that output was saved; never clears an earlier True."""
        if saved:
            with contextlib.suppress(Exception):
                exc.download_progress_saved = True  # type: ignore[attr-defined]

    @staticmethod
    def _already_committed(plan: _artifact_download.GzipPlan) -> bool:
        """True when a completed file already holds exactly the planned chunks."""
        if plan.new_paths or plan.promote:
            return False
        try:
            return plan.destination.stat().st_size == plan.committed
        except OSError:
            return False

    def _promote_partial(self, plan: _artifact_download.GzipPlan) -> None:
        """Replace the destination with a completed partial file."""
        manifest = self._resume_manifest_path(plan.destination)
        manifest.unlink(missing_ok=True)
        os.replace(plan.data_path, plan.destination)
        os.replace(plan.manifest_path, manifest)

    @staticmethod
    def _artifact_progress_status(
        artifact_name: str, done: int, total: int, sub_job_id: str
    ) -> str:
        width = 20
        filled = (width * done) // total
        bar = "#" * filled + "-" * (width - filled)
        return f"{artifact_name}: [{bar}] {done}/{total} {sub_job_id}"

    @staticmethod
    def _report_artifact_progress(progress: Callable[[str], None] | None, message: str) -> None:
        if progress is not None:
            progress(message)

    @contextlib.contextmanager
    def _artifact_progress_line(
        self, progress: Callable[[str], None] | None
    ) -> Iterator[None]:
        """Close the status line with ``"\\n"`` whether the body succeeds or fails.

        On failure the download error wins over any error from the newline.
        """
        try:
            yield
        except BaseException:
            try:
                self._report_artifact_progress(progress, "\n")
            except BaseException:
                pass
            raise
        self._report_artifact_progress(progress, "\n")

    def _write_gzip_plans(
        self,
        shared: _artifact_download.SharedConnection,
        run_uri: str,
        artifact_name: str,
        plans: list[_artifact_download.GzipPlan],
        *,
        max_workers: int,
        temporary_prefix: str,
        progress: Callable[[str], None] | None = None,
        on_chunk: Callable[[str, int, int, int], None] | None = None,
        strict: bool = False,
        warnings_by_sub_job: Mapping[str, list[str]] | None = None,
        on_commit: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch every new chunk concurrently and append them in plan order.

        Fetches run ahead of the writer across sub-job boundaries, up to a
        bounded window, so a short sub-job never leaves workers idle.
        ``warnings_by_sub_job`` is copied into each sub-job's result.

        For each sub-job, ``progress`` gets a status line before anything in
        the output directory changes, an update after each appended chunk, and
        ``"\\n"`` once the manifest is written or the sub-job fails; a sub-job
        with nothing new gets one line ending in a newline instead.
        ``on_chunk`` gets ``(sub_job_id, done, total, bytes)`` counts, with
        ``done == total`` only after the file is committed. ``on_commit`` then
        gets a copy of that sub-job's result entry.
        """
        work = [path for plan in plans for path in plan.new_paths]
        window = max_workers * self._DOWNLOAD_WINDOW_PER_WORKER
        values = self._operation_values()
        for key in ("chunk_count", "chunks_skipped", "bytes_written"):
            values.setdefault(key, 0)
        results: list[dict[str, Any]] = []
        cancel = threading.Event()
        # A resumed plan starts from output an earlier download saved.
        saved_progress = any(plan.known for plan in plans)
        spool_prefix = f"cortex-training-{artifact_name}-artifacts-"
        _artifact_download.Spool.sweep(spool_prefix, self._TEMP_ORPHAN_MIN_AGE_SECONDS)
        with _artifact_download.Spool(spool_prefix) as spool:
            spool_root = spool.path
            pool = ThreadPoolExecutor(
                max_workers=max_workers, thread_name_prefix="cortex-training-download"
            )
            pending: deque[Future[Path]] = deque()
            # The fetch the writer is waiting on; it has left ``pending``.
            head: Future[Path] | None = None
            submitted = 0
            try:
                for plan in plans:
                    total = len(plan.chunk_paths)
                    new_total = len(plan.new_paths)
                    warnings = list((warnings_by_sub_job or {}).get(plan.sub_job_id, []))
                    if plan.skipped:
                        warnings.append(
                            f"{len(plan.skipped)} damaged chunk(s) skipped by an earlier "
                            "download are still missing from the saved file; download "
                            "again without resume to retry them"
                        )
                    if not new_total:
                        self._report_artifact_progress(
                            progress, f"{artifact_name}: no new chunks for {plan.sub_job_id}\n"
                        )

                    def report(done: int, plan=plan, new_total=new_total) -> None:
                        if new_total:
                            self._report_artifact_progress(
                                progress,
                                self._artifact_progress_status(
                                    artifact_name, done, new_total, plan.sub_job_id
                                ),
                            )

                    with self._artifact_progress_line(progress if new_total else None):
                        # Reported before anything in the output directory changes.
                        report(0)
                        self._prepare_gzip_plan(plan, temporary_prefix)
                        if self._already_committed(plan):
                            # Nothing to write: leave the file alone (it may be
                            # read-only) and report it as it is.
                            saved_progress = True
                            if on_chunk is not None:
                                on_chunk(plan.sub_job_id, total, total, plan.committed)
                            results.append(
                                self._artifact_result(
                                    plan.destination,
                                    run_uri,
                                    plan.known,
                                    skipped=plan.skipped,
                                    warnings=warnings,
                                )
                            )
                            if on_commit is not None:
                                on_commit(dict(results[-1]))
                            continue
                        writer = _artifact_download.ChunkWriter(
                            plan.data_path,
                            plan.manifest_path,
                            plan.known,
                            plan.committed,
                            buffer_bytes=_STREAM_COPY_BUFFER_BYTES,
                            checkpoint_chunks=self._DOWNLOAD_CHECKPOINT_CHUNKS,
                            checkpoint_seconds=self._DOWNLOAD_CHECKPOINT_SECONDS,
                            skipped=plan.skipped,
                        )
                        saved_progress = saved_progress or bool(writer.names)
                        try:
                            if on_chunk is not None and len(writer.names) < total:
                                on_chunk(plan.sub_job_id, len(writer.names), total, writer.offset)
                            for done, relative_path in enumerate(plan.new_paths, start=1):
                                while submitted < len(work) and len(pending) < window:
                                    pending.append(
                                        pool.submit(
                                            self._fetch_artifact_chunk,
                                            shared,
                                            run_uri,
                                            work[submitted],
                                            spool_root,
                                            cancel,
                                        )
                                    )
                                    submitted += 1
                                head = pending.popleft()
                                offset, skipped = writer.offset, len(writer.skipped)
                                self._commit_chunk(
                                    writer,
                                    relative_path,
                                    head.result,
                                    functools.partial(
                                        self._fetch_artifact_chunk,
                                        shared,
                                        run_uri,
                                        relative_path,
                                        spool_root,
                                        cancel,
                                    ),
                                    run_uri=run_uri,
                                    strict=strict,
                                    warnings=warnings,
                                )
                                saved_progress = True
                                values["chunk_count"] += 1
                                values["chunks_skipped"] += len(writer.skipped) - skipped
                                values["bytes_written"] += writer.offset - offset
                                report(done)
                                # done == total is reported once the file is committed.
                                if on_chunk is not None and len(writer.names) < total:
                                    on_chunk(
                                        plan.sub_job_id, len(writer.names), total, writer.offset
                                    )
                            writer.finish()
                            saved_progress = True
                        except BaseException:
                            try:
                                writer.abort()
                            except Exception as abort_exc:
                                logger.warning(
                                    "could not save download progress for %s: %s",
                                    plan.destination,
                                    abort_exc,
                                )
                                logger.debug("saving download progress failed", exc_info=True)
                            raise
                        if plan.promote:
                            self._promote_partial(plan)
                        if on_chunk is not None:
                            on_chunk(plan.sub_job_id, total, total, writer.offset)
                    results.append(
                        self._artifact_result(
                            plan.destination,
                            run_uri,
                            writer.names,
                            skipped=writer.skipped,
                            warnings=warnings,
                        )
                    )
                    if on_commit is not None:
                        on_commit(dict(results[-1]))
                # Inside the handler, so an interrupt here is still marked saved.
                pool.shutdown(wait=True)
                spool.remove_after([])
            except BaseException as exc:
                # Workers stop at their next attempt or backoff; only a GET
                # already on the wire runs to completion, and it is not awaited.
                cancel.set()
                pool.shutdown(wait=False, cancel_futures=True)
                spool.remove_after([*([head] if head is not None else []), *pending])
                with contextlib.suppress(Exception):
                    if not getattr(exc, "download_progress_saved", False):
                        exc.download_progress_saved = saved_progress  # type: ignore[attr-defined]
                raise
        return results

    def _list_gzip_chunks(
        self,
        shared: _artifact_download.SharedConnection,
        run_uri: str,
        run_name: str,
        job_id: str,
        artifact_name: str,
        chunk_pattern: re.Pattern[str],
        *,
        report_merges: bool = True,
        cancel: threading.Event | None = None,
    ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        """Return each sub-job's chunk paths in time order, plus listing warnings.

        Chunks normally live under ``_<name>/``; a run whose stage is anchored
        under ``checkpoints/`` uses ``checkpoints/_<name>/``. If one sub-job has
        chunks under both, they are merged by chunk name (which starts with the
        chunk's time), and a chunk name present in both keeps the
        ``_<name>/`` copy, so no output is dropped or repeated.
        """
        policy = self._artifact_retry_policy
        artifact_roots = (f"_{artifact_name}", f"checkpoints/_{artifact_name}")
        by_root: dict[str, dict[str, dict[str, str]]] = {}
        for artifact_root in artifact_roots:
            paths = _artifact_download.call_with_retry(
                shared,
                policy,
                f"LIST {artifact_root}",
                lambda connection, artifact_root=artifact_root: (
                    self._list_experiment_artifacts(
                        connection, run_uri, run_name, artifact_root
                    )
                ),
                cancel,
            )
            prefix = artifact_root + "/"
            for relative_path in paths:
                after_root = relative_path.removeprefix(prefix)
                sub_job_id, separator, filename = after_root.partition("/")
                if (
                    not separator
                    or sub_job_id in {"", ".", ".."}
                    or not (
                        sub_job_id == job_id
                        or sub_job_id.startswith(f"{job_id}:")
                    )
                    or "/" in filename
                    or "\\" in sub_job_id
                    or chunk_pattern.fullmatch(filename) is None
                ):
                    continue
                by_root.setdefault(sub_job_id, {}).setdefault(artifact_root, {})[
                    filename
                ] = relative_path

        chunks: dict[str, list[str]] = {}
        warnings: dict[str, list[str]] = {}
        for sub_job_id, roots in by_root.items():
            merged: dict[str, str] = {}
            for artifact_root in reversed(artifact_roots):
                merged.update(roots.get(artifact_root, {}))
            chunks[sub_job_id] = [merged[name] for name in sorted(merged)]
            if len(roots) > 1 and report_merges:
                message = (
                    f"{artifact_name} chunks for {sub_job_id} were saved under both "
                    f"{artifact_roots[0]}/ and {artifact_roots[1]}/; they were merged "
                    "in chunk time order"
                )
                logger.warning(message)
                warnings[sub_job_id] = [message]
        return chunks, warnings

    # A metrics chunk is uploaded about once a minute; until a sub-job has run
    # for twice that, having none yet is expected.
    _MISSING_METRICS_GRACE_SECONDS = 120.0
    # Allowance for a local clock behind the server's when deciding whether the
    # job status is worth fetching; the decision itself uses the server's clock.
    _STALE_PRECHECK_SKEW_SECONDS = 300.0
    # A job's last chunks can land this long after it ends; a download in that
    # window says so.
    _RECENTLY_ENDED_SECONDS = 120.0

    def _job_for_upload_warnings(self, job_id: str) -> tuple[dict, float | None]:
        """Return the job and the server's clock (from the ``Date`` header), if any.

        Tried once: the lookup only adds notices, and it runs in the background,
        where a retried failure would keep logging after the download returned.
        """
        resp = self._send("GET", f"{self._prefix}/{job_id}", max_retries=0)
        server_now = None
        try:
            server_now = email.utils.parsedate_to_datetime(resp.headers["Date"]).timestamp()
        except Exception:
            logger.debug("job response has no usable Date header", exc_info=True)
        job = resp.json()
        return (job if isinstance(job, dict) else {}), server_now

    @staticmethod
    def _parse_api_time(value: Any) -> float | None:
        if not isinstance(value, str) or not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()

    @staticmethod
    def _chunk_time(chunk_pattern: re.Pattern[str], relative_path: str) -> float | None:
        """Return the UTC time in a chunk's name, or None if it has none."""
        match = chunk_pattern.fullmatch(PurePosixPath(relative_path).name)
        if match is None:
            return None
        try:
            stamp = datetime.strptime(match.group("timestamp"), "%Y%m%d-%H%M%S")
        except ValueError:
            return None
        return stamp.replace(tzinfo=timezone.utc).timestamp()

    def _stale_upload_warnings(
        self,
        shared: _artifact_download.SharedConnection,
        run_uri: str,
        run_name: str,
        job_id: str,
        artifact_name: str,
        chunk_pattern: re.Pattern[str],
        chunks_by_sub_job: Mapping[str, list[str]],
        wanted: set[str] | None,
        stale_after_seconds: float | None,
        cancel: threading.Event | None = None,
    ) -> dict[str, list[str]]:
        """Warn when a running job's uploads look stalled; never raises.

        ``cancel`` is set when the download returns; after that the check
        stops, makes no further calls, and logs nothing.

        Two signs: the newest chunk's name time is older than
        ``stale_after_seconds``, or (for metrics) a sub-job has saved stdout but
        no metrics at all. A sub-job is checked only while it (or, if the job
        does not report sub-job status, the job) is ``running``, and only once
        it has been running long enough for the sign to mean anything. Ages
        use the server's clock when the response carries one.

        For metrics of a job that has ended, a sub-job with saved stdout but no
        metrics gets a warning saying why there may be none, so an empty
        result is not silent.
        """
        if stale_after_seconds is None:
            return {}
        local_now = self._wall_clock()
        candidates: dict[str, float] = {}
        for sub_job_id, paths in chunks_by_sub_job.items():
            if wanted is not None and sub_job_id not in wanted:
                continue
            newest = self._chunk_time(chunk_pattern, paths[-1])
            if newest is not None and local_now - newest > max(
                stale_after_seconds - self._STALE_PRECHECK_SKEW_SECONDS, 0
            ):
                candidates[sub_job_id] = newest
        check_missing = artifact_name == "metrics"
        # A chunk this recent means the job may have only just ended.
        recent_cutoff = local_now - self._RECENTLY_ENDED_SECONDS - self._STALE_PRECHECK_SKEW_SECONDS
        recently_active = any(
            (stamp := self._chunk_time(chunk_pattern, paths[-1])) is not None and stamp >= recent_cutoff
            for sub_job_id, paths in chunks_by_sub_job.items()
            if paths and (wanted is None or sub_job_id in wanted)
        )
        if not candidates and not check_missing and not recently_active:
            return {}
        try:
            job, server_now = self._job_for_upload_warnings(job_id)
        except Exception:
            logger.debug("job status lookup for upload warnings failed", exc_info=True)
            return {}
        status = self._normalize_job_status(job.get("status"))
        if status in _ENDED_JOB_STATES:
            warnings = self._recently_ended_warnings(
                job, local_now if server_now is None else server_now, artifact_name, chunks_by_sub_job, wanted
            )
            if check_missing:
                for sub_job_id, messages in self._ended_without_metrics_warnings(
                    shared, run_uri, run_name, job_id, chunks_by_sub_job, wanted
                ).items():
                    warnings.setdefault(sub_job_id, []).extend(messages)
            return warnings
        if status != "running":
            return {}
        now = local_now if server_now is None else server_now
        # When the job entered its current state; absent on older services.
        running_since = self._parse_api_time(job.get("state_changed_at"))
        sub_job_status = {
            sub_job.get("sub_job_id"): self._normalize_job_status(sub_job.get("status"))
            for sub_job in job.get("sub_jobs") or []
            if isinstance(sub_job, dict) and sub_job.get("status")
        }

        def running_for(sub_job_id: str, started: float | None = None) -> float | None:
            """Seconds the sub-job has been running, None if unknown, -1 if not running."""
            status = sub_job_status.get(sub_job_id)
            if status is not None and status != "running":
                return -1.0
            starts = [t for t in (running_since, started) if t is not None]
            return now - max(starts) if starts else None

        label = "stdout output" if artifact_name == "stdout" else "GPU metrics"
        warnings: dict[str, list[str]] = {}
        for sub_job_id, newest in candidates.items():
            age = now - newest
            ran = running_for(sub_job_id)
            if age <= stale_after_seconds or (ran is not None and ran < stale_after_seconds):
                continue
            stamp = datetime.fromtimestamp(newest, tz=timezone.utc)
            message = (
                f"newest saved {label} for {sub_job_id} is from "
                f"{stamp:%Y-%m-%d %H:%M:%S} UTC ({int(age // 60)} min ago) while the job "
                "is still running; uploads may be stalled. Contact support if this persists."
            )
            if artifact_name == "stdout":
                message += " A job that prints nothing for a while also looks like this."
            warnings[sub_job_id] = [message]
        if check_missing:
            try:
                stdout_chunks, _ = self._list_gzip_chunks(
                    shared,
                    run_uri,
                    run_name,
                    job_id,
                    "stdout",
                    _STDOUT_CHUNK_RE,
                    report_merges=False,
                    cancel=cancel,
                )
            except Exception:
                logger.debug("stdout listing for upload warnings failed", exc_info=True)
                stdout_chunks = {}
            for sub_job_id, paths in stdout_chunks.items():
                if sub_job_id in chunks_by_sub_job or (
                    wanted is not None and sub_job_id not in wanted
                ):
                    continue
                # The first stdout chunk is the latest evidence of when it started.
                ran = running_for(sub_job_id, self._chunk_time(_STDOUT_CHUNK_RE, paths[0]))
                if ran is not None and ran < self._MISSING_METRICS_GRACE_SECONDS:
                    continue
                warnings[sub_job_id] = [
                    f"no GPU metrics have been saved for {sub_job_id} while the job is "
                    "still running, although its stdout output has been; metric "
                    "uploads may be stalled. Contact support if this persists."
                ]
        if cancel is not None and cancel.is_set():
            return {}
        for messages in warnings.values():
            for message in messages:
                logger.warning(message)
        return warnings

    def _recently_ended_warnings(
        self,
        job: Mapping[str, Any],
        now: float,
        artifact_name: str,
        chunks_by_sub_job: Mapping[str, list[str]],
        wanted: set[str] | None,
    ) -> dict[str, list[str]]:
        """Say that a job that ended moments ago may still be uploading."""
        ended_at = self._parse_api_time(job.get("state_changed_at"))
        if ended_at is None or now - ended_at >= self._RECENTLY_ENDED_SECONDS:
            return {}
        label = "stdout output" if artifact_name == "stdout" else "GPU metrics"
        warnings: dict[str, list[str]] = {}
        for sub_job_id in sorted(chunks_by_sub_job):
            if wanted is not None and sub_job_id not in wanted:
                continue
            message = (
                f"the job ended {max(0, int(now - ended_at))}s ago, so the last {label} for "
                f"{sub_job_id} may still be uploading; download again with resume in a "
                "minute or two to pick it up."
            )
            logger.warning(message)
            warnings[sub_job_id] = [message]
        return warnings

    def _ended_without_metrics_warnings(
        self,
        shared: _artifact_download.SharedConnection,
        run_uri: str,
        run_name: str,
        job_id: str,
        metrics_by_sub_job: Mapping[str, list[str]],
        wanted: set[str] | None,
    ) -> dict[str, list[str]]:
        """Explain each sub-job of an ended job that saved stdout but no metrics."""
        try:
            stdout_chunks, _ = self._list_gzip_chunks(
                shared, run_uri, run_name, job_id, "stdout", _STDOUT_CHUNK_RE, report_merges=False
            )
        except Exception:
            logger.debug("stdout listing for missing-metrics warnings failed", exc_info=True)
            return {}
        warnings: dict[str, list[str]] = {}
        for sub_job_id in sorted(stdout_chunks):
            if sub_job_id in metrics_by_sub_job or (wanted is not None and sub_job_id not in wanted):
                continue
            message = (
                f"no GPU metrics were saved for {sub_job_id}, although its stdout output "
                "was. Jobs that ran on images from before GPU metrics were uploaded, or "
                "that ended before their first metrics upload, have none."
            )
            logger.warning(message)
            warnings[sub_job_id] = [message]
        return warnings

    def _download_gzip_artifacts(
        self,
        job_id: str,
        output_dir: str | os.PathLike[str],
        *,
        artifact_name: str,
        chunk_pattern: re.Pattern[str],
        destination_name: str,
        temporary_prefix: str,
        resume: bool = False,
        max_workers: int | None = None,
        progress: Callable[[str], None] | None = None,
        on_chunk: Callable[[str, int, int, int], None] | None = None,
        sub_job_ids: Iterable[str] | None = None,
        strict: bool = False,
        stale_after_seconds: float | None = _DOWNLOAD_STALE_AFTER_SECONDS,
        transfer_timeout_seconds: float | None = _DOWNLOAD_TRANSFER_TIMEOUT_SECONDS,
        priority: Iterable[str] = (),
        on_commit: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Reconstruct staged gzip chunks into one file per sub-job.

        Chunks are fetched concurrently and appended in sorted name order. A
        sub-job's destination is replaced only when every chunk is written;
        until then progress is checkpointed to a hidden ``.partial`` file so
        ``resume=True`` can continue an interrupted first download. With
        ``resume=True`` a completed file is extended when its manifest names
        are a prefix of the sorted chunk list, and rebuilt otherwise. A chunk
        that will not decompress is downloaded once more and then skipped,
        unless ``strict``.

        Sub-jobs are written in id order, those named in ``priority`` first
        (in that order); the returned list stays in id order. ``on_commit``
        gets a copy of each sub-job's result entry once its file is committed.
        """
        if max_workers is None:
            max_workers = self._DOWNLOAD_DEFAULT_WORKERS
        if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
            raise ValueError("max_workers must be a positive integer")
        if isinstance(sub_job_ids, (str, bytes)):
            raise TypeError("sub_job_ids must be an iterable of sub-job ids, not a single string")
        wanted = None if sub_job_ids is None else set(sub_job_ids)
        if wanted is not None and not wanted:
            raise ValueError("sub_job_ids must name at least one sub-job")
        if isinstance(priority, (str, bytes)):
            raise TypeError("priority must be an iterable of sub-job ids, not a single string")
        first = {sub_job_id: rank for rank, sub_job_id in enumerate(dict.fromkeys(priority or ()))}
        if stale_after_seconds is not None and (
            isinstance(stale_after_seconds, bool)
            or not isinstance(stale_after_seconds, (int, float))
            or stale_after_seconds <= 0
        ):
            raise ValueError("stale_after_seconds must be a positive number or None")
        if transfer_timeout_seconds is not None and (
            isinstance(transfer_timeout_seconds, bool)
            or not isinstance(transfer_timeout_seconds, (int, float))
            or not transfer_timeout_seconds > 0
        ):
            raise ValueError("transfer_timeout_seconds must be a positive number or None")
        values = self._operation_values()
        values.update(
            artifact=artifact_name,
            max_workers=max_workers,
            resume=bool(resume),
            resume_mode="none",
            sub_job_count=0,
            chunk_count=0,
            chunks_skipped=0,
            bytes_written=0,
            list_ms=0.0,
            fetch_ms=0.0,
            artifact_retry_count=0,
            transfer_timeout_count=0,
            cancelled=False,
        )
        run_uri, run_name = self._experiment_run_uri(job_id)
        shared = _artifact_download.SharedConnection(
            self._open_experiment_artifact_connection, transfer_timeout_seconds
        )
        root = Path(output_dir).expanduser()
        stall_cancel = threading.Event()
        # True once output is on disk that a resumed download would keep.
        saved = False
        try:
            started = time.monotonic()
            chunks_by_sub_job, warnings_by_sub_job = self._list_gzip_chunks(
                shared, run_uri, run_name, job_id, artifact_name, chunk_pattern
            )
            values["list_ms"] = round((time.monotonic() - started) * 1000, 3)
            # The stall check reads the job status (one REST call, plus a
            # stdout listing for metrics); it runs while chunks are fetched so
            # it adds no time to the download.
            stall_check = None
            if stale_after_seconds is not None:
                stall_check = _artifact_download.Background(
                    functools.partial(
                        self._stale_upload_warnings,
                        shared,
                        run_uri,
                        run_name,
                        job_id,
                        artifact_name,
                        chunk_pattern,
                        dict(chunks_by_sub_job),
                        wanted,
                        stale_after_seconds,
                        stall_cancel,
                    ),
                    name="cortex-training-stall-check",
                )

            def stale_warnings() -> dict[str, list[str]]:
                if stall_check is None:
                    return {}
                return stall_check.result(self._STALL_CHECK_WAIT_SECONDS, {})

            if wanted is not None and wanted - set(chunks_by_sub_job):
                # A named sub-job without chunks is fine only if it has a warning.
                unsaved = set(stale_warnings()) - set(chunks_by_sub_job)
                missing = sorted(wanted - set(chunks_by_sub_job) - unsaved)
                if missing:
                    available = sorted(chunks_by_sub_job)
                    raise NoChunksError(
                        f"no {artifact_name} chunks for sub-job(s) "
                        f"{', '.join(missing)}; sub-jobs with {artifact_name}: "
                        f"{', '.join(available) or 'none'}",
                        sub_job_ids=missing,
                        available=available,
                    )
            plans = [
                self._plan_gzip_download(
                    sub_job_id,
                    root / sub_job_id / destination_name,
                    chunks_by_sub_job[sub_job_id],
                    resume=resume,
                )
                for sub_job_id in sorted(chunks_by_sub_job)
                if wanted is None or sub_job_id in wanted
            ]
            # Those named in priority first, in that order; the sort is stable,
            # so the rest stay in id order.
            plans.sort(key=lambda plan: first.get(plan.sub_job_id, len(first)))
            values["sub_job_count"] = len(plans)
            modes = {
                "append"
                if not plan.promote
                else "partial"
                if plan.known
                else "rebuild"
                if plan.destination.exists()
                else "fresh"
                for plan in plans
            }
            values["resume_mode"] = modes.pop() if len(modes) == 1 else "mixed" if modes else "none"
            results: list[dict[str, Any]] = []
            if not plans:
                self._report_artifact_progress(progress, f"{artifact_name}: found 0 chunks\n")
            else:
                started = time.monotonic()
                try:
                    results = self._write_gzip_plans(
                        shared,
                        run_uri,
                        artifact_name,
                        plans,
                        max_workers=max_workers,
                        temporary_prefix=temporary_prefix,
                        progress=progress,
                        on_chunk=on_chunk,
                        strict=strict,
                        warnings_by_sub_job=warnings_by_sub_job,
                        on_commit=on_commit,
                    )
                finally:
                    values["fetch_ms"] = round((time.monotonic() - started) * 1000, 3)
                saved = True
            stale = stale_warnings()
            for result in results:
                messages = stale.get(result["sub_job_id"])
                if messages:
                    result["warnings"] = [*result.get("warnings", []), *messages]
            for sub_job_id, messages in stale.items():
                warnings_by_sub_job.setdefault(sub_job_id, []).extend(messages)
            # Sub-jobs with a warning but no chunks yet still get a result entry.
            unsaved = sorted(set(stale) - set(chunks_by_sub_job))
            results.extend(
                {
                    "sub_job_id": sub_job_id,
                    "filename": destination_name,
                    "saved_path": None,
                    "chunk_count": 0,
                    "first_artifact_uri": None,
                    "last_artifact_uri": None,
                    "warnings": warnings_by_sub_job[sub_job_id],
                }
                for sub_job_id in unsaved
            )
            return sorted(results, key=lambda result: result["sub_job_id"])
        except BaseException as exc:
            if isinstance(exc, (DownloadCancelled, KeyboardInterrupt)):
                values["cancelled"] = True
            # An interrupt in the stall-check wait comes after every file is
            # committed; one from the writes carries its own flag.
            saved = saved or getattr(exc, "download_progress_saved", False)
            self._mark_progress_saved(exc, saved)
            raise
        finally:
            # Stop a stall check that is still running before its session goes.
            stall_cancel.set()
            values["artifact_retry_count"] = shared.retry_count
            values["transfer_timeout_count"] = shared.transfer_timeout_count
            try:
                shared.close()
            except BaseException as exc:
                # Closing the session can block; Ctrl-C there keeps the flag.
                self._mark_progress_saved(exc, saved)
                raise

    @_track_operation("download_stdout_logs")
    def download_stdout_logs(
        self,
        job_id: str,
        output_dir: str | os.PathLike[str],
        *,
        resume: bool = False,
        max_workers: int | None = None,
        progress: Callable[[str], None] | None = None,
        on_chunk: Callable[[str, int, int, int], None] | None = None,
        sub_job_ids: Iterable[str] | None = None,
        strict: bool = False,
        stale_after_seconds: float | None = _DOWNLOAD_STALE_AFTER_SECONDS,
        transfer_timeout_seconds: float | None = _DOWNLOAD_TRANSFER_TIMEOUT_SECONDS,
    ) -> list[dict[str, Any]]:
        """Reconstruct sealed console artifacts into one file per sub-job.

        Chunks are fetched with up to ``max_workers`` concurrent GETs (default
        16). ``resume=True`` appends chunks that are not already recorded beside
        the file, and continues an interrupted download; progress is saved as
        the download runs whether or not ``resume`` is set.

        ``progress`` receives status updates without a trailing newline, then
        ``"\\n"`` when that sub-job finishes or fails. A status that needs no
        updating (nothing new, no chunks) arrives as one message ending in a
        newline. ``on_chunk`` is called as
        ``on_chunk(sub_job_id, chunks_done, chunks_total, bytes)`` when a
        sub-job starts and after each appended chunk; ``chunks_done ==
        chunks_total`` arrives only after that sub-job's file is committed.
        Both run on the calling thread. An error raised by either stops the
        download, keeps the chunks already written for ``resume=True``, and is
        re-raised; raise :class:`DownloadCancelled` to cancel without the call
        being counted as a failure. The return value is unchanged.
        ``sub_job_ids`` limits the download to those sub-jobs. A chunk that will
        not decompress is downloaded once more and then skipped; it is listed in
        that sub-job's ``skipped_chunks`` and ``warnings``. ``strict=True``
        raises instead. While the job is running, a sub-job whose newest chunk is
        older than ``stale_after_seconds`` (default 600; ``None`` disables) gets
        a warning that uploads may be stalled; the download still completes.
        A LIST or GET attempt still running after ``transfer_timeout_seconds``
        (default 300; ``None`` disables) is abandoned and retried. Two
        downloads of the same output directory at once are unsupported.
        """
        return self._download_gzip_artifacts(
            job_id,
            output_dir,
            artifact_name="stdout",
            chunk_pattern=_STDOUT_CHUNK_RE,
            destination_name="stdout.log",
            temporary_prefix=".stdout-",
            resume=resume,
            max_workers=max_workers,
            progress=progress,
            on_chunk=on_chunk,
            sub_job_ids=sub_job_ids,
            strict=strict,
            stale_after_seconds=stale_after_seconds,
            transfer_timeout_seconds=transfer_timeout_seconds,
        )

    @_track_operation("download_metrics")
    def download_metrics(
        self,
        job_id: str,
        output_dir: str | os.PathLike[str],
        *,
        resume: bool = False,
        max_workers: int | None = None,
        progress: Callable[[str], None] | None = None,
        on_chunk: Callable[[str, int, int, int], None] | None = None,
        sub_job_ids: Iterable[str] | None = None,
        strict: bool = False,
        stale_after_seconds: float | None = _DOWNLOAD_STALE_AFTER_SECONDS,
        transfer_timeout_seconds: float | None = _DOWNLOAD_TRANSFER_TIMEOUT_SECONDS,
        priority: Iterable[str] = (),
        on_commit: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Reconstruct staged GPU metric chunks into one JSONL file per sub-job.

        Accepts the same ``resume``, ``max_workers``, ``progress``, ``on_chunk``,
        ``sub_job_ids``, ``strict``, ``stale_after_seconds``, and
        ``transfer_timeout_seconds`` options as
        :meth:`download_stdout_logs`. Sub-jobs named in ``priority`` are
        downloaded first, in that order, then the rest in id order; the
        returned list is in id order either way. ``on_commit`` is called, on
        the calling thread, with a copy of each sub-job's result entry once its
        file is committed; an error it raises stops the download as one from
        ``on_chunk`` does. While the job is running, a sub-job with
        saved stdout but no metrics gets a result entry with ``chunk_count`` 0,
        ``saved_path`` ``None``, and a warning. Two downloads
        of the same output directory at once are unsupported.
        """
        return self._download_gzip_artifacts(
            job_id,
            output_dir,
            artifact_name="metrics",
            chunk_pattern=_GPU_METRICS_CHUNK_RE,
            destination_name="gpu.jsonl",
            temporary_prefix=".gpu-metrics-",
            resume=resume,
            max_workers=max_workers,
            progress=progress,
            on_chunk=on_chunk,
            sub_job_ids=sub_job_ids,
            strict=strict,
            stale_after_seconds=stale_after_seconds,
            transfer_timeout_seconds=transfer_timeout_seconds,
            priority=priority,
            on_commit=on_commit,
        )

    # ─── Data-plane async operations ─────────────────────────────────────

    # Maximum payload size for forward-backward requests.
    # Payloads beyond this limit will fail fast with a message.
    _MAX_FWD_BWD_BYTES = 60 * 1024 * 1024  # 60 MB

    # Maximum payload size for generate / generate_stream request bodies.
    # The encoded frame must stay under this cap; generate splits an oversized
    # frame into request chunks, while generate_stream fails fast client-side
    # with a clear message.
    _MAX_GENERATE_BYTES = 60 * 1024 * 1024  # 60 MB

    def _build_generate_frame(
        self,
        op: str,
        job_id: str,
        prompts,
        sampling_params,
        routing_key,
        strict,
    ) -> bytes:
        """Encode a generate request body, for both the unary and stream calls.

        The two calls take the same logical fields and the server reads them
        from the same request model, so any difference in how they encode is a
        bug waiting to happen -- and was one: ``generate_stream`` built its
        body with ``json.dumps`` long after ``generate`` had moved token ids
        into the frame's tensor section, so streaming callers silently kept
        paying the JSON encoding for every token.

        ``op`` only names the caller in the length-check error message.
        """
        self._check_prompt_lengths(op, job_id, prompts)
        # Packed after validation so the check above still reads plain lists,
        # and so a rejected prompt is never converted for nothing.
        payload: dict = {"prompts": _pack_token_prompts(prompts)}
        if sampling_params is not None:
            payload["sampling_params"] = sampling_params
        if routing_key is not None:
            payload["routing_key"] = routing_key
        if strict is not None:
            payload["strict"] = strict
        return wire.dumps(
            payload,
            metadata={"response_options": {"format": "dssst1", "delivery": "chunked"}},
        )

    def _check_generate_payload(self, op: str, body: bytes) -> None:
        if len(body) > self._MAX_GENERATE_BYTES:
            size_mb = len(body) / (1024 * 1024)
            raise ValueError(
                f"{op} payload is {size_mb:.1f} MB, which exceeds the "
                "maximum allowed size of "
                f"{self._MAX_GENERATE_BYTES / (1024 * 1024):.0f} MB. "
                "Reduce the number of prompts or shorten each prompt to "
                "bring the serialized request under the limit."
            )

    def _resolve_sampling_max_seq_len(self, job_id: str) -> int | None:
        """Return the sampling sub-job's ``max_seq_len``, or ``None`` if unknown.

        The value is read from the job's sampling ``inference_config`` (the same
        number the backend launches vLLM with as ``max_model_len``) and cached
        per ``job_id``. A successful lookup — including a definitive "no sampling
        sub-job" miss — is cached; a transient ``get_job`` failure is not, so a
        later call retries rather than disabling validation permanently.
        """
        if job_id in self._sampling_max_seq_len:
            return self._sampling_max_seq_len[job_id]
        try:
            job = self.get_job(job_id)
        except Exception:
            # Don't let a transient lookup failure break generate, and don't
            # cache the miss — the real request will surface any hard error.
            return None
        max_seq_len: int | None = None
        for sub in job.get("sub_jobs", []) or []:
            if not isinstance(sub, dict):
                continue
            cfg = sub.get("inference_config")
            if not isinstance(cfg, dict):
                continue
            raw = cfg.get("max_seq_len")
            # Proto numbers can arrive as floats (e.g. 2048.0); coerce to int.
            if isinstance(raw, (int, float)) and raw > 0:
                max_seq_len = int(raw)
                break
        self._sampling_max_seq_len[job_id] = max_seq_len
        return max_seq_len

    def _check_prompt_lengths(self, op: str, job_id: str, prompts: list[str | list[int]]) -> None:
        """Fail fast when a pre-tokenized prompt cannot fit the sampling window.

        Mirrors vLLM's decoder-prompt check (``vllm/v1/engine/input_processor``):
        a generation prompt must be strictly shorter than ``max_model_len`` so
        at least one output token fits, i.e. it is rejected when
        ``len(prompt_token_ids) >= max_seq_len``. Only pre-tokenized prompts
        are validated here, since their token count is exact; string prompts
        would require replicating the server tokenizer and are left to fail
        server-side.

        Tensors count as pre-tokenized. They are a supported wire shape for
        token ids, and matching on ``list`` alone let one through unchecked --
        silently trading a clear client-side error for vLLM's much later
        ``maximum model length`` failure.
        """
        torch = _load_torch()

        def _pretokenized(p) -> bool:
            return isinstance(p, list) or torch.is_tensor(p)

        # ``prompts`` is either one prompt or a batch of them. Iterating one
        # prompt yields its tokens, which the loop below would then read as
        # prompts: a flat ``list[int]`` lands on ``int``, which is not
        # pre-tokenized, so the whole check is skipped; a bare tensor lands on
        # 0-d scalars, which are tensors, so the check proceeds and ``len()``
        # raises on them. Wrapping first means one prompt is checked as one
        # prompt in either shape, and reads the argument the same way
        # ``_pack_token_prompts`` does.
        if torch.is_tensor(prompts) or (
            isinstance(prompts, list) and prompts and _is_token_id(prompts[0])
        ):
            prompts = [prompts]

        if not any(_pretokenized(p) for p in prompts):
            return
        max_seq_len = self._resolve_sampling_max_seq_len(job_id)
        if max_seq_len is None:
            return
        for i, prompt in enumerate(prompts):
            if not _pretokenized(prompt):
                continue
            n = len(prompt)
            if n >= max_seq_len:
                raise ValueError(
                    f"{op} prompt at index {i} has {n} tokens, which does not "
                    f"fit the sampling job's max_seq_len of {max_seq_len}: a "
                    "prompt must be shorter than max_seq_len to leave room for "
                    "at least one generated token (vLLM rejects "
                    "len(prompt) >= max_model_len). Shorten the prompt to at "
                    f"most {max_seq_len - 1} tokens, or recreate the job with a "
                    "larger max_seq_len."
                )

    def _post_octet_request_chunks(
        self,
        *,
        job_id: str,
        path_suffix: str,
        operation: str,
        frame: bytes,
        max_bytes: int,
        force_chunk: bool = False,
        allow_group_restart: bool = False,
        debug_context: dict[str, Any] | None = None,
    ) -> dict:
        chunks = wire.encode_byte_chunks(
            frame,
            kind="request",
            operation=operation,
            max_bytes=max_bytes,
            force_chunk=force_chunk,
        )
        group_metadata = wire.read_byte_chunk_metadata(chunks[0])
        chunk_group_id = (
            str(group_metadata["chunk_group_id"])
            if group_metadata is not None and group_metadata.get("chunk_group_id")
            else None
        )
        group_restarts = 0
        idx = 0
        final_body: dict | None = None
        while idx < len(chunks):
            chunk = chunks[idx]
            try:
                resp = self._send(
                    "POST",
                    f"{self._prefix}/{job_id}/{path_suffix}",
                    data=chunk,
                    headers={"Content-Type": "application/octet-stream"},
                    retry_on=(_is_chunk_post_transient if allow_group_restart else _is_transient),
                    debug_context=debug_context,
                )
            except requests.exceptions.HTTPError as exc:
                detail = _chunk_group_error_detail(exc.response)
                if detail is None:
                    raise
                code = detail.get("code")
                if allow_group_restart and code == _CHUNK_GROUP_RESTART_REQUIRED and group_restarts < 1:
                    group_restarts += 1
                    idx = 0
                    final_body = None
                    logger.warning(
                        "%s chunk group %s lost server state; replaying once from chunk 0",
                        operation,
                        chunk_group_id,
                    )
                    continue
                error_type = (
                    ChunkGroupRestartError if code == _CHUNK_GROUP_RESTART_REQUIRED else ChunkGroupConflictError
                )
                raise error_type(
                    str(detail.get("message") or code),
                    response=exc.response,
                    detail=detail,
                ) from exc

            body = resp.json()
            if idx < len(chunks) - 1:
                if isinstance(body, dict) and body.get("request_id"):
                    raise RuntimeError(f"{operation} chunk {idx} unexpectedly returned request_id")
            else:
                final_body = body
            idx += 1
        if final_body is None:
            raise RuntimeError(f"{operation} produced no request body")
        return final_body

    @_track_operation("forward_backward")
    def forward_backward(self, job_id: str, data: bytes) -> str:
        """Submit a forward+backward pass. Returns request_id."""
        self._fwd_bwd_send_count += 1
        call = self._fwd_bwd_send_count
        debug_context = {
            "phase": "submit",
            "call": call,
            "job_id": job_id,
            "payload_bytes": len(data),
        }
        debug_label = self._debug_context_label(debug_context)
        logger.debug("%s payload total bytes: %d", debug_label, len(data))
        body = self._post_octet_request_chunks(
            job_id=job_id,
            path_suffix="forward-backward",
            operation="fwd-bwd",
            frame=data,
            max_bytes=self._MAX_FWD_BWD_BYTES,
            force_chunk=True,
            allow_group_restart=True,
            debug_context=debug_context,
        )
        request_id = body["request_id"]
        request_debug_context = {
            "phase": "poll",
            "call": call,
            "job_id": job_id,
            "request_id": request_id,
            "payload_bytes": len(data),
        }
        self._fwd_bwd_request_debug[request_id] = request_debug_context
        logger.debug(
            "%s submitted request_id=%s body=%s",
            debug_label,
            request_id,
            self._debug_json_summary(body),
        )
        return request_id

    @_track_operation("generate")
    def generate(
        self,
        job_id: str,
        prompts: list[str | list[int]],
        sampling_params: dict | list[dict | None] | None = None,
        routing_key: str | list[str | None] | None = None,
        strict: bool | None = None,
    ) -> str:
        """Submit a generate request to a sampling sub-job. Returns request_id.

        The typed fields are serialized as a DSSST1 safetensors frame and sent
        as ``application/octet-stream``. Oversized frames are byte-chunked
        across multiple POSTs; the final POST returns the pollable request id.
        Poll with :meth:`poll_request`.

        Prompt-length validation is asymmetric by input type:

        * **Pre-tokenized prompts** are checked client-side against the
          sampling sub-job's ``max_seq_len`` and fail fast with a
          ``ValueError`` when ``len(prompt) >= max_seq_len`` — the token count
          is exact, so this never rejects a prompt vLLM would accept. A flat
          ``list[int]``, and a bare 1-D tensor, each count as one prompt.
        * **String prompts** are *not* validated here. Counting their tokens
          would require replicating the server tokenizer (including special
          tokens), so an over-long string instead fails server-side in vLLM
          with a ``maximum model length`` error rather than client-side.
        """
        body = self._build_generate_frame("generate", job_id, prompts, sampling_params, routing_key, strict)
        response = self._post_octet_request_chunks(
            job_id=job_id,
            path_suffix="generate",
            operation="generate",
            frame=body,
            max_bytes=self._MAX_GENERATE_BYTES,
        )
        request_id = response["request_id"]
        self._generate_request_ids.add(request_id)
        return request_id

    @_track_operation("generate_stream")
    def generate_stream(
        self,
        job_id: str,
        prompts: list[str | list[int]],
        sampling_params: dict | list[dict | None] | None = None,
        routing_key: str | list[str | None] | None = None,
        strict: bool | None = None,
    ) -> dict:
        """Start a streaming generate request. Returns the response body.

        The body is the same DSSST1 safetensors frame :meth:`generate` sends,
        under ``application/octet-stream``. Unlike ``generate`` it is never
        split into request chunks: the frame goes out in one POST and must fit
        under 60 MiB.

        Progress is read with :meth:`get_request_status` using the returned
        ``request_id``. Cancellation uses :meth:`cancel_request`.

        Prompt-length validation matches :meth:`generate`: pre-tokenized
        prompts are checked client-side against the sampling sub-job's
        ``max_seq_len`` and fail fast, while over-long string prompts are left
        to fail server-side in vLLM (see :meth:`generate`).
        """
        body = self._build_generate_frame("generate_stream", job_id, prompts, sampling_params, routing_key, strict)
        self._check_generate_payload("generate_stream", body)
        resp = self._send(
            "POST",
            f"{self._prefix}/{job_id}/generate-stream",
            data=body,
            headers={"Content-Type": "application/octet-stream"},
        )
        return resp.json()

    @_track_operation("step")
    def step(self, job_id: str, learning_rate: float | None = None) -> str:
        """Submit an optimizer step. Returns request_id."""
        body: dict = {}
        if learning_rate is not None:
            body["learning_rate"] = learning_rate
        resp = self._send("POST", f"{self._prefix}/{job_id}/step", json=body)
        return resp.json()["request_id"]

    @_track_operation("save")
    def save(
        self,
        job_id: str,
        checkpoint_id: str | None = None,
        checkpoint_type: str | None = None,
    ) -> str:
        """Submit a checkpoint save. Returns request_id."""
        body: dict = {}
        if checkpoint_id is not None:
            body["checkpoint_id"] = checkpoint_id
        if checkpoint_type is not None:
            normalized_type = checkpoint_type.lower()
            if normalized_type not in ("resumable", "weights-only"):
                raise ValueError("checkpoint_type must be 'resumable' or 'weights-only'")
            body["checkpoint_type"] = normalized_type
        resp = self._send("POST", f"{self._prefix}/{job_id}/save", json=body)
        return resp.json()["request_id"]

    @_track_operation("load")
    def load(
        self,
        job_id: str,
        checkpoint_id: str,
        source_job_id: str | None = None,
        *,
        target_sub_job_id: str | None = None,
    ) -> str:
        """Submit a checkpoint load into an existing job. Returns request_id.

        Args:
            job_id: The job to load the checkpoint into.
            checkpoint_id: Checkpoint identifier (from a prior save).
            source_job_id: Optional. Load from another job's checkpoint store.
            target_sub_job_id: Optional. Route the load to a specific training
                sub-job. Format: ``"{job_id}:training:{index}"``. Omit to use
                the session's default training sub-job.

        Returns:
            request_id for polling via ``poll_request()``.

        **When to use target_sub_job_id:**

        A job has at most one training sub-job, so omitting this routes to that
        sub-job. Pass it when you want explicit routing rather than relying on
        the server's default resolution.

        **Discovering sub-job IDs:**

        .. code-block:: python

            job = client.get_job(job_id)
            for sj in job["sub_jobs"]:
                if sj["job_type"] == "training":
                    sub_job_id = sj["sub_job_id"]
                    n_gpus = sj["training_config"]["n_gpus"]
                    print(f"Training sub-job {sub_job_id}: {n_gpus} GPUs")

        **DP size compatibility:**

        If the target sub-job has a different DP size (``n_gpus``) than the
        checkpoint was saved from, you **must** have created the job with
        ``load_optimizer_states=False`` in the ``SubJobConfig``. The optimizer
        states are DP-sharded and cannot be resized. This setting is configured
        at job creation time and cannot be changed at runtime.

        **Server validation:**

        The server validates that ``target_sub_job_id`` belongs to this session
        (returns 400 if not) and is a training sub-job (returns 501 for
        non-training sub-jobs like sampling).
        """
        body: dict = {"checkpoint_id": checkpoint_id}
        if source_job_id is not None:
            body["source_job_id"] = source_job_id
        if target_sub_job_id is not None:
            body["target_sub_job_id"] = target_sub_job_id
        resp = self._send("POST", f"{self._prefix}/{job_id}/load", json=body)
        return resp.json()["request_id"]

    def _operation(
        self,
        job_id: str,
        operation_type: str,
        *,
        payload: dict | bytes | bytearray | memoryview | None = None,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
        max_json_bytes: int | None = None,
        retry_on: Callable[[BaseException], bool] | None = None,
    ) -> dict:
        """Submit a generic data-plane operation to the routed sub-job.

        ``max_json_bytes``, when set, fails fast if the encoded body exceeds it.
        ``retry_on`` overrides the retry predicate when passed; chunk posts use
        one that excludes structured chunk-group conflicts.
        """
        body: dict = {"operation_type": operation_type}
        if sub_job_id is not None:
            body["sub_job_id"] = sub_job_id
        if sub_job_type is not None:
            body["sub_job_type"] = sub_job_type
        if isinstance(payload, (bytes, bytearray, memoryview)):
            payload = {
                "payload_b64": base64.b64encode(bytes(payload)).decode("ascii"),
                "content_type": "application/octet-stream",
            }
        if payload is not None:
            body["payload"] = payload
        if max_json_bytes is not None:
            encoded_bytes = _operation_body_json_bytes(body)
            if encoded_bytes > max_json_bytes:
                raise ValueError(
                    f"{operation_type} operation body is {encoded_bytes} bytes, which "
                    f"exceeds the {max_json_bytes}-byte limit for one /operation "
                    f"request. Lower {_OPERATION_MAX_JSON_BYTES_ENV} so the payload is "
                    f"split into more chunks, or raise it if the deployment allows "
                    f"larger JSON bodies."
                )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "POST operation %s/%s/operation body=%s",
                self._prefix,
                job_id,
                self._debug_json_summary(_log_safe_operation_body(body)),
            )
        kwargs: dict[str, Any] = {}
        if retry_on is not None:
            kwargs["retry_on"] = retry_on
        resp = self._send(
            "POST",
            f"{self._prefix}/{job_id}/operation",
            json=body,
            **kwargs,
        )
        return resp.json()

    def _post_operation_request_chunks(
        self,
        job_id: str,
        *,
        operation_type: str,
        frame: bytes,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
    ) -> dict:
        """Send a DSSST1 frame through ``/operation``, byte-chunked if oversized."""
        max_json_bytes = _operation_max_json_bytes()
        # The server assembler rejects total_chunks <= 1, so a fitting frame
        # is posted raw.
        chunks = wire.encode_byte_chunks(
            frame,
            kind="request",
            operation=operation_type,
            max_bytes=_operation_chunk_max_bytes(max_json_bytes),
        )
        final_body: dict | None = None
        for idx, chunk in enumerate(chunks):
            try:
                body = self._operation(
                    job_id,
                    operation_type,
                    payload=chunk,
                    sub_job_id=sub_job_id,
                    sub_job_type=sub_job_type,
                    max_json_bytes=max_json_bytes,
                    retry_on=_is_chunk_post_transient,
                )
            except requests.exceptions.HTTPError as exc:
                detail = _chunk_group_error_detail(exc.response)
                if detail is None:
                    raise
                raise ChunkGroupConflictError(
                    str(detail.get("message") or detail.get("code")),
                    response=exc.response,
                    detail=detail,
                ) from exc
            if idx < len(chunks) - 1:
                if isinstance(body, dict) and body.get("request_id"):
                    raise RuntimeError(
                        f"{operation_type} chunk {idx} unexpectedly returned request_id"
                    )
            else:
                final_body = body
        if final_body is None:
            raise RuntimeError(f"{operation_type} produced no request body")
        return final_body

    @_track_operation("forward")
    def forward(
        self,
        job_id: str,
        payload: dict | bytes | bytearray | memoryview | None = None,
        *,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
    ) -> dict:
        """Submit a generic forward operation. Returns the response body.

        A ``bytes`` payload is a DSSST1 frame; it is byte-chunked across as many
        ``/operation`` POSTs as the JSON body ceiling requires and reassembled by
        the backend. Only the final POST returns the pollable ``request_id``;
        poll it with :meth:`poll_request`. Dict payloads stay a single plain
        ``/operation`` POST.
        """
        if isinstance(payload, (bytes, bytearray, memoryview)):
            body = self._post_operation_request_chunks(
                job_id,
                operation_type="forward",
                frame=bytes(payload),
                sub_job_id=sub_job_id,
                sub_job_type=sub_job_type,
            )
        else:
            body = self._operation(
                job_id,
                "forward",
                payload=payload,
                sub_job_id=sub_job_id,
                sub_job_type=sub_job_type,
            )
        request_id = body.get("request_id") if isinstance(body, dict) else None
        if isinstance(request_id, str) and request_id:
            self._forward_request_ids.add(request_id)
        return body

    def fwd(
        self,
        job_id: str,
        payload: dict | bytes | bytearray | memoryview | None = None,
        *,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
    ) -> dict:
        """Compatibility alias for :meth:`forward`."""
        return self.forward(
            job_id,
            payload=payload,
            sub_job_id=sub_job_id,
            sub_job_type=sub_job_type,
        )

    def fwd_no_grad(
        self,
        job_id: str,
        payload: dict | bytes | bytearray | memoryview | None = None,
        *,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
    ) -> dict:
        """Compatibility alias for :meth:`forward`."""
        return self.forward(
            job_id,
            payload=payload,
            sub_job_id=sub_job_id,
            sub_job_type=sub_job_type,
        )

    @_track_operation("weight_sync")
    def weight_sync(
        self,
        job_id: str,
        source_sub_job_id: str,
        target_sub_job_ids: list[str],
        *,
        weight_format: str | None = None,
        bucket_size: int | None = None,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
    ) -> str:
        """Sync model weights from a training sub-job into one or more sampling
        sub-jobs (used in RL training loops). Returns request_id.

        Multi-sub-job sessions require the operation envelope to include a
        routing hint; by default the operation is routed through the source
        training sub-job.

        Args:
            job_id: The job owning both sub-jobs.
            source_sub_job_id: Training sub-job to read weights from, e.g.
                ``f"{job_id}:training:0"``.
            target_sub_job_ids: Sampling sub-jobs to update, e.g.
                ``[f"{job_id}:sampling:0"]``.
            weight_format: ``"vllm"`` (server default) or ``"hf"`` for full
                weights, or ``"lora"`` to broadcast only the adapter tensors.
            bucket_size: Optional transfer-bucket size in bytes. Increase this
                when one model tensor exceeds the server default bucket.

        Example::

            request_id = client.weight_sync(
                job_id,
                source_sub_job_id=f"{job_id}:training:0",
                target_sub_job_ids=[f"{job_id}:sampling:0"],
                weight_format="lora",
            )
            client.poll_request(job_id, request_id)
        """
        body = {
            "source_sub_job_id": source_sub_job_id,  # training
            "target_sub_job_ids": list(target_sub_job_ids),  # sampling
        }
        if weight_format is not None:
            # "lora" broadcasts only the trained adapter tensors.
            body["weight_format"] = weight_format
        if bucket_size is not None:
            if bucket_size <= 0:
                raise ValueError("bucket_size must be positive")
            body["bucket_size"] = int(bucket_size)
        return self._operation(
            job_id,
            "weight-sync",
            payload=body,
            sub_job_id=sub_job_id or source_sub_job_id,  # training
            sub_job_type=(sub_job_type if sub_job_type is not None else "training"),
        )["request_id"]

    @_track_operation("bootstrap_router_replay")
    def bootstrap_router_replay(
        self,
        job_id: str,
        source_sub_job_id: str,
        target_sub_job_id: str,
        *,
        max_cache_bytes: int | None = None,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
    ) -> dict:
        """Bootstrap router replay from one sub-job to another."""
        payload: dict = {
            "source_sub_job_id": source_sub_job_id,
            "target_sub_job_id": target_sub_job_id,
        }
        if max_cache_bytes is not None:
            payload["max_cache_bytes"] = max_cache_bytes
        return self._operation(
            job_id,
            "bootstrap-router-replay",
            payload=payload,
            sub_job_id=sub_job_id,
            sub_job_type=sub_job_type,
        )

    @_track_operation("router_replay_discard")
    def router_replay_discard(
        self,
        job_id: str,
        sample_ids: list[str] | None = None,
        *,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
        extra: dict | None = None,
    ) -> dict:
        """Discard router replay samples on a routed sub-job."""
        payload = dict(extra or {})
        payload.setdefault("sample_ids", list(sample_ids or []))
        return self._operation(
            job_id,
            "router-replay-discard",
            payload=payload,
            sub_job_id=sub_job_id,
            sub_job_type=sub_job_type,
        )

    @_track_operation("reset_prefix_cache")
    def reset_prefix_cache(
        self,
        job_id: str,
        *,
        sub_job_id: str | None = None,
        sub_job_type: str | None = "sampling",
        drain: bool = True,
        timeout_s: float = 60.0,
        retry_interval_s: float = 0.1,
        extra: dict | None = None,
    ) -> dict:
        """Reset the sampling prefix cache for a Cortex Training job.

        Operation-specific fields are carried inside the opaque operation
        payload. ``extra`` allows new reset options to flow through without a
        cortex-training release.
        """
        payload = dict(extra or {})
        payload.setdefault("drain", drain)
        payload.setdefault("timeout_s", timeout_s)
        payload.setdefault("retry_interval_s", retry_interval_s)
        return self._operation(
            job_id,
            "reset-prefix-cache",
            payload=payload,
            sub_job_id=sub_job_id,
            sub_job_type=sub_job_type,
        )

    # ─── Async request polling ────────────────────────────────────────────

    @staticmethod
    def _normalize_request_status(status: str) -> str:
        """Normalize full enum names (``REQUEST_STATE_DONE``) to short form (``done``)."""
        return status.lower().removeprefix("request_state_")

    @staticmethod
    def _decode_result_payload(result: dict) -> dict | None:
        if not isinstance(result, dict):
            return None
        if result.get("wire_format") != wire.WIRE_FORMAT_VERSION:
            return None
        if result.get("encoding") != "base64":
            raise RuntimeError("DSSST1 result payload must use base64 encoding")
        raw = result.get("payload_b64")
        if not isinstance(raw, str) or not raw:
            raise RuntimeError("DSSST1 result payload missing payload_b64")
        payload = base64.b64decode(raw)
        decoded = wire.loads(payload)
        if not isinstance(decoded, dict):
            raise RuntimeError("DSSST1 result payload did not decode to a dict")
        return CortexTrainingClient._merge_transport_metrics(decoded, result)

    @staticmethod
    def _merge_transport_metrics(decoded: dict, envelope: dict) -> dict:
        transport_metrics = envelope.get("metrics")
        if not isinstance(transport_metrics, dict):
            return decoded
        decoded = dict(decoded)
        payload_metrics = decoded.get("metrics")
        if isinstance(payload_metrics, dict):
            decoded["metrics"] = {**payload_metrics, **transport_metrics}
        else:
            decoded["metrics"] = dict(transport_metrics)
        return decoded

    @staticmethod
    def _decode_result_chunk_event(event: dict) -> bytes | None:
        if not isinstance(event, dict) or event.get("type") != "result_chunk":
            return None
        raw = event.get("payload_b64")
        if not isinstance(raw, str) or not raw:
            raise RuntimeError("result_chunk event missing payload_b64")
        payload = base64.b64decode(raw)
        expected = event.get("payload_sha256")
        if expected is not None and hashlib.sha256(payload).hexdigest() != expected:
            raise RuntimeError("result_chunk payload_sha256 mismatch")
        return payload

    @staticmethod
    def _restore_generate_result_lists(value: Any, torch_module: Any | None = None) -> Any:
        torch_module = _load_torch() if torch_module is None else torch_module
        if torch_module.is_tensor(value):
            return value.cpu().tolist()
        if isinstance(value, dict):
            return {
                key: CortexTrainingClient._restore_generate_result_lists(item, torch_module)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [CortexTrainingClient._restore_generate_result_lists(item, torch_module) for item in value]
        if isinstance(value, tuple):
            return tuple(CortexTrainingClient._restore_generate_result_lists(item, torch_module) for item in value)
        return value

    def _normalize_generate_result_if_needed(self, request_id: str, result: dict) -> dict:
        if request_id not in self._generate_request_ids:
            return result
        self._generate_request_ids.discard(request_id)
        if isinstance(result, dict) and "results" in result:
            result = dict(result)
            result["results"] = self._restore_generate_result_lists(result["results"])
        return result

    def _normalize_forward_result_if_needed(self, request_id: str, result: dict) -> dict:
        """Decode a forward result that is only a base64 DSSST1 blob.

        A current backend returns a self-describing envelope that
        :meth:`_decode_result_payload` already handles. An older backend
        returns ``{"job_id", "payload_b64"}`` with no ``wire_format``.
        """
        if request_id not in self._forward_request_ids:
            return result
        self._forward_request_ids.discard(request_id)
        if not isinstance(result, dict) or result.get("wire_format"):
            return result
        raw = result.get("payload_b64")
        if not isinstance(raw, str) or not raw:
            return result
        decoded = wire.loads(base64.b64decode(raw))
        if not isinstance(decoded, dict):
            raise RuntimeError("legacy forward payload_b64 did not decode to a dict")
        return self._merge_transport_metrics(decoded, result)

    @staticmethod
    def _decode_stream_result_event(event: Any) -> Any:
        if not isinstance(event, dict) or event.get("type") != "result":
            return event
        result = event.get("result")
        if not isinstance(result, dict):
            return event
        decoded = CortexTrainingClient._decode_result_payload(result)
        if decoded is None:
            return event
        event = dict(event)
        event["result"] = CortexTrainingClient._restore_generate_result_lists(decoded)
        return event

    @staticmethod
    def _decode_stream_result_events(status: dict) -> dict:
        events = status.get("events")
        if not isinstance(events, list):
            return status
        decoded = [CortexTrainingClient._decode_stream_result_event(event) for event in events]
        if decoded == events:
            return status
        status = dict(status)
        status["events"] = decoded
        return status

    @_track_operation("poll_request")
    def poll_request(self, job_id: str, request_id: str) -> dict:
        """Poll until the async request completes. Returns the result dict."""
        deadline = time.monotonic() + self.poll_timeout
        delay = self.poll_interval
        debug_context = self._fwd_bwd_request_debug.get(request_id)
        debug_label = self._debug_context_label(debug_context) if debug_context is not None else None
        result_chunks: list[bytes] = []
        cursor: str | None = None
        while time.monotonic() < deadline:
            status = self.get_request_status(job_id, request_id, cursor=cursor)
            state = self._normalize_request_status(status.get("status", ""))
            events = status.get("events") or []
            received_chunk = False
            for event in events:
                with _server_payload():
                    chunk = self._decode_result_chunk_event(event)
                if chunk is not None:
                    result_chunks.append(chunk)
                    received_chunk = True
            next_cursor = status.get("next_cursor")
            if isinstance(next_cursor, str) and next_cursor:
                cursor = next_cursor
                # Drain available result chunks without backing off.
                continue
            if state in ("completed", "done", "succeeded"):
                if result_chunks:
                    with _server_payload():
                        result = wire.decode_result_chunks(result_chunks)
                    result = self._merge_transport_metrics(
                        result,
                        status.get("result") or {},
                    )
                else:
                    result = status.get("result") or {}
                    with _server_payload():
                        decoded = self._decode_result_payload(result)
                    if decoded is not None:
                        result = decoded
                result = self._normalize_generate_result_if_needed(request_id, result)
                result = self._normalize_forward_result_if_needed(request_id, result)
                if debug_label is not None:
                    logger.debug(
                        "%s completed state=%s result=%s",
                        debug_label,
                        state,
                        self._debug_json_summary(result),
                    )
                    self._fwd_bwd_request_debug.pop(request_id, None)
                return result
            if state in ("failed", "cancelled", "canceled"):
                error = status.get("error", "")
                if debug_label is not None:
                    logger.debug(
                        "%s failed state=%s error=%s body=%s",
                        debug_label,
                        state,
                        error,
                        self._debug_json_summary(status),
                    )
                    self._fwd_bwd_request_debug.pop(request_id, None)
                self._generate_request_ids.discard(request_id)
                self._forward_request_ids.discard(request_id)
                raise _server_failure(
                    RuntimeError(f"Request {request_id} ended with state '{state}': {error}"),
                    failed=state == "failed",
                )
            if received_chunk:
                # Defensive: if the server returns a chunk without next_cursor
                # before terminal status, poll again immediately rather than
                # sleeping.
                continue
            delay = self._sleep_with_backoff(delay, deadline)
        if debug_label is not None:
            logger.debug("%s timed out after %ss", debug_label, self.poll_timeout)
            self._fwd_bwd_request_debug.pop(request_id, None)
        self._generate_request_ids.discard(request_id)
        self._forward_request_ids.discard(request_id)
        raise _server_failure(
            TimeoutError(f"Request {request_id} did not complete within {self.poll_timeout}s")
        )

    @_track_operation("get_request_status")
    def get_request_status(
        self,
        job_id: str,
        request_id: str,
        *,
        max_events: int | None = None,
        cursor: str | None = None,
    ) -> dict:
        params = {}
        if max_events is not None:
            params["max_events"] = max_events
        if cursor:
            params["cursor"] = cursor
        debug_context = self._fwd_bwd_request_debug.get(request_id)
        resp = self._send(
            "GET",
            f"{self._prefix}/{job_id}/requests/{request_id}",
            params=params or None,
            debug_context=debug_context,
        )
        with _server_payload():
            return self._decode_stream_result_events(resp.json())

    @_track_operation("cancel_request")
    def cancel_request(
        self,
        job_id: str,
        request_id: str,
        *,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
    ) -> dict:
        return self._operation(
            job_id,
            "cancel-request",
            payload={"request_id": request_id},
            sub_job_id=sub_job_id,
            sub_job_type=sub_job_type,
        )

    # ─── Logs (read-only) ────────────────────────────────────────────────

    def tail_logs(
        self,
        job_id: str,
        *,
        cursor: str | None = None,
        max_lines: int | None = None,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
    ) -> dict:
        """Fetch one page of a sub-job's logs from ``cursor`` (read-only).

        The target is the sub-job's zone-manager (Ray head) pod, whose stdout
        already includes worker output (Ray ``log_to_driver``); identify it via
        ``sub_job_id`` (or ``sub_job_type`` when unambiguous). Returns
        ``{"entries": [...], "next_cursor": str, "eof": bool}``. Re-call with
        ``next_cursor`` to continue; an empty ``cursor`` reads from the start.

        A finished job's pods are gone, so this raises at once, with a pointer
        to :meth:`download_stdout_logs` for the saved console; so does a
        ``sub_job_id`` that is not part of the job.
        """
        payload: dict = {}
        if cursor is not None:
            payload["cursor"] = cursor
        if max_lines is not None:
            payload["max_lines"] = max_lines
        checked: dict[str, Any] = {}

        def retry_on(exc: BaseException) -> bool:
            return _is_transient(exc) and self._tail_hint(exc, job_id, sub_job_id, checked) is None

        try:
            return self._operation(
                job_id,
                "tail-logs",
                payload=payload,
                sub_job_id=sub_job_id,
                sub_job_type=sub_job_type,
                retry_on=retry_on,
            )
        except requests.exceptions.HTTPError as exc:
            hint = self._tail_hint(exc, job_id, sub_job_id, checked)
            if hint is None:
                raise
            error = requests.exceptions.HTTPError(
                f"{exc}. {hint}", response=exc.response, request=exc.request
            )
            # Lets stream_logs end a tail cleanly when the job ends under it.
            error.job_ended = bool(checked.get("ended"))  # type: ignore[attr-defined]
            raise error from exc

    def _tail_hint(
        self,
        exc: BaseException,
        job_id: str,
        sub_job_id: str | None,
        checked: dict[str, Any],
    ) -> str | None:
        """Explain a tail-logs failure that no retry can fix, else ``None``.

        A 404, 409 or 410 from tail-logs means the sub-job's session is not there.
        The job is looked up once per call: if it has ended, its live logs are
        gone; if ``sub_job_id`` is not one of its sub-jobs, it never had any.
        Otherwise (for example, a job still starting) the error stays retryable.
        """
        if not isinstance(exc, requests.exceptions.HTTPError):
            return None
        response = exc.response
        if response is None or response.status_code not in (404, 409, 410):
            return None
        if _response_error_code(response) == _JOB_NOT_FOUND_CODE:
            return None
        if "job" not in checked:
            try:
                # Tried once: the lookup only improves the error.
                checked["job"] = self._send("GET", f"{self._prefix}/{job_id}", max_retries=0).json()
            except Exception:
                logger.debug("job lookup for a tail-logs error failed", exc_info=True)
                checked["job"] = None
        job = checked["job"]
        if not isinstance(job, dict):
            return None
        status = self._normalize_job_status(job.get("status"))
        if status in _ENDED_JOB_STATES:
            checked["ended"] = True
            return (
                f"Job {job_id} is {status}, so its live logs are gone. Download the "
                f"saved console with `cortex-training download-log {job_id} --log-type "
                "stdout` (SDK: download_stdout_logs)"
            )
        known = [
            str(sub.get("sub_job_id") or sub.get("id"))
            for sub in job.get("sub_jobs") or []
            if isinstance(sub, dict) and (sub.get("sub_job_id") or sub.get("id"))
        ]
        if sub_job_id and known and sub_job_id not in known:
            return f"Sub-job {sub_job_id} is not part of job {job_id}; its sub-jobs are: {', '.join(known)}"
        return None

    def _stream_pages(
        self,
        fetch,
        entries_key: str,
        *,
        follow: bool,
        cursor: str | None = None,
        poll_interval: float | None = None,
        stop_at_eof: bool = False,
    ) -> Iterator[dict]:
        """Drive a cursor-paged tail: call ``fetch(cursor)`` for a page, yield
        its entries (under ``entries_key``), advance the cursor, and poll with
        backoff. With ``follow`` False, stop at the first empty page; with
        ``stop_at_eof``, also stop at an empty page that reports ``eof``.
        """
        base_delay = poll_interval or self.poll_interval
        delay = base_delay
        while True:
            page = fetch(cursor)
            cursor = page.get("next_cursor", cursor)
            entries = page.get(entries_key) or []
            for entry in entries:
                yield entry
            if entries:
                delay = base_delay  # made progress; poll again promptly
                continue
            # An empty page means the reader has caught up. Without follow,
            # stop there: older deployments never report eof.
            if not follow or (stop_at_eof and page.get("eof", False)):
                return
            time.sleep(delay)
            delay = min(delay * self.poll_backoff_multiplier, self.poll_max_interval)

    def stream_logs(
        self,
        job_id: str,
        *,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
        cursor: str | None = None,
        max_lines: int | None = None,
        poll_interval: float | None = None,
        follow: bool = True,
    ) -> Iterator[dict]:
        """Yield a sub-job's log entries, polling with backoff.

        Drains the log then, with ``follow=True`` (the default), keeps polling
        for new entries until the caller stops iterating — suitable for a live
        tail. With ``follow=False`` it stops at the first empty page, once it
        has caught up with the log (whether or not the server reports ``eof``).

        The stream also ends, without an error, once the log can grow no more:
        at an empty page that reports ``eof``, or when the job ends after the
        stream has read from it. A job that had already ended before the first
        read raises, as :meth:`tail_logs` does.
        """
        read_once = False

        def fetch(cur: str | None) -> dict:
            nonlocal read_once
            try:
                page = self.tail_logs(
                    job_id,
                    cursor=cur,
                    max_lines=max_lines,
                    sub_job_id=sub_job_id,
                    sub_job_type=sub_job_type,
                )
            except requests.exceptions.HTTPError as exc:
                if read_once and getattr(exc, "job_ended", False):
                    raise _LogEnded() from exc
                raise
            read_once = True
            return page

        def entries() -> Iterator[dict]:
            try:
                yield from self._stream_pages(
                    fetch,
                    "entries",
                    follow=follow,
                    cursor=cursor,
                    poll_interval=poll_interval,
                    stop_at_eof=True,
                )
            except _LogEnded:
                return

        return entries()

    # ─── Zone scheduling events (read-only) ───────────────────────────────

    def tail_events(
        self,
        job_id: str,
        *,
        cursor: str | None = None,
        max_events: int | None = None,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
    ) -> dict:
        """Fetch one page of zone scheduling events for the session.

        Returns ``{"events": [...], "next_cursor": str, "eof": bool}``. Served
        by the zone dispatcher itself, not routed to a Zone Manager.

        Note: the server does not currently accept this operation; see
        ``docs/reference/rest-api.md`` section 7.9.
        """
        payload: dict = {}
        if cursor is not None:
            payload["cursor"] = cursor
        if max_events is not None:
            payload["max_events"] = max_events
        return self._operation(
            job_id,
            "zmd-events",
            payload=payload or None,
            sub_job_id=sub_job_id,
            sub_job_type=sub_job_type,
        )

    def stream_events(
        self,
        job_id: str,
        *,
        cursor: str | None = None,
        max_events: int | None = None,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
        poll_interval: float | None = None,
        follow: bool = True,
    ) -> Iterator[dict]:
        """Yield zone scheduling events for the session, polling with backoff."""
        return self._stream_pages(
            lambda cur: self.tail_events(
                job_id,
                cursor=cur,
                max_events=max_events,
                sub_job_id=sub_job_id,
                sub_job_type=sub_job_type,
            ),
            "events",
            follow=follow,
            cursor=cursor,
            poll_interval=poll_interval,
        )

    # ─── Checkpoints ─────────────────────────────────────────────────────

    @_track_operation("list_checkpoints")
    def list_checkpoints(self, job_id: str) -> list:
        resp = self._send("GET", f"{self._prefix}/{job_id}/checkpoints")
        return resp.json().get("checkpoints", [])

    @_track_operation("export_checkpoint")
    def export_checkpoint(self, job_id: str, checkpoint_id: str) -> dict:
        # The REST API uses colon-action syntax: /{jobId}/checkpoints/{cpId}:export
        resp = self._send(
            "POST",
            f"{self._prefix}/{job_id}/checkpoints/{checkpoint_id}:export",
        )
        return resp.json()

    @_track_operation("delete_checkpoint")
    def delete_checkpoint(self, job_id: str, checkpoint_id: str) -> None:
        """Delete a checkpoint. Returns None on success."""
        self._send("DELETE", f"{self._prefix}/{job_id}/checkpoints/{checkpoint_id}")
