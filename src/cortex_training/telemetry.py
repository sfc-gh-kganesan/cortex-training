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

"""Best-effort client telemetry over Snowflake's OTLP endpoint.

Telemetry is deliberately isolated from the Cortex Training request session
because the two paths use different authentication schemes:

* Cortex Training APIs accept the PAT directly.
* Observability APIs require a short-lived Snowflake session token.

All metric emission failures are swallowed. Telemetry must never change the
outcome of a caller's training or inference operation.
"""

from __future__ import annotations

import atexit
import logging
import math
import os
import queue
import threading
import time
import weakref
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

_PAT_TOKEN_TYPE = "PROGRAMMATIC_ACCESS_TOKEN"
_DEFAULT_SESSION_VALIDITY_SECONDS = 3600.0
_MIN_SESSION_VALIDITY_SECONDS = 60.0
_DEFAULT_TIMEOUT_SECONDS = 3.0
_FAILURE_COOLDOWN_SECONDS = 300.0
_MAX_CONSECUTIVE_FAILURES = 3
_SNOWFLAKE_HOST_SUFFIX = ".snowflakecomputing.com"
_AUTH_REJECTED_STATUSES = {401, 403}
_DEFAULT_QUEUE_SIZE = 1024
_WORKER_IDLE_SECONDS = 30.0
_STOP = object()
_EMITTERS: weakref.WeakSet["OtlpMetricEmitter"] = weakref.WeakSet()


def _flush_emitters_at_exit() -> None:
    deadline = time.monotonic() + 1.0
    for emitter in list(_EMITTERS):
        remaining = max(deadline - time.monotonic(), 0.0)
        emitter.close(timeout=remaining)


atexit.register(_flush_emitters_at_exit)


def _snapshot(value: Any) -> Any:
    """Copy JSON-like values before a background worker observes them."""
    if isinstance(value, Mapping):
        return {str(key): _snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_snapshot(item) for item in value]
    return value


def _otlp_value(value: Any) -> dict[str, Any]:
    """Convert a JSON-like Python value to an OTLP ``AnyValue``."""
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return {"stringValue": str(value)}
        return {"doubleValue": value}
    if isinstance(value, Mapping):
        return {
            "kvlistValue": {
                "values": [
                    {"key": str(key), "value": _otlp_value(item)}
                    for key, item in value.items()
                    if item is not None
                ]
            }
        }
    if isinstance(value, (list, tuple)):
        return {
            "arrayValue": {
                "values": [_otlp_value(item) for item in value if item is not None]
            }
        }
    return {"stringValue": str(value)}


def _telemetry_base_url(hostname: str) -> str:
    """Build an OTLP base URL from the observability hostname response.

    GS returns a host, optionally with a ``/system`` path. Posts go to
    ``https://{hostname}/v1/logs`` without stripping that path, so a value of
    ``telemetry.example.snowflakecomputing.com/system`` becomes
    ``https://telemetry.example.snowflakecomputing.com/system``.
    """
    hostname = hostname.strip().rstrip("/")
    if not hostname:
        raise ValueError("observability response did not contain a hostname")
    if not hostname.startswith(("http://", "https://")):
        hostname = f"https://{hostname}"
    parsed = urlparse(hostname)
    host = parsed.hostname or ""
    if not host.endswith(_SNOWFLAKE_HOST_SUFFIX):
        raise ValueError("observability hostname is not a Snowflake collector")
    return hostname.rstrip("/")


class CachedSessionTokenProvider:
    """Exchange a PAT for a Snowflake session token and cache it until expiry."""

    def __init__(
        self,
        base_url: str,
        pat: str,
        *,
        verify_ssl: bool = True,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        refresh_buffer_seconds: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.refresh_buffer_seconds = refresh_buffer_seconds
        self._pat = pat
        self._verify_ssl = verify_ssl
        self._pid = os.getpid()
        self._session = self._new_session()
        self._token: str | None = None
        self._expires_at = 0.0
        self._lock = threading.Lock()

    def _new_session(self) -> requests.Session:
        session = requests.Session()
        session.verify = self._verify_ssl
        session.headers.update(
            {
                "Authorization": f"Bearer {self._pat}",
                "X-Snowflake-Authorization-Token-Type": _PAT_TOKEN_TYPE,
                "Accept": "application/json",
            }
        )
        return session

    def _ensure_process(self) -> None:
        if self._pid == os.getpid():
            return
        self._pid = os.getpid()
        self._session = self._new_session()
        self._token = None
        self._expires_at = 0.0
        self._lock = threading.Lock()

    def invalidate(self) -> None:
        """Drop the cached token so the next call re-exchanges the PAT."""
        self._ensure_process()
        with self._lock:
            self._token = None
            self._expires_at = 0.0

    def get_token(self) -> str:
        """Return a cached session token, refreshing it shortly before expiry."""
        self._ensure_process()
        now = time.monotonic()
        if self._token is not None and now < self._expires_at:
            return self._token

        with self._lock:
            now = time.monotonic()
            if self._token is not None and now < self._expires_at:
                return self._token

            response = self._session.post(
                f"{self.base_url}/api/v2/sessions",
                json={},
                timeout=self.timeout,
            )
            response.raise_for_status()
            body = response.json()
            token = body["token"]
            if not isinstance(token, str) or not token:
                raise ValueError("session response did not contain a token")

            validity = body.get("validityInSeconds", _DEFAULT_SESSION_VALIDITY_SECONDS)
            try:
                validity_seconds = float(validity)
            except (TypeError, ValueError):
                validity_seconds = _DEFAULT_SESSION_VALIDITY_SECONDS
            validity_seconds = max(validity_seconds, _MIN_SESSION_VALIDITY_SECONDS)
            refresh_buffer = min(
                self.refresh_buffer_seconds,
                validity_seconds * 0.1,
            )
            self._token = token
            self._expires_at = now + validity_seconds - refresh_buffer
            return token

    def close(self) -> None:
        self._session.close()


class OtlpMetricEmitter:
    """Emit OTLP log records as best-effort client-side metrics."""

    def __init__(
        self,
        base_url: str,
        token_provider: CachedSessionTokenProvider,
        *,
        service_name: str = "cortex-training",
        service_version: str | None = None,
        verify_ssl: bool = True,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        max_consecutive_failures: int = _MAX_CONSECUTIVE_FAILURES,
        failure_cooldown_seconds: float = _FAILURE_COOLDOWN_SECONDS,
        background: bool = True,
        queue_size: int = _DEFAULT_QUEUE_SIZE,
    ):
        self.base_url = base_url.rstrip("/")
        self.account_host = urlparse(self.base_url).hostname
        if not self.account_host:
            raise ValueError("base_url must contain an account hostname")
        self.token_provider = token_provider
        self.service_name = service_name
        self.service_version = service_version
        self.timeout = timeout
        self._max_consecutive_failures = max_consecutive_failures
        self._failure_cooldown_seconds = failure_cooldown_seconds
        self._verify_ssl = verify_ssl
        self._queue_size = queue_size
        self._pid = os.getpid()
        self._session = self._new_session()
        self._telemetry_base_url: str | None = None
        self._hostname_lock = threading.Lock()
        self._consecutive_failures = 0
        self._disabled_until = 0.0
        self._background = background
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=queue_size)
        self._worker: threading.Thread | None = None
        self._worker_lock = threading.Lock()
        self._pending_condition = threading.Condition()
        self._pending = 0
        self._closed = False
        _EMITTERS.add(self)

    def _new_session(self) -> requests.Session:
        session = requests.Session()
        session.verify = self._verify_ssl
        return session

    def _ensure_process(self) -> None:
        if self._pid == os.getpid():
            return
        self._pid = os.getpid()
        self._session = self._new_session()
        self._telemetry_base_url = None
        self._hostname_lock = threading.Lock()
        self._consecutive_failures = 0
        self._disabled_until = 0.0
        self._queue = queue.Queue(maxsize=self._queue_size)
        self._worker = None
        self._worker_lock = threading.Lock()
        self._pending_condition = threading.Condition()
        self._pending = 0
        self.token_provider._ensure_process()

    def _is_disabled(self) -> bool:
        return time.monotonic() < self._disabled_until

    def _note_success(self) -> None:
        self._consecutive_failures = 0

    def _note_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._max_consecutive_failures:
            self._disabled_until = time.monotonic() + self._failure_cooldown_seconds

    def _invalidate_auth(self) -> None:
        self.token_provider.invalidate()
        with self._hostname_lock:
            self._telemetry_base_url = None

    def _get_telemetry_base_url(self, token: str) -> str:
        if self._telemetry_base_url is not None:
            return self._telemetry_base_url

        with self._hostname_lock:
            if self._telemetry_base_url is not None:
                return self._telemetry_base_url
            response = self._session.get(
                f"{self.base_url}/observability/system/hostname",
                headers={"Authorization": f'Snowflake Token="{token}"'},
                timeout=self.timeout,
            )
            response.raise_for_status()
            hostname = response.json()["hostname"]
            if not isinstance(hostname, str) or not hostname:
                raise ValueError("observability response did not contain a hostname")
            self._telemetry_base_url = _telemetry_base_url(hostname)
            return self._telemetry_base_url

    def _ensure_worker(self) -> None:
        worker = self._worker
        if worker is not None and worker.is_alive():
            return
        with self._worker_lock:
            worker = self._worker
            if worker is not None and worker.is_alive():
                return
            self._worker = None
            worker = threading.Thread(
                target=self._run,
                name="cortex-training-telemetry",
                daemon=True,
            )
            worker.start()
            self._worker = worker

    def _run(self) -> None:
        try:
            while True:
                try:
                    item = self._queue.get(timeout=_WORKER_IDLE_SECONDS)
                except queue.Empty:
                    return
                if item is _STOP:
                    self._queue.task_done()
                    return
                operation, value, attributes, time_unix_nano = item
                try:
                    self._emit_sync(
                        operation,
                        value,
                        attributes=attributes,
                        time_unix_nano=time_unix_nano,
                    )
                finally:
                    self._queue.task_done()
                    with self._pending_condition:
                        self._pending -= 1
                        self._pending_condition.notify_all()
        finally:
            with self._worker_lock:
                if self._worker is threading.current_thread():
                    self._worker = None
            if not self._closed and not self._queue.empty():
                self._ensure_worker()

    def flush(self, timeout: float = 1.0) -> bool:
        """Wait briefly for queued metrics, primarily during interpreter exit."""
        deadline = time.monotonic() + timeout
        with self._pending_condition:
            while self._pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._pending_condition.wait(remaining)
            return True

    def close(self, timeout: float = 1.0) -> None:
        """Flush queued events and release this emitter's worker and sessions."""
        self._ensure_process()
        worker = self._worker
        if self._closed and (worker is None or not worker.is_alive()):
            self._session.close()
            self.token_provider.close()
            _EMITTERS.discard(self)
            return
        self._closed = True
        deadline = time.monotonic() + timeout
        self.flush(timeout=max(deadline - time.monotonic(), 0.0))
        worker = self._worker
        if worker is not None and worker.is_alive():
            try:
                self._queue.put_nowait(_STOP)
            except queue.Full:
                pass
            worker.join(timeout=max(deadline - time.monotonic(), 0.0))
        if worker is None or not worker.is_alive():
            self._session.close()
            self.token_provider.close()
            _EMITTERS.discard(self)

    def emit(
        self,
        operation: str,
        value: Any = 1,
        *,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        """Queue one operation metric without blocking or raising into the caller."""
        try:
            self._ensure_process()
            if self._closed:
                return
            time_unix_nano = time.time_ns()
            if not self._background:
                self._emit_sync(
                    operation,
                    value,
                    attributes=attributes,
                    time_unix_nano=time_unix_nano,
                )
                return
            item = (
                operation,
                _snapshot(value),
                _snapshot(attributes) if attributes is not None else None,
                time_unix_nano,
            )
            with self._pending_condition:
                self._pending += 1
            try:
                self._queue.put_nowait(item)
            except Exception:
                with self._pending_condition:
                    self._pending -= 1
                    self._pending_condition.notify_all()
                raise
            self._ensure_worker()
        except Exception:
            logger.debug("client telemetry enqueue failed", exc_info=True)

    def _emit_sync(
        self,
        operation: str,
        value: Any = 1,
        *,
        attributes: Mapping[str, Any] | None = None,
        time_unix_nano: int | None = None,
    ) -> None:
        """Emit one operation metric, swallowing setup and export failures."""
        self._ensure_process()
        if self._is_disabled():
            return
        try:
            token = self.token_provider.get_token()
            telemetry_base_url = self._get_telemetry_base_url(token)
            record_attributes = {
                **dict(attributes or {}),
                "event.name": operation,
            }
            resource_attributes = [
                {
                    "key": "service.name",
                    "value": {"stringValue": self.service_name},
                },
                {
                    "key": "snowflake.account_host",
                    "value": {"stringValue": self.account_host},
                },
            ]
            if self.service_version:
                resource_attributes.append(
                    {
                        "key": "service.version",
                        "value": {"stringValue": self.service_version},
                    }
                )
            response = self._session.post(
                f"{telemetry_base_url}/v1/logs",
                headers={
                    "Authorization": f'Snowflake Token="{token}"',
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                json={
                    "resourceLogs": [
                        {
                            "resource": {"attributes": resource_attributes},
                            "scopeLogs": [
                                {
                                    "scope": {"name": self.service_name},
                                    "logRecords": [
                                        {
                                            "timeUnixNano": str(
                                                time.time_ns()
                                                if time_unix_nano is None
                                                else time_unix_nano
                                            ),
                                            "severityText": "INFO",
                                            "body": _otlp_value(value),
                                            "attributes": [
                                                {
                                                    "key": str(key),
                                                    "value": _otlp_value(item),
                                                }
                                                for key, item in record_attributes.items()
                                                if item is not None
                                            ],
                                        }
                                    ],
                                }
                            ],
                        }
                    ]
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            self._note_success()
        except Exception as exc:
            logger.debug("client telemetry emit failed", exc_info=True)
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None)
            if status in _AUTH_REJECTED_STATUSES:
                self._invalidate_auth()
            self._note_failure()
