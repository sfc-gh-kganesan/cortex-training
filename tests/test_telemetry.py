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

"""Unit tests for best-effort OTLP client telemetry."""

import math
import threading
from unittest.mock import MagicMock

import pytest
import requests

from cortex_training.telemetry import (
    CachedSessionTokenProvider,
    OtlpMetricEmitter,
    _otlp_value,
    _telemetry_base_url,
)

COLLECTOR = "telemetry.example.snowflakecomputing.com"


def _response(body=None, status_code=200):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = body or {}
    if status_code >= 400:
        response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=response
        )
    return response


def test_otlp_value_bool_is_not_int():
    assert _otlp_value(True) == {"boolValue": True}
    assert _otlp_value(False) == {"boolValue": False}
    assert _otlp_value(1) == {"intValue": "1"}


def test_otlp_value_nested_and_non_finite():
    assert _otlp_value(1.5) == {"doubleValue": 1.5}
    assert _otlp_value(math.nan) == {"stringValue": "nan"}
    assert _otlp_value((1, None, "x")) == {
        "arrayValue": {
            "values": [{"intValue": "1"}, {"stringValue": "x"}],
        }
    }
    assert _otlp_value({"a": 2, "skip": None}) == {
        "kvlistValue": {
            "values": [{"key": "a", "value": {"intValue": "2"}}],
        }
    }


def test_telemetry_base_url_keeps_system_path():
    assert _telemetry_base_url(f"{COLLECTOR}/system") == f"https://{COLLECTOR}/system"
    assert _telemetry_base_url(f"https://{COLLECTOR}") == f"https://{COLLECTOR}"
    with pytest.raises(ValueError, match="Snowflake collector"):
        _telemetry_base_url("evil.example.com/system")


def test_session_token_is_cached_until_refresh(monkeypatch):
    provider = CachedSessionTokenProvider("https://account.test", "pat")
    provider._session = MagicMock()
    provider._session.post.return_value = _response(
        {"token": "session-1", "validityInSeconds": 100}
    )
    now = iter([10.0, 10.0, 20.0])
    monkeypatch.setattr("cortex_training.telemetry.time.monotonic", lambda: next(now))

    assert provider.get_token() == "session-1"
    assert provider.get_token() == "session-1"
    provider._session.post.assert_called_once_with(
        "https://account.test/api/v2/sessions",
        json={},
        timeout=3.0,
    )


def test_session_token_refreshes_after_expiry(monkeypatch):
    provider = CachedSessionTokenProvider(
        "https://account.test",
        "pat",
        refresh_buffer_seconds=0,
    )
    provider._session = MagicMock()
    provider._session.post.side_effect = [
        _response({"token": "session-1", "validityInSeconds": 100}),
        _response({"token": "session-2", "validityInSeconds": 100}),
    ]
    now = iter([0.0, 0.0, 101.0, 101.0])
    monkeypatch.setattr("cortex_training.telemetry.time.monotonic", lambda: next(now))

    assert provider.get_token() == "session-1"
    assert provider.get_token() == "session-2"
    assert provider._session.post.call_count == 2


def test_zero_validity_still_caches_token(monkeypatch):
    provider = CachedSessionTokenProvider(
        "https://account.test",
        "pat",
        refresh_buffer_seconds=0,
    )
    provider._session = MagicMock()
    provider._session.post.return_value = _response(
        {"token": "session-1", "validityInSeconds": 0}
    )
    now = iter([0.0, 0.0, 1.0])
    monkeypatch.setattr("cortex_training.telemetry.time.monotonic", lambda: next(now))

    assert provider.get_token() == "session-1"
    assert provider.get_token() == "session-1"
    provider._session.post.assert_called_once()


def test_emitter_discovers_hostname_once_and_posts_otlp_payload(monkeypatch):
    token_provider = MagicMock()
    token_provider.get_token.return_value = "session"
    emitter = OtlpMetricEmitter(
        "https://account.test",
        token_provider,
        service_version="0.0.2",
        background=False,
    )
    emitter._session = MagicMock()
    emitter._session.get.return_value = _response({"hostname": f"{COLLECTOR}/system"})
    emitter._session.post.return_value = _response()
    monkeypatch.setattr("cortex_training.telemetry.time.time_ns", lambda: 123)

    emitter.emit(
        "generate",
        {"duration_ms": 42, "success": True},
        attributes={
            "operation": "generate",
            "job_id": "job-1",
            "event.name": "should-not-win",
        },
    )
    emitter.emit("generate", 1)

    emitter._session.get.assert_called_once_with(
        "https://account.test/observability/system/hostname",
        headers={"Authorization": 'Snowflake Token="session"'},
        timeout=3.0,
    )
    assert emitter._session.post.call_count == 2
    args, kwargs = emitter._session.post.call_args_list[0]
    assert args[0] == f"https://{COLLECTOR}/system/v1/logs"
    assert kwargs["headers"]["Authorization"] == 'Snowflake Token="session"'
    resource = kwargs["json"]["resourceLogs"][0]
    assert resource["resource"]["attributes"] == [
        {"key": "service.name", "value": {"stringValue": "cortex-training"}},
        {
            "key": "snowflake.account_host",
            "value": {"stringValue": "account.test"},
        },
        {"key": "service.version", "value": {"stringValue": "0.0.2"}},
    ]
    assert resource["scopeLogs"][0]["scope"] == {"name": "cortex-training"}
    record = resource["scopeLogs"][0]["logRecords"][0]
    assert record["timeUnixNano"] == "123"
    assert record["body"]["kvlistValue"]["values"] == [
        {"key": "duration_ms", "value": {"intValue": "42"}},
        {"key": "success", "value": {"boolValue": True}},
    ]
    assert record["attributes"] == [
        {"key": "operation", "value": {"stringValue": "generate"}},
        {"key": "job_id", "value": {"stringValue": "job-1"}},
        {"key": "event.name", "value": {"stringValue": "generate"}},
    ]


def test_emitter_session_does_not_carry_pat():
    provider = CachedSessionTokenProvider("https://account.test", "secret-pat")
    emitter = OtlpMetricEmitter("https://account.test", provider)
    assert "secret-pat" not in str(emitter._session.headers)
    assert "PROGRAMMATIC_ACCESS_TOKEN" not in str(emitter._session.headers)


def test_background_emitter_queues_without_waiting_for_export():
    token_provider = MagicMock()
    emitter = OtlpMetricEmitter("https://account.test", token_provider)
    started = threading.Event()
    release = threading.Event()
    exported = []

    def blocking_export(operation, value, *, attributes, time_unix_nano):
        started.set()
        release.wait(timeout=1)
        exported.append((operation, value, attributes, time_unix_nano))

    emitter._emit_sync = blocking_export
    value = {"duration_ms": 42}
    attributes = {
        "job_id": "job-1",
        "target_sub_job_ids": ["job-1:sampling:0"],
    }

    emitter.emit("generate", value, attributes=attributes)
    assert started.wait(timeout=1)
    value["duration_ms"] = 99
    attributes["job_id"] = "changed"
    attributes["target_sub_job_ids"].append("changed")
    release.set()
    assert emitter.flush(timeout=1)

    assert exported[0][:3] == (
        "generate",
        {"duration_ms": 42},
        {
            "job_id": "job-1",
            "target_sub_job_ids": ["job-1:sampling:0"],
        },
    )
    assert isinstance(exported[0][3], int)
    emitter.close()


def test_dead_worker_is_restarted():
    token_provider = MagicMock()
    emitter = OtlpMetricEmitter("https://account.test", token_provider)
    emitter._worker = threading.Thread()
    exported = []
    emitter._emit_sync = lambda operation, value, **kwargs: exported.append(operation)

    emitter.emit("generate")

    assert emitter.flush(timeout=1)
    assert exported == ["generate"]
    emitter.close()


def test_process_change_resets_inherited_worker_and_queue():
    token_provider = MagicMock()
    emitter = OtlpMetricEmitter(
        "https://account.test",
        token_provider,
        background=False,
    )
    inherited_queue = emitter._queue
    inherited_worker = threading.Thread()
    emitter._worker = inherited_worker
    emitter._pid = -1
    emitter._emit_sync = MagicMock()

    emitter.emit("generate")

    assert emitter._queue is not inherited_queue
    assert emitter._worker is None
    token_provider._ensure_process.assert_called()
    emitter._emit_sync.assert_called_once()
    emitter.close()


def test_full_queue_drops_without_raising_or_leaking_pending_count():
    token_provider = MagicMock()
    emitter = OtlpMetricEmitter(
        "https://account.test",
        token_provider,
        queue_size=1,
    )
    emitter._ensure_worker = MagicMock()

    assert emitter.emit("first") is None
    assert emitter.emit("dropped") is None
    assert emitter._pending == 1
    emitter.close(timeout=0)


def test_emitter_swallows_session_discovery_and_export_errors():
    token_provider = MagicMock()
    token_provider.get_token.side_effect = RuntimeError("auth failed")
    emitter = OtlpMetricEmitter(
        "https://account.test", token_provider, background=False
    )

    assert emitter.emit("event") is None

    token_provider.get_token.side_effect = None
    token_provider.get_token.return_value = "session"
    emitter._session = MagicMock()
    emitter._session.get.return_value = _response({"hostname": f"{COLLECTOR}/system"})
    emitter._session.post.side_effect = RuntimeError("export failed")

    assert emitter.emit("event") is None


def test_auth_rejection_invalidates_cached_token():
    token_provider = MagicMock()
    token_provider.get_token.return_value = "session"
    emitter = OtlpMetricEmitter(
        "https://account.test", token_provider, background=False
    )
    emitter._session = MagicMock()
    emitter._session.get.return_value = _response({"hostname": f"{COLLECTOR}/system"})
    emitter._session.post.return_value = _response(status_code=401)

    assert emitter.emit("event") is None
    token_provider.invalidate.assert_called_once()
    assert emitter._telemetry_base_url is None


def test_consecutive_failures_disable_further_network_calls():
    token_provider = MagicMock()
    token_provider.get_token.side_effect = RuntimeError("auth failed")
    emitter = OtlpMetricEmitter(
        "https://account.test",
        token_provider,
        max_consecutive_failures=3,
        failure_cooldown_seconds=300,
        background=False,
    )

    for _ in range(3):
        emitter.emit("event")
    assert token_provider.get_token.call_count == 3

    emitter.emit("event")
    assert token_provider.get_token.call_count == 3
