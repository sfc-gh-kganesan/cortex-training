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

"""``CortexTrainingClient.list_jobs_page``: one page of the account-level job
list, its page token, and the errors that say why the list is unavailable."""

from __future__ import annotations

import copy
import json
import pickle
from datetime import date
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from urllib.parse import parse_qsl
from urllib.parse import urlsplit

import pytest
import requests
from requests.adapters import BaseAdapter
from tenacity import wait_none

import cortex_training.client as client_module
from cortex_training import AccountJobListUnavailable
from cortex_training import CortexTrainingClient
from cortex_training import InvalidPageToken

NOT_ENABLED = "Account-level job listing is not enabled for this account"


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    monkeypatch.setattr(client_module, "wait_exponential_jitter", lambda **_kwargs: wait_none())


class _Service(BaseAdapter):
    """Answers the n-th request with the n-th ``(status, body)`` (the last one
    from then on), the body JSON, raw bytes or None; records what was sent."""

    def __init__(self, answers):
        super().__init__()
        self.answers, self.urls, self.timeouts = answers, [], []

    @property
    def queries(self) -> list[dict[str, str]]:
        return [dict(parse_qsl(urlsplit(url).query)) for url in self.urls]

    def send(self, request, timeout=None, **_kwargs):
        self.urls.append(request.url)
        self.timeouts.append(timeout)
        status, body = self.answers[min(len(self.urls), len(self.answers)) - 1]
        resp = requests.Response()
        resp.status_code, resp.request, resp.url, resp.encoding = status, request, request.url, "utf-8"
        resp._content = body if isinstance(body, bytes) else b"" if body is None else json.dumps(body).encode()
        return resp

    def close(self):
        pass


def _client(*answers, **kwargs) -> tuple[CortexTrainingClient, _Service]:
    service = _Service(list(answers))
    client = CortexTrainingClient(base_url="http://test.local", database="DB", schema="SCH", **kwargs)
    client._session.mount("http://", service)
    return client, service


def _page(*job_ids, token=None):
    return 200, {"jobs": [{"job_id": job_id} for job_id in job_ids], "next_page_token": token}


# --- One page, and its token ---


def test_list_jobs_page_reads_one_page_and_takes_its_token_back():
    client, service = _client(_page("b", token="t+/="), (200, {"next_page_token": ""}))

    first = client.list_jobs_page(status="active", page_size=1)
    last = client.list_jobs_page(status="active", page_size=1, page_token=first["next_page_token"])

    assert first == {"jobs": [{"job_id": "b"}], "next_page_token": "t+/="}
    assert last == {"jobs": [], "next_page_token": None}
    # The route iter_jobs reads; the token goes back as it came.
    assert {urlsplit(url).path for url in service.urls} == {"/api/v2/cortex-training"}
    assert service.queries == [
        {"status": "active", "limit": "1"},
        {"status": "active", "limit": "1", "page_token": "t+/="},
    ]
    assert service.timeouts == [client.request_timeout] * 2  # the client's own, unless bounded


def test_list_jobs_page_sends_the_filters_as_iter_jobs_does():
    client, service = _client(_page(), _page())
    filters = dict(
        database="DB",
        schema='"Exp 1"',
        status=["running", "FAILED"],
        submitted_by="jdoe",
        created_after=datetime(2026, 10, 1, 1, 0, tzinfo=timezone(timedelta(hours=-7))),
        created_before=date(2026, 10, 2),
        page_size=50,
    )
    client.list_jobs_page(**filters)
    list(client.iter_jobs(**filters))
    assert service.queries[0] == {
        "database": "DB",
        "schema": '"Exp 1"',
        "status": "running,FAILED",
        "submitted_by": "jdoe",
        "created_after": "2026-10-01T08:00:00Z",
        "created_before": "2026-10-02",
        "limit": "50",
    }
    assert service.queries[1] == service.queries[0]


@pytest.mark.parametrize("body", [[{"job_id": "b"}], {"jobs": {"job_id": "b"}}])
def test_a_page_that_is_not_a_job_list_is_refused(body):
    client, _ = _client((200, body))
    with pytest.raises(ValueError, match="the account-level job list answered with a non-"):
        client.list_jobs_page()


@pytest.mark.parametrize(
    ("kwargs", "words"),
    [
        ({"page_size": 0}, "page_size"),
        ({"page_size": 1001}, "page_size"),
        ({"schema": "SCH"}, "list_jobs_page\\(schema=...\\) needs database"),
        ({"max_retries": -1}, "max_retries"),
        ({"max_retries": True}, "max_retries"),
        ({"request_timeout": 0}, "request_timeout"),
    ],
)
def test_bad_arguments_are_refused_before_any_request(kwargs, words):
    client, service = _client(_page())
    with pytest.raises(ValueError, match=words):
        client.list_jobs_page(**kwargs)
    assert service.urls == []


# --- An unavailable list, and a page token the service can't open ---


@pytest.mark.parametrize(
    ("status", "body", "reason", "words"),
    [
        (404, None, "not_deployed", "answered 404 with no error body"),
        (404, b"<html>Not Found</html>", "not_deployed", "or a proxy in front of the service answered"),
        (404, {"code": "000404", "message": NOT_ENABLED}, "disabled", "not enabled for this account (HTTP 404)"),
        # The service's reason, quoted on one line and cut short.
        (
            403,
            {"message": "refused\nbecause " + "x" * 1000},
            "forbidden",
            "(HTTP 403): refused because " + "x" * 281 + "...",
        ),
        (403, None, "forbidden", "(HTTP 403) with no error message"),
        (405, None, "rest_disabled", "REST API is turned off for this account (HTTP 405)"),
    ],
)
def test_an_unavailable_list_raises_why_at_once(status, body, reason, words):
    client, service = _client((status, body))
    with pytest.raises(AccountJobListUnavailable) as raised:
        client.list_jobs_page()
    assert (raised.value.reason, raised.value.response.status_code) == (reason, status)
    assert words in str(raised.value) and "\n" not in str(raised.value)
    assert len(service.urls) == 1  # not retried


def test_the_unavailable_error_is_an_http_error_and_survives_copy_and_pickle():
    client, _ = _client((404, {"message": NOT_ENABLED}))
    with pytest.raises(requests.HTTPError) as raised:  # what callers catch for HTTP errors still catches it
        client.list_jobs_page()
    error = raised.value
    for copied in (copy.copy(error), copy.deepcopy(error), pickle.loads(pickle.dumps(error))):
        assert (type(copied), copied.reason, str(copied)) == (AccountJobListUnavailable, "disabled", str(error))


def test_iter_jobs_keeps_its_own_404_error():
    client, _ = _client((404, {"message": NOT_ENABLED}))
    with pytest.raises(requests.HTTPError) as raised:
        list(client.iter_jobs())
    assert type(raised.value) is requests.HTTPError
    assert str(raised.value) == "account-level job listing is not enabled for this account"


@pytest.mark.parametrize(
    ("message", "invalid"),
    [
        ("Invalid page_token", True),
        ("Invalid page_token: sealed with a retired key", True),
        # A token sent with other filters is the caller's to fix: starting again would hide it.
        ("page_token does not match the request filters", False),
    ],
)
def test_an_invalid_page_token_is_raised_for_the_caller_to_start_again(message, invalid):
    client, service = _client((400, {"message": message}))
    with pytest.raises(requests.HTTPError) as raised:
        client.list_jobs_page(page_token="opaque")
    assert isinstance(raised.value, InvalidPageToken) is invalid
    assert len(service.urls) == 1


def test_a_400_without_a_page_token_is_not_an_invalid_page_token():
    client, _ = _client((400, {"message": "Invalid page_token"}))
    with pytest.raises(requests.HTTPError) as raised:
        client.list_jobs_page()
    assert not isinstance(raised.value, InvalidPageToken)


# --- Per-call bounds ---


def test_max_retries_and_request_timeout_bound_one_page():
    client, service = _client((503, None))
    with pytest.raises(requests.HTTPError):
        client.list_jobs_page(max_retries=2, request_timeout=(5, 45))
    assert service.timeouts == [(5.0, 45.0)] * 3  # one attempt and two retries, not eleven
    assert (client.max_retries, client.request_timeout) == (10, (30.0, 600.0))


def test_the_page_never_creates_the_database():
    client, service = _client((400, {"code": "002003", "message": "Database 'DB' does not exist or not authorized."}))
    with pytest.raises(requests.HTTPError):
        client.list_jobs_page()
    assert len(service.urls) == 1
