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

"""The job list's account-wide list: its fallback to the configured schema, and when it is asked again."""

import json
from datetime import datetime
from datetime import timezone

import pytest
import requests

from cortex_training import AccountJobListUnavailable
from cortex_training.tui.account import ACCOUNT
from cortex_training.tui.account import ACCOUNT_MANUAL_REPROBE_SECONDS
from cortex_training.tui.account import ACCOUNT_REPROBE_SECONDS
from cortex_training.tui.account import FALLBACK
from cortex_training.tui.account import JobLister
from cortex_training.tui.account import JobListError
from cortex_training.tui.account import ListScope

NOW = datetime(2026, 10, 1, tzinfo=timezone.utc)
NOT_DEPLOYED = AccountJobListUnavailable.NOT_DEPLOYED
DISABLED = AccountJobListUnavailable.DISABLED
FORBIDDEN = AccountJobListUnavailable.FORBIDDEN
REST_DISABLED = AccountJobListUnavailable.REST_DISABLED


def _response(status: int, body=None) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response.reason = {403: "Forbidden", 404: "Not Found", 405: "Method Not Allowed"}.get(status, "")
    response._content = json.dumps(body).encode() if body is not None else b""
    return response


def _unavailable(reason: str, body=None) -> AccountJobListUnavailable:
    status = {NOT_DEPLOYED: 404, DISABLED: 404, FORBIDDEN: 403, REST_DISABLED: 405}[reason]
    return AccountJobListUnavailable(f"the account-level job list is unavailable ({reason})", reason=reason,
                                     response=_response(status, body))


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


class _Client:
    """``list_jobs_page`` raises ``account_error`` or answers one page; ``list_jobs`` raises ``schema_error``
    or lists the configured schema. Each counts its calls."""

    def __init__(self, account_error=None, schema_error=None) -> None:
        self.account_error, self.schema_error = account_error, schema_error
        self.account_asks = self.schema_asks = 0

    def list_jobs_page(self, **filters) -> dict:
        self.account_asks += 1
        if self.account_error is not None:
            raise self.account_error
        return {"jobs": [{"job_id": "account-job", "database_name": "DB", "schema_name": "S"}], "next_page_token": None}

    def list_jobs(self, status=None) -> list:  # the real client's signature: bounds passed here fail
        self.schema_asks += 1
        if self.schema_error is not None:
            raise self.schema_error
        return [{"job_id": "schema-job"}, "not a job"]


def test_an_unavailable_account_list_falls_back_and_is_asked_again_only_when_due():
    clock, client = _Clock(), _Client(account_error=_unavailable(NOT_DEPLOYED))
    lister = JobLister(client, account_wide=True, clock=clock)
    listing = lister.load(NOW)
    assert listing.scope == ListScope(FALLBACK, reason=NOT_DEPLOYED) and listing.jobs == [{"job_id": "schema-job"}]
    assert lister._fallback == NOT_DEPLOYED and client.account_asks == 1

    clock.now += ACCOUNT_MANUAL_REPROBE_SECONDS / 2
    lister.load(NOW, manual=True)  # r, too soon after the last ask
    clock.now += ACCOUNT_MANUAL_REPROBE_SECONDS / 2
    lister.load(NOW)  # a timed refresh waits longer than r does
    assert client.account_asks == 1
    lister.load(NOW, manual=True)
    assert client.account_asks == 2  # still unavailable: that ask restarts the wait

    clock.now += ACCOUNT_REPROBE_SECONDS
    client.account_error = None
    listing = lister.load(NOW)
    assert client.account_asks == 3 and listing.scope == ListScope(ACCOUNT) and lister._fallback is None


def test_asking_again_while_falling_back_never_raises_and_restarts_the_wait():
    clock, client = _Clock(), _Client(account_error=_unavailable(DISABLED))
    lister = JobLister(client, account_wide=True, clock=clock)
    lister.load(NOW)
    clock.now += ACCOUNT_REPROBE_SECONDS
    client.account_error = requests.exceptions.ConnectionError("connection reset")
    assert lister.load(NOW).scope == ListScope(FALLBACK, reason=DISABLED)  # the earlier reason stands
    clock.now += ACCOUNT_REPROBE_SECONDS / 2
    lister.load(NOW)
    assert client.account_asks == 2
    clock.now += ACCOUNT_REPROBE_SECONDS
    client.account_error = _unavailable(REST_DISABLED)  # refused now: the list says so, still without an error
    assert lister.load(NOW).scope == ListScope(FALLBACK, reason=REST_DISABLED) and client.account_asks == 3


def test_a_asks_again_once_the_last_answer_is_a_minute_old_and_then_raises_what_it_gets():
    clock, client = _Clock(), _Client(account_error=_unavailable(NOT_DEPLOYED))
    lister = JobLister(client, account_wide=True, clock=clock)
    lister.load(NOW)
    assert lister.load(NOW, manual=True, asked=True).scope.reason == NOT_DEPLOYED  # the answer it just had
    assert client.account_asks == 1
    clock.now += ACCOUNT_MANUAL_REPROBE_SECONDS
    client.account_error = requests.exceptions.ConnectionError("connection reset")
    with pytest.raises(JobListError) as failed:
        lister.load(NOW, manual=True, asked=True)
    assert not failed.value.refused and str(failed.value).startswith("the account-wide job list failed:")


@pytest.mark.parametrize("refusal", [_unavailable(REST_DISABLED), _unavailable(FORBIDDEN, {"message": "not enabled"})])
def test_a_refusal_from_a_gate_the_schema_list_shares_is_the_lists_error(refusal):
    client = _Client(account_error=refusal)
    with pytest.raises(JobListError) as refused:
        JobLister(client, account_wide=True).load(NOW)
    assert refused.value.refused and client.schema_asks == 0


def test_a_403_without_the_services_message_asks_the_configured_schema_once():
    client = _Client(account_error=_unavailable(FORBIDDEN))  # a proxy or a network policy may have sent it
    lister = JobLister(client, account_wide=True)
    assert lister.load(NOW).scope == ListScope(FALLBACK, reason=FORBIDDEN) and client.schema_asks == 1
    assert lister._fallback == FORBIDDEN

    refused = requests.exceptions.HTTPError("403 Client Error", response=_response(403))
    client = _Client(account_error=_unavailable(FORBIDDEN), schema_error=refused)
    with pytest.raises(JobListError) as failed:
        JobLister(client, account_wide=True).load(NOW)
    assert not failed.value.refused and str(failed.value).endswith("; and list_jobs answered HTTP 403 Forbidden")

