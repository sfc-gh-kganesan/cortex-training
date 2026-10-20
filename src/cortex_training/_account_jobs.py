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

"""The account-level job list's answers, read one page at a time.

:meth:`CortexTrainingClient.list_jobs_page` sends the request; this module
holds what does not depend on the client: the errors that say why the list
can't be read, how a failed answer maps to them, and the page's shape. Error
bodies arrive here as the ``message`` strings the client found in them.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import Any

import requests

# The service's message when the route is deployed but not enabled for the
# account. Any other 404 means the route is missing; the service sends those
# with an empty body.
_NOT_ENABLED_MESSAGE = "account-level job listing is not enabled for this account"
_INVALID_PAGE_TOKEN_MESSAGE = "invalid page_token"
# How much of the service's error message an exception's own message quotes.
_SERVICE_MESSAGE_LIMIT = 300


class AccountJobListUnavailable(requests.exceptions.HTTPError):
    """The account-level job list is not available to this caller.

    Raised by :meth:`CortexTrainingClient.list_jobs_page` at once, without
    retries. ``reason`` is one of:

    - ``"not_deployed"``: 404 without the service's not-enabled error, normally
      with an empty body: the service does not have the route yet. With some
      other body, a proxy in front of the service may have answered instead.
    - ``"disabled"``: 404 whose JSON body says account-level job listing is not
      enabled for this account.
    - ``"forbidden"``: 403. When the answer carries the service's reason, the
      message quotes it. A 403 without one can also come from a network policy
      or a proxy in front of the service.
    - ``"rest_disabled"``: 405. The Cortex Training REST API is turned off for
      the account.

    ``not_deployed`` and ``disabled`` concern this route only, so
    :meth:`CortexTrainingClient.list_jobs` still lists the configured schema.
    ``forbidden`` and ``rest_disabled`` mostly come from gates the per-schema
    routes share. A subclass of ``requests.HTTPError``; ``response`` is the
    service's answer, and ``str()`` is one line.
    """

    NOT_DEPLOYED = "not_deployed"
    DISABLED = "disabled"
    FORBIDDEN = "forbidden"
    REST_DISABLED = "rest_disabled"
    REASONS = frozenset({NOT_DEPLOYED, DISABLED, FORBIDDEN, REST_DISABLED})

    def __init__(
        self,
        message: str,
        *,
        reason: str,
        response: requests.Response | None = None,
        request: requests.Request | requests.PreparedRequest | None = None,
    ):
        super().__init__(message, response=response, request=request)
        self.reason = reason

    def __reduce__(self):
        # copy and pickle rebuild an exception from its args, which leave out
        # the keyword-only reason.
        return (functools.partial(type(self), reason=self.reason), self.args, self.__dict__)


class InvalidPageToken(requests.exceptions.HTTPError):
    """The service could not open a ``page_token``: 400 ``Invalid page_token``.

    It happens, for example, after the service rotates the key that seals its
    tokens. Start the walk again from the first page.
    """


def page_error(
    response: requests.Response,
    request: requests.Request | requests.PreparedRequest | None,
    *,
    messages: Sequence[str],
    page_token_sent: bool,
) -> requests.exceptions.HTTPError | None:
    """The typed error for a failed page, or None when the answer is not one of them.

    ``messages`` are the ``message`` strings found in the answer's JSON body.
    """
    reason = _unavailable_reason(response.status_code, messages)
    if reason is not None:
        return AccountJobListUnavailable(
            _unavailable_message(reason, response, messages),
            reason=reason,
            response=response,
            request=request,
        )
    if page_token_sent and response.status_code == 400 and _says_invalid_page_token(messages):
        return InvalidPageToken(
            "the service could not open page_token (HTTP 400 Invalid page_token); "
            "start again from the first page",
            response=response,
            request=request,
        )
    return None


def parse_page(body: Any) -> dict:
    """``{"jobs": [...], "next_page_token": str | None}`` from a page's JSON body."""
    if not isinstance(body, dict):
        raise ValueError("the account-level job list answered with a non-object JSON body")
    jobs = body.get("jobs")
    if jobs is None:
        jobs = []
    if not isinstance(jobs, list):
        raise ValueError("the account-level job list answered with a non-list 'jobs'")
    token = body.get("next_page_token")
    return {"jobs": jobs, "next_page_token": token if isinstance(token, str) and token else None}


def _unavailable_reason(status_code: int | None, messages: Sequence[str]) -> str | None:
    if status_code == 405:
        return AccountJobListUnavailable.REST_DISABLED
    if status_code == 403:
        return AccountJobListUnavailable.FORBIDDEN
    if status_code != 404:
        return None
    if any(_NOT_ENABLED_MESSAGE in message.lower() for message in messages):
        return AccountJobListUnavailable.DISABLED
    return AccountJobListUnavailable.NOT_DEPLOYED


def _unavailable_message(reason: str, response: requests.Response, messages: Sequence[str]) -> str:
    """Why the account-level list is unavailable, claiming no more than the answer shows."""
    if reason == AccountJobListUnavailable.NOT_DEPLOYED:
        if not _has_body(response):
            return (
                "account-level job listing is not available on this service yet: "
                "the account-level route answered 404 with no error body, so it is "
                "not deployed"
            )
        # A body without the service's error: often a proxy's or gateway's page,
        # which says nothing about whether the route is deployed.
        return (
            "account-level job listing is not available: the account-level route "
            "answered 404 without the service's not-enabled error; it may not be "
            "deployed yet, or a proxy in front of the service answered"
        )
    if reason == AccountJobListUnavailable.DISABLED:
        return (
            "account-level job listing is not enabled for this account (HTTP 404); "
            "Snowflake enables it per account"
        )
    if reason == AccountJobListUnavailable.FORBIDDEN:
        message = _service_message(messages)
        if message is not None:
            return f"account-level job listing was refused (HTTP 403): {message}"
        return (
            "account-level job listing was refused (HTTP 403) with no error "
            "message: a network policy or a proxy may have blocked the request, "
            "or Cortex Training is not enabled for this account or its trial has "
            "ended"
        )
    return (
        "account-level job listing is unavailable: the Cortex Training REST API "
        "is turned off for this account (HTTP 405)"
    )


def _says_invalid_page_token(messages: Sequence[str]) -> bool:
    # A prefix, should the service add detail. Not "page_token does not match
    # the request filters": that one is a caller sending different filters,
    # which a restart would only hide.
    return any(
        message.strip().lower().startswith(_INVALID_PAGE_TOKEN_MESSAGE) for message in messages
    )


def _service_message(messages: Sequence[str]) -> str | None:
    """The service's first error message, on one bounded line."""
    for message in messages:
        if message.strip():
            message = " ".join(message.split())
            if len(message) > _SERVICE_MESSAGE_LIMIT:
                message = message[: _SERVICE_MESSAGE_LIMIT - 3] + "..."
            return message
    return None


def _has_body(response: requests.Response) -> bool:
    try:
        return bool((response.content or b"").strip())
    except Exception:
        return True
