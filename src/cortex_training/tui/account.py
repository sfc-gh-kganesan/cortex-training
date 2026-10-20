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

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from typing import Any
from typing import Callable
from typing import Optional

from cortex_training import AccountJobListUnavailable
from cortex_training import InvalidPageToken
from cortex_training.tui import views

logger = logging.getLogger(__name__)

# Account-list bounds for one refresh.
ACCOUNT_LIST_CAP = 2_000
ACCOUNT_PAGE_SIZE = 500
ACCOUNT_LIST_MAX_PAGES = 20
# Fallback reprobe cadence (manual refreshes may ask sooner).
ACCOUNT_REPROBE_SECONDS = 600
ACCOUNT_MANUAL_REPROBE_SECONDS = 60
_ASK_SLACK_SECONDS = 5.0
ACTIVE = "active"

ACCOUNT = "account"  # the account-level list answered
SCHEMA = "schema"  # the configured schema's list: the default, while the account-level one isn't asked for
FALLBACK = "fallback"  # the account-level list was asked for and isn't there: the configured schema's list

CAPPED = "cap"  # a walk read ACCOUNT_LIST_CAP jobs, and more exist
PAGE_LIMIT = "pages"  # a walk read ACCOUNT_LIST_MAX_PAGES pages, and the last one carried a token

# These failures affect only the account route, so schema listing can continue.
_FALLBACK_REASONS = frozenset({AccountJobListUnavailable.NOT_DEPLOYED, AccountJobListUnavailable.DISABLED})
_FALLBACK_WHY = {
    AccountJobListUnavailable.NOT_DEPLOYED: "no account-wide list yet",
    AccountJobListUnavailable.DISABLED: "account-wide list not enabled",
    AccountJobListUnavailable.FORBIDDEN: "account-wide list: HTTP 403",
    AccountJobListUnavailable.REST_DISABLED: "account-wide list: HTTP 405",
}
_FAILURE_LIMIT = 300  # cells of a failure's one-line text
_ACCOUNT_ROUTE, _SCHEMA_ROUTE = "the account-wide job list", "list_jobs"


class JobListError(Exception):
    """A one-line list failure; ``refused`` means the schema was not attempted."""

    def __init__(self, text: str, *, refused: bool = False):
        super().__init__(text)
        self.refused = refused


@dataclass(frozen=True)
class ListScope:
    """What one listing covers."""

    kind: str  # ACCOUNT, SCHEMA or FALLBACK
    stopped: Optional[str] = None
    walked: int = 0  # ACCOUNT, stopped: the jobs that walk read, the newest
    pages: int = 0  # ACCOUNT, stopped at PAGE_LIMIT: the pages it read
    active_missing: bool = False  # ACCOUNT, stopped: the active-job walk failed or stopped early; some may be missing
    reason: Optional[str] = None  # FALLBACK: why the account-level list is unavailable

    @property
    def truncated(self) -> bool:
        """The list holds only the newest jobs: more exist, or may exist."""
        return self.stopped is not None


@dataclass
class Listing:
    """One refresh's jobs, as a plain list, and what they cover."""

    jobs: list
    scope: ListScope


def _more(scope: ListScope) -> str:
    more = "more exist" if scope.stopped == CAPPED else "more may exist"
    return more + ("; older active jobs may be missing" if scope.active_missing else "")


def scope_text(scope: ListScope, *, location: str, only: str = "") -> str:
    """A short header for this scope; ``only`` names its sole location."""
    if scope.kind == ACCOUNT:
        if scope.stopped is None:
            return f"all schemas (all in {only})" if only else "all schemas"
        held = f"newest {scope.walked:,}" if scope.stopped == CAPPED else (
            f"first {scope.pages} pages: {scope.walked:,} job{'' if scope.walked == 1 else 's'}")
        return held + ("" if scope.active_missing else " + every active job") + f" ({_more(scope)})"
    if scope.kind == SCHEMA:
        return f"Current schema: {location}"
    return f"Current schema only: {location} ({unavailable_text(scope)})"


def unavailable_text(scope: ListScope) -> str:
    """Why a :data:`FALLBACK` listing holds the configured schema's jobs, in a few words."""
    return _FALLBACK_WHY.get(scope.reason or "", "")


def loaded_text(scope: Optional[ListScope]) -> str:
    """What a list cut short holds, for a search without results; "" when it holds every job."""
    if scope is None or scope.stopped is None:
        return ""
    held = (f"the newest {scope.walked:,} of the last 6 months" if scope.stopped == CAPPED
            else f"what the first {scope.pages} pages held")
    return held + ("" if scope.active_missing else " and every active job") + f"; {_more(scope)}"


def empty_text(scope: ListScope) -> str:
    """The list's message when a walk cut short left no job to show."""
    held = (f"in its first {scope.pages} pages" if scope.stopped == PAGE_LIMIT
            else f"among its newest {scope.walked:,}")
    return f"The account-wide list returned no jobs to show {held}; more may exist. r reads it again."


def _one_line(text: str) -> str:
    return views.fit_cells(views.collapse(text), _FAILURE_LIMIT)


def _failure_text(route: str, exc: BaseException) -> str:
    """What failed, on one line: the route, then the answer's status and message or the error's own text."""
    answered = isinstance(getattr(getattr(exc, "response", None), "status_code", None), int)
    return _one_line(f"{route} {'answered' if answered else 'failed:'} {views.describe_error(exc)}")


def _page(page: Any) -> tuple:
    """``(jobs, next_page_token)`` of one ``list_jobs_page`` answer, as it documents them."""
    if not isinstance(page, dict) or not isinstance(page.get("jobs"), list):
        raise ValueError("the account-level job list answered a page that is not a job list")
    token = page.get("next_page_token")
    return [job for job in page["jobs"] if isinstance(job, dict)], token if isinstance(token, str) and token else None


def _has_message(value: Any, depth: int = 0) -> bool:
    if depth > 4:
        return False
    if isinstance(value, dict):
        message = value.get("message")
        if isinstance(message, str) and message.strip():
            return True
        return any(_has_message(item, depth + 1) for item in value.values())
    if isinstance(value, list):
        return any(_has_message(item, depth + 1) for item in value)
    return False


def _service_explained(exc: BaseException) -> bool:
    """Whether a refusal carries the service's own JSON error message."""
    try:
        body = exc.response.json()  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001 - no response, or no JSON in it
        return False
    return _has_message(body)


class JobLister:
    def __init__(self, client: Any, *, account_wide: bool = False, clock: Callable[[], float] = time.monotonic):
        self._client = client
        self.account_wide = account_wide
        self._clock = clock
        self._fallback: Optional[str] = None  # why the account-level list is unavailable, while it is
        self._asked_at = 0.0  # in fallback: when the account-level list last answered (or failed)

    def load(self, now: datetime, *, is_cancelled: Callable[[], bool] = lambda: False,
             manual: bool = False, account: Optional[bool] = None, asked: bool = False) -> Optional[Listing]:
        """Load one refresh; flags override scope, reprobe, and fallback behavior."""
        if not (self.account_wide if account is None else account):
            return self._schema(ListScope(SCHEMA))
        if self._fallback is not None and not self._ask_again(manual):
            return self._schema(ListScope(FALLBACK, reason=self._fallback))
        if asked:
            self._fallback = None  # asked again for a: the earlier answer no longer counts
        probing = self._fallback is not None
        try:
            listing = self._walk(now, is_cancelled)
        except AccountJobListUnavailable as exc:
            return self._unavailable(exc, probing)
        except Exception as exc:
            if not probing:
                raise JobListError(_failure_text(_ACCOUNT_ROUTE, exc)) from exc
            # Asking again failed some other way (a timeout, a 503): keep listing
            # the configured schema, and ask again later.
            self._asked_at = self._clock()
            logger.info("job list: the account-level list failed again; listing the configured schema", exc_info=True)
            return self._schema(ListScope(FALLBACK, reason=self._fallback))
        if listing is not None:
            self._fallback = None
        return listing

    def _ask_again(self, manual: bool) -> bool:
        waited = self._clock() - self._asked_at + _ASK_SLACK_SECONDS
        return waited >= ACCOUNT_REPROBE_SECONDS or (manual and waited >= ACCOUNT_MANUAL_REPROBE_SECONDS)

    def _fall_back(self, exc: AccountJobListUnavailable) -> None:
        self._fallback = exc.reason
        self._asked_at = self._clock()
        logger.info("job list: %s; listing the configured schema instead", exc)

    def _unavailable(self, exc: AccountJobListUnavailable, probing: bool) -> Listing:
        if exc.reason in _FALLBACK_REASONS or probing:
            self._fall_back(exc)
            return self._schema(ListScope(FALLBACK, reason=exc.reason))
        refused = JobListError(_one_line(str(exc)), refused=True)
        # An unexplained 403 may affect only this route; probe the schema once.
        if exc.reason != AccountJobListUnavailable.FORBIDDEN or _service_explained(exc):
            raise refused from exc
        try:
            listing = self._schema(ListScope(FALLBACK, reason=exc.reason))
        except JobListError as schema_failed:
            raise JobListError(_one_line(f"{exc}; and {schema_failed}")) from exc
        self._fall_back(exc)
        return listing

    def _schema(self, scope: ListScope) -> Listing:
        try:
            jobs = self._client.list_jobs()
        except Exception as exc:
            raise JobListError(_failure_text(_SCHEMA_ROUTE, exc)) from exc
        return Listing([job for job in (jobs or []) if isinstance(job, dict)], scope)

    def _walk(self, now: datetime, is_cancelled: Callable[[], bool]) -> Optional[Listing]:
        """The newest jobs of the account-level list, and the active ones when more exist; None when cancelled."""
        created_after = now - timedelta(days=views.RETENTION_DAYS)
        walked = self._pages(created_after, None, is_cancelled)
        if walked is None:
            return None
        jobs, stopped = walked
        if stopped is None:
            return Listing(jobs, ListScope(ACCOUNT))
        try:
            active = self._pages(created_after, ACTIVE, is_cancelled)
        except Exception:  # noqa: BLE001 - the newest jobs stand; the scope says active ones may be missing
            logger.info("job list: reading the active jobs failed; older active jobs may be missing", exc_info=True)
            active = ([], "failed")
        if active is None:
            return None
        rows, active_stopped = active
        listed = {views.job_key(job) for job in jobs}
        added = [job for job in rows if views.job_key(job) not in listed]
        scope = ListScope(ACCOUNT, stopped=stopped, walked=len(jobs),
                          pages=ACCOUNT_LIST_MAX_PAGES if stopped == PAGE_LIMIT else 0,
                          active_missing=active_stopped is not None)
        return Listing(jobs + added, scope)

    def _pages(self, created_after: datetime, status: Optional[str],
               is_cancelled: Callable[[], bool]) -> Optional[tuple]:
        """One walk; it starts again from the first page, once, when the service can't open its token."""
        try:
            return self._read_pages(created_after, status, is_cancelled)
        except InvalidPageToken:
            logger.info("job list: the service could not open its page token; reading again from the first page")
            return self._read_pages(created_after, status, is_cancelled)

    def _read_pages(self, created_after: datetime, status: Optional[str],
                    is_cancelled: Callable[[], bool]) -> Optional[tuple]:
        """``(jobs, stopped)``: up to the cap or the page limit, newest first; None when cancelled."""
        filters: dict = {"created_after": created_after, **({"status": status} if status else {})}
        jobs: list = []
        seen: set = set()
        token = None
        for _ in range(ACCOUNT_LIST_MAX_PAGES):
            if is_cancelled():
                return None
            page = self._client.list_jobs_page(
                **filters,
                page_size=ACCOUNT_PAGE_SIZE,
                page_token=token,
            )
            rows, token = _page(page)
            for job in rows:
                job_id = views.job_id_of(job)
                key = views.job_key(job)
                if job_id and key in seen:
                    continue
                seen.add(key)
                jobs.append(job)
            if len(jobs) >= ACCOUNT_LIST_CAP:
                more = token is not None or len(jobs) > ACCOUNT_LIST_CAP
                return jobs[:ACCOUNT_LIST_CAP], CAPPED if more else None
            if token is None:
                return jobs, None
        return jobs, PAGE_LIMIT  # stopped at the page limit with pages left


class JobClients:
    """The base client plus one cached child per account-list location."""

    def __init__(self, client: Any):
        self.base = client
        self.home = views.client_location(client)
        self._children: dict = {}
        self._lock = threading.Lock()

    def elsewhere(self, job: Any) -> bool:
        """Whether ``job`` names a location other than the configured one (or the client doesn't say which)."""
        names = views.location_names(job)
        return names is not None and names != self.home

    def for_job(self, job: Any) -> Any:
        """The location-bound client for ``job``."""
        location = views.location_names(job)
        if location is None:  # the configured schema's own list, or a job opened by id
            return self.base
        with self._lock:
            child = self._children.get(location)
            if child is None:
                child = self._children[location] = self.base.for_job(job)
            return child
