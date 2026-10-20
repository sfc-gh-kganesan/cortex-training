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

"""Wrap job-record reads so untrusted strings are inert before screens see them."""

from __future__ import annotations

import functools
from typing import Any

from cortex_training.tui.views import printable_data

# The client methods whose answers are job records.
_RECORD_READS = frozenset({"get_job", "list_jobs", "list_jobs_page"})

# A job's location, which the account-wide list names. These are SQL identifiers that
# route the job's calls (JobClients.for_job), not text to draw: made printable, a name
# holding a control or zero-width character becomes a different identifier and every
# call for that job answers 404. They are kept as the service stored them, and what
# draws them makes them printable itself (views.location_label, app._cell).
_ROUTING_KEYS = ("database_name", "schema_name")


def _keep_routing(clean: Any, raw: Any) -> Any:
    """``clean`` with every record's :data:`_ROUTING_KEYS` taken from ``raw``."""
    if isinstance(clean, dict) and isinstance(raw, dict):
        kept = {key: _keep_routing(value, raw.get(key)) for key, value in clean.items()}
        for key in _ROUTING_KEYS:
            if isinstance(raw.get(key), str):
                kept[key] = raw[key]
        return kept
    if isinstance(clean, list) and isinstance(raw, list) and len(clean) == len(raw):
        return [_keep_routing(item, original) for item, original in zip(clean, raw)]
    return clean


class PrintableClient:
    """A transparent client proxy that sanitizes job reads and child clients."""

    def __init__(self, client: Any) -> None:
        object.__setattr__(self, "wrapped", client)

    def __getattr__(self, name: str) -> Any:
        if name == "wrapped":  # not set yet (a copy being made): don't recurse
            raise AttributeError(name)
        value = getattr(self.wrapped, name)
        if not callable(value):
            return value
        if name in _RECORD_READS:
            return _printable_answer(value)
        if name == "for_job":
            return _printable_child(value)
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self.wrapped, name, value)

    def __repr__(self) -> str:
        return f"PrintableClient({self.wrapped!r})"


def printable_client(client: Any) -> Any:
    """``client`` wrapped in a :class:`PrintableClient`, once (None stays None)."""
    if client is None or isinstance(client, PrintableClient):
        return client
    return PrintableClient(client)


def _printable_answer(method):
    @functools.wraps(method)
    def read(*args, **kwargs):
        answer = method(*args, **kwargs)
        return _keep_routing(printable_data(answer), answer)

    return read


def _printable_child(method):
    @functools.wraps(method)
    def child(*args, **kwargs):
        return printable_client(method(*args, **kwargs))

    return child
