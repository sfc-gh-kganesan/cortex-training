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

"""The job list: the screen that opens first, and the table it draws.

It reads each refresh through JobLister (tui/account.py), groups the jobs into LIVE
and HISTORY, folds each job's sub-jobs under it, and opens a job's preview.
"""

from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from rich.text import Text
from textual import events
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.content import Content
from textual.coordinate import Coordinate
from textual.css.query import NoMatches
from textual.screen import Screen
from textual.widgets import DataTable
from textual.widgets import Footer
from textual.widgets import Header
from textual.widgets import Input
from textual.widgets import Static
from textual.worker import get_current_worker
from cortex_training.tui import views
from cortex_training.tui.account import ACCOUNT
from cortex_training.tui.logs import LogScreen
from cortex_training.tui.account import SCHEMA
from cortex_training.tui.account import JobClients
from cortex_training.tui.account import JobLister
from cortex_training.tui.account import JobListError
from cortex_training.tui.account import Listing
from cortex_training.tui.account import ListScope
from cortex_training.tui.account import empty_text
from cortex_training.tui.account import loaded_text
from cortex_training.tui.account import scope_text
from cortex_training.tui.account import unavailable_text
from cortex_training.tui.format import format_created_at
from cortex_training.tui.widgets import FollowsPointer
from cortex_training.tui.widgets import _BLANK
from cortex_training.tui.widgets import _ERROR_LOG
from cortex_training.tui.widgets import log_traceback
from cortex_training.tui.widgets import KEYS_BINDING
from cortex_training.tui.widgets import SelectableTable
from cortex_training.tui.widgets import _FixedCell
from cortex_training.tui.widgets import _cell
from cortex_training.tui.widgets import after_refresh
from cortex_training.tui.widgets import call_on_screen
from cortex_training.tui.widgets import close_keys_panel
from cortex_training.tui.widgets import hanging
from cortex_training.tui.widgets import is_shown
from cortex_training.tui.widgets import wrapped_parts


def _sort_part(key: str, descending: bool) -> Content:
    """The clickable ``sort: created ↓`` control."""
    arrow = "↓" if descending else "↑"
    return Content.from_markup(
        f"[dim]sort:[/dim] [b][@click=screen.cycle_sort]{key}[/] [@click=screen.toggle_order]{arrow}[/][/b]")


_SORT_PART_CELLS = max(_sort_part(key, descending).cell_length
                       for key in views.SORT_KEYS for descending in (True, False))


@dataclass(frozen=True)
class _ListLayout:
    """The job table's columns for one terminal width, sort and set of locations."""

    columns: tuple  # (label, width) in display order
    compact_times: bool  # Created and Updated as MM-DD HH:MM

    @property
    def labels(self) -> tuple:
        return tuple(label for label, _width in self.columns)


class _JobRows:
    """One job's table rows as last built, and what they were built from."""

    # Slots: one of these per listed job, kept between renders.
    __slots__ = ("job", "compact", "cells", "sub_cells", "sub_keys", "layout", "expanded", "row", "subs")

    def __init__(self, job: dict, compact: bool, cells: dict) -> None:
        self.job = job  # the record (compared by identity: a refresh keeps an unchanged record)
        self.compact = compact  # built with MM-DD HH:MM times
        self.cells = cells  # column label -> cell: the job's row, but for its Job ID
        self.sub_cells: tuple | None = None  # ((row key suffix, {label: cell}), ...): its sub-job rows (and note)
        self.sub_keys: tuple = ()  # ((job id, sub-job id) or None for the note, ...) beside sub_cells
        self.layout = None  # the layout row and subs are in
        self.expanded: bool | None = None  # whether row and subs are for the job unfolded
        self.row: tuple = ()
        self.subs: tuple = ()  # ((row key suffix, cells), ...); none while folded


class _JobTable(SelectableTable):
    """The selectable job table with fold-aware navigation."""

    BINDINGS = [
        Binding("home", "first_job", "First job", show=False),
        Binding("end", "last_job", "Last job", show=False),
        Binding("right", "cursor_right", "Unfold sub-jobs, else scroll right", show=False),
        Binding("left", "cursor_left", "Fold sub-jobs, else scroll left", show=False),
    ]
    _MARKERS = ("▸ ", "▾ ")

    def action_first_job(self) -> None:
        self.move_cursor(row=min(1, self.row_count - 1), animate=False)  # row 0 is the LIVE/HISTORY header
        self.scroll_home(animate=False)

    def action_last_job(self) -> None:
        self.move_cursor(row=self.row_count - 1, animate=False)

    def action_cursor_right(self) -> None:
        if not self.screen.unfold_highlighted():
            super().action_cursor_right()  # a row cursor: the table scrolls right

    def action_cursor_left(self) -> None:
        if not self.screen.fold_highlighted():
            super().action_cursor_left()

    def on_click(self, event: events.Click) -> None:
        """A click on a job's ▸/▾ marker unfolds or folds it; any other click is the table's own, unless swallowed."""
        if self.swallowed(event):
            return
        key = self._marker_at(event)
        if key is not None and self.screen.toggle_job_at(key):
            event.prevent_default()  # the table's own handler would open the job on a second click
            event.stop()

    def _marker_at(self, event: events.Click) -> str | None:
        """The row key of the job whose ▸/▾ marker ``event`` is on, else None."""
        meta = event.style.meta
        row, column = meta.get("row"), meta.get("column")
        columns = self.ordered_columns
        if not (isinstance(row, int) and isinstance(column, int) and 0 <= row < self.row_count
                and 0 <= column < min(len(columns), self.fixed_columns)):
            return None  # the header, or no fixed column (the Job ID column is one; it never scrolls)
        if columns[column].key.value != "Job ID" or not str(self.get_cell_at(Coordinate(row, column))).startswith(
                self._MARKERS):
            return None
        offset = event.get_content_offset(self)
        if offset is None:
            return None
        left = getattr(self, "_row_label_column_width", 0) + sum(c.get_render_width(self) for c in columns[:column])
        if offset.x - left != self.cell_padding:  # the arrow: ▸ and ▾ are one cell wide, after the padding
            return None
        return self.ordered_rows[row].key.value

    def replace_rows(self, rows: list) -> None:
        """Show ``rows`` (``(key, cells)`` pairs, in order) in place of the table's rows."""
        self.clear()
        for key, cells in rows:
            self.add_row(*cells, key=key)


class JobListScreen(FollowsPointer, Screen):
    """Every job in the retention window: live first, then history, with sub-job rows.

    ``a`` switches between the configured schema's jobs and the account-wide list
    (see :class:`cortex_training.tui.account.JobLister`); search takes ids, locations
    and ``field<op>value`` filters (see :func:`cortex_training.tui.views.parse_query`).
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        KEYS_BINDING,
        # Priority keeps burst input on the focus selected by the first key.
        Binding("/", "filter", "Search", priority=True),
        Binding("escape", "clear_filter", "Clear search", show=False, priority=True),
        Binding("enter", "submit_search", "Show matches", show=False, priority=True),
        Binding("a", "show_list('account')", "All schemas"),
        Binding("a", "show_list('schema')", "Current schema"),
        ("s", "cycle_sort", "Sort"),
        ("o", "toggle_order", "Order"),
        ("e", "expand_all", "Expand/collapse all"),
        ("r", "refresh", "Refresh"),
    ]
    _REFRESH_SECONDS = 60
    _COLUMNS = [(" ", 1), ("Job ID", 38), ("Status", 12), ("Type", 13), ("Created", 16),
                ("Owner", 14), ("Model", 18), ("GPUs", 9), ("Updated", 16)]
    _FIXED_COLUMNS = 2  # the marker and the job id stay in view while the table scrolls sideways
    _FIRST_COLUMNS = (" ", "Job ID", "Status", "Type", "Created", "Owner")
    _NAME_MIN = 16
    _TIME_COMPACT = 11
    _SEARCH_DEBOUNCE_SECONDS = 0.12
    _LOCATION_AT = [label for label, _width in _COLUMNS].index("GPUs") + 1
    _LOCATION_WIDTHS = (10, 32)

    def __init__(self, client, *, poll_interval=1.0, now=None, clients=None, lister=None):
        """``clients`` and ``lister`` are the app's (shared with a list opened again
        later); without them the screen makes its own for ``client``."""
        super().__init__()
        self._client = client
        self._clients = clients if clients is not None else JobClients(client)
        self._lister = lister if lister is not None else JobLister(client)
        self._poll_interval = poll_interval
        self._now = now  # fixed clock for tests/screenshots; None = wall clock
        self._job_by_item: dict[str, tuple] = {}
        self._jobs_by_key: dict[tuple, dict] = {}
        self._sub_by_item: dict[str, tuple] = {}
        self._all_jobs: list = []
        self._filter = ""
        self._sort = "created"
        self._descending = True
        self._expanded: set = set()
        self._collapsed: set = set()  # live jobs the user folded
        self._last_refreshed = None
        self._loaded_at: float | None = None  # monotonic time of the last good load
        self._failed_at: str | None = None  # HH:MM:SS of the last failed load, while it stands
        self._refreshing = False
        self._refresh_started: float | None = None  # monotonic time the refresh in flight began
        self._list_error: str | None = None  # last failed load, on one line; cleared by a good one
        self._list_refused = False  # that failure was the account-wide list refusing this caller (403/405)
        self._scope: ListScope | None = None  # what the list on screen covers
        self._asked = self._account_wanted()
        self._search_timer = None
        self._layout: _ListLayout | None = None  # the table's columns, once they exist
        self._notice: Content | None = None  # the stale-list notice, before wrapping
        self._info: Content | list | None = None  # the line(s) under the header, before wrapping
        self._nomatch: str | None = None  # the no-match message (plain text), before wrapping
        self._rows: dict[tuple, _JobRows] = {}
        self._hay: dict[tuple, tuple] = {}  # job key -> (record, search_haystack)
        self._resolved: tuple | None = None  # (search text, the jobs list it was resolved against, query)
        self._headers: dict[str, tuple] = {}  # LIVE/HISTORY -> ((count, label, layout), cells)
        self._keys: dict[tuple, str] = {}  # job key -> its row's table key
        self._shown: tuple | None = None  # what the table holds: (layout, ((row key, cells), ...))

    def _job_key(self, job: dict) -> tuple:
        return views.job_key(job, self._clients.home)

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("", id="listinfo")
        yield Static("", id="listnotice")
        yield _JobTable(id="jobs", cursor_type="row", zebra_stripes=False, fixed_columns=self._FIXED_COLUMNS)
        with VerticalScroll(id="nomatch-box"):
            yield Static("", id="nomatch")
        yield Input(
            placeholder="search ids…  or fields: status:RUNNING gpu:H200",
            id="jobfilter",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#nomatch", Static).display = False
        self.query_one("#nomatch-box", VerticalScroll).display = False
        self.query_one("#listnotice", Static).display = False
        self._set_columns(self.query_one("#jobs", DataTable), self._layout_for(0))
        self._refresh_jobs()
        # DOMNode reserves _auto_refresh, so this timer has another name.
        self.set_interval(self._REFRESH_SECONDS, self._refresh_tick)

    def on_screen_resume(self) -> None:
        self._update_subtitle()
        self._render_jobs()  # with a status the preview read; a list as it was isn't drawn again
        if self._loaded_at is not None and time.monotonic() - self._loaded_at >= self._REFRESH_SECONDS:
            self._refresh_jobs()

    def on_resize(self, event) -> None:
        after_refresh(self, self._fit_to_width)

    def _fit_to_width(self) -> None:
        """After a resize: re-wrap the lines around the list, and re-render if the columns change."""
        self._paint_info()
        self._paint_notice()
        self._paint_nomatch()
        if self._layout != self._layout_for(self._location_column_width(self._recent())):
            self._render_jobs()

    def _layout_for(self, location_width: int) -> _ListLayout:
        """Fit columns, compacting dates and the job id while keeping the sorted time visible."""
        table = self.query_one("#jobs", DataTable)
        pad = 2 * table.cell_padding
        room = (self.size.width or self.app.size.width) - table.styles.scrollbar_size_vertical
        columns = list(self._COLUMNS)
        if location_width:
            columns.insert(self._LOCATION_AT, ("Location", location_width))
        compact = sum(width + pad for _label, width in columns) > room
        if compact:
            columns = [(label, self._TIME_COMPACT if label in ("Created", "Updated") else width)
                       for label, width in columns]
        name_base = dict(self._COLUMNS)["Job ID"]
        spare = room - pad - sum(width + pad for label, width in columns if label != "Job ID")
        if spare >= name_base:
            name = name_base
        else:
            first = set(self._FIRST_COLUMNS) - {"Job ID"}
            needed = sum(width + pad for label, width in columns if label in first)
            if room - pad - needed < self._NAME_MIN:  # the last of them can't fit: the job id takes its room
                needed -= dict(columns)[self._FIRST_COLUMNS[-1]] + pad
            name = max(self._NAME_MIN, min(name_base, room - pad - needed))
        columns = [(label, name if label == "Job ID" else width) for label, width in columns]
        labels = [label for label, _width in columns]
        if self._sort == "updated":
            ends = sum(width + pad for _label, width in columns[: labels.index("Updated") + 1])
            if ends > room:  # out of view: put it where Created is (the two have the same width)
                at_created, at_updated = labels.index("Created"), labels.index("Updated")
                columns[at_created], columns[at_updated] = columns[at_updated], columns[at_created]
        return _ListLayout(tuple(columns), compact)

    def _set_columns(self, table: DataTable, layout: _ListLayout) -> None:
        """Set the table's columns up for ``layout`` (clearing it) when they aren't the ones it has."""
        if layout == self._layout:
            return
        self._layout = layout
        table.clear(columns=True)
        for label, width in layout.columns:
            table.add_column(label, width=width, key=label)

    def _recent(self) -> list:
        """The jobs of the retention window (the configured schema's list has older ones too)."""
        now = self._clock()
        return [j for j in self._all_jobs if views.within_retention(j, now)]

    def _location_column_width(self, jobs: list) -> int:
        """The Location column's width for ``jobs``; 0 (hidden) when they all share one location."""
        labels = {views.location_label(job) for job in jobs}
        if len(labels) <= 1:
            return 0
        low, high = self._LOCATION_WIDTHS
        return max(low, min(high, max(views.cell_width(label) for label in labels)))

    def _clock(self) -> datetime:
        return self._now or datetime.now(timezone.utc)

    def _update_subtitle(self) -> None:
        if self.app.screen is not self:
            return  # a late refresh must not retitle the screen on top
        if self._refreshing:
            refresh_status = "refreshing…"
            if self._last_refreshed:
                refresh_status += f" (last refreshed {self._last_refreshed})"
        elif self._list_error and self._failed_at and self._loaded_at is None:
            refresh_status = f"load failed {self._failed_at}"  # no list yet: the error is in its place
        elif self._list_error and self._failed_at:
            refresh_status = f"refresh failed {self._failed_at}"
            if self._last_refreshed:
                refresh_status += f" (list from {self._last_refreshed[:5]})"
        else:
            refresh_status = f"last refreshed {self._last_refreshed or '—'}"
        bits = [refresh_status, self._scope_text()]
        self.app.sub_title = "  ·  ".join(bit for bit in bits if bit)

    def _scope_text(self) -> str:
        """What the list on screen covers ("all schemas", ...); "" before the first list."""
        if self._scope is None:
            return ""
        labels = {views.location_label(job) for job in self._all_jobs}
        only = next(iter(labels)) if len(labels) == 1 else ""
        return scope_text(self._scope, location=self._schema_label(), only=only)

    def _refresh_jobs(self, *, manual: bool = False, quiet: bool = False) -> None:
        """Refresh; ``manual`` permits reprobe and ``quiet`` suppresses an in-flight notice."""
        if self._refreshing:
            if manual and not quiet:
                waited = int(time.monotonic() - (self._refresh_started or time.monotonic()))
                self.notify(f"Still waiting for the last refresh ({waited}s); r asks again once it answers.",
                            markup=False)
            return
        self._refreshing = True
        self._refresh_started = time.monotonic()
        self._update_subtitle()
        account = self._account_wanted()
        self._load_jobs(manual, account, account and self._asked)

    def _account_wanted(self) -> bool:
        """Whether the list to show is the account-wide one (a), not the configured schema's."""
        return self._lister.account_wide

    def action_show_list(self, which: str) -> None:
        """``a``: show the account-wide list (``which`` "account") or the configured schema's ("schema")."""
        account = which == "account"
        if account == self._lister.account_wide:
            return
        self._lister.account_wide = account
        self._asked = account
        self.refresh_bindings()  # the footer's a names the other list now
        self._refresh_jobs(manual=True, quiet=True)

    def _refresh_tick(self) -> None:
        if self.app.screen is self:
            self._refresh_jobs()

    def action_refresh(self) -> None:
        self._refresh_jobs(manual=True)

    def _search_focused(self) -> bool:
        try:
            return self.focused is self.query_one("#jobfilter", Input)
        except NoMatches:  # not composed yet
            return False

    def check_action(self, action: str, parameters: tuple) -> bool | None:
        if action == "filter":
            return not self._search_focused()  # in the box, / is part of the query
        if action == "submit_search":
            return self._search_focused()  # on the list, Enter opens the highlighted row
        if action == "show_list":
            # One of a's two bindings, the one for the other list.
            return (parameters[:1] == ("account",)) != self._lister.account_wide
        return True

    def action_filter(self) -> None:
        self.set_focus(self.query_one("#jobfilter", Input))

    async def _typed_ahead(self, box: Input) -> None:
        """Wait for keys already sent to ``box`` before acting on its text."""
        done = asyncio.get_running_loop().create_future()

        def handled() -> None:
            if not done.done():
                done.set_result(None)

        if not box.call_later(handled):  # after what is already in the box's queue
            return
        try:
            await asyncio.wait_for(done, timeout=1.0)
        except asyncio.TimeoutError:  # never expected: act on what the box holds
            pass

    async def action_submit_search(self) -> None:
        """Enter in the search box: show the matches of what was typed now, and move to the list."""
        box = self.query_one("#jobfilter", Input)
        await self._typed_ahead(box)
        text = box.value.strip()
        if text != self._filter or self._search_timer is not None:
            # Enter right after typing: show the matches before Enter can pick a row.
            self._filter = text
            self._search_now()
        self.set_focus(self.query_one("#jobs", DataTable))

    async def action_clear_filter(self) -> None:
        if close_keys_panel(self):
            return
        box = self.query_one("#jobfilter", Input)
        await self._typed_ahead(box)
        if self.focused is box or self._filter or box.value:
            box.value = ""  # its Changed comes after, and finds the search already cleared
            if self._filter or self._search_timer is not None:
                self._filter = ""
                self._search_now()
            self.set_focus(self.query_one("#jobs", DataTable))

    def action_cycle_sort(self) -> None:
        keys = views.SORT_KEYS
        self._sort = keys[(keys.index(self._sort) + 1) % len(keys)]
        self._render_jobs(top=True)  # a new order: show its start, not the old row wherever it went

    def action_toggle_order(self) -> None:
        self._descending = not self._descending
        self._render_jobs(top=True)

    def whole_id(self, key) -> str | None:
        """The whole id of the job or sub-job on row ``key`` (a sub-job's row shows its label there)."""
        if key in self._job_by_item:
            job = self._jobs_by_key.get(self._job_by_item[key])
            return views.job_id_of(job) if job is not None else None
        sub = self._sub_by_item.get(key)  # (job key, sub-job id)
        if sub:
            job = self._jobs_by_key.get(sub[0])
            return sub[1] if sub[1] else (views.job_id_of(job) if job is not None else None)
        return None

    def action_copy_text(self) -> None:
        """ctrl+c: the text the mouse highlighted (in the list too), else everything
        the screen shows, the whole list with whole ids."""
        if not self.app.copy_highlight(self):
            self.app.copy_everything(self)

    def _highlighted(self) -> tuple:
        table = self.query_one("#jobs", DataTable)
        try:
            key = table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value
        except Exception:  # noqa: BLE001 - empty table
            return None, None
        if key in self._job_by_item:
            return key, self._job_by_item[key]
        if key in self._sub_by_item:
            return key, self._sub_by_item[key][0]
        if str(key).startswith("note-"):  # an UNKNOWN job's "status unavailable" line
            return key, self._job_by_item.get(str(key).removeprefix("note-"))
        return key, None

    def _is_expanded(self, job: dict) -> bool:
        job_key = self._job_key(job)
        if job_key in self._collapsed:
            return False
        return job_key in self._expanded or views.is_live(job)

    @staticmethod
    def _expandable(job: dict) -> bool:
        # An UNKNOWN job unfolds into a "status unavailable" line even without sub-jobs.
        return bool(views.sub_jobs(job)) or views.is_unknown(job)

    def _unfold(self, job_key: tuple, unfolded: bool) -> None:
        """Toggle a job's sub-rows while preserving cursor and scroll positions."""
        if unfolded:
            self._collapsed.discard(job_key)
            self._expanded.add(job_key)
        else:
            self._expanded.discard(job_key)
            self._collapsed.add(job_key)
        table = self.query_one("#jobs", DataTable)
        scrolled = table.scroll_x
        key = self._row_key(job_key)
        try:
            on_screen = table.get_row_index(key) - int(table.scroll_y)  # rows below the top of the view
        except Exception:  # noqa: BLE001 - not in the table
            on_screen = None
        self._render_jobs(keep=job_key)
        if scrolled:
            after_refresh(self, table.scroll_to, x=scrolled, animate=False, immediate=True)
        if on_screen is not None:
            after_refresh(self, self._keep_job_row, key, on_screen, unfolded)

    def _keep_job_row(self, key: str, on_screen: int, unfolded: bool, tries: int = 10) -> None:
        """Restore a toggled job row's screen position."""
        table = self.query_one("#jobs", DataTable)
        try:
            row = table.get_row_index(key)
        except Exception:  # noqa: BLE001 - gone with a refresh meanwhile
            return
        view = table.scrollable_content_region.height - (table.header_height if table.show_header else 0)
        top = row - max(0, on_screen)
        if unfolded and view > 0:
            rows = table.ordered_rows
            last = row
            while last + 1 < len(rows) and str(rows[last + 1].key.value).startswith((f"sub-{key}-", f"note-{key}")):
                last += 1
            if last - top >= view:
                top = min(row, last - view + 1)
        top = max(0, top)
        if (view <= 0 or top > table.max_scroll_y) and tries:
            # Not laid out, or not yet as tall as its new rows: once it is.
            after_refresh(self, self._keep_job_row, key, on_screen, unfolded, tries - 1)
            return
        table.scroll_to(y=top, animate=False, immediate=True)

    def unfold_highlighted(self) -> bool:
        """→ on a job's own row: unfold it; whether it did (where it didn't, → scrolls the table)."""
        key, job_key = self._highlighted()
        return key in self._job_by_item and self._toggle(job_key, True)

    def fold_highlighted(self) -> bool:
        """← on a job's row or a sub-job row: fold the job; whether it did (else ← scrolls the table)."""
        return self._toggle(self._highlighted()[1], False)

    def toggle_job_at(self, key) -> bool:
        """A click on the ▸/▾ marker of the job row ``key``: unfold or fold it; whether it did."""
        return self._toggle(self._job_by_item.get(key), None)

    def _toggle(self, job_key, unfold: bool | None) -> bool:
        """Unfold (True) or fold (False) a job, or (None) switch it; whether it did."""
        job = self._jobs_by_key.get(job_key)
        if job is None or not self._expandable(job) or self._is_expanded(job) == unfold:
            return False
        self._unfold(job_key, not self._is_expanded(job))
        return True

    def action_expand_all(self) -> None:
        with_subs = [j for j in self._all_jobs if self._expandable(j)]
        if with_subs and all(self._is_expanded(j) for j in with_subs):
            self._expanded.clear()
            self._collapsed = {self._job_key(j) for j in self._all_jobs if views.is_live(j)}
        else:
            self._collapsed.clear()
            self._expanded = {self._job_key(j) for j in self._all_jobs}
        self._render_jobs()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "jobfilter":
            return
        if event.value != event.input.value:
            return  # the box has changed again since: a Changed with what it holds is on its way
        text = event.value.strip()
        if text == self._filter and self._search_timer is None:
            return  # the list shows this search already (Enter or Esc applied it, or a space was typed)
        self._filter = text
        # Rebuild once typing pauses; a rebuild per keystroke lagged with ~900 jobs.
        if self._search_timer is not None:
            self._search_timer.stop()
        self._search_timer = self.set_timer(self._SEARCH_DEBOUNCE_SECONDS, self._search_now)

    def _search_now(self) -> None:
        """Show the matches of the search (``_filter``), without waiting for typing to pause."""
        if self._search_timer is not None:
            self._search_timer.stop()
            self._search_timer = None
        self._render_jobs()

    @work(thread=True, group="jobs")
    def _load_jobs(self, manual: bool = False, account: bool = False, asked: bool = False) -> None:
        """Read one schema or account-wide listing off the UI thread."""
        worker = get_current_worker()
        try:
            listing = self._lister.load(self._clock(), is_cancelled=lambda: worker.is_cancelled, manual=manual,
                                        account=account, asked=asked)
            error = None
        except JobListError as exc:
            listing, error = None, exc
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            listing, error = None, JobListError(views.collapse(f"the job list failed: {views.describe_error(exc)}"))
        if worker.is_cancelled or (listing is None and error is None):
            return
        call_on_screen(self, self._on_jobs, listing, error, account)

    def _on_jobs(self, listing, error: JobListError | None, account: bool = False) -> None:
        """A load's answer: ``listing``, or ``error``; ``account``: it read the account-wide list."""
        self._refreshing = False
        if account != self._account_wanted():
            # a switched lists while this one was read: read the one asked for now.
            self._refresh_jobs(manual=True, quiet=True)
            return
        if account and self._asked:
            self._asked = False
            if error is not None or listing.scope.kind != ACCOUNT:
                listing = self._stay_on_schema(listing, error)
                if listing is None:
                    return
                error = None
        if error is not None:
            # Keep the last good list on screen, saying so, until a load succeeds.
            self._list_error = str(error) or "the job list failed"
            self._list_refused = error.refused
            self._failed_at = time.strftime("%H:%M:%S")
            self._update_subtitle()
            self._render_jobs()
            return
        self._list_error = None
        self._list_refused = False
        self._failed_at = None
        self._loaded_at = time.monotonic()
        self._scope = listing.scope
        self._all_jobs = self._merged(listing.jobs)
        self._jobs_by_key = {self._job_key(job): job for job in self._all_jobs}
        self._rows = {key: rows for key, rows in self._rows.items() if key in self._jobs_by_key}
        self._hay = {key: hay for key, hay in self._hay.items() if key in self._jobs_by_key}
        self._last_refreshed = time.strftime("%H:%M:%S")
        self._update_subtitle()
        self._render_jobs()
        self.app.job_list_drawn(self)

    def _stay_on_schema(self, listing: Listing | None, error: JobListError | None) -> Listing | None:
        """Handle an unavailable account list by retaining the configured schema."""
        self._lister.account_wide = False
        self.refresh_bindings()  # the footer's a offers all schemas again
        why = (str(error) or "the account-wide job list failed") if error is not None else unavailable_text(
            listing.scope)
        self.notify(f"Can't show all schemas ({why}); the list stays on {self._schema_label()}.",
                    severity="warning", timeout=10, markup=False)
        if error is None:
            return Listing(listing.jobs, ListScope(SCHEMA))
        self._update_subtitle()
        if self._loaded_at is None:
            self._refresh_jobs()  # nothing listed yet (a came first): read the configured schema's list
        return None

    def _merged(self, rows: list) -> list:
        """Reuse unchanged records and retain an UNKNOWN row's known creation time."""
        merged = []
        for job in rows:
            if not (isinstance(job, dict) and views.job_id_of(job)):
                continue
            old = self._jobs_by_key.get(self._job_key(job))
            if old is not None:
                if not job.get("created_at") and old.get("created_at"):
                    job = {**job, "created_at": old["created_at"]}
                if job == old:
                    job = old
            merged.append(job)
        return merged

    def job_fetched(self, job: dict) -> None:
        """Merge a detail screen's fresh job into the list."""
        if not isinstance(job, dict):
            return
        key = self._job_key(job)
        current = self._jobs_by_key.get(key)
        if current is None:
            return
        fresh = views.refresh_job(current, job)
        if fresh == current:
            return
        self._all_jobs = [fresh if j is current else j for j in self._all_jobs]
        self._jobs_by_key[key] = fresh

    # ─── rows ────────────────────────────────────────────────────────────────

    @staticmethod
    def _name_cell(job: dict, unfolded: bool | None = None) -> _FixedCell:
        """A plain job id with its optional fold marker."""
        parts = [(views.job_id_of(job), "grey50")]
        if unfolded is not None:
            parts.insert(0, ("▾ " if unfolded else "▸ ", "grey50"))
        return _FixedCell.assemble(*parts)

    @staticmethod
    def _time_cell(raw, compact: bool) -> _FixedCell:
        shown = format_created_at(raw)
        if shown and compact:
            shown = shown[5:]  # MM-DD HH:MM: the list spans six months, so the year is never in doubt
        return _cell(shown or "—")

    @staticmethod
    def _owner_cell(job: dict) -> _FixedCell:
        """The submitting user; a job that records none shows its owning role, dimmed and named a role."""
        user = views.collapse(job.get("submitted_by") or "")
        if user:
            return _cell(user)
        role = views.collapse(job.get("owner_role") or "")
        return _cell(f"role {role}", "grey62") if role else _cell("—")

    def _job_cells(self, job: dict, compact: bool) -> dict:
        """A job's row but for its Job ID: column label -> cell."""
        live = views.is_live(job)
        word = views.status_word(job.get("status"))
        unknown = views.is_unknown(job)
        status_style = {"running": "bold green", "failed": "bold red", "error": "bold red"}.get(
            word, "yellow" if live or unknown else "grey62")
        models = views.job_models(job)
        model = views.short_model(models[0]) if models else "—"
        if len(models) > 1:
            model += f" +{len(models) - 1}"
        if unknown:
            marker = _cell("?", "bold yellow")  # no live state: neither running nor ended, as far as known
        else:
            marker = _cell("●", "bold green") if live else _cell("○", "grey50")
        # Plain text throughout: DataTable parses a str cell as markup.
        return {
            " ": marker,
            "Status": _cell(word.upper(), status_style),
            "Type": _cell(views.job_type_label(job)),
            "Created": self._time_cell(job.get("created_at"), compact),
            "Owner": self._owner_cell(job),
            "Model": _cell(model),
            # Requested GPUs; "shared" for a job that runs only on another job's GPUs.
            "GPUs": _cell(views.job_gpu_cell(job, str(job.get("hardware") or ""))),
            # Plain text: a quoted name can hold anything, brackets included.
            "Location": _cell(views.location_label(job) or "—", "grey70"),
            "Updated": self._time_cell(job.get("updated_at"), compact),
        }

    @staticmethod
    def _row_or_note(job: dict, build) -> dict:
        """Build a row, isolating one malformed job from the rest of the list."""
        try:
            return build()
        except Exception:  # noqa: BLE001 - the traceback goes to the error log
            jid = views.job_id_of(job) if isinstance(job, dict) else "?"
            log_traceback(f"job list row for {jid}")
            return {"Job ID": _cell(jid, "grey50"), "Status": _cell("? can't show", "yellow")}

    @staticmethod
    def _sub_cells(sj: dict, last: bool) -> dict:
        word = views.status_word(sj.get("status"))
        style = {"running": "green", "failed": "red", "error": "red"}.get(word, "grey62")
        branch = "└─" if last else "├─"
        return {
            "Job ID": _cell(f"   {branch} {views.sub_job_label(sj)}", "grey70"),
            "Status": _cell(word.upper(), style),
            "Type": _cell(views.sub_job_type(sj), "grey70"),
            "Model": _cell(views.short_model(sj.get("model_name")) or "—", "grey70"),
            "GPUs": _cell(views.sub_job_gpu_text(sj), "grey70"),
        }

    def _sub_rows(self, job: dict) -> tuple:
        """An expanded job's status note and sub-job rows."""
        job_key = self._job_key(job)
        entries = views.sub_jobs(job)
        subs, keys = [], []
        if views.is_unknown(job):
            subs.append(("note", {"Job ID": _cell(f"   {'├─' if entries else '└─'} ? status unavailable", "yellow")}))
            keys.append(None)
        for si, sj in enumerate(entries):
            last = si == len(entries) - 1
            subs.append((str(si), self._row_or_note(job, lambda: self._sub_cells(sj, last))))
            keys.append((job_key, sj.get("sub_job_id")))
        return tuple(subs), tuple(keys)

    def _header(self, group: str, count: int, layout: _ListLayout) -> tuple:
        """A LIVE/HISTORY row, kept while it reads the same: its title in the Job ID column."""
        kept = self._headers.get(group)
        if kept is not None and kept[0] == (count, layout):
            return kept[1]
        style = "bold green" if group == "LIVE" else "bold grey62"
        cells = {"Job ID": _cell(f"{group} ({count})", style)}
        cells = tuple(cells.get(column, _BLANK) for column in layout.labels)
        self._headers[group] = ((count, layout), cells)
        return cells

    def _job_rows(self, job: dict, layout: _ListLayout) -> _JobRows:
        """Cached rows for a job, layout, and expansion state."""
        job_key = self._job_key(job)
        compact = layout.compact_times
        rows = self._rows.get(job_key)
        if not (rows is not None and rows.job is job and rows.compact == compact):
            rows = _JobRows(job, compact, self._row_or_note(job, lambda: self._job_cells(job, compact)))
            self._rows[job_key] = rows
        expanded = self._is_expanded(job)
        if rows.layout != layout or rows.expanded != expanded:
            unfolded = expanded if self._expandable(job) else None
            name = rows.cells.get("Job ID") or self._row_or_note(
                job, lambda: {"Job ID": self._name_cell(job, unfolded)})["Job ID"]
            labels = layout.labels
            cells = rows.cells
            rows.row = tuple(name if label == "Job ID" else cells.get(label, _BLANK) for label in labels)
            if expanded:
                if rows.sub_cells is None:
                    rows.sub_cells, rows.sub_keys = self._sub_rows(job)
                rows.subs = tuple((suffix, tuple(sub.get(label, _BLANK) for label in labels))
                                  for suffix, sub in rows.sub_cells)
            else:
                rows.subs = ()
            rows.layout, rows.expanded = layout, expanded
        return rows

    def _query(self) -> views.Query:
        """The resolved search query, cached with its input and job records."""
        kept = self._resolved
        if kept is not None and kept[0] == self._filter and kept[1] is self._all_jobs:
            return kept[2]
        query = views.resolve_query(views.parse_query(self._filter), self._all_jobs)
        self._resolved = (self._filter, self._all_jobs, query)
        return query

    def _haystack(self, job: dict) -> str:
        """``views.search_haystack`` of ``job``, kept while its record stays the same."""
        key = self._job_key(job)
        kept = self._hay.get(key)
        if kept is not None and kept[0] is job:
            return kept[1]
        hay = views.search_haystack(job)
        self._hay[key] = (job, hay)
        return hay

    # ─── the lines around the table ──────────────────────────────────────────

    def _show_notice(self) -> None:
        """Show a failed refresh above the retained list."""
        widget = self.query_one("#listnotice", Static)
        stale = bool(self._list_error and self._all_jobs)
        widget.display = stale
        self._notice = None
        if not stale:
            return
        self._notice = Content.from_markup("   [red]refresh failed $failed: $error; showing the list from $at[/red]",
                                           failed=self._failed_at or "", error=self._list_error or "",
                                           at=(self._last_refreshed or "—")[:5])
        self._paint_notice()

    def _paint_notice(self) -> None:
        if self._notice is None:
            return
        widget = self.query_one("#listnotice", Static)
        widget.update(hanging(self._notice, widget.size.width or self.size.width or self.app.size.width, 3))

    def _paint_info(self) -> None:
        if self._info is None:
            return
        widget = self.query_one("#listinfo", Static)
        width = widget.content_size.width or (self.size.width or self.app.size.width) - 2
        if isinstance(self._info, Content):
            widget.update(hanging(self._info, width, 2))
        else:
            widget.update(wrapped_parts(self._info, width, 2, Content.from_markup("   [dim]·[/dim]  ")))

    def _show_nomatch(self, message: str | None) -> None:
        """The message in place of the table (it takes the table's room, so the search box stays put)."""
        widget = self.query_one("#nomatch", Static)
        box = self.query_one("#nomatch-box", VerticalScroll)
        if message != self._nomatch:
            box.scroll_home(animate=False)  # a new message is read from its start
        self._nomatch = message
        widget.display = box.display = message is not None
        self._paint_nomatch()

    def _paint_nomatch(self) -> None:
        if self._nomatch is None:
            return
        widget = self.query_one("#nomatch", Static)
        width = widget.content_size.width
        if not width:
            width = (self.size.width or self.app.size.width) - 6
            self.call_after_refresh(self._repaint_nomatch, width)
        message = self._nomatch
        lines = [Content(line) for line in message.split("\n")]
        widget.update(Content("\n").join(hanging(line, width, 2) for line in lines))

    def _repaint_nomatch(self, painted_for: int) -> None:
        if self.query_one("#nomatch", Static).content_size.width not in (0, painted_for):
            self._paint_nomatch()

    def _older_matches(self, query) -> list:
        """Ids of the jobs older than six months (hidden) that ``query`` matches, newest first."""
        now = self._clock()
        home = self._clients.home
        found = [job for job in self._all_jobs if not views.within_retention(job, now)
                 and views.job_matches_query(job, query, home=home, hay=self._haystack(job))]
        return [views.job_id_of(j) for j in views.sort_job_list(found, "created", True)]

    # ─── the table ───────────────────────────────────────────────────────────

    def _render_jobs(self, *, keep: tuple | None = None, top: bool = False) -> None:
        """Draw the list, optionally keeping one job or starting at the top."""
        table = self.query_one("#jobs", DataTable)
        info_line = self.query_one("#listinfo", Static)
        keep_sub = None
        if keep is None and not top:
            key, keep = self._highlighted()
            keep_sub = self._sub_by_item.get(key)  # a sub-job row keeps its own cursor
        recent = self._recent()
        layout = self._layout_for(self._location_column_width(recent))
        self._show_notice()
        info_line.set_class(False, "error")
        if self._loaded_at is None and not self._list_error:
            self._info = Content.from_markup("  [dim]loading jobs…[/dim]")
            self._paint_info()
            self._hide_table(table, layout, "Loading jobs…")
            return
        if self._list_error and not self._all_jobs:
            info_line.set_class(True, "error")
            hint = f" · a lists {self._schema_label()} only" if self._list_refused else ""
            self._info = None
            info_line.update(Text(f"[error] {self._list_error}  (r retries{hint})", style="red"))
            self._hide_table(table, layout, "")  # an empty message keeps the search box at the bottom
            return
        hidden = len(self._all_jobs) - len(recent)
        query = self._query()
        home = self._clients.home
        if query.empty and not query.errors:
            visible = recent
        else:
            visible = [job for job in recent if views.job_matches_query(job, query, home=home, hay=self._haystack(job))]
        ordered = views.sort_job_list(visible, self._sort, self._descending)
        live = [j for j in ordered if views.is_live(j)]
        history = [j for j in ordered if not views.is_live(j)]

        parts = [Content.from_markup(f"[bold green]● {len(live):,} live[/]   [grey62]○ {len(history):,} historical[/]")]
        if self._filter:
            parts.append(Content.from_markup(f"[b]{len(visible):,}[/b] [dim]match(es) for[/dim] [b]$query[/b]",
                                             query=self._filter))
        sort = _sort_part(self._sort, self._descending)
        parts.append(sort + Content(" " * (_SORT_PART_CELLS - sort.cell_length)))
        self._info = parts
        self._paint_info()

        if not visible:
            one = self._scope is not None and self._scope.kind != ACCOUNT
            widen = self._scope is not None and self._scope.kind == SCHEMA
            if self._filter:
                found = dict(cut=loaded_text(self._scope), older=self._older_matches(query) if hidden else (),
                             location=self._schema_label() if one else "", widen=widen)
                message = views.no_match_message(self._filter, query, len(recent), **found)
            elif self._scope is not None and self._scope.truncated:
                message = empty_text(self._scope)
            elif self._scope is not None and self._scope.kind == ACCOUNT:
                message = "No jobs created in the last 6 months that this role can see, in any database and schema."
            else:
                message = f"No jobs created in the last 6 months in {self._schema_label()} that this role can see."
                if widen:
                    message += "\n  a lists every database and schema."
            self._hide_table(table, layout, message)
            return
        self._show_nomatch(None)
        was_hidden = not table.display
        table.display = True

        plan: list = []
        job_by_item: dict = {}
        sub_by_item: dict = {}
        target_row = None
        groups = (("LIVE", live), ("HISTORY", history))
        if self._sort == "status" and not self._descending:
            groups = groups[::-1]  # status ↑ ended first: the ended jobs are what comes first
        for group, jobs in groups:
            if not jobs:
                continue
            plan.append((f"hdr-{group.lower()}", self._header(group, len(jobs), layout)))
            for job in jobs:
                job_key = self._job_key(job)
                item_id = self._row_key(job_key)
                job_by_item[item_id] = job_key
                rows = self._job_rows(job, layout)
                plan.append((item_id, rows.row))
                if job_key == keep and target_row is None:
                    target_row = len(plan) - 1
                for (suffix, cells), sub_key in zip(rows.subs, rows.sub_keys):
                    if sub_key is None:
                        plan.append((f"note-{item_id}", cells))
                        continue
                    row_key = f"sub-{item_id}-{suffix}"
                    sub_by_item[row_key] = sub_key
                    plan.append((row_key, cells))
                    if keep_sub is not None and sub_key == keep_sub:
                        target_row = len(plan) - 1
        self._job_by_item = job_by_item
        self._sub_by_item = sub_by_item
        # Reuse identical row objects, or reorder them without rebuilding.
        shown = self._shown
        same = (shown is not None and shown[0] == layout and len(shown[1]) == len(plan)
                and table.row_count == len(plan)
                and all(a[0] == b[0] and a[1] is b[1] for a, b in zip(shown[1], plan)))
        if not same:
            if (shown is not None and shown[0] == layout and table.row_count == len(plan)
                    and {key: id(cells) for key, cells in shown[1]} == {key: id(cells) for key, cells in plan}):
                at = layout.labels.index("Job ID")
                position = {id(cells[at]): index for index, (_key, cells) in enumerate(plan)}
                table.sort("Job ID", key=lambda name: position[id(name)])
            else:
                self._set_columns(table, layout)
                table.replace_rows(plan)
            self._shown = (layout, tuple(plan))
        if target_row is None:
            target_row = 1  # first job row (row 0 is a group header)
        target_row = min(target_row, table.row_count - 1)
        if was_hidden:
            # Place again after layout without initially scrolling off the group header.
            table.move_cursor(row=target_row, animate=False, scroll=False)
            after_refresh(self, self._place_cursor, target_row)
        elif not same or top or table.cursor_row != target_row:
            self._place_cursor(target_row)

    def _row_key(self, job_key: tuple) -> str:
        """A stable table key for this screen."""
        key = self._keys.get(job_key)
        if key is None:
            key = self._keys[job_key] = f"job-{len(self._keys)}"
        return key

    def _hide_table(self, table: DataTable, layout: _ListLayout, message: str) -> None:
        """The table off, with ``message`` in its place."""
        self._set_columns(table, layout)
        table.clear()
        self._shown = None
        self._job_by_item, self._sub_by_item = {}, {}
        table.display = False
        self._show_nomatch(message)

    def _schema_label(self) -> str:
        names = [getattr(self._client, attr, None) for attr in ("database", "schema")]
        if all(isinstance(n, str) and n for n in names):
            return ".".join(names)
        return "the configured schema"

    def _place_cursor(self, row: int, tries: int = 20) -> None:
        table = self.query_one("#jobs", DataTable)
        if table.size.height == 0 and tries:
            after_refresh(self, self._place_cursor, row, tries - 1)  # not laid out yet
            return
        table.move_cursor(row=row, animate=False)
        if row <= 1:
            table.scroll_home(animate=False)  # keep the group header above the first job

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        from cortex_training.tui.screens import JobSummaryScreen

        if not is_shown(self):
            return  # a second quick Enter: the first already opened the job
        key = event.row_key.value
        if str(key).startswith("note-"):  # an UNKNOWN job's note line stands for its job
            key = str(key).removeprefix("note-")
        if key in self._job_by_item:
            job = self._jobs_by_key.get(self._job_by_item[key])
            # Every call about the job goes to its own database and schema.
            client = self._job_client(job)
            if client is not None:
                jid = views.job_id_of(job)
                self.app.push_screen(JobSummaryScreen(client, jid, job=job,
                                                      poll_interval=self._poll_interval, on_job=self.job_fetched))
        elif key in self._sub_by_item:
            job_key, sid = self._sub_by_item[key]
            job = self._jobs_by_key.get(job_key)
            client = self._job_client(job)
            if client is not None:
                jid = views.job_id_of(job)
                self.app.push_screen(LogScreen(client, jid, job=job, sub_job_id=sid, poll_interval=self._poll_interval))

    def _job_client(self, job: dict | None):
        """The job's location-bound client, or None after notifying."""
        try:
            return self._clients.for_job(job)
        except Exception as exc:  # noqa: BLE001 - shown, with the traceback in the error log
            log_traceback(f"no client for {views.location_label(job)}")
            self.notify(f"Can't open the job: no client for {views.location_label(job)} "
                        f"({views.describe_error(exc)}; details in {_ERROR_LOG}).", severity="error", markup=False)
            return None
