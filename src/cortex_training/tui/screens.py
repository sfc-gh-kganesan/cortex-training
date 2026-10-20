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

"""Read-only job preview between the list and Logs/Resources details."""

from __future__ import annotations

import time
from datetime import datetime
from datetime import timezone
from time import monotonic

from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.content import Content
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import DataTable
from textual.widgets import Footer
from textual.widgets import Header
from textual.widgets import Static

from cortex_training.tui import views as v
from cortex_training.tui.widgets import ENDED_STYLE
from cortex_training.tui.widgets import KEYS_BINDING
from cortex_training.tui.widgets import FollowsPointer
from cortex_training.tui.logs import LogScreen
from cortex_training.tui.widgets import SelectableTable
from cortex_training.tui.widgets import after_refresh
from cortex_training.tui.widgets import call_on_screen
from cortex_training.tui.widgets import close_keys_panel
from cortex_training.tui.widgets import hanging
from cortex_training.tui.widgets import is_shown

STATUS_STYLE = {
    "running": "bold green",
    "initializing": "yellow",
    "placing": "yellow",
    "pending": "yellow",
    "queued": "yellow",
    "creating": "yellow",
    "failed": "bold red",
    "error": "bold red",
    "cancelled": ENDED_STYLE,
    "canceled": ENDED_STYLE,
    "terminated": ENDED_STYLE,
    "done": ENDED_STYLE,
    "succeeded": ENDED_STYLE,
    "completed": ENDED_STYLE,
    "unknown": "yellow",
}
# What a waiting job is doing, by state, when the service sends no reason_details.
_WAITING = {
    "pending": "waiting to be placed on GPUs",
    "queued": "waiting to be placed on GPUs",
    "creating": "waiting to be placed on GPUs",
    "placing": "waiting for its GPUs to come up",
    "initializing": "placed on GPUs; loading the model and starting the engines",
}


def status_text(status) -> Text:
    word = v.status_word(status)
    return Text(word.upper(), style=STATUS_STYLE.get(word, ""))


_SUBJOB_COLUMNS = ("Sub-job", "Type", "Status", "Model", "GPUs", "Config")
_TYPE, _MODEL, _CONFIG = 1, 3, 5
# Columns dropped and minimum widths tried before horizontal scrolling.
_TABLE_FITS = (
    ((), ((_MODEL, 36), (_CONFIG, 24))),
    ((_TYPE,), ((_MODEL, 36), (_CONFIG, 24), (_MODEL, 10), (_CONFIG, 12))),
)


def _column_widths(labels, rows: list) -> list:
    """Each column's width with nothing cut: its widest cell, or its label."""
    return [max([v.cell_width(label)] + [cells[i].cell_len for _key, cells in rows]) for i, label in enumerate(labels)]


def fit_columns(labels, rows: list, width: int, pad: int = 2) -> tuple:
    """Return fitted labels/rows, whether cut, and whether scrolling remains."""
    natural = _column_widths(labels, rows)
    for dropped, cuts in _TABLE_FITS:
        keep = [i for i in range(len(labels)) if i not in dropped]
        widths = {i: natural[i] for i in keep}
        over = sum(widths[i] + pad for i in keep) - width
        for column, least in cuts:
            if over <= 0:
                break
            take = min(over, max(0, widths[column] - least))
            widths[column] -= take
            over -= take
        if over <= 0:
            break
    cut = len(keep) < len(labels) or any(widths[i] < natural[i] for i in keep)
    fitted = [(key, [_fit_text(cells[i], widths[i]) for i in keep]) for key, cells in rows]
    return [labels[i] for i in keep], fitted, cut, over > 0


def _fit_text(text: Text, width: int) -> Text:
    """A one-line cell cut to ``width`` terminal cells, ending in ``…`` when cut
    (never in ``·…``), its style kept."""
    if text.cell_len <= width:
        return text
    cut = v.fit_cells(text.plain, width)
    return Text(cut[:-1].rstrip(" ·") + "…", style=text.style)


# ─── job preview ─────────────────────────────────────────────────────────────


class JobSummaryScreen(FollowsPointer, Screen):
    """Preview of one job: identity, config, status, and a sub-job breakdown."""

    BINDINGS = [
        ("b", "back", "Back"),
        Binding("escape", "close_or_back", "Back", show=False),
        ("q", "app.quit", "Quit"),
        KEYS_BINDING,
        ("l", "open('logs')", "Logs"),
        ("g", "open('resource')", "Resources"),
        ("r", "refresh", "Refresh"),
        # check_action exposes the applicable expansion binding.
        Binding("e", "expand_values", "Full text"),
        Binding("e", "collapse_values", "Less text"),
    ]

    def __init__(self, client, job_id: str, *, job: dict | None = None,
                 poll_interval: float = 1.0, on_job=None) -> None:
        """``on_job(record)`` is called with the job each time a fresh read of it
        arrives (the job list then shows that status too)."""
        super().__init__()
        self._client = client
        self._job_id = job_id
        self._job = job or {"job_id": job_id}
        self._on_job = on_job
        self._poll_interval = poll_interval
        self._loading = False
        self._fetch_since: float | None = None  # monotonic start of the get_job in flight
        self._fetched_at: str | None = None  # local HH:MM:SS of the last good get_job answer
        self._fetch_error: str | None = None  # why the last get_job gave no copy, until one does
        self._scope_by_row: dict = {}
        self._reread_on_resume = False  # set when the details open over this preview
        self._expanded = False  # e: every value whole, however long, and the whole sub-job table
        self._long = False  # some value or table cell is (or, expanded, would be) cut
        self._table_shown = None  # what the sub-job table holds, so an unchanged refresh leaves it be

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="summary-body"):
            yield Static("", id="preview")
            yield Static("", id="fetch")  # reading, refreshing, or why the copy shown is older
            yield Static("", id="health")
            yield Static("", id="subjobs-title")  # says what a click on a row opens
            yield SelectableTable(id="subjobs", cursor_type="row", zebra_stripes=True)  # its text highlights to copy
        yield Footer()

    _REFRESH_SECONDS = 60

    def on_mount(self) -> None:
        table = self.query_one("#subjobs", DataTable)
        self._refresh_view()
        self._start_fetch()
        self.set_focus(table, scroll_visible=False)
        self.set_interval(self._REFRESH_SECONDS, self._refresh_tick)
        self.set_interval(1, self._fetch_tick)

    def _refresh_tick(self) -> None:
        if self.app.screen is self and v.is_live(self._job):
            self._start_fetch()

    def on_screen_resume(self) -> None:
        self._refresh_view()
        # Back from the details, which may have seen the status move on: re-read the job now.
        reread, self._reread_on_resume = self._reread_on_resume, False
        if reread and v.is_live(self._job):
            self._start_fetch()

    _LABEL_WIDTH = 15  # "  " + a 12-column label + " "

    def _value_width(self) -> int:
        """Columns left for a preview value after its label (100 before the first layout)."""
        try:
            width = self.query_one("#preview", Static).size.width
        except Exception:  # noqa: BLE001 - not mounted yet
            width = 0
        return max(20, (width or 100) - self._LABEL_WIDTH)

    def _value_lines(self, value: str) -> list:
        """A printable preview value wrapped to its cell-aware column."""
        width = self._value_width()
        return [row for line in v.printable(value).split("\n") for row in v.wrap_cells(line, width)]

    def _cut_rows(self, rows: list, keep: int) -> list:
        """The first ``keep`` rows, the last ending in ``…``: the value goes on."""
        kept = rows[:keep]
        kept[-1] = v.fit_cells(f"{kept[-1].rstrip()} …", self._value_width())
        return kept

    def _value_block(self, rows: list) -> str:
        """Rows of a value, the continuation rows under the value column."""
        return ("\n" + " " * self._LABEL_WIDTH).join(rows)

    _VALUE_ROWS = 3
    _MORE = "\n" + " " * _LABEL_WIDTH + "[dim](+$more rows · e shows the whole {what})[/dim]"

    def check_action(self, action: str, parameters) -> bool | None:
        if action == "expand_values":
            return self._long and not self._expanded
        if action == "collapse_values":
            return self._long and self._expanded
        return True

    def action_expand_values(self) -> None:
        self._expanded = True
        self._refresh_view()
        self.refresh_bindings()

    def action_collapse_values(self) -> None:
        self._expanded = False
        self._refresh_view()
        self.refresh_bindings()

    def _refresh_view(self) -> None:
        job = self._job
        if self.app.screen is self:  # a late load must not retitle the details on top
            self.app.sub_title = f"job preview  ·  {v.job_id_of(job)}"
        now = datetime.now(timezone.utc)
        rows = []
        long = False
        for label, value in v.summary_fields(job, now):
            more = 0  # rows of the value left out
            if label == "Status":
                word = v.status_word(job.get("status"))
                style = STATUS_STYLE.get(word, "")
                if v.is_unknown(job):  # no live state: the health line below says why
                    live = "  [yellow]? status unavailable[/yellow]"
                else:
                    live = "  [green]● live[/green]" if v.is_live(job) else "  [dim]historical[/dim]"
                value_markup = (f"[{style}]$value[/]" if style else "$value") + live
            else:
                lines = self._value_lines(value)
                if len(lines) > self._VALUE_ROWS:
                    long = True
                    if not self._expanded:
                        more = len(lines) - self._VALUE_ROWS
                        lines = self._cut_rows(lines, self._VALUE_ROWS)
                value = self._value_block(lines)
                value_markup = (("[b]$value[/b]" if label == "Job ID" else "$value")
                                + (self._MORE.format(what="value") if more else ""))
            rows.append(Content.from_markup(f"  [dim]{label:<12}[/dim] {value_markup}", value=value, more=str(more)))
        self._render_fetch()
        width = self._value_width() + self._LABEL_WIDTH
        health = hanging(self._health(), width, 4)
        self.query_one("#health", Static).update(health)

        table = self.query_one("#subjobs", DataTable)
        subjob_rows = self._subjob_rows(job)
        table_width = self._table_width(table, len(subjob_rows))
        labels, fitted, cut, wider = fit_columns(_SUBJOB_COLUMNS, subjob_rows, table_width, 2 * table.cell_padding)
        if self._expanded:  # every column, every cell whole: the table scrolls sideways when it must
            labels, fitted = list(_SUBJOB_COLUMNS), subjob_rows
            wider = sum(w + 2 * table.cell_padding for w in _column_widths(labels, fitted)) > table_width
        long = long or cut
        self._show_table(table, labels, fitted, wider)
        hints = ["Click a job to open its telemetry dashboard"]
        if cut:
            hints.append("e fit to screen" if self._expanded else "e whole table")
        if wider:
            hints.append("← → scroll")
        title = self._subjobs_title(hints, width)
        self.query_one("#subjobs-title", Static).update(title)
        self.query_one("#preview", Static).update(Content("\n").join(rows))
        if long != self._long:
            self._long = long
            self.refresh_bindings()  # e appears in the footer, or goes

    _TITLE = "Sub-jobs  ("

    def _subjobs_title(self, hints: list, width: int) -> Content:
        """Fit ``Sub-jobs  (hint · hint)`` with hanging continuation lines."""
        indent = v.cell_width(self._TITLE)
        rows = [self._TITLE + hints[0]]
        for hint in hints[1:]:
            if v.cell_width(rows[-1]) + v.cell_width(f" · {hint})") <= width:
                rows[-1] += f" · {hint}"
            else:
                rows.append(" " * indent + hint)
        rows[-1] += ")"
        text = Content("\n").join(hanging(Content(row), width, indent) for row in rows).plain
        bold = len("Sub-jobs")
        return Content.styled(text[:bold], "bold") + Content.styled(text[bold:], "dim")

    def _show_table(self, table: DataTable, labels: list, rows: list, wider: bool) -> None:
        """Update changed table rows while retaining cursor and horizontal scroll."""
        shown = (tuple(labels), wider,
                 tuple((key, tuple((cell.plain, str(cell.style)) for cell in cells)) for key, cells in rows))
        if shown == self._table_shown:
            return
        self._table_shown = shown
        # Updating overflow forces Textual to re-check its horizontal scrollbar.
        table.styles.overflow_x = "auto" if wider else "hidden"
        keep = table.cursor_row
        table.clear(columns=True)
        table.add_columns(*labels)
        for key, cells in rows:
            table.add_row(*cells, key=key)
        table.move_cursor(row=min(keep, table.row_count - 1), animate=False)  # keep the row on refresh

    def _subjob_rows(self, job: dict) -> list:
        """The sub-job table's rows as ``(key, cells)``; also maps each row to its scope."""
        self._scope_by_row = {"all": None}
        rows = [("all", [Text("▸ entire job", style="bold"), Text(v.job_type_label(job)), status_text(job.get("status")),
                         Text(", ".join(v.job_models(job)) or "—"), Text(v.job_gpu_cell(job, "requested")),
                         Text("all sub-jobs", style="dim")])]
        subs = v.sub_jobs(job)
        if v.is_unknown(job):
            rows.append(("note", [Text("  ? status unavailable", style="yellow"), Text(""), Text("UNKNOWN", style="yellow"),
                                  Text(""), Text(""), Text("configs at creation" if subs else
                                                           "the service sent no sub-job configs", style="dim")]))
        for i, sj in enumerate(subs):
            key = f"sj-{i}"
            self._scope_by_row[key] = sj.get("sub_job_id")
            config = v.subjob_config_summary(sj)
            target = v.multiplex_target(sj)
            if target:
                config = " · ".join(p for p in (config, f"runs on {v.short_sub_job_id(target)}") if p)
            rows.append((key, [Text(f"  {v.sub_job_label(sj)}"), Text(v.sub_job_type(sj)),
                               status_text(sj.get("status")), Text(str(sj.get("model_name") or "—")),
                               Text(v.sub_job_gpu_text(sj)), Text(config)]))
        return rows

    def _table_width(self, table: DataTable, rows: int) -> int:
        """Available table cells, accounting for a needed vertical scrollbar."""
        width = table.size.width or max(40, self.app.size.width - 6)
        limit = table.styles.max_height
        if limit is not None and limit.is_cells and rows + 1 > limit.value:
            width -= table.styles.scrollbar_size_vertical
        return width

    def on_resize(self, event) -> None:
        if self.query_one("#subjobs", DataTable).columns:  # mounted: re-wrap the values, refit the table
            after_refresh(self, self._refresh_view)

    _TABLE_MOVES = {"up", "down", "pageup", "pagedown", "ctrl+home", "ctrl+end"}

    def on_key(self, event) -> None:
        table = self.query_one("#subjobs", DataTable)
        if self.focused is not table or event.key not in self._TABLE_MOVES:
            return
        body = self.query_one("#summary-body", VerticalScroll)
        if event.key == "ctrl+home" or (event.key == "up" and table.cursor_row <= 0):
            body.scroll_home(animate=False)
        else:
            after_refresh(self, body.scroll_to_widget, table, animate=False)

    def _health(self) -> Content:
        job = self._job
        word = v.status_word(job.get("status"))
        reason, details = v.collapse(job.get("reason") or ""), v.collapse(job.get("reason_details") or "")
        if word == "failed" and not reason:
            reason = "not reported"  # a failure always gets its reason line
        why = ("  reason: [b]$reason[/b]" if reason else "") + ((" — $details" if reason else "  $details")
                                                                 if details else "")

        def line(lead: str, tail: str = "", **values) -> Content:
            return Content.from_markup(f"  {lead}{why}{tail}", reason=reason, details=details, **values)

        if word == "failed":
            hint = ("it never got GPUs, so no sub-job started or wrote logs" if v.never_started(job)
                    else "check Logs (l) for the failing sub-job")
            return line("[bold red]✖ Job failed[/]", "  — $hint", hint=hint)
        if word in ("cancelled", "canceled"):
            return line(f"[{ENDED_STYLE}]■ Cancelled[/]")
        if word == "terminated":
            return line(f"[{ENDED_STYLE}]■ Terminated by the platform[/]")
        if word in ("succeeded", "completed", "done"):
            return line(f"[{ENDED_STYLE}]■ Completed[/]")
        if word in _WAITING:
            return line(f"[yellow]◔ {word.capitalize()}[/]", "" if details else " — $doing", doing=_WAITING[word])
        if word == "running":
            return Content.from_markup(
                "  [green]● Running[/] — logs stream live; resources refresh automatically")
        if word == "unknown":
            text = v.status_unavailable_text(job)
            hint = "" if "not found" in text.lower() else " — the job may still be running"
            return Content.from_markup(f"  [yellow]? $text[/]{hint}", text=text)
        # A status this client doesn't know: the service's reason, if it sent one.
        return Content.from_markup("  " + why.lstrip(), reason=reason, details=details) if why else Content("")

    def _start_fetch(self) -> None:
        if not self._loading:
            self._loading = True
            self._fetch_since = monotonic()
            self._fetch_job()
        self._render_fetch()  # "refreshing…", or how long the read in flight has taken

    @work(thread=True, group="summary")
    def _fetch_job(self) -> None:
        try:
            job, error = self._client.get_job(self._job_id), None
        except Exception as exc:  # noqa: BLE001 - said under the fields; the copy on screen stays
            job, error = None, v.collapse(v.describe_error(exc))
        if not (isinstance(job, dict) and job):
            job, error = None, error or "the service sent an empty answer"
        call_on_screen(self, self._apply_job, job, error)

    def _apply_job(self, job: dict | None, error: str | None = None) -> None:
        self._loading = False
        self._fetch_since = None
        if job:
            # The service's fields come only from the fresh answer: a reason it stopped sending disappears.
            self._job = v.refresh_job(self._job, job)
            self._fetched_at, self._fetch_error = time.strftime("%H:%M:%S"), None
            if self._on_job is not None:
                self._on_job(self._job)
        else:
            self._fetch_error = error or "the service sent an empty answer"
        self._refresh_view()

    # A read in flight longer than this shows how long it has taken.
    _FETCH_SLOW_SECONDS = 5

    def _fetch_tick(self) -> None:
        if self._loading and self.app.screen is self:
            self._render_fetch()

    def _fetch_line(self) -> Content:
        """The line under the fields: a read in flight, the last one's failure, or when the copy was read."""
        took = ""
        if self._loading and self._fetch_since is not None:
            seconds = int(monotonic() - self._fetch_since)
            took = f" ({seconds}s)" if seconds >= self._FETCH_SLOW_SECONDS else ""
        error, at = self._fetch_error or "", self._fetched_at or ""
        if self._loading:
            if error:
                return Content.from_markup(f"  [dim]refreshing…{took}[/dim]  [yellow](last try: $error)[/yellow]",
                                           error=error)
            if at:
                return Content.from_markup(f"  [dim]refreshing…{took}  (last read at $at)[/dim]", at=at)
            return Content.from_markup(f"  [dim]reading the job from the service…{took}[/dim]")
        if error:
            # The fields above are the last good copy: this preview's, or the job list's.
            if at:
                return Content.from_markup("  [yellow]couldn't refresh: $error; showing the copy read at $at[/yellow]",
                                           error=error, at=at)
            return Content.from_markup("  [yellow]couldn't read the job: $error; showing the job list's copy[/yellow]",
                                       error=error)
        if at:
            return Content.from_markup("  [dim]read from the service at $at[/dim]", at=at)
        return Content("")

    def _render_fetch(self) -> None:
        try:
            line = self.query_one("#fetch", Static)
        except Exception:  # noqa: BLE001 - not mounted, or already removed
            return
        content = self._fetch_line()
        line.display = bool(content.plain)  # no blank row before the first read
        # A long error continues under its text, after the two-cell indent.
        line.update(hanging(content, self._value_width() + self._LABEL_WIDTH, 4))

    def action_refresh(self) -> None:
        if self._loading:
            # The read is in flight (a thread can't be cancelled): say so, not nothing.
            self.notify("Still reading this job; r asks again once it answers.")
            return
        self._start_fetch()

    def _highlighted_scope(self):
        table = self.query_one("#subjobs", DataTable)
        try:
            key = table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value
        except Exception:  # noqa: BLE001 - empty table
            key = "all"
        return self._scope_by_row.get(key)

    def whole_id(self, key) -> str | None:
        """The whole id of the sub-job on row ``key``, the job's on the "all" row."""
        return self._scope_by_row.get(key) or (self._job_id if key == "all" else None)

    def action_copy_text(self) -> None:
        """ctrl+c: what the mouse highlighted, else the whole preview."""
        if not self.app.copy_highlight(self):
            self.app.copy_everything(self, "the preview")

    def action_back(self) -> None:
        if is_shown(self):  # not while still mounting, nor the screen under a pushed one
            self.app.pop_screen()

    def action_close_or_back(self) -> None:
        """Esc: close the keys panel when it is open, else go back."""
        if is_shown(self) and not close_keys_panel(self):
            self.app.pop_screen()

    def action_open(self, tab: str) -> None:
        if is_shown(self):
            self._open_detail(self._highlighted_scope(), tab)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        # Not a second quick Enter, after the first opened the details.
        if is_shown(self) and event.row_key.value in self._scope_by_row:  # not the "status unavailable" line
            self._open_detail(self._scope_by_row[event.row_key.value], "logs")

    def _open_detail(self, sub_job_id, tab: str) -> None:
        self._reread_on_resume = True
        self.app.push_screen(
            LogScreen(self._client, self._job_id, job=self._job, sub_job_id=sub_job_id, initial_tab=tab,
                      poll_interval=self._poll_interval)
        )
