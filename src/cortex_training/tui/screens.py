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

"""Read-only job preview between the list and the Logs/Resources/Checkpoints details."""

from __future__ import annotations

import re
import time
from datetime import datetime
from datetime import timezone
from time import monotonic

from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.content import Content
from textual.containers import Vertical
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import DataTable
from textual.widgets import Footer
from textual.widgets import Header
from textual.widgets import Static

from cortex_training.tui import theme
from cortex_training.tui import views as v
from cortex_training.tui.widgets import KEYS_BINDING
from cortex_training.tui.widgets import FollowsPointer
from cortex_training.tui.logs import LogScreen
from cortex_training.tui.widgets import SelectableTable
from cortex_training.tui.widgets import after_refresh
from cortex_training.tui.widgets import call_on_screen
from cortex_training.tui.widgets import close_keys_panel
from cortex_training.tui.widgets import hanging
from cortex_training.tui.widgets import is_shown

def status_text(status, word=None) -> Text:
    """A status as the sub-job table shows it, a badge: ``● RUNNING``, in the
    theme's glyph and color for it (:func:`~cortex_training.tui.theme.status_badge`).
    ``word`` overrides the word shown (a deployment's own status words), colors
    still come from ``status`` itself."""
    word = word if word is not None else v.status_word(status)
    look = theme.status_look(status)
    return Text.assemble((f"{look.glyph} ", look.color), (word.upper(), look.style)) if word != "?" else Text("?")


# The highlighted sub-job's config has a card of its own, under the table.
_SUBJOB_COLUMNS = ("Sub-job", "Type", "Status", "Model", "GPUs")
_TYPE, _MODEL = 1, 3
# Columns dropped and minimum widths tried before horizontal scrolling.
_TABLE_FITS = (
    ((), ((_MODEL, 36),)),
    ((_TYPE,), ((_MODEL, 36), (_MODEL, 10))),
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
    """Preview of one job: identity, status, a sub-job breakdown, and the
    highlighted sub-job's config."""

    BINDINGS = [
        ("b", "back", "Back"),
        Binding("escape", "close_or_back", "Back", show=False),
        ("q", "app.quit", "Quit"),
        KEYS_BINDING,
        ("l", "open('logs')", "Logs"),
        ("g", "open('resource')", "Resources"),
        ("m", "open('training')", "Training metrics"),
        ("c", "open('checkpoints')", "Checkpoints"),
        ("v", "open('serving')", "Serving"),
        ("r", "refresh", "Refresh"),
        # check_action exposes the applicable expansion binding.
        Binding("e", "expand_values", "Full text"),
        Binding("e", "collapse_values", "Less text"),
    ]

    # Cards: the job (its head, facts and lifecycle), the callout (its border
    # colored by _render_health), the sub-jobs and the highlighted row's config
    # (the job card's border and padding); _inner_width counts the cells
    # their borders and paddings take. The scrollbar's room is kept while the
    # preview fits, so its text never changes width (and re-wraps at the edge)
    # when a refresh makes it scroll.
    DEFAULT_CSS = """
    #summary-body { padding: 0 1; scrollbar-gutter: stable; scrollbar-size-vertical: 1; }
    #job-card { height: auto; border: round $border; padding: 0 1; }
    #hero { height: auto; margin-bottom: 1; }
    #preview, #fetch { height: auto; }
    #health { height: auto; margin-top: 1; padding: 0 1; border-title-align: left; }
    #subjobs-card { height: auto; margin-top: 1; border: round $border; padding: 0 1;
                    border-title-align: left; border-subtitle-align: right; }
    #subjobs { height: auto; max-height: 12; }
    #subjobs > .datatable--header { background: $surface; color: $text-muted; }
    #config-card { height: auto; margin-top: 1; border: round $border; padding: 0 1;
                   border-title-align: left; border-subtitle-align: right; }
    #config { height: auto; }
    """

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
        self._sub_job_by_row: dict = {}  # each sub-job row's entry, for the config card
        self._experiment: str | None = None  # the job's experiment, for the config card, once read
        self._experiment_error: str | None = None  # why the last try to read it failed
        self._reread_on_resume = False  # set when the details open over this preview
        self._expanded = False  # e: every value whole, however long, and the whole sub-job table
        self._long = False  # some value or table cell is (or, expanded, would be) cut
        self._footer_keys: tuple | None = None  # (long, is a deployment) the footer was last built for
        self._table_shown = None  # what the sub-job table holds, so an unchanged refresh leaves it be

    def compose(self) -> ComposeResult:
        yield Header(icon=theme.BRAND_ICON)
        with VerticalScroll(id="summary-body"):
            with Vertical(id="job-card"):
                yield Static("", id="hero")  # the job id, status pill, elapsed time and next steps
                yield Static("", id="preview")  # the facts and the lifecycle
                yield Static("", id="fetch")  # reading, refreshing, or why the copy shown is older
            yield Static("", id="health")  # why the job is where it is
            with Vertical(id="subjobs-card"):  # its border says what Enter on a row opens
                yield SelectableTable(id="subjobs", cursor_type="row")  # its text highlights to copy
            with Vertical(id="config-card"):  # its title names whose config it is
                yield Static("", id="config")  # the highlighted row's config, key by key
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

    # The grid's label column: a 12-cell label and a space. Its values wrap
    # under where they start.
    _LABEL_WIDTH = 13
    _GUTTER = 4  # between the grid's two columns
    # Two columns of facts from this many cells inside the card: each value then
    # has at least 28 cells ("16 × B200 requested", a model name).
    _TWO_COLUMNS_FROM = 2 * (_LABEL_WIDTH + 28) + _GUTTER
    _TITLE_LEAST = 12  # the fewest cells the job id is cut to, to keep the status beside it

    def _inner_width(self) -> int:
        """The cells inside the job card (the preview's width); before the first
        layout, the screen's less the body's padding and scrollbar gutter and the
        card's border and padding (this screen's CSS)."""
        try:
            width = self.query_one("#preview", Static).size.width
        except Exception:  # noqa: BLE001 - not mounted yet
            width = 0
        return max(30, width or self.app.size.width - 7)

    def _value_lines(self, value: str, width: int) -> list:
        """A printable preview value wrapped to ``width`` cells, by terminal cells."""
        return [row for line in v.printable(value).split("\n") for row in v.wrap_cells(line, max(10, width))]

    def _cut_rows(self, rows: list, keep: int, width: int) -> list:
        """The first ``keep`` rows, the last ending in ``…``: the value goes on."""
        kept = rows[:keep]
        kept[-1] = v.fit_cells(f"{kept[-1].rstrip()} …", width)
        return kept

    _VALUE_ROWS = 3

    def check_action(self, action: str, parameters) -> bool | None:
        if action == "expand_values":
            return self._long and not self._expanded
        if action == "collapse_values":
            return self._long and self._expanded
        if action == "open" and parameters:
            return parameters[0] in v.job_tabs(self._job)
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
        width = self._inner_width()
        # Every value (ids, the service's text) goes in as a styled run, never as
        # markup, so it shows literally. Each wrapped row's indent is a run of its
        # own too, so the text after it starts where the indent ends.
        self.query_one("#hero", Static).update(Content("\n").join(self._hero(width, now)))
        base_url = self._client.model_deployment_base_url if v.is_deployment(job) else ""
        grid, long = self._grid(v.preview_facts(job, now, base_url=base_url), width)
        self.query_one("#preview", Static).update(Content("\n").join(grid + self._lifecycle_rows(now, width)))
        self._render_fetch()
        self._render_health()

        table = self.query_one("#subjobs", DataTable)
        subjob_rows = self._subjob_rows(job)
        table_width = self._table_width(table, len(subjob_rows))
        labels, fitted, cut, wider = fit_columns(_SUBJOB_COLUMNS, subjob_rows, table_width, 2 * table.cell_padding)
        if self._expanded:  # every column, every cell whole: the table scrolls sideways when it must
            labels, fitted = list(_SUBJOB_COLUMNS), subjob_rows
            wider = sum(w + 2 * table.cell_padding for w in _column_widths(labels, fitted)) > table_width
        self._show_table(table, labels, fitted, wider)
        long = self._render_config(width) or long or cut
        # What Enter (or a click on the highlighted row) opens, then the keys for a
        # table that is cut to fit, or wider than the screen: on the card's border,
        # so they take no row of their own.
        hints = ["⏎ opens its Logs"]
        if cut:
            hints.append("e fit to screen" if self._expanded else "e whole table")
        if wider:
            hints.append("← → scroll")
        card = self.query_one("#subjobs-card")
        card.border_title = Content.styled(" Sub-jobs ", f"bold {theme.TEXT}")
        card.border_subtitle = Content.styled(f" {' · '.join(hints)} ", theme.MUTED)
        self._long = long
        keys = (long, v.is_deployment(self._job))
        if keys != self._footer_keys:
            # e comes or goes, or a job opened by id turned out to be a deployment.
            self._footer_keys = keys
            self.refresh_bindings()

    # The next steps from here, under the title: the footer has them too, but it
    # cuts its end on a narrow screen, and these say what each one opens: the tabs, in the job screen's order.
    _ACTIONS = (("serving", "v", "Serving"), ("logs", "l", "Logs"), ("resource", "g", "Resources"),
                ("training", "m", "Training metrics"), ("checkpoints", "c", "Checkpoints"))

    def _actions(self) -> tuple:
        tabs = v.job_tabs(self._job)
        return tuple((key, label) for tab, key, label in self._ACTIONS if tab in tabs)

    def _hero(self, width: int, now: datetime) -> list:
        """The card's head, as rows: the job id, its status pill and how long it
        has been going; then the next steps' keys."""
        job = self._job
        word = v.status_word_for(job, job.get("status")) if v.is_deployment(job) else None
        pill = theme.status_pill(job.get("status"), word=word)
        elapsed = v.elapsed_text(job, now)
        tail = Content.assemble("   ", pill, *(("   ", (elapsed, theme.MUTED)) if elapsed else ()))
        rows = []
        # The id gives way first, down to a few cells; then the time goes under it.
        if elapsed and width - tail.cell_length < self._TITLE_LEAST:
            tail = Content.assemble("   ", pill)
            rows.append(Content.styled(elapsed, theme.MUTED))
        title = v.fit_cells(v.job_id_of(job) or self._job_id, max(self._TITLE_LEAST, width - tail.cell_length))
        rows.insert(0, Content.styled(title, f"bold {theme.TEXT}") + tail)
        actions = Content("   ").join(
            Content.assemble((key, f"bold {theme.ACCENT}"), (f" {label}", theme.MUTED))
            for key, label in self._actions())
        return [*rows, actions]

    def _fact_rows(self, label: str, value: str, width: int) -> tuple:
        """``(rows, long)``: a fact in ``width`` cells, its label, then its value
        wrapped under the value column and, without e, cut after a few rows with a
        note saying so; and whether it is too long to show whole."""
        value_width = max(10, width - self._LABEL_WIDTH)
        lines = self._value_lines(value, value_width)
        long = len(lines) > self._VALUE_ROWS
        more = 0
        if long and not self._expanded:
            more = len(lines) - self._VALUE_ROWS
            lines = self._cut_rows(lines, self._VALUE_ROWS, value_width)
        pad = Content(" " * self._LABEL_WIDTH)
        rows = [Content.assemble((f"{label:<{self._LABEL_WIDTH - 1}} ", theme.MUTED), (lines[0], theme.TEXT))]
        rows += [pad + Content.styled(line, theme.TEXT) for line in lines[1:]]
        if more:
            # A cut value says so under it, with the key that shows the rest (a
            # narrow footer has no room for that key).
            note = v.fit_cells(f"(+{more} rows · e shows the whole value)", value_width)
            rows.append(pad + Content.styled(note, theme.FAINT))
        return rows, long

    def _grid(self, facts: list, width: int) -> tuple:
        """``(rows, long)``: the facts in two columns, down the first and then the
        second, when the card is wide enough, else in one."""
        two = width >= self._TWO_COLUMNS_FROM
        column = (width - self._GUTTER) // 2 if two else width
        cells, long = [], False
        for label, value in facts:
            rows, cut = self._fact_rows(label, value, column)
            cells.append(rows)
            long = long or cut
        if not two:
            return [row for rows in cells for row in rows], long
        half = (len(cells) + 1) // 2
        left, right = cells[:half], cells[half:] + [[]]
        # The second column starts just after the first's widest row, so short
        # values don't leave a gulf between the two.
        start = max(row.cell_length for rows in left for row in rows) + self._GUTTER
        out = []
        for a, b in zip(left, right):  # a pair of facts side by side, as tall as the taller
            for i in range(max(len(a), len(b))):
                row = a[i] if i < len(a) else Content("")
                if i < len(b):
                    row = row + Content(" " * (start - row.cell_length)) + b[i]
                out.append(row)
        return out, long

    _LIFE_GLYPHS = {"created": ("◇ ", theme.MUTED), "now": ("", theme.ACCENT)}  # "now" has the arrow

    def _lifecycle_rows(self, now: datetime, width: int) -> list:
        """The job's lifecycle (:func:`~cortex_training.tui.views.lifecycle`) in
        ``width`` cells: its steps joined by arrows that say how long each took, on
        as few rows as they fit, each row after the first under the value column."""
        steps = v.lifecycle(self._job, now)
        if not steps:
            return []
        parts = []
        for before, step in zip([None, *steps], steps):
            look = theme.status_look(step.status)
            glyph, color = self._LIFE_GLYPHS.get(step.status, (f"{look.glyph} ", look.color))
            current = step.status not in self._LIFE_GLYPHS  # the job's own state, in its color
            # The arrow from the step before, with how long that step took when the record tells.
            if before is None:
                arrow = []
            elif before.then:
                arrow = [(" ──", theme.FAINT), (f" {before.then} ", theme.MUTED), ("─▸ ", theme.FAINT)]
            else:
                arrow = [(" ──▸ ", theme.FAINT)]
            parts.append(Content.assemble(*arrow, (glyph, color),
                                          (step.label, f"bold {color}" if current else theme.TEXT),
                                          *([(f" {step.at}", theme.MUTED)] if step.at else [])))
        value_width = max(10, width - self._LABEL_WIDTH)
        lines = [parts[0]]
        for part in parts[1:]:
            if lines[-1].cell_length + part.cell_length <= value_width:
                lines[-1] += part
            else:
                lines.append(part[1:])  # a row starts at the connector's line, not a space
        pad = Content(" " * self._LABEL_WIDTH)
        label = Content.styled(f"{'Lifecycle':<{self._LABEL_WIDTH - 1}} ", theme.MUTED)
        return [label + lines[0], *(pad + line for line in lines[1:])]

    def _show_table(self, table: DataTable, labels: list, rows: list, wider: bool) -> None:
        """Update changed table rows while retaining cursor and horizontal scroll."""
        shown = (tuple(labels), wider,
                 tuple((key, tuple((cell.plain, str(cell.spans), str(cell.style)) for cell in cells))
                       for key, cells in rows))
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
        self._sub_job_by_row = {}
        rows = [("all", [Text("▸ entire job", style="bold"), Text(v.job_type_label(job)),
                         status_text(job.get("status"), word=v.status_word_for(job, job.get("status"))),
                         Text(", ".join(v.job_models(job)) or "—"), Text(v.job_gpu_cell(job, "requested"))])]
        if v.is_unknown(job):
            rows.append(("note", [Text("  ? status unavailable", style=theme.WARN), Text(""), status_text("unknown"),
                                  Text(""), Text("")]))
        for i, sj in enumerate(v.sub_jobs(job)):
            key = f"sj-{i}"
            self._scope_by_row[key] = sj.get("sub_job_id")
            self._sub_job_by_row[key] = sj
            rows.append((key, [Text(f"  {v.sub_job_label(sj)}"), Text(v.sub_job_type(sj)),
                               status_text(sj.get("status"), word=v.status_word_for(job, sj.get("status"))),
                               Text(str(sj.get("model_name") or "—")), Text(v.sub_job_gpu_text(sj))]))
        return rows

    _JSON_KEY = re.compile(r'^(\s*)("(?:[^"\\]|\\.)*")(: )(.*)$')

    def _render_config(self, width: int) -> bool:
        """Fill the config card for the highlighted row: the job's config as
        ``cortex-training get`` prints it, a body that ``submit`` takes as it is,
        with only that sub-job on a sub-job's row. Return whether a line is (or,
        expanded, would be) cut."""
        card = self.query_one("#config-card")
        picked = self._sub_job_by_row.get(self._highlighted_key())
        whose = v.sub_job_label(picked) if picked is not None else "entire job"
        card.border_title = Content.assemble((" Config ", f"bold {theme.TEXT}"), (f"· {whose} ", theme.MUTED))
        if self._experiment or v.is_deployment(self._job):
            card.border_subtitle = Content.styled(" as cortex-training get prints it, for submit ", theme.MUTED)
        elif self._experiment_error:
            card.border_subtitle = Content.styled(
                f" couldn't read the experiment: {self._experiment_error}; null gives a copy its own ", theme.WARN)
        else:
            card.border_subtitle = Content.styled(" reading the job's experiment… ", theme.MUTED)
        body = self.query_one("#config", Static)
        if not self._sub_job_by_row:
            body.update(Content.styled("the service sent no sub-job configs", theme.FAINT))
            return False
        try:
            lines = v.job_config_json(self._job, picked, experiment_name=self._experiment)
        except ValueError as exc:
            body.update(Content.styled(f"no config to show: {exc}", theme.FAINT))
            return False
        long = any(v.cell_width(line) > width for line in lines)
        body.update(Content("\n").join(
            self._json_row(line if self._expanded else v.fit_cells(line, width)) for line in lines))
        return long

    def _json_row(self, line: str) -> Content:
        """A line of the config's JSON: its key dimmer than its value, its brackets faint."""
        match = self._JSON_KEY.match(line)
        if match:
            indent, key, colon, rest = match.groups()
            return Content.assemble(indent, (key, theme.MUTED), (colon, theme.FAINT),
                                    (rest, theme.FAINT if rest in ("{", "[") else theme.TEXT))
        return Content.styled(line, theme.FAINT if line.strip().rstrip(",") in ("{", "}", "[", "]") else theme.TEXT)

    def _table_width(self, table: DataTable, rows: int) -> int:
        """Available table cells, accounting for a needed vertical scrollbar."""
        # Before the first layout: the card's inside, as the job card's (the same
        # border and padding).
        width = table.size.width or self._inner_width()
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

    # The callout's color, by its tone: red for a failure, orange while waiting (or
    # when the state is unavailable), calm once the job ended without failing.
    _TONES = {"failed": theme.ERR, "waiting": theme.WARN, "unknown": theme.WARN, "starting": theme.INFO,
              "running": theme.OK, "ended": theme.ENDED}

    def _render_health(self) -> None:
        """Draw why the job is where it is (:func:`~cortex_training.tui.views.health_callout`)
        in its box; no box for a running job the service says nothing more about."""
        box = self.query_one("#health", Static)
        callout = v.health_callout(self._job)
        box.display = callout is not None
        if callout is None:
            return
        color = self._TONES.get(callout.tone, theme.WARN)
        box.styles.border = ("round", color)
        box.border_title = Content.styled(f" {theme.status_look(callout.status).glyph} {callout.title} ",
                                          f"bold {color}")
        lines = []
        if callout.reason:
            lines.append(Content.assemble(("reason  ", theme.MUTED), (callout.reason, f"bold {color}")))
        if callout.details:
            lines.append(Content.styled(callout.details, theme.TEXT))
        if callout.hint:
            lines.append(Content.styled(callout.hint, theme.MUTED))
        box.update(Content("\n").join(lines))  # wrapped rows stay in the box, under the line's start

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
        # Then the config card's experiment_name, once the job shows: read once, as
        # a job's never changes. A deployment has none.
        if job is not None and self._experiment is None and not v.is_deployment(job):
            call_on_screen(self, self._apply_experiment, *self._read_experiment())

    def _read_experiment(self) -> tuple:
        """(the job's experiment, None), or (None, why it could not be read)."""
        try:
            run = self._client.get_experiment_run(self._job_id)
        except Exception as exc:  # noqa: BLE001 - said on the config card
            return None, v.collapse(v.describe_error(exc))
        name = run.get("experiment_name") if isinstance(run, dict) else None
        return (name, None) if isinstance(name, str) and name else (None, "the service sent no experiment name")

    def _apply_experiment(self, name: str | None, error: str | None) -> None:
        self._experiment, self._experiment_error = name, error
        self._refresh_view()

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
        """The job card's last line: a read in flight, the last one's failure, or when the copy was read."""
        took = ""
        if self._loading and self._fetch_since is not None:
            seconds = int(monotonic() - self._fetch_since)
            took = f" ({seconds}s)" if seconds >= self._FETCH_SLOW_SECONDS else ""
        error, at = self._fetch_error or "", self._fetched_at or ""
        if self._loading:
            if error:
                return Content.from_markup(f"[{theme.FAINT}]refreshing…{took}[/]  [{theme.WARN}](last try: $error)[/]",
                                           error=error)
            if at:
                return Content.from_markup(f"[{theme.FAINT}]refreshing…{took}  (last read at $at)[/]", at=at)
            return Content.from_markup(f"[{theme.FAINT}]reading the job from the service…{took}[/]")
        if error:
            # The fields above are the last good copy: this preview's, or the job list's.
            if at:
                return Content.from_markup(f"[{theme.WARN}]couldn't refresh: $error; showing the copy read at $at[/]",
                                           error=error, at=at)
            return Content.from_markup(f"[{theme.WARN}]couldn't read the job: $error; showing the job list's copy[/]",
                                       error=error)
        if at:
            return Content.from_markup(f"[{theme.FAINT}]read from the service at $at[/]", at=at)
        return Content("")

    def _render_fetch(self) -> None:
        try:
            line = self.query_one("#fetch", Static)
        except Exception:  # noqa: BLE001 - not mounted, or already removed
            return
        content = self._fetch_line()
        line.display = bool(content.plain)  # no blank row before the first read
        # A long error continues under its text, after the two-cell indent.
        line.update(hanging(content, self._inner_width(), 2))

    def action_refresh(self) -> None:
        if self._loading:
            # The read is in flight (a thread can't be cancelled): say so, not nothing.
            self.notify("Still reading this job; r asks again once it answers.")
            return
        self._start_fetch()

    def _highlighted_key(self) -> str:
        table = self.query_one("#subjobs", DataTable)
        try:
            return table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value
        except Exception:  # noqa: BLE001 - empty table
            return "all"

    def _highlighted_scope(self):
        return self._scope_by_row.get(self._highlighted_key())

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if self.query_one("#subjobs", DataTable).columns:  # the config card follows the highlighted row
            self._refresh_view()

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
