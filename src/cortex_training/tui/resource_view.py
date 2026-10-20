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

"""The job details screen's Resources tab.

``ResourceTab`` is mixed into ``LogScreen`` (tui/logs.py): the tab's own state lives
on that screen, and these are the methods that download the telemetry, pick the
sub-job and the view, and draw the header, the stats and the charts.
"""

from __future__ import annotations
from datetime import datetime
from datetime import timezone
from textual import work
from textual.content import Content
from textual.css.query import NoMatches
from textual.widgets import Label
from textual.widgets import ListItem
from textual.widgets import ListView
from textual.widgets import Static
from textual.worker import get_current_worker
from cortex_training._download_lock import DownloadLock
from cortex_training.tui import views
from cortex_training.tui.format import safe_text
from cortex_training.tui.widgets import _has_telemetry
from cortex_training.tui.widgets import _quiet_download
from cortex_training.tui.widgets import _result_warnings
from cortex_training.tui.widgets import _same_telemetry
from cortex_training.tui.widgets import _warning_text
from cortex_training.tui.log_style import sub_job_color
from cortex_training.tui.resource import CHART_MUTED
from cortex_training.tui.resource import GPU
from cortex_training.tui.resource import LOADING_OLDER
from cortex_training.tui.resource import OLDER_NOT_LOADED
from cortex_training.tui.resource import RANGES
from cortex_training.tui.resource import SYSTEM
from cortex_training.tui.resource import KindResource
from cortex_training.tui.resource import KindView
from cortex_training.tui.resource import ResourceData
from cortex_training.tui.resource import age_slack
from cortex_training.tui.resource import allocation_text
from cortex_training.tui.resource import chart_hue
from cortex_training.tui.resource import format_value
from cortex_training.tui.resource import freshness_text
from cortex_training.tui.resource import gpu_allocation
from cortex_training.tui.resource import has_metric_samples
from cortex_training.tui.resource import headline_metric
from cortex_training.tui.resource import level_glyph
from cortex_training.tui.resource import metric_chart_spans
from cortex_training.tui.resource import metric_stats_spans
from cortex_training.tui.resource import no_metrics_text
from cortex_training.tui.resource import node_allocation_text
from cortex_training.tui.resource import resource_loader
from cortex_training.tui.resource import scrape_text
from cortex_training.tui.resource import skipped_rows_text
from cortex_training.tui.resource import window_complete
from cortex_training.tui.resource import window_label
from cortex_training.tui.widgets import log_traceback
from cortex_training.tui.widgets import _SEPARATOR
from cortex_training.tui.widgets import _fit_line
from cortex_training.tui.widgets import after_refresh
from cortex_training.tui.widgets import spans_content


class ResourceTab:
    """The Resources tab of :class:`~cortex_training.tui.logs.LogScreen`."""

    # ─── resources ───────────────────────────────────────────────────────
    # Two charts to a row from this pane width, one under the other below it.
    _RESOURCE_SIDE_BY_SIDE = 90
    _RESOURCE_CHART_GAP = 3  # cells between the two charts of a row
    # Rows of bars in every chart, by the terminal's height: (fewest terminal rows, bar
    # rows), tallest first; shorter than all of them, the last. Each even, so the row
    # labelled with half the scale tops out at exactly half of it.
    _RESOURCE_BAR_ROWS_BY_HEIGHT = ((48, 12), (40, 10), (34, 8), (28, 6), (0, 4))
    _RESOURCE_TICK_SECONDS = 5  # how often the newest sample's age is redrawn
    # A running sub-job's chunks reach the stage about every minute, and an upload
    # can take a minute more: a newest sample older than this (or six sampling
    # intervals, if longer) means no new samples are arriving.
    _RESOURCE_STALE_SECONDS = 180
    # The panel's two views, and what the header calls each view's telemetry.
    _RESOURCE_VIEWS = {GPU: "GPU Resources", SYSTEM: "System Resources"}
    _RESOURCE_WHAT = {GPU: "GPU metrics", SYSTEM: "system metrics"}
    _RESOURCE_SAMPLES = {GPU: "GPU samples", SYSTEM: "system samples"}

    def action_resource_range(self, name: str) -> None:
        if not self._resource_active() or name not in RANGES:
            return
        self._resource_range = name
        self._render_resource()

    def action_cycle_window(self) -> None:
        names = list(RANGES)
        index = names.index(self._resource_range) if self._resource_range in names else -1
        self.action_resource_range(names[(index + 1) % len(names)])

    def _start_resource_refresh(self, *, force: bool) -> None:
        if (
            not self._screen_open
            or not self._sources_ready
            or not self._resource_source()
            or self._sub_job_missing
            or self._resource_loading
            or (self._resource_refresh_succeeded and not force)
        ):
            return
        self._resource_loading = True
        if not self._resource_data.kinds:
            self.query_one("#resource-status", Static).update(f"loading {self._RESOURCE_WHAT[self._resource_kind]}…")
        elif self._resource_active():
            self._render_resource()  # "cached · refreshing…" while it runs
        self._refresh_resource()

    @work(thread=True, group="resource")
    def _refresh_resource(self) -> None:
        worker = get_current_worker()
        lock = DownloadLock(self._metrics_root)
        acquired = False
        # Shared per cache directory: a refresh, or reopening the job, reads only new bytes.
        loader = resource_loader(self._metrics_root)
        shown_here: set[str] = set()  # sub-jobs this refresh has put on screen

        def accept(data: ResourceData) -> None:
            shown_here.update(data.kinds)
            self._post(self._accept_cached_resource, data)

        def show_partial(data: ResourceData) -> None:
            if not worker.is_cancelled:
                accept(data)

        def load() -> ResourceData:
            # The selected sub-job first. A large file read from the start is read
            # newest-first, and that partial view is shown only while the screen has
            # nothing for the sub-job: download_metrics rebuilds the file when a chunk
            # lands out of order, and the re-read must not replace a complete view
            # with its newest few minutes.
            source = self._resource_source()
            blank = source not in self._resource_data.kinds and source not in shown_here
            return loader.load(priority=source, window=RANGES.get(self._resource_range),
                               on_partial=show_partial if blank else None)

        try:
            last = loader.last
            if not self._resource_loaded and _has_telemetry(last):
                # Reopened while another screen's refresh holds the lock: paint what the
                # shared loader read last instead of waiting for that download.
                accept(last)
            acquired = lock.acquire(lambda: self._screen_open and not worker.is_cancelled)
            if not acquired or not self._screen_open or worker.is_cancelled:
                return
            # Paint what is cached while the download runs. Every load is incremental:
            # it reads what was committed since the shared loader last read (another
            # screen's download, say, that closed before reading it), and nothing when
            # nothing was. The screen already shows an unchanged result.
            cached = load()
            if _has_telemetry(cached) and not worker.is_cancelled and not _same_telemetry(cached, self._resource_data):
                accept(cached)

            def committed(_result: dict) -> None:
                # Each sub-job is painted as it lands, the one on screen first.
                if worker.is_cancelled or not self._screen_open:
                    return
                data = load()
                if _has_telemetry(data) and not _same_telemetry(data, self._resource_data):
                    accept(data)

            source = self._resource_source()
            results = self._client.download_metrics(
                self._job_id,
                self._metrics_root,
                resume=True,
                priority=[source] if source else [],
                on_commit=committed,
                **_quiet_download(self._client.download_metrics),
            )
            # Chunks the download gave up on: say so, with the command that retries them.
            self._post(self._accept_download_warnings, "metrics",
                       _result_warnings(results, kind="metrics", job_id=self._job_id,
                                        output_dir=self._metrics_root))
            if worker.is_cancelled or not self._screen_open:
                return  # the screen closed during the download; the next one to open reads it
            fresh = load()  # reads only what the download appended
            if not worker.is_cancelled:
                self._post(self._finish_resource_refresh, fresh, None)
        except Exception as exc:  # noqa: BLE001 - surfaced in the Resource tab
            # The tab's header has room for the start of the error; the log keeps it whole.
            log_traceback(f"GPU metrics refresh for {self._job_id} failed")
            if not worker.is_cancelled:
                try:
                    # Only the lock holder may load; without it, show what the loader read last.
                    partial = loader.load() if acquired else (loader.last or ResourceData({}))
                except Exception:  # noqa: BLE001 - preserve the original refresh error
                    partial = ResourceData({})
                self._post(
                    self._finish_resource_refresh,
                    partial if _has_telemetry(partial) else None,
                    views.access_hint(exc, "GPU metrics") or views.describe_error(exc),
                )
        finally:
            if acquired:
                lock.release()

    def _accept_cached_resource(self, data: ResourceData) -> None:
        if not self._screen_open:
            return
        self._resource_data = data
        self._resource_loaded = True
        self._resource_error = None
        if self._resource_active():
            self._render_resource()

    def _finish_resource_refresh(
        self,
        data: ResourceData | None,
        error: str | None,
    ) -> None:
        if not self._screen_open:
            return
        self._resource_loading = False
        self._resource_refresh_succeeded = error is None
        if error is None:
            self._resource_downloaded = True  # every sub-job's rows are in: one without a kind has none
        if data is not None:
            self._resource_data = data
            self._resource_loaded = True
            self._resource_error = error
            if self._resource_active():
                self._render_resource()
            return
        self._resource_error = error
        self._resource_loaded = _has_telemetry(self._resource_data)
        if self._resource_active():
            self._render_resource()

    def _resource_sub_job_ids(self) -> list[str]:
        """The sub-jobs < and > switch between, in the order the Logs tab lists them."""
        return list(dict.fromkeys(self._source_by_item.values()))

    def _resource_source(self) -> str | None:
        """The sub-job the Resources tab shows: the one picked with < or >, else the one selected in Logs."""
        pick = self._resource_pick
        if pick and pick in self._source_by_item.values():
            return pick
        return self._current_source

    def _resource_kind_data(self, kind: str | None = None) -> KindResource | None:
        """The shown sub-job's telemetry of ``kind`` (default: the view shown); None without rows of it."""
        return self._resource_data.kind(self._resource_source(), kind or self._resource_kind)

    def action_resource_sub_job(self, step: int) -> None:
        """Show the previous or next sub-job without changing Logs."""
        ids = self._resource_sub_job_ids()
        current = self._resource_source()
        index = ids.index(current) if current in ids else 0
        self._resource_pick = ids[(index + step) % len(ids)]
        self.refresh_bindings()
        self._render_resource()
        if self._resource_pick not in self._resource_data.kinds:
            # Not downloaded yet (a cold cache downloads the shown sub-job first): fetch it now.
            self._start_resource_refresh(force=True)

    def _follow_resource_source(self, source: str | None) -> None:
        """Apply the preferred view and mean unit to a newly shown sub-job."""
        if source != self._resource_shown:
            self._resource_shown = source
            self._resource_kind = self._resource_kind_pref
            self._resource_settled = None
            self._resource_unit = None
        kinds = self._resource_data.kinds.get(source or "")
        if kinds and self._resource_settled != source:
            self._resource_settled = source
            if self._resource_kind == GPU and GPU not in kinds and SYSTEM in kinds:
                self._resource_kind = SYSTEM

    def _resource_list_moved(self, index: int | None) -> None:
        """Show the view or unit highlighted in the resource panel."""
        if self._resource_placing or index is None or not 0 <= index < len(self._resource_entries):
            return  # the redraw put it on the entry shown (or on a view's while it has no units)
        if not self._screen_open:
            return
        role, which = (self._resource_entries[index] + (None,))[:2]
        if role == "view":
            self._resource_on_view = True
            if which == self._resource_kind and self._resource_unit is None:
                return
            self._resource_kind = self._resource_kind_pref = which
            self._resource_settled = self._resource_shown  # picked here: it stays for this sub-job
            self._resource_unit = None
        elif role == "unit":
            self._resource_on_view = False
            if which == self._resource_unit:
                return
            self._resource_unit = which
        else:
            return
        self._render_resource()

    def _resource_down_from(self, index: int | None) -> int | None:
        """Route ↓ from GPU Resources to its ALL row."""
        entries = self._resource_entries
        if index is None or not 0 <= index < len(entries) or entries[index] != ("view", GPU):
            return None
        return entries.index(("unit", None)) if ("unit", None) in entries else None

    def _resource_cursor(self) -> int | None:
        """The panel entry of what is shown: the unit's (ALL for the mean), or the view's."""
        entries = self._resource_entries
        if not self._resource_on_view and ("unit", self._resource_unit) in entries:
            return entries.index(("unit", self._resource_unit))
        view = ("view", self._resource_kind)
        return entries.index(view) if view in entries else None

    def _resource_list_rows(self, source: str | None, now: datetime) -> list[tuple[tuple, Content, bool]]:
        """Resource panel rows as ``(entry, label, disabled)``."""
        kinds = self._resource_data.kinds.get(source or "", {})
        # A kind missing now has no rows: once the sub-job has some, or once a download
        # found none for a sub-job that isn't still to send its first samples.
        known = bool(kinds) or (self._resource_downloaded and self._resource_waiting() is None
                                and not self._resource_first_samples_due(gpu_allocation(self._job, source)))
        heads = []
        for kind, name in self._RESOURCE_VIEWS.items():
            none = known and kind not in kinds
            shown = kind == self._resource_kind
            text = f"{'●' if shown else '○'} {name}{' (none)' if none else ''}"
            heads.append((("view", kind), text, "dim" if none else "bold" if shown else ""))
        resource = kinds.get(self._resource_kind)
        units: list[tuple[tuple, str, str, str]] = []
        hue = ""
        if resource is not None:
            metric = headline_metric(resource)
            stale = self._resource_stale(resource, now)
            metrics = list(resource.metrics)
            hue = chart_hue(metrics.index(metric)) if metric in metrics else ""

            def value(latest) -> str:
                if metric is None:
                    return ""
                found = latest.get(metric.key)
                return f"{level_glyph(found, metric.scale)} {format_value(found, metric.unit)}"

            units.append((("unit", None), "ALL", value(resource.latest), "dim" if stale else ""))
            for unit in resource.units:
                dim = stale or not unit.reporting
                units.append((("unit", unit.id), unit.name + (" (stale)" if dim else ""), value(unit.latest),
                              "dim" if dim else ""))
        value_width = max((views.cell_width(value) for *_head, value, _style in units), default=0)
        inner = max([views.cell_width(text) for _entry, text, _style in heads]
                    + [views.cell_width(name) + 2 + value_width for _entry, name, _value, _style in units])
        rows = [(entry, Content.styled(text, style) if style else Content(text), False)
                for entry, text, style in heads]
        if units:
            rows.append((("rule",), Content("─" * inner), True))
        for entry, name, value, style in units:
            gap = " " * (inner - views.cell_width(name) - views.cell_width(value))
            if style:  # dimmed: a unit that stopped reporting
                rows.append((entry, Content.styled(name + gap + value, style), False))
            else:
                # The bar in the chart's colour; the value plain, readable under the highlight too.
                bar, _space, number = value.partition(" ")
                rows.append((entry, Content.assemble(name, gap, (bar, hue) if hue else bar, _space + number), False))
        return rows

    def _sync_resource_list(self, source: str | None, now: datetime) -> None:
        """Show the panel's entries in its rows, and put its highlight on the entry shown."""
        try:
            panel = self.query_one("#resource-list", ListView)
        except NoMatches:
            return
        rows = self._resource_list_rows(source, now)
        kept = self._resource_rows
        for position, (item, label) in enumerate(kept):
            if position < len(rows):
                _entry, content, disabled = rows[position]
                label.update(content)
                if item.disabled != disabled:
                    item.disabled = disabled
                if not item.display:
                    item.display = True
            else:
                if not item.disabled:
                    item.disabled = True  # passed over by the arrow keys
                if item.display:
                    item.display = False
        for _entry, content, disabled in rows[len(kept):]:
            label = Label(content)
            item = ListItem(label, disabled=disabled)
            kept.append((item, label))
            panel.append(item)
        self._resource_entries = [entry for entry, _content, _disabled in rows]
        target = self._resource_cursor()
        if target is not None and panel.index != target:
            self._resource_placing = True
            try:
                panel.index = target
            finally:
                self._resource_placing = False
        self._apply_resource_list_width(max(content.cell_length for _entry, content, _disabled in rows))

    def _apply_resource_list_width(self, longest: int) -> None:
        """Size the panel as the Logs tab's sub-job list is sized: to its labels, within the same bounds."""
        width = min(self._SOURCES_AUTO_MAX, max(self._SOURCES_AUTO_MIN, longest + 4))
        width = max(self._SOURCES_MIN, min(width, self._sources_cap()))
        if width == self._resource_list_width:
            return
        self._resource_list_width = width
        try:
            self.query_one("#resource-list", ListView).styles.width = width
        except NoMatches:
            return
        after_refresh(self, self._render_resource)  # the pane beside it changed width: draw to it

    _RESOURCE_REFRESHING = ("cached · refreshing…", "refreshing…")

    def _resource_state(self, resource: KindResource | None) -> tuple[str, ...]:
        """Refresh-state forms, fullest first, or empty when idle."""
        source = self._resource_source() or ""
        cached = bool(self._resource_data.kinds.get(source)) or source in self._resource_data.ignored_rows_by_sub_job
        what = self._RESOURCE_WHAT[self._resource_kind]
        if self._resource_missing(resource) == LOADING_OLDER:
            return (LOADING_OLDER,)
        if resource is not None and resource.partial:
            if self._resource_error:
                return ("[$warning]older samples failed to load: $problem[/]",
                        "[$warning]older samples failed: $problem[/]")
            return (f"[$warning]{OLDER_NOT_LOADED}[/]",)
        if self._resource_loading:
            if cached:
                return self._RESOURCE_REFRESHING
            # A sub-job with nothing of its own after a whole download has none to load.
            return ("refreshing…",) if self._resource_downloaded else (f"loading {what}…",)
        if self._resource_error:
            return (("cached · [$warning]refresh failed: $problem[/]", "[$warning]refresh failed: $problem[/]")
                    if cached else (f"[$warning]{what} unavailable: $problem[/]",))
        return ()

    def _resource_missing(self, resource: KindResource | None) -> str | None:
        """What stands in for the stats that need the shown sub-job's older samples; None once they are read."""
        if resource is None or not resource.partial:
            return None
        return LOADING_OLDER if self._resource_loading else OLDER_NOT_LOADED

    def _resource_waiting(self) -> str | None:
        """The status word while the shown sub-job (or its job) has not started running, else None."""
        source = self._resource_source()
        word = self._tail_status(source) if source else ""
        return word if word in self._PRE_PLACEMENT | self._STARTING else None

    def _resource_first_samples_due(self, allocation) -> bool:
        """Whether a running, GPU-owning sub-job still owes its first samples."""
        source = self._resource_source()
        if not source or self._tail_status(source) != "running":
            return False
        if allocation.log_prob or allocation.shared_with or allocation.requested == 0:
            return False
        return not self._resource_data.kinds.get(source)

    def _resource_stale(self, resource: KindResource | None, now: datetime) -> bool:
        """Whether a running sub-job has sent no samples of the view's kind for well past their usual delay."""
        source = self._resource_source()
        if resource is None or not source or self._tail_status(source) != "running":
            return False
        interval = resource.sample_interval_s or resource.reported_interval_s or 0
        age = (now - resource.latest_sample_ts).total_seconds()
        return age > max(self._RESOURCE_STALE_SECONDS, 6 * interval)

    def _resource_empty_text(self, allocation, ignored: int) -> str:
        """Why the view shown has nothing to chart, once the telemetry has loaded."""
        kind = self._resource_kind
        source = self._resource_source()
        if kind == GPU and ignored:
            return "no usable GPU metrics for this sub-job"
        waiting = self._resource_waiting()
        if waiting is None and self._resource_first_samples_due(allocation):
            waiting = "running"  # only said: unlike a sub-job still placing, this one is polled
        job_waiting = views.status_word(self._job_status) in self._PRE_PLACEMENT | self._STARTING
        who = "the job" if job_waiting else views.sub_job_label(source) if source else "it"
        return no_metrics_text(allocation, waiting=waiting, who=who, kind=kind)

    def _resource_status(self, resource: KindResource | None, width: int = 0) -> Content:
        """Two fitted header lines for scope, sampling, freshness, and refresh state."""
        kind = self._resource_kind
        source = self._resource_source() or ""
        ignored = self._resource_data.ignored_rows_by_sub_job.get(source, 0)
        states = self._resource_state(resource)
        allocation = gpu_allocation(self._job, source)
        now = datetime.now(timezone.utc)
        stale = self._resource_stale(resource, now)
        values: dict = {"problem": views.collapse(self._resource_error or "")}
        if ignored:
            values.update(note=skipped_rows_text(ignored), note_short=skipped_rows_text(ignored, short=True))
        note = [("$note", 0), ("$note_short", 0), ("", 0)] if ignored else [("", 0)]
        state = [(form, 0) for form in states]
        # Which sub-job this is, by name (the job's id is on the line above), and where it is among them.
        ids = self._resource_sub_job_ids()
        values["label"] = views.sub_job_label(source) if source else ""
        # In the colour of its dot in the Logs tab's list and its tag on the log lines.
        named = f"[b {sub_job_color(values['label'])}]$label[/]"
        if source in ids and len(ids) > 1:
            values.update(position=f"{ids.index(source) + 1} of {len(ids)}", pos=f"{ids.index(source) + 1}/{len(ids)}")
            # A click on it shows the next sub-job, as space does.
            source_forms = [(f"[@click=screen.resource_sub_job(1)]{form}[/]", 0)
                            for form in (f"{named} ($position)", f"{named} ($pos)", named)]
        else:
            source_forms = [(named, 0)]
        if resource is None:
            if kind == GPU:
                values.update(allocation=allocation_text(allocation),
                              allocation_short=allocation_text(allocation, hardware=False))
                first = _fit_line([source_forms, [("$allocation", 0), ("$allocation_short", 0)]], [0, 0, 1], width,
                                  values)
            else:
                first = _fit_line([source_forms], [0, 0], width, values)
            cached = bool(self._resource_data.kinds.get(source))
            # A download that gave up on chunks is why there may be nothing here: say so.
            warned = self._source_warnings()
            if warned:
                values["warning"] = _warning_text(warned)
            alert = [("$warning", 0)] if warned else [("", 0)]
            if self._resource_loaded and (self._resource_downloaded or cached or not states):
                # A message, not the charts: it may wrap. A refresh's state follows it.
                values["empty"] = self._resource_empty_text(allocation, ignored)
                second = _fit_line([[("$empty", 0)], state or [("", 0)], note, alert], [], 0, values)
            elif states or warned:
                # Nothing to chart yet: the state (an error, say) is the message, whole; it may wrap.
                second = _fit_line([state, note, alert], [], 0, values)
            else:
                second = Content("")
            return Content("\n").join([first, second])
        # A running sub-job that stopped sending samples has no GPU or node reporting now.
        reporting = 0 if stale else len(resource.reporting_unit_ids)
        if kind == GPU:
            values.update(allocation=allocation_text(allocation, reporting, len(resource.units)),
                          allocation_short=allocation_text(allocation, reporting, len(resource.units), hardware=False))
            allocation_forms = [("$allocation", 0), ("$allocation_short", 0)]
        else:
            values.update(nodes=node_allocation_text(resource, reporting),
                          nodes_short=node_allocation_text(resource, reporting, sizes=False))
            allocation_forms = [("$nodes", 0), ("$nodes_short", 0)]
        order = [0, 1, 1, 0, 2]
        samples = self._RESOURCE_SAMPLES[kind]
        values.update(
            window=window_label(self._resource_range),
            key=self._resource_range,  # "10m", "all": the window keys' own names
            scraped=scrape_text(resource),
            scraped_short=scrape_text(resource, notes=False),
            fresh=freshness_text(resource, now, stale=stale, what=samples),
            fresh_short=freshness_text(resource, now, stale=stale, clock=False, what=samples),
        )
        # The window goes before the place: the stats and the chart titles name it too.
        # A click on it shows the next window, as w does.
        window_forms = [("[@click=screen.cycle_window]$window[/]", 0), ("[@click=screen.cycle_window]$key[/]", 0),
                        ("", 0)]
        first = _fit_line([source_forms, window_forms, allocation_forms], order, width, values)
        grows = age_slack(resource, now)
        warn = "[$warning]{}[/]" if stale else "{}"
        fresh = [(warn.format("$fresh"), grows), (warn.format("$fresh_short"), grows)]
        if not states:
            # Nothing shows, but the room a refresh's state needs stays free.
            state = [("", len(_SEPARATOR) + len(text)) for text in self._RESOURCE_REFRESHING]
        scraped = [("$scraped", 0), ("$scraped_short", 0), ("", 0)]
        # A download that gave up on chunks: the telemetry on screen has holes, so say so
        # here rather than let it look complete. It goes last and is never dropped.
        warnings = self._source_warnings()
        if warnings:
            values["warning"] = _warning_text(warnings)
        alert = [("$warning", 0)] if warnings else [("", 0)]
        # Parts: 0 scraped, 1 freshness, 2 state, 3 skipped rows, 4 warning.
        second = _fit_line([scraped, fresh, state, note, alert], [3, 0, 1, 2, 3, 0], width, values, cut="problem")
        return Content("\n").join([first, second])

    def _tick_resource_status(self) -> None:
        # Keeps "newest N s ago" current between refreshes, without redrawing the rest
        # (unless the newest sample just became stale, which marks the units too).
        if not self._screen_open or not self._resource_active() or self.app.screen is not self:
            return
        resource = self._resource_kind_data()
        if resource is not None:
            if self._resource_stale(resource, datetime.now(timezone.utc)) != self._resource_stale_shown:
                self._render_resource()
                return
            status = self.query_one("#resource-status", Static)
            status.update(self._resource_status(resource, status.content_size.width))

    def _resource_charts(self, resource: KindResource, view: KindView, width: int, complete: bool,
                         missing: str | None) -> Content:
        """Every metric's chart, in the metrics' order: two to a row in a pane at least 90 cells wide, titled as
        docs/reference/cli.md says; where one doesn't fit, every title leaves out what the values are, then the
        window, and a name that still doesn't fit ends in ``…`` (so the charts of a row stay level). A window
        whose older samples aren't read yet says so after each title, or on a line of its own under each."""
        metrics = list(resource.metrics)
        if not metrics:
            return Content("")
        per_row = 2 if width >= self._RESOURCE_SIDE_BY_SIDE else 1
        gap = self._RESOURCE_CHART_GAP
        column = max(1, (width - gap * (per_row - 1)) // per_row)
        bars = next(n for rows, n in self._RESOURCE_BAR_ROWS_BY_HEIGHT if self.app.size.height >= rows)
        hues = [chart_hue(index) for index in range(len(metrics))]
        # Each title ends with the metric's latest value, at the chart's right edge; the
        # rest of the title gets the room before it.
        latest = []
        for metric in metrics:
            stats = view.stats.get(metric.key)
            latest.append(format_value(stats.latest, metric.unit) if stats is not None else "")
        room = max(1, column - max((views.cell_width(value) for value in latest), default=0) - 2)
        # The header, pinned above them, names the sub-job: the titles needn't.
        suffix = f" · {window_label(view.name)}" + (f" · {view.label}" if view.label else "")
        tails = [suffix] * len(metrics)
        if any(views.cell_width(metric.name + suffix) > room for metric in metrics):
            tails = [f" · {window_label(view.name)}"] * len(metrics)
        if any(views.cell_width(metric.name + tail) > room for metric, tail in zip(metrics, tails)):
            tails = [""] * len(metrics)
        notes: list[str | None] = [None] * len(metrics)
        if not complete:
            loading = missing == LOADING_OLDER
            marker = " (loading…)" if loading else f" ({OLDER_NOT_LOADED})"
            if all(views.cell_width(metric.name + tail + marker) <= room for metric, tail in zip(metrics, tails)):
                tails = [tail + marker for tail in tails]
            else:  # on a line of its own under every title, so the charts of a row stay level
                note = views.fit_cells("(older samples loading…)" if loading else f"({OLDER_NOT_LOADED})", column)
                notes = [note] * len(metrics)
        blocks = []
        for metric, hue, tail, value, note in zip(metrics, hues, tails, latest, notes):
            name = views.fit_cells(metric.name, room)
            tail = views.fit_cells(tail, max(0, room - views.cell_width(name)))
            used = views.cell_width(name) + views.cell_width(tail)
            head = [Content.assemble((name, f"bold {hue}"), (tail, CHART_MUTED),
                                     " " * max(1, column - used - views.cell_width(value)), (value, f"bold {hue}"))]
            if note is not None:
                head.append(Content.styled(note, CHART_MUTED))
            if has_metric_samples(view, metric.key):
                body = [spans_content(line) for line in metric_chart_spans(view, metric, column, height=bars,
                                                                              hue=hue)]
            else:
                body = [Content.styled(views.fit_cells(f"no {metric.name} samples in this range", column),
                                       CHART_MUTED)]
            blocks.append(head + body)
        height = max(len(block) for block in blocks)
        rows = []
        for start in range(0, len(blocks), per_row):
            group = blocks[start:start + per_row]
            lines = []
            for index in range(height):
                cells = []
                for block in group:
                    line = block[index] if index < len(block) else Content("")
                    if line.cell_length > column:
                        line = line.truncate(column)
                    cells.append(line.pad_right(column - line.cell_length))
                lines.append(Content(" " * gap).join(cells).rstrip())
            rows.append(Content("\n").join(lines))
        return Content("\n\n").join(rows)

    def _render_resource(self, *, retry_layout: bool = True) -> None:
        if not self._screen_open or not self._resource_active():
            return
        try:
            status = self.query_one("#resource-status", Static)
            stats = self.query_one("#resource-stats", Static)
            charts = self.query_one("#resource-charts", Static)
        except NoMatches:
            return
        source = self._resource_source()
        now = datetime.now(timezone.utc)
        self._follow_resource_source(source)
        resource = self._resource_kind_data() if source else None
        drawn = (source, self._resource_kind, self._resource_unit, self._resource_range)
        if drawn != self._resource_drawn:
            # Another sub-job, view, GPU or node, or window: a highlight would sit over
            # other text, or past the end of a shorter block.
            self._resource_drawn = drawn
            self.clear_selection()
        if self._resource_unit is not None and (resource is None or self._resource_unit not in resource.unit_ids):
            self._resource_unit = None  # not one of this view's units (any more)
        self._sync_resource_list(source, now)
        if not source and self._no_sources:
            # Nothing to load or lay out: say why at once (the pane may not have a width yet).
            status.update(self._resource_no_source_text())
        pane_width = stats.content_size.width
        if pane_width <= 0:
            if retry_layout:
                self.call_after_refresh(self._render_resource, retry_layout=False)
            return
        if resource is None:
            status.update(self._resource_status(None, pane_width) if source else self._resource_no_source_text())
            stats.update("")
            charts.update("")
            self._resource_stale_shown = False
            return
        if self._resource_ticker is None:
            self._resource_ticker = self.set_interval(self._RESOURCE_TICK_SECONDS, self._tick_resource_status)
        unit = self._resource_unit
        view = resource.view(self._resource_range, unit) or resource.view(self._resource_range)
        whole = resource.view("all", unit) or resource.view("all")
        if view is None:
            return
        complete = window_complete(resource, view)
        missing = self._resource_missing(resource)  # the older samples: loading, or not loaded
        self._resource_stale_shown = self._resource_stale(resource, now)
        status.update(self._resource_status(resource, pane_width))
        stats.update(Content("\n").join(spans_content(line) for line in metric_stats_spans(
            view, whole, resource.metrics, width=pane_width, partial=resource.partial, window_partial=not complete,
            missing=missing or LOADING_OLDER)))
        charts.update(self._resource_charts(resource, view, pane_width, complete, missing))

    def _resource_no_source_text(self) -> Content | str:
        """The Resources header when no sub-job is selected: why not."""
        if self._sub_job_missing and self._sub_job_id:
            return Content.from_markup("sub-job not found: $sub_job", sub_job=self._sub_job_id)
        if not self._sources_ready:
            return "loading log sources…"
        if self._source_by_item:
            return "select a log source first"
        job = self._job
        if views.is_unknown(job):
            # The service can't say which sub-jobs the job has while its status is unavailable.
            return Content.from_markup("$why; no GPU metrics are cached on this machine",
                                       why=views.status_unavailable_text(job))
        if self._no_sources and self._no_sources != self._NO_SUB_JOBS:
            # The job couldn't be read, or the service answered nothing: the reason the Logs pane gives.
            return Content(safe_text(f"no sub-jobs to show GPU metrics for: {self._no_sources}"))
        return "this job has no sub-jobs, and no GPU metrics are cached on this machine"
