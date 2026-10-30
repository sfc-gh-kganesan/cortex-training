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

"""The job details screen: its Logs tab, and the screen the Resources and Checkpoints tabs live on.

One screen holds every tab, so the job's status, its sub-jobs and its downloads are
read once. The Resources tab's own methods are ``ResourceTab``
(tui/resource_view.py), the Checkpoints tab's ``CheckpointTab``
(tui/checkpoint_view.py); everything here is the log tail, the filter, the saved
console and what the screen shares.
"""

from __future__ import annotations
import codecs
import inspect
import os
import re
import shlex
import shutil
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from functools import partial
from datetime import datetime
from datetime import timezone
from pathlib import Path
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.containers import VerticalScroll
from textual.content import Content
from textual.css.query import NoMatches
from textual.screen import Screen
from textual.widgets import DataTable
from textual.widgets import Header
from textual.widgets import HelpPanel
from textual.widgets import Input
from textual.widgets import Label
from textual.widgets import ListItem
from textual.widgets import ListView
from textual.widgets import Log
from textual.widgets import Static
from textual.widgets import TabbedContent
from textual.widgets import TabPane
from textual.worker import get_current_worker
from cortex_training._artifact_download import DownloadCancelled
from cortex_training._artifact_download import NoChunksError
from cortex_training.tui import theme
from cortex_training.tui import format as log_format
from cortex_training.tui import views
from cortex_training.tui.format import compact_stamp
from cortex_training.tui.format import day_rule
from cortex_training.tui.format import display_day
from cortex_training._download_lock import DownloadLock
from cortex_training.tui.detail_tabs import TrainingPane
from cortex_training.tui.format import display_line
from cortex_training.tui.widgets import _TimedTail
from cortex_training.tui.widgets import _cache_job_id
from cortex_training.tui.widgets import _download_options
from cortex_training.tui.widgets import _quiet_download
from cortex_training.tui.widgets import _result_warnings
from cortex_training.tui.widgets import _sleep_until_cancelled
from cortex_training.tui.widgets import _warning_text
from cortex_training.tui.format import entry_at_level
from cortex_training.tui.format import entry_matches
from cortex_training.tui.format import format_log_entry
from cortex_training.tui.format import one_line
from cortex_training.tui.format import safe_text
from cortex_training.tui.format import wrap_log_line
from cortex_training.tui.cache_cleanup import DEFAULT_CACHE_LIMIT
from cortex_training.tui.cache_cleanup import LEASE_HEARTBEAT_SECONDS
from cortex_training.tui.cache_cleanup import enforce_cache_limit
from cortex_training.tui.cache_cleanup import telemetry_bytes
from cortex_training.tui.cache_cleanup import make_stage
from cortex_training.tui.cache_cleanup import release_stage
from cortex_training.tui.cache_cleanup import JobLease
from cortex_training.tui.log_cache import LogCache
from cortex_training.tui.log_cache import cached_log_pages
from cortex_training.tui.log_cache import catching_up
from cortex_training.tui.log_cache import job_cache_dir
from cortex_training.tui.log_style import DAY_KIND
from cortex_training.tui.log_style import ERROR_KINDS
from cortex_training.tui.log_style import ISSUE_KINDS
from cortex_training.tui.log_style import NO_TRACEBACK
from cortex_training.tui.log_style import LogHighlighter
from cortex_training.tui.log_style import in_context
from cortex_training.tui.log_style import line_kind
from cortex_training.tui.log_style import next_line_of
from cortex_training.tui.log_style import place_of
from cortex_training.tui.resource import DEFAULT_RANGE
from cortex_training.tui.resource import GPU
from cortex_training.tui.resource import ResourceData
from cortex_training.tui.resource import cached_metric_sources
from cortex_training.tui.saved_log import committed_size
from cortex_training.tui.saved_log import read_saved_tail
from cortex_training.tui.serving_view import ServingPane
from cortex_training.tui.widgets import _ERROR_LOG
from cortex_training.tui.widgets import log_traceback
from cortex_training.tui.widgets import FittedFooter
from cortex_training.tui.widgets import FollowsPointer
from cortex_training.tui.widgets import JobTabs
from cortex_training.tui.widgets import KEYS_BINDING
from cortex_training.tui.widgets import _ColoredLinks
from cortex_training.tui.widgets import _LogPane
from cortex_training.tui.widgets import _ResourceList
from cortex_training.tui.widgets import _SourceList
from cortex_training.tui.widgets import after_refresh
from cortex_training.tui.widgets import call_on_screen
from cortex_training.tui.widgets import close_keys_panel
from cortex_training.tui.widgets import fit_line
from cortex_training.tui.widgets import is_shown
from cortex_training.tui.widgets import on_stack
from cortex_training.tui.resource_view import ResourceTab
from cortex_training.tui.checkpoint_view import CheckpointTab


class LogScreen(ResourceTab, CheckpointTab, FollowsPointer, Screen):
    """Read-only Logs, Resource dashboard and Checkpoints for one job."""

    BINDINGS = [
        ("b", "back", "Back"),
        Binding("escape", "dismiss_or_back", "Back/clear", show=False),  # the filter box says Esc clears it
        ("q", "quit", "Quit"),
        KEYS_BINDING,
        Binding("t", "cycle_tab", "Tab", show=False),
        Binding("/", "filter", "Filter", priority=True),
        ("l", "cycle_level", "Level"),
        Binding("l", "show_tab('logs')", "Logs", show=False),
        Binding("L", "cycle_level", "Level", show=False),  # the key it had before
        # Logs: the next error (a traceback is one), or the one before; n and N stop at warnings too.
        Binding("e", "jump('error', 1)", "Error"),
        Binding("E", "jump('error', -1)", "Prev error", show=False),
        Binding("n", "jump('issue', 1)", "Next warning/error", show=False),
        Binding("N", "jump('issue', -1)", "Prev warning/error", show=False),
        ("p", "toggle_pause", "Pause"),
        ("s", "save_log", "Save"),
        ("r", "refresh_sources", "Refresh"),
        Binding("[", "shrink_sources", "Narrower", show=False),
        Binding("]", "grow_sources", "Wider", show=False),
        Binding("w", "cycle_window", "Window"),  # Resources: the next window
        # Resources: Space shows the next sub-job (round to the first after the last), no Shift
        # needed; shown there with two or more sub-jobs (see check_action). < and > (and , and .)
        # still go back and forth.
        Binding("space", "resource_sub_job(1)", "Sub-job", key_display="space"),
        Binding("less_than_sign", "resource_sub_job(-1)", "Prev sub-job", key_display="<", show=False),
        Binding("greater_than_sign", "resource_sub_job(1)", "Next sub-job", key_display=">", show=False),
        Binding("comma", "resource_sub_job(-1)", "Prev sub-job", show=False),
        Binding("full_stop", "resource_sub_job(1)", "Next sub-job", show=False),
        Binding("g", "show_tab('resource')", "Resources", show=False),
        Binding("m", "show_tab('training')", "Training metrics", show=False),
        Binding("c", "show_tab('checkpoints')", "Checkpoints", show=False),
        Binding("v", "show_tab('serving')", "Serving", show=False),
        # Checkpoints: s and o sort the list (on Logs, s saves the log; see check_action).
        Binding("s", "checkpoint_sort", "Sort"),
        Binding("o", "checkpoint_order", "Order"),
        Binding("d", "checkpoint_download", "Download"),
    ]

    # Auto-fit the source labels until [ or ], while preserving the log's minimum.
    _SOURCES_MIN = 12
    _SOURCES_AUTO_MIN = 16
    _SOURCES_AUTO_MAX = 48
    _SOURCES_STEP = 8
    _LOG_MIN_WIDTH = 40
    _LEVEL_CYCLE = [None, "INFO", "WARNING", "ERROR"]
    _KEEP_CONSOLE_AFTER_SECONDS = 600
    # A saved console can take minutes for a long job: the sub-job on screen first shows
    # its last half hour (where a job's end, and its error, are), until the whole is in.
    _CONSOLE_PREVIEW_SECONDS = 30 * 60
    _DATA_REFRESH_SECONDS = 30
    _STATUS_REFRESH_SECONDS = 60
    # The warm-up (see _warm_artifact_session) waits for the Logs tab's first lines, which
    # its import of the Snowflake Connector would slow, or this long after the screen opens.
    _WARM_UP_AFTER_SECONDS = 1.5
    _LOG_ONLY = {"save_log", "cycle_level", "toggle_pause", "shrink_sources", "grow_sources", "filter", "jump"}
    # Every tab in compose order; t cycles the job's kind's own (views.job_tabs).
    _TABS = ("serving", "logs", "resource", "training", "checkpoints")
    _TAB_NAMES = {"serving": "Serving", "logs": "Logs", "resource": "Resources", "training": "Training metrics",
                  "checkpoints": "Checkpoints"}

    class _Note(str):
        """A TUI-authored pane line distinguished by type, not text."""

        __slots__ = ()

    class _CallbackError(Exception):
        """A worker's UI callback failed, not its request."""

    def __init__(self, client, job_id, *, job: dict | None = None, sub_job_id=None, initial_tab: str = "logs",
                 poll_interval=1.0):
        super().__init__()
        self._client = client
        self._job_id = job_id
        self._sub_job_id = sub_job_id
        self._poll_interval = poll_interval
        self._job = job or {"job_id": job_id}
        self._job_status = self._job.get("status")
        self._job_reason = None  # failure/cancel reason, when terminal
        self._sources_width = self._SOURCES_AUTO_MIN
        self._sources_auto = True  # sized to its labels until [ or ]
        cache_job_id = _cache_job_id(client, job_id)
        job_dir = job_cache_dir(cache_job_id)
        self._cache = LogCache(cache_job_id)
        self._lease = JobLease(job_dir)
        self._cache_limit = DEFAULT_CACHE_LIMIT
        self._source_by_item: dict[str, str] = {}
        self._source_items: dict[str, tuple[ListItem, Label]] = {}  # each sub-job's row (see _paint_sources)
        self._source_labels: dict[str, str] = {}  # each sub-job's name in the list, from its record (job_type)
        self._logview: Log | None = None
        self._filter = ""  # grep substring for the log pane
        self._min_level = None  # None | INFO | WARNING | ERROR
        self._paused = False  # auto-scroll paused?
        self._current_source = None  # source currently being tailed
        self._shown_lines: list[str] = []  # unwrapped lines on screen (for re-wrap); the TUI's are _Notes
        self._wrap_width = -1  # content width the buffer is wrapped to (-1: unknown or mixed)
        self._rewrap_timer = None  # wraps the pane again once a resize settles (see _schedule_rewrap)
        self._filter_timer = None
        self._last_line_at: str | None = None  # "last line HH:MM:SS": the newest live line's own time
        self._unmounted = False
        self._stage: tempfile.TemporaryDirectory[str] | None = None
        self._stage_owner: int | None = None  # the lock that says this stage is in use (cache cleanup)
        self._console_dir = job_dir / "console"
        self._state_changed_at = None  # when the job entered its current (terminal) state
        self._job_created_at = None  # ties a kept console to this job, not a reused id
        self._job_reads = 0
        self._job_reads_lock = threading.Lock()
        self._job_seq = 0  # the number of the read whose answer the view shows (0: none yet)
        self._stdout_inflight = False
        self._stdout_ready = False
        self._stdout_have: set[str] = set()  # sub-jobs whose saved console is in while the rest download
        self._stdout_preview: dict[str, Path] = {}  # sub-jobs showing their console's newest chunks until it is in
        self._stdout_first: str | None = None  # the sub-job a download fetches before the others
        self._stdout_others = False  # ... and whether the job has others
        self._stdout_failed = False
        self._warm_up_started = False
        self._held: tuple[list[str], bool] | None = None
        self._init_log_state()
        self._active_tab = initial_tab if initial_tab in self._TABS else "logs"
        self._init_checkpoint_state()
        self._log_focus_id = "sources"
        self._metrics_root = job_dir / "metrics"
        self._serving_root = job_dir / "serving"
        self._tabs_shown: tuple | None = None  # the kind's tabs last applied (see _apply_tabs)
        # Per sub-job, what a download warned about (chunks it gave up on). The Logs
        # status line and the Resources header show the selected sub-job's.
        self._stdout_warnings: dict = {}
        self._metrics_warnings: dict = {}
        self._resource_data = ResourceData({})
        self._resource_loaded = False
        self._resource_loading = False
        self._resource_refresh_succeeded = False
        self._resource_error: str | None = None
        # The window whose rebuild failed (see _rebuild_resource_window) and why, until it is rebuilt.
        self._resource_window_error: tuple[str, str] | None = None
        self._resource_range = DEFAULT_RANGE
        self._resource_downloaded = False
        # When (monotonic) the newest successful whole-job download started and, per sub-job, the
        # newest successful download of it alone; the sub-jobs that whole-job download found chunks
        # for, less any a download alone has found none for since (see _resource_scope); the
        # sub-job the download running now is of, alone; and an r to do once it ends.
        self._resource_full_at: float | None = None
        self._resource_alone_at: dict[str, float] = {}
        self._resource_with_chunks: set[str] = set()
        self._resource_loading_scope: str | None = None
        self._resource_whole_pending = False
        self._resource_progress: dict[str, tuple[int, int]] = {}  # per sub-job, a download's (chunks done, of them)
        self._resource_ticker = None
        self._resource_pick: str | None = None
        self._resource_kind = GPU
        self._resource_kind_pref = GPU
        self._resource_unit: int | None = None
        self._resource_drawn: tuple | None = None  # (sub-job, view, unit, window) last drawn
        self._resource_on_view = False  # the panel's highlight is on a view's entry, not a unit's
        self._resource_shown: str | None = None  # the sub-job the panel and unit were set for
        self._resource_settled: str | None = None  # ... and whose view was chosen (see _follow_resource_source)
        self._resource_entries: list[tuple] = []
        self._resource_rows: list[tuple[ListItem, Label]] = []
        self._resource_list_width = 0
        self._resource_placing = False  # the redraw itself is moving the panel's highlight
        self._resource_stale_shown = False  # the panel marks every unit stale (see _resource_stale)
        self._sources_ready = False
        self._sub_job_missing = False
        self._job_read_started: float | None = None  # monotonic time the job read in flight began
        self._pending_top: int | None = None  # the paused reader's line a re-wrap is scrolling back to
        self._job_read_ticker = None  # repaints the status line's wait while that read is slow
        self._no_sources: str | None = None  # why the job has no sources to show (it couldn't be read, say)

    def compose(self) -> ComposeResult:
        yield Header(icon=theme.BRAND_ICON)
        yield Static("", id="summary")
        with JobTabs(initial=self._active_tab, id="job-tabs"):
            with TabPane("Serving", id="serving"):
                yield ServingPane(self._client, self._job_id, self._serving_root, job=self._tab_job,
                                  download=self._download_serving_console, id="serving-tab")
            with TabPane("Logs", id="logs"):
                with Horizontal(id="logs-body"):
                    yield _SourceList(id="sources")
                    # Log does not soft-wrap, so writes arrive pre-wrapped.
                    yield _LogPane(id="logview", highlight=False, max_lines=self._MAX_ROWS)
                status = Static("", id="log-status")
                status.display = False
                yield status
                # One row under the pane: whether it follows the log (see _log_live), the filter
                # box, and what a jump (e, n), the level and the filter found (see _log_chips).
                with Horizontal(id="log-bar"):
                    yield Static("", id="log-live")
                    yield Input(placeholder="/ filter   Esc clears", id="logfilter")
                    yield Static("", id="log-chips")
            with TabPane("Resources", id="resource"):
                with Horizontal(id="resource-body"):
                    yield _ResourceList(id="resource-list")
                    with VerticalScroll(id="resource-pane"):
                        yield _ColoredLinks("Select Resource to load GPU metrics.", id="resource-status")
                        yield Static("", id="resource-kpis")
                        yield Static("", id="resource-stats")
                        yield Static("", id="resource-charts")
                yield _ColoredLinks("", id="resource-keys")  # the window selector, under the pane
            with TabPane("Training metrics", id="training"):
                yield TrainingPane(self._client, self._job_id, self._metrics_root, job=self._tab_job,
                                   download=self._download_training_metrics, id="training-tab")
            with TabPane("Checkpoints", id="checkpoints"):
                yield from self._compose_checkpoint_tab()
        yield FittedFooter(id="footer")

    def on_mount(self) -> None:
        # First: no tail or download may write into the job's cache before it is leased,
        # or another TUI's cache limit could remove it under them.
        self._lease.acquire()
        self.set_interval(LEASE_HEARTBEAT_SECONDS, self._lease.heartbeat)
        self._cache_limit = getattr(self.app, "_CACHE_LIMIT", DEFAULT_CACHE_LIMIT)
        self._render_summary_line()
        self._apply_tabs()
        self.set_interval(self._DATA_REFRESH_SECONDS, self._auto_refresh_data)
        self.set_interval(self._STATUS_REFRESH_SECONDS, self._refresh_status)
        self._logview = self.query_one("#logview", Log)
        self._logview.auto_scroll = False
        self._logview.highlighter = LogHighlighter()
        self._logview.highlight = True
        if isinstance(self._logview, _LogPane):
            self._logview.on_scrolled = self._log_scrolled  # the bar says whether the reader follows the log
            self._logview.copy_text = self._copied_text  # a highlight copies the lines as logged
        self._apply_sources_width()
        self._update_subtitle()
        self._load_sources()
        self.set_timer(self._WARM_UP_AFTER_SECONDS, self._warm_up)
        after_refresh(self, self._focus_active_tab)
        self.set_interval(self._IDLE_CHECK_SECONDS, self._check_idle)

    def on_screen_resume(self) -> None:
        self._update_subtitle()  # a screen pushed over this one may have replaced it

    def _focus_active_tab(self) -> None:
        try:
            # Queued at mount (see on_mount). A tab key pressed before it runs has
            # already switched the tabs, and focusing the tab left behind would put the
            # focus in a hidden pane, which re-activates that pane and undoes the key
            # (the alternating activations on_tabbed_content_tab_activated guards
            # against). That activation focuses its own pane, so there is nothing to do.
            if (self.query_one("#job-tabs", TabbedContent).active or "logs") != self._active_tab:
                return
            pane = self._detail_pane()
            if pane is not None:
                pane.activate()  # it claims the focus itself (see detail_tabs.claim_focus)
                return
            focus = {"resource": "#resource-list", "checkpoints": "#checkpoint-table"}.get(self._active_tab, "#sources")
            self.set_focus(self.query_one(focus))
        except Exception:  # noqa: BLE001 - not mounted yet
            pass

    def on_unmount(self) -> None:
        # A running download owns its temp directory until it returns.
        self._unmounted = True
        self.workers.cancel_node(self)
        if self._logview is not None:
            self.workers.cancel_node(self._logview)
        if not self._stdout_inflight:
            self._cleanup_stage()
        # Not at once: this screen's threads can't be stopped mid-request, and a late answer
        # still writes into the job's cache, which must not be removed under it.
        self._lease.release_later()

    @property
    def _screen_open(self) -> bool:
        """Whether this screen remains mounted on the app stack."""
        return not self._unmounted and on_stack(self)

    def _logs_active(self) -> bool:
        return self._active_tab == "logs"

    def _resource_active(self) -> bool:
        return self._active_tab == "resource"

    def _detail_pane(self):
        """The Training metrics or Serving tab's pane, when it is the active tab (see detail_tabs.py)."""
        if self._active_tab not in ("training", "serving"):
            return None
        try:
            return self.query_one(f"#{self._active_tab}-tab")
        except NoMatches:  # not mounted yet
            return None

    def _tab_job(self) -> dict:
        """The job as this screen knows it, with its newest status, for the Training metrics and Serving tabs."""
        return {**self._job, "status": self._job_status or self._job.get("status")}

    def _warm_up(self) -> None:
        """UI thread. Start _warm_artifact_session, once."""
        if not self._warm_up_started:
            self._warm_up_started = True
            self._warm_artifact_session()

    @work(thread=True, exclusive=True, group="warm-up")
    def _warm_artifact_session(self) -> None:
        """Worker thread. Once the Logs tab has drawn its first lines (see _warm_up), the job's
        experiment run and a Snowflake session are made ready, so the first download a tab asks
        for (Resources, Training metrics, a saved console) starts with its listing, not a login
        (see the client's warm_artifact_session). A client without it, or a failure, leaves
        that to the download."""
        warm = getattr(self._client, "warm_artifact_session", None)
        if warm is None or not self._screen_open:
            return
        try:
            warm(self._job_id)
        except Exception:  # noqa: BLE001 - the download logs in itself, and says what failed
            # Not on screen, where the download says what failed, but in the error log, once per
            # job screen: the place to look when a job's first chart is slow.
            log_traceback(f"getting {self._job_id}'s downloads ready failed")

    def _download_training_metrics(self, *, priority, on_commit, cancelled, sub_job_ids=None) -> bool:
        """Worker thread. The Training metrics tab's download: the zone's own training rows,
        into the same cache directory as the Resources tab's GPU file and under the same
        :class:`DownloadLock`, so the two tabs (and the CLI) don't write that directory at
        once. The streams are separate, so this fetches its own chunks, only those of
        ``sub_job_ids`` when given (and the client takes them). False when the
        screen closed (or ``cancelled()``) before the lock was taken."""
        lock = DownloadLock(self._metrics_root)
        if not lock.acquire(lambda: self._screen_open and not cancelled()):
            return False
        before = telemetry_bytes(self._metrics_root)
        alone = {"sub_job_ids": sub_job_ids} if sub_job_ids else {}
        try:
            self._client.download_training_metrics(
                self._job_id,
                self._metrics_root,
                resume=True,
                priority=priority,
                on_commit=on_commit,
                **_quiet_download(self._client.download_training_metrics),
                **_download_options(self._client.download_training_metrics, alone),
            )
        finally:
            lock.release()
        self._after_telemetry_download(before)
        return True

    def _tabs(self) -> tuple:
        return views.job_tabs(self._job)

    def _apply_tabs(self) -> None:
        """UI thread. Show the tabs of the job's kind; a job opened by id learns it from its record."""
        shown = self._tabs()
        if shown == self._tabs_shown:
            return
        try:
            tabs = self.query_one("#job-tabs", TabbedContent)
        except NoMatches:  # not mounted yet: on_mount applies them
            return
        self._tabs_shown = shown
        if self._active_tab not in shown:
            tabs.active = "logs"
        for tab in self._TABS:
            if tab in shown:
                tabs.show_tab(tab)
            else:
                tabs.hide_tab(tab)
        self.refresh_bindings()

    def _download_serving_console(self, *, sub_job_id, cancelled, progress=None) -> list | None:
        """Worker thread. The Serving tab's download: the deployment's console, resumed, in a directory
        and DownloadLock of its own. The Logs tab's saved console is kept or dropped by rules of its own
        (_keep_console), so the two don't share files. Returns the download's warnings, or None when it
        didn't run. ``progress(done, total)`` gets the chunk count a few times a second."""
        lock = DownloadLock(self._serving_root)
        if not lock.acquire(lambda: self._screen_open and not cancelled()):
            return None
        before = telemetry_bytes(self._serving_root)

        last = [0.0]

        def on_chunk(_sub_job_id: str, done: int, total: int, _bytes: int = 0) -> None:
            if not self._screen_open or cancelled():
                raise DownloadCancelled("the Serving tab closed")
            now = time.monotonic()
            if progress is None or last[0] < 0 or (done < total and now - last[0] < self._PROGRESS_SECONDS):
                return
            last[0] = now
            try:
                progress(done, total)
            except Exception:  # noqa: BLE001 - a bug in drawing the count must not fail the download
                last[0] = -1.0  # no more counts
                log_traceback("showing the Serving download's progress failed (a bug in the TUI)")

        try:
            results = self._client.download_stdout_logs(
                self._job_id, self._serving_root, resume=True, sub_job_ids=[sub_job_id],
                on_chunk=on_chunk, **_quiet_download(self._client.download_stdout_logs))
        except DownloadCancelled:  # the next open resumes from the chunks saved so far
            results = None
        finally:
            lock.release()
        self._after_telemetry_download(before, self._serving_root)
        if results is None:
            return None
        warnings = _result_warnings(results, kind="stdout", job_id=self._job_id, output_dir=self._serving_root)
        return warnings.get(sub_job_id, [])

    def action_show_tab(self, tab: str) -> None:
        self.query_one("#job-tabs", TabbedContent).active = tab

    def action_cycle_tab(self) -> None:
        tabs = self.query_one("#job-tabs", TabbedContent)
        shown = self._tabs()
        index = shown.index(self._active_tab) if self._active_tab in shown else -1
        tabs.active = shown[(index + 1) % len(shown)]

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        # Ignore queued stale activations, which would re-activate each other.
        if event.tabbed_content.active != event.pane.id or not self._screen_open:
            return  # (or a tab key just before a back key: the screen is being removed)
        self._active_tab = event.pane.id or "logs"
        self.clear_selection()  # what was highlighted is on the tab left behind
        self.refresh_bindings()  # check_action scopes the footer's keys to the tab
        self._update_subtitle()  # the Logs tab's state shows only on Logs
        focused = self.focused
        if not self._logs_active() and focused is not None and focused.id in {"sources", "logview", "logfilter"}:
            self._log_focus_id = focused.id
        pane = self._detail_pane()
        if pane is not None:
            reopened = isinstance(pane, ServingPane) and pane.opened
            pane.activate()  # it claims the focus itself (see detail_tabs.claim_focus)
            if reopened and (self._serving_may_change() or pane.status_changed_since_read):
                # Only the tab on screen polls, so drains (or the job's end) may have come while it was off
                # screen. The download resumes: this fetches only new chunks.
                pane.reload()
            return
        if self._resource_active():
            self.set_focus(self.query_one("#resource-list", ListView))  # its arrows pick a view or unit
            self.call_after_refresh(self._render_resource)
            self._start_resource_refresh(force=False)
            return
        if self._checkpoints_active():
            focused = self.focused
            if focused is not None and focused.id in {"sources", "logview", "logfilter"}:
                self._log_focus_id = focused.id
            self.set_focus(self.query_one("#checkpoint-table", DataTable))
            self._render_checkpoints()
            self._start_checkpoint_refresh(force=False)
            return
        try:
            target = self.query_one(f"#{self._log_focus_id}")
        except Exception:  # noqa: BLE001 - stale focus id after a remount
            target = self.query_one("#sources", ListView)
        self.set_focus(target)
        # Re-wraps only if rows were written at another width while Logs was hidden.
        self.call_after_refresh(self._rewrap)

    def _sources_cap(self) -> int:
        """The widest the sources panel may be: the log keeps _LOG_MIN_WIDTH cells."""
        width = getattr(getattr(self, "size", None), "width", 0) or 0
        return max(self._SOURCES_MIN, width - self._LOG_MIN_WIDTH) if width else self._SOURCES_AUTO_MAX

    def _fitted_sources_width(self) -> int:
        labels = [self._source_label(sid) for sid in self._source_by_item.values()]
        longest = max((views.cell_width(label) for label in labels), default=0) + 2  # with the "● " before it
        return min(self._SOURCES_AUTO_MAX, max(self._SOURCES_AUTO_MIN, longest + 4))

    def _apply_sources_width(self) -> None:
        width = self._fitted_sources_width() if self._sources_auto else self._sources_width
        width = max(self._SOURCES_MIN, min(width, self._sources_cap()))
        changed = width != self._sources_width
        self._sources_width = width
        try:
            self.query_one("#sources", ListView).styles.width = width
        except Exception:  # noqa: BLE001 - not mounted yet
            return
        if changed:
            after_refresh(self, self._rewrap)  # log pane width changed

    def action_grow_sources(self) -> None:
        if not self._logs_active():
            return
        self._sources_auto = False
        self._sources_width = min(self._sources_cap(), self._sources_width + self._SOURCES_STEP)
        self._apply_sources_width()
        self.call_after_refresh(self._rewrap)  # log pane width changed

    def action_shrink_sources(self) -> None:
        if not self._logs_active():
            return
        self._sources_auto = False
        self._sources_width = max(self._SOURCES_MIN, self._sources_width - self._SOURCES_STEP)
        self._apply_sources_width()
        self.call_after_refresh(self._rewrap)  # log pane width changed

    def on_resize(self, event) -> None:
        self.call_after_refresh(self._render_summary_line)
        # Reflow only the in-memory views; resize never performs I/O.
        self._apply_sources_width()  # fitted again, and never wider than the screen allows
        self._schedule_rewrap()
        if self._resource_active():
            self.call_after_refresh(self._render_resource)

    def _tabs_resized(self) -> None:
        """Reflow the log and charts after any tab-width change."""
        self._schedule_rewrap()
        if self._resource_active():
            after_refresh(self, self._render_resource)

    # Debounce drag-resize rewrapping.
    _REWRAP_SETTLE_SECONDS = 0.15

    def _schedule_rewrap(self) -> None:
        """UI thread. Wrap the log pane again for its new width once a resize settles (see _rewrap)."""
        if self._rewrap_timer is not None:
            self._rewrap_timer.stop()
        self._rewrap_timer = self.set_timer(self._REWRAP_SETTLE_SECONDS, partial(after_refresh, self, self._rewrap))

    def action_refresh_sources(self) -> None:
        pane = self._detail_pane()
        if pane is not None:
            pane.action_refresh()
            return
        if self._resource_active():
            self._start_resource_refresh(force=True, whole=True)  # every sub-job, as a first download
            return
        if self._checkpoints_active():
            self._start_checkpoint_refresh(force=True)
            return
        if self._job_read_started is not None:
            # The job is still being read (a thread can't be cancelled): a second read
            # would only wait beside it. Say so, rather than nothing.
            waited = int(time.monotonic() - self._job_read_started)
            self.notify(f"Still reading this job ({waited}s); r reads it again once it answers.", markup=False)
            return
        self._stdout_failed = False  # r retries a saved-console download that failed
        if self._stdout_ready and self._stdout_recent:
            # Fetched moments after the job ended: add the chunks uploaded since,
            # and read the execution logs again (shown for sub-jobs without a console).
            self._stdout_ready = False
            self._stdout_have.clear()
            self._stdout_again = True
            self._exec_logs.clear()
            self._exec_failed.clear()
        self._load_sources()

    def _update_subtitle(self) -> None:
        # Dynamic status line: everything that changes over the session lives
        # here, leaving the #summary line for the job's identity (its status badge
        # aside). The status comes first, so a narrow header still shows it.
        if not self._screen_open or self.app.screen is not self:
            return  # a late tail update must not overwrite another screen's subtitle
        bits = []
        st = (self._job_status or "").strip()
        if st:
            bits.append(st.removeprefix("JOB_STATE_"))
        bits.append(str(self._job_id))
        if self._logs_active():  # the Logs tab's own state
            if self._last_line_at:
                bits.append(self._last_line_at)
            if self._filter:
                bits.append(f"filter:{self._filter}")
            if self._min_level:
                bits.append(f"≥{self._min_level}")
            if self._paused:
                bits.append("PAUSED")
        warnings = self._source_warnings()
        if warnings:
            bits.append(_warning_text(warnings))
        # Plain text (the header doesn't read markup), but service text could
        # still carry an escape sequence or a line break.
        self.app.sub_title = "  ·  ".join(" ".join(safe_text(bit).split()) for bit in bits)
        self._render_log_bar()  # the pause, the filter and the level show there too

    def _source_warnings(self) -> list:
        """What the downloads warned about for the sub-job on screen, each warning once."""
        source = self._resource_source() if self._resource_active() else self._current_source
        if not source:
            return []
        merged: dict = {}
        for table in (self._stdout_warnings, self._metrics_warnings):
            merged.update(dict.fromkeys(table.get(source, [])))
        return list(merged)

    def _accept_download_warnings(self, kind: str, warnings: dict, covered: list | None = None) -> None:
        """UI thread. Keep one download's warnings and show them on the lines that carry them.
        A metrics download of the ``covered`` sub-jobs alone leaves the others' as they were."""
        if not self._screen_open:
            return
        if kind == "stdout":
            self._stdout_warnings.update(warnings)
        elif covered is None:
            self._metrics_warnings = warnings
        else:
            kept = {sub_job_id: messages for sub_job_id, messages in self._metrics_warnings.items()
                    if sub_job_id not in covered}
            self._metrics_warnings = {**kept, **warnings}
        self._update_subtitle()
        if self._resource_active():
            self._render_resource()

    # The fewest cells the models are cut to so the summary line fits on one row.
    _SUMMARY_LEAST = 12

    def _render_summary_line(self) -> None:
        job = self._ended_job()  # the newest status and reason: the tail can see the job end first
        job_id = views.job_id_of(job) or self._job_id
        # A sub-job's id starts with its job's (JOB_ID:TYPE:INDEX): the chip names
        # the rest, beside the job's id.
        if not self._sub_job_id:
            scope = "all sub-jobs"
        elif str(self._sub_job_id).startswith(f"{job_id}:"):
            scope = f"sub-job {str(self._sub_job_id)[len(job_id) + 1:]}"
        else:
            scope = f"sub-job {self._sub_job_id}"
        # Why the job is where it is, in the service's words: reason and reason_details,
        # or the sentence it sends when the job's status is unavailable.
        why = views.status_unavailable_text(job) if views.is_unknown(job) else views.reason_text(job)
        word = views.status_word(job.get("status"))
        style = theme.ERR if word == "failed" else theme.WARN if views.is_live(job) else theme.ENDED
        # Where the job is (its calls go there), when the account-wide list says it is
        # somewhere other than the configured location.
        location = views.location_label(job) if self.app.job_elsewhere(job) else ""
        least = self._SUMMARY_LEAST
        # The job id in bold, its status badge, then chips for what the job is, and
        # the reason. The models are cut to keep the line on one row; the rest stays whole.
        # A deployment's badge in its own words (ready, ...); a job's as the badge words it.
        badge_word = views.status_word_for(job, job.get("status")) if views.is_deployment(job) else None
        segments = [(job_id, f"bold {theme.TEXT}", None, False), theme.status_badge(job.get("status"), word=badge_word),
                    (views.job_type_label(job), "", None, True),
                    (", ".join(views.job_models(job)) or "—", "", least, True),
                    (scope, "", None, True), *([(location, "", None, True)] if location else []),
                    *([(why, style, None, False)] if why else [])]
        width = self._summary_width()
        line = fit_line(segments, width)
        if "\n" in line.plain:
            # The header's subtitle names the job too: a narrow line leaves the id
            # out rather than wrap.
            line = fit_line(segments[1:], width)
        self._set_summary(line)

    def _summary_width(self) -> int:
        """The cells the summary line has: the screen's, less its padding, before the first layout."""
        try:
            width = self.query_one("#summary", Static).size.width
        except Exception:  # noqa: BLE001 - not mounted, or already removed
            width = 0
        return width or max(40, self.app.size.width - 2)

    def _set_summary(self, text: Content) -> None:
        try:
            self.query_one("#summary", Static).update(text)
        except Exception:  # noqa: BLE001 - not mounted yet
            pass

    # ─── filter / level / pause ──────────────────────────────────────────
    def check_action(self, action: str, parameters: tuple) -> bool | None:
        if action == "filter" and isinstance(self.focused, Input):
            return False  # in a text box, / is typed
        if action in self._LOG_ONLY:
            return self._logs_active()
        if action == "resource_sub_job":
            # Only where it does something: on Resources, with another sub-job to switch to.
            return self._resource_active() and len(self._resource_sub_job_ids()) > 1
        if action == "cycle_window":
            return self._resource_active()
        if action in ("checkpoint_sort", "checkpoint_order", "checkpoint_download"):
            return self._checkpoints_active()
        if action == "show_tab":
            return not parameters or parameters[0] in self._tabs()  # a hidden tab's key does nothing
        return True

    def action_filter(self) -> None:
        if not self._logs_active():
            return
        self.set_focus(self.query_one("#logfilter", Input))

    def action_back(self) -> None:
        if not is_shown(self):
            return  # still being put up, or no longer on top (see is_shown)
        # Opened straight from the command line, nothing is underneath: show the
        # job list rather than an empty screen.
        if len(self.app.screen_stack) > 2:
            self.app.pop_screen()
        else:
            # The app's list (its own client, not this job's, and its account-list state).
            self.app.switch_screen(self.app.job_list_screen())

    def action_dismiss_or_back(self) -> None:
        if not is_shown(self) or close_keys_panel(self):
            return
        if not self._logs_active():
            self.action_back()
            return
        # Esc clears an active filter (and unfocuses the input); otherwise backs out.
        inp = self.query_one("#logfilter", Input)
        if self.focused is inp or self._filter:
            inp.value = ""  # fires on_input_changed → clears the filter
            self.set_focus(self.query_one("#sources", ListView))
        else:
            self.action_back()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "logfilter" and self._screen_open:
            self._filter = event.value.strip()
            self._update_subtitle()
            # Repaint once typing pauses. The filter applies to the lines already
            # here, so a key press never restarts the tail or sends a request.
            if self._filter_timer is not None:
                self._filter_timer.stop()
            self._filter_timer = self.set_timer(self._FILTER_DEBOUNCE_SECONDS, self._refilter)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "logfilter" and self._screen_open:
            self.set_focus(self.query_one("#sources", ListView))

    def action_cycle_level(self) -> None:
        if not self._logs_active():
            return
        i = self._LEVEL_CYCLE.index(self._min_level) if self._min_level in self._LEVEL_CYCLE else 0
        self._min_level = self._LEVEL_CYCLE[(i + 1) % len(self._LEVEL_CYCLE)]
        self._update_subtitle()
        self._refilter()

    def action_toggle_pause(self) -> None:
        if not self._logs_active():
            return
        if self._idle_paused:
            # p resumes a tail paused for lack of input (it picks up from its
            # cursor); auto-scroll stays as it was.
            self._idle_paused = False
            self._resumed_at = time.monotonic()
            self._render_log_status()
            return
        # Scrolled up from the end of a log that goes on, the bar says p follows: it
        # goes back to the end and follows again, as from a pause.
        self._paused = not self._paused and not (self._tail_live and self._away_from_end())
        if not self._paused:
            self._unseen = 0
            self._jump = None
            if isinstance(self._logview, _LogPane):
                self._logview.set_mark(None)
        # Resuming jumps to the end and follows again; pausing just stops
        # following (the user can scroll freely). A saved console that arrived
        # while paused replaces the pane now.
        if not self._paused:
            held = self._held
            self._held = None
            if held is not None:
                lines, empty_note = held
                if lines:
                    # Wrapped off the UI thread (see _repaint): a console can be 20,000 lines.
                    self._repaint([(None, line) for line in lines])
                else:
                    self._clear_pane()
                    if empty_note:
                        self._write_line(self._no_log_output())
                self._render_log_status()
            if self._logview is not None:
                self._logview.scroll_end(animate=False)
        self._update_subtitle()

    _JUMPS = {"error": (ERROR_KINDS, "error"), "issue": (ISSUE_KINDS, "problem")}

    def action_jump(self, which: str, step: int) -> None:
        """e / E (n / N): the pane's next error, or the one before (warnings too), marked, a third
        of the way down the view, and the reader's place kept there as if paused (p resumes). From
        the end of the log, the newest one; round past either end, as a pager's search goes."""
        lv = self._logview
        if not self._logs_active() or which not in self._JUMPS or not isinstance(lv, _LogPane):
            return
        wanted, noun = self._JUMPS[which]
        kinds = self._shown_kinds
        if lv.mark is not None:
            start = self._line_at_row(lv.mark[0])
        elif self._away_from_end():
            top = self._line_at_row(int(lv.scroll_offset.y))
            start = None if top is None else top - 1 if step > 0 else top
        else:
            start, step = None, -1  # following the log: the newest one first
        index = next_line_of(kinds, start, step, wanted)
        if index is None:
            self._jump = (noun, 0, 0)
            self._render_log_bar()
            return
        n, total = place_of(kinds, index, wanted)
        self._jump = (noun, n, total)
        first, end = self._line_rows(index)
        if not self._paused:
            self._paused = True  # the reader stays here as lines come (p resumes)
            self._update_subtitle()
        lv.set_mark((first, end))
        lv.scroll_to(y=max(0, first - lv.scrollable_content_region.height // 3), animate=False)
        self._render_log_bar()

    def _recount_jump(self) -> None:
        """UI thread. The last jump's place among the lines the pane holds now: which of how many
        the marked line is, or only how many once that line has left the pane."""
        noun = self._jump[0]
        wanted = next(kinds for kinds, name in self._JUMPS.values() if name == noun)
        lv = self._logview
        index = self._line_at_row(lv.mark[0]) if isinstance(lv, _LogPane) and lv.mark is not None else None
        if index is None:
            self._jump = (noun, 0, sum(kind in wanted for kind in self._shown_kinds))
        else:
            self._jump = (noun, *place_of(self._shown_kinds, index, wanted))

    def _line_rows(self, index: int) -> tuple[int, int]:
        """The pane rows [start, end) of line ``index`` of _shown_lines: its own, not its day's rule."""
        first = sum(self._shown_rows[:index])
        rows = self._shown_rows[index]
        kinds = getattr(self._logview, "row_kinds", [])
        if rows > 1 and 0 <= first < len(kinds) and kinds[first] == DAY_KIND:
            first, rows = first + 1, rows - 1
        return first, first + rows

    def _refilter(self) -> None:
        """Apply the filter and level to what this view already has; never re-tail.

        A saved console is re-read from its local file; a live or cached tail is
        repainted from the lines it has received. Either way the work is done
        off the UI thread, and it lands on the newest lines.
        """
        if not self._current_source:
            return
        highlighter = getattr(self._logview, "highlighter", None)
        if isinstance(highlighter, LogHighlighter):
            highlighter.query = self._filter  # the repaint below marks the new filter's matches
        self._held = None
        self._paused = False
        self._update_subtitle()
        if self._console_shown(self._current_source):
            self._paint_saved()
        else:
            # Older lines may have left the buffer before this repaint.
            self._repaint(list(self._entries), self._filter, self._min_level, truncated=bool(self._entries_dropped))
        if self._logview is not None:
            self._logview.scroll_end(animate=False)
        self._render_log_status()

    # ─── export / copy ───────────────────────────────────────────────────
    # The most a copy of the whole log (ctrl+c, nothing highlighted) takes. OSC 52 carries the text
    # base64-encoded in one escape sequence; many terminals drop a much bigger
    # one without a word, and writing it holds the screen up meanwhile.
    _COPY_MAX_BYTES = 1024 * 1024

    def _export_target(self) -> tuple[str, str, list[str]] | None:
        """UI thread. ``(source, path, shown)``: ``path`` is the saved console
        (empty: export the tail cache); ``shown`` is the execution log the pane
        shows for a sub-job without a console, the export when the cache is empty."""
        src = self._current_source
        if not src:
            return None
        if self._saved_shown(src):
            path = self._stdout_path(src)
            try:
                if path.is_file() and path.stat().st_size > 0:
                    return src, str(path), []
            except OSError:
                pass
        notes, records = self._exec_logs.get(src, ([], []))
        shown = [*notes, *(line for _, lines in records for line in lines)] if records else []
        return src, "", shown

    def _exported_text(self, src: str, shown: list[str]) -> str:
        entries = self._cache.load_entries(src)
        if not entries and shown:
            return "\n".join(shown)
        return "\n".join(format_log_entry(entry) for entry in entries)

    def _export_name(self, src: str) -> str:
        """``~/cortex-training-<job8>-<type>-<index>.log`` for sub-job ``<job>:<type>:<index>``."""
        job = str(self._job_id)
        rest = src[len(job) + 1:] if src.startswith(f"{job}:") else src
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in rest).strip(".-") or "log"
        return os.path.expanduser(f"~/cortex-training-{job[:8]}-{safe}.log")

    _EXPORT_NOTE_SECONDS = 15  # how long what s did stays on the status line

    def _show_export_note(self, text: str) -> None:
        """UI thread. What s did, on the status line (not in the log), for a while."""
        self._export_note = text
        self._render_log_status()
        if self._export_timer is not None:
            self._export_timer.stop()
        self._export_timer = self.set_timer(self._EXPORT_NOTE_SECONDS, self._clear_export_note)

    def _clear_export_note(self) -> None:
        self._export_timer = None
        if self._export_note is not None:
            self._export_note = None
            self._render_log_status()

    def action_save_log(self) -> None:
        if not self._logs_active():
            return
        target = self._export_target()  # what the pane shows now, for the worker
        if target is not None:
            self._save_log(*target)

    @work(thread=True, exclusive=True, group="export")
    def _save_log(self, src: str, file_path: str, shown: list[str]) -> None:
        """Worker thread. Save what :meth:`_export_target` named: the saved console at
        ``file_path`` copied as it is, never read into memory (it can be gigabytes), else
        the lines the tail cached, else ``shown``."""
        path = self._export_name(src)
        try:
            if file_path:
                if not os.path.getsize(file_path):  # emptied since: as for an empty log, below
                    self._post(self._show_export_note, "nothing to save yet for this sub-job")
                    return
                shutil.copyfile(file_path, path)
                count = self._count_lines(path)
            else:
                text = self._exported_text(src, shown)
                if not text:
                    # Nothing yet: never write (or overwrite an earlier save with) an empty file.
                    self._post(self._show_export_note, "nothing to save yet for this sub-job")
                    return
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(text + "\n")
                count = len(text.splitlines())
            msg = f"saved {self._count(count, 'line')} to {path}"
        except OSError as exc:  # noqa: BLE001
            msg = f"save failed: {exc}"
        self._post(self._show_export_note, msg)

    # What ends a line for str.splitlines(), besides \r\n.
    _LINE_BREAKS = ("\n", "\r", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x85", "\N{LINE SEPARATOR}",
                    "\N{PARAGRAPH SEPARATOR}")

    @classmethod
    def _count_lines(cls, path: str) -> int:
        """Worker thread. ``len(text.splitlines())`` for the file at ``path`` decoded as the
        pane decodes it, read a block at a time rather than whole (a console can be
        gigabytes). ``newline=""`` leaves each line break as it is."""
        count, last = 0, ""
        with open(path, encoding="utf-8", errors="replace", newline="") as handle:
            while block := handle.read(1 << 20):
                count += sum(block.count(mark) for mark in cls._LINE_BREAKS) - block.count("\r\n")
                if last == "\r" and block.startswith("\n"):
                    count -= 1  # one \r\n, split between two blocks
                last = block[-1]
        return count + (last != "" and last not in cls._LINE_BREAKS)  # a last line without a break

    @work(thread=True, exclusive=True, group="copy")
    def _copy_log(self) -> None:
        try:
            target = self.app.call_from_thread(self._export_target)
        except RuntimeError:
            return
        if target is None:
            return
        src, path, shown = target
        if path:
            tail = read_saved_tail(
                Path(path),
                max_lines=10**9,
                max_bytes=self._COPY_MAX_BYTES,
                max_line_bytes=self._COPY_MAX_BYTES,
            )
            text, truncated = ("\n".join(tail.lines), tail.truncated) if tail is not None else ("", False)
        else:
            text, truncated = self._exported_text(src, shown), False
        if not text:
            self._post(self.notify, "nothing to copy yet for this sub-job", timeout=3)
            return
        # Without escape sequences or control characters: pasted into another
        # terminal, those would act there. The newest lines, up to the cap.
        text = safe_text(text).removesuffix("\n")
        raw = text.encode("utf-8")
        if len(raw) > self._COPY_MAX_BYTES:
            truncated = True
            raw = raw[-self._COPY_MAX_BYTES:]
            newline = raw.find(b"\n")
            if newline >= 0:
                raw = raw[newline + 1:]
                prefix = ""
            else:
                raw = raw[3:]  # room for the ellipsis below
                while raw and raw[0] & 0xC0 == 0x80:
                    raw = raw[1:]
                prefix = "…"
            text = prefix + raw.decode("utf-8")
        lines = text.split("\n")
        what = (f"the log's last {self._count(len(lines), 'line')} (s saves them all)"
                if truncated else f"the whole log ({self._count(len(lines), 'line')})")
        self._post(self.app.copy_out, text, what)

    def action_copy_text(self) -> None:
        """ctrl+c, on every tab: what the mouse highlighted, else
        everything the tab holds: the log (its newest lines, up to the cap) on Logs; on
        Resources the panel, the header, every stats row and every chart; on Checkpoints the
        list and the highlighted checkpoint's details."""
        if self.app.copy_highlight(self):
            return
        if self.query(HelpPanel):
            self.app.copy_everything(self)  # the key list, with the tab beside it
        elif self._logs_active():
            self._copy_log()
        elif self._checkpoints_active():
            self.app.copy_everything(self, "the Checkpoints tab")
        else:
            self.app.copy_everything(self, f"the {self._TAB_NAMES[self._active_tab]} tab")

    def _load_sources(self) -> None:
        """UI thread. Read the job, then show its sub-jobs (see _read_sources)."""
        # The row this screen was opened with (the list's or the preview's copy), as it is now.
        self._read_sources(dict(self._job))

    @work(thread=True, exclusive=True, group="sources")
    def _read_sources(self, row: dict) -> None:
        worker = get_current_worker()
        # Fetch the job once (GS metadata — survives after the zone is gone) for
        # the summary header, status (drives follow/cache), and the sub-job list.
        failed: Exception | None = None
        self._post(self._job_read_began)
        seq = self._next_job_read()
        try:
            job = self._client.get_job(self._job_id) or {}
        except Exception as exc:  # noqa: BLE001 - best-effort; said in the pane when nothing else can show
            job, failed = {}, exc
            log_traceback(f"reading job {self._job_id} failed")
        finally:
            self._post(self._job_read_ended)
        if not isinstance(job, dict):
            job = {}
        # An answer without a status or sub-jobs ({}) says no more than a failed call.
        answered = bool(job.get("status") or job.get("sub_jobs"))
        if answered:
            # Status, reason and times are the UI thread's to take, in the order the reads
            # were sent (see _take_job).
            self._post(self._take_job, job, seq)
        # get_job failed or answered nothing: the row this screen was opened with still
        # names the sub-jobs, or says the status is unavailable.
        listed = job if answered else row

        # Each sub-job is a tailable source (its zone-manager / Ray-head pod,
        # whose stdout already includes worker output via Ray log_to_driver).
        sources = []
        for sj in views.sub_jobs(listed):  # an UNKNOWN row's ids are derived from its configs
            if isinstance(sj, dict):
                sid = sj.get("sub_job_id") or sj.get("id")
                if sid:
                    sources.append({"sub_job_id": sid, "job_type": sj.get("job_type") or sj.get("type"),
                                    "status": sj.get("status")})
        # Training first: the service may list sampling first, and training is where
        # a job's progress (and most of its GPUs) show.
        sources.sort(key=lambda s: str(s.get("job_type") or "").lower() != "training")
        if views.is_unknown(listed) and self._sub_job_id and all(s["sub_job_id"] != self._sub_job_id for s in sources):
            # While its status is unavailable the service can't say which sub-jobs exist: try the one asked for.
            sources.append({"sub_job_id": self._sub_job_id})
        if not sources:
            # get_job failed/empty (e.g. offline): fall back to cached sub-jobs.
            source_ids = self._cache.cached_sources()
            source_ids.extend(cached_metric_sources(self._metrics_root))
            sources = [{"sub_job_id": sid} for sid in dict.fromkeys(source_ids)]
        why = None if sources else self._no_sources_text(listed, failed, answered)
        if worker.is_cancelled:
            return
        self._post(self._populate_sources, sources, why)

    def _no_sources_text(self, listed: dict, error: Exception | None, answered: bool) -> tuple[str, str]:
        """Worker thread. Why there is nothing to show (no sub-jobs, and no cached logs or
        metrics): the pane line's tag, ``[error]`` when the read itself failed, and the reason."""
        if views.is_unknown(listed):
            return "[no logs]", f"{views.status_unavailable_text(listed)}; no logs are cached on this machine"
        if getattr(getattr(error, "response", None), "status_code", None) == 404:
            # The job isn't in this client's database and schema: with the account-wide
            # job list, a job id copied from it may well be in another one.
            where = self._configured_location()
            place = ".".join((where["database"], where["schema"])) if where["database"] and where["schema"] else (
                "the configured database and schema")
            return "[no logs]", (
                f"job {self._job_id} was not found in {place} ({views.describe_error(error)}), and no logs are "
                "cached on this machine. If the job list shows it in another location, start the TUI with that "
                "--database and --schema.")
        if error is not None:
            # The read failed (an outage, a timeout): the job may well have sub-jobs and logs.
            bounds = self._read_bounds()
            tries = f" ({bounds[0]} tries)" if bounds and bounds[0] > 1 else ""
            return "[error]", (f"couldn't read job {self._job_id}: {views.describe_error(error)}{tries}; no logs "
                               f"are cached on this machine  (r tries again · details in {_ERROR_LOG})")
        if not answered:
            return "[no logs]", ("the service sent no details for this job (an empty answer), so its sub-jobs "
                                 "aren't known, and no logs are cached on this machine; r asks again")
        return "[no logs]", self._NO_SUB_JOBS

    _NO_SUB_JOBS = "this job has no sub-jobs and no cached logs are available"

    def _read_bounds(self) -> tuple[int, float] | None:
        """(tries, read timeout in seconds) of each request this screen's client sends,
        when the client says (the TUI's own: 3 tries of up to 45 s each)."""
        retries = getattr(self._client, "max_retries", None)
        timeout = getattr(self._client, "request_timeout", None)
        read = timeout[-1] if isinstance(timeout, tuple) and timeout else timeout
        if type(retries) is int and isinstance(read, (int, float)) and not isinstance(read, bool) and read > 0:
            return retries + 1, float(read)
        return None

    def _job_read_began(self) -> None:
        """UI thread. The job read that names the sources started: show that the pane waits for it."""
        self._job_read_started = time.monotonic()
        if not self._shown_lines:  # nothing on screen yet: say what it is waiting for
            self._write_line(f"— reading job {self._job_id} —")
        if self._job_read_ticker is None:
            self._job_read_ticker = self.set_interval(1.0, self._render_log_status)
        else:
            self._job_read_ticker.resume()

    def _job_read_ended(self) -> None:
        """UI thread. The job read answered, or gave up."""
        self._job_read_started = None
        if self._job_read_ticker is not None:
            self._job_read_ticker.pause()
        self._render_log_status()

    # A read that answers within this long shows nothing on the status line.
    _JOB_READ_QUIET_SECONDS = 2.0

    def _job_read_note(self) -> str | None:
        """The status line's note while the job read is slow: how long it has waited,
        and how long it may still (a stalled service is asked again after each timeout)."""
        if self._job_read_started is None:
            return None
        waited = time.monotonic() - self._job_read_started
        if waited < self._JOB_READ_QUIET_SECONDS:
            return None
        note = f"reading the job: no answer yet after {int(waited)} s"
        bounds = self._read_bounds()
        if bounds:
            tries, read = bounds
            note += f" (up to {tries} tries of {read:g} s each)" if tries > 1 else f" (up to {read:g} s)"
        return note

    async def _populate_sources(self, sources, why: tuple[str, str] | None = None) -> None:
        """UI thread. ``why``: the pane line's tag and the reason there are no sources
        (the job couldn't be read, say); see :meth:`_no_sources_text`."""
        lv = self.query_one("#sources", ListView)
        await lv.clear()  # await removal before re-appending src-* ids (see _populate)
        self._source_by_item = {}
        self._source_items = {}
        self._source_labels = {}
        self._sources_ready = True
        self._sub_job_missing = False
        self._no_sources = why[1] if why is not None else None
        items = [s for s in sources if s.get("sub_job_id")]
        if not self._job_seq:
            # No get_job answer taken yet: the row the view was opened with says how the
            # sub-jobs are. Once one is, _take_job keeps them as the newest answer says.
            self._sub_job_status = {s["sub_job_id"]: s["status"] for s in items if s.get("status")}
        for source in self._exec_failed:  # r retries an execution log that couldn't be read
            self._exec_logs.pop(source, None)
        self._exec_failed.clear()
        if why is not None:
            self._clear_pane()
            self._write_line(" ".join(why))
        if self._sub_job_id:
            known = [s["sub_job_id"] for s in items]
            items = [s for s in items if s["sub_job_id"] == self._sub_job_id]
            if not items:
                self._current_source = None
                if why is None:
                    self._sub_job_missing = True
                    lv.append(ListItem(Label(Content.from_markup("error: sub-job not found: $sub_job",
                                                                 sub_job=self._sub_job_id))))
                    # The sources column is narrow: say it in full in the log pane too, with the ids there are.
                    self._clear_pane()
                    ids = ", ".join(known) if known else "none it can name"
                    self._write_line(views.collapse(f"error: sub-job not found: {self._sub_job_id}; this job's "
                                                    f"sub-jobs: {ids} (--sub-job-id takes one of them)"))
                if self._resource_active():
                    self._render_resource()
                return
        for i, s in enumerate(items):
            sid = s["sub_job_id"]
            item_id = f"src-{i}"
            self._source_by_item[item_id] = sid
            self._source_labels[sid] = views.sub_job_label(s)  # its type as the preview and Resources name it
            label = Label(self._source_text(sid))
            item = ListItem(label, id=item_id)
            self._source_items[sid] = (item, label)  # (the label, before the item mounts it)
            lv.append(item)
        self._apply_sources_width()  # fitted to these labels
        self.refresh_bindings()  # < and > need two or more sub-jobs
        if not items:
            self._current_source = None
        if items:
            # A refresh (r) stays on the source the user picked.
            keep = next((i for i, s in enumerate(items) if s["sub_job_id"] == self._current_source), 0)
            lv.index = keep
            self._start_tail(items[keep]["sub_job_id"])
        if self._resource_active():
            self._render_resource()
            self._start_resource_refresh(force=False)

    def _source_label(self, sid: str) -> str:
        return self._source_labels.get(sid) or views.sub_job_label(sid)

    def _source_text(self, sid: str) -> Content:
        """A sub-job's row in the list: its status's glyph and name, and its status under them
        (the name bold for the one the pane shows)."""
        word = self._tail_status(sid)
        look = theme.status_look(word)
        shown = sid == self._current_source
        return Content.assemble((f"{look.glyph} ", look.color),
                                (self._source_label(sid), f"bold {theme.TEXT}" if shown else ""),
                                ("\n  " + views.status_word_for(self._job, word),
                                 theme.TEXT if shown else theme.FAINT))

    def _paint_sources(self) -> None:
        """UI thread. Draw the sub-job list again: which one the pane shows, and each one's status."""
        for sid, (item, label) in self._source_items.items():
            item.set_class(sid == self._current_source, "-shown")
            label.update(self._source_text(sid))

    def _display(self, line: str) -> str:
        """How a buffered line is shown: a log line's display text (see
        ``format.display_line``), the TUI's own notes only made safe to draw.
        The buffer (and save and copy) keep the line as it came."""
        if isinstance(line, self._Note):
            return safe_text(line)  # never cached: the cache is by text, and a log line may read the same
        text = self._display_cache.get(line)
        if text is None:
            text = self._display_cache[line] = display_line(line)
        return text

    # A log line's later rows (wrapped, or a record's traceback) start under the end of its
    # time ("HH:MM:SS "), so the times stay one column down the pane's left edge.
    _HANGING_INDENT = " " * 9

    def _line_prefixes(self) -> tuple[str, str]:
        """What starts the first row of a log line, and each row after it. No sub-job tag:
        the pane shows one sub-job, the one selected beside it."""
        return "", self._HANGING_INDENT

    def _prefixes(self, line: str) -> tuple[str, str]:
        """The row prefixes of ``line``: none for the TUI's own notes."""
        return ("", "") if isinstance(line, self._Note) else self._line_prefixes()

    def _wrap(self, line: str, width: int) -> list[str]:
        """The rows ``line`` takes in a pane ``width`` cells wide."""
        first, rest = self._prefixes(line)
        text = self._display(line)
        if not isinstance(line, self._Note):
            text = compact_stamp(text)  # the date is on the day's rule (see in_context)
        return wrap_log_line(text, width, first=first, rest=rest, max_rows=self._MAX_ROWS_PER_LINE)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if not self._screen_open:
            return
        sid = self._source_by_item.get(event.item.id or "")
        # Re-clicking the source you're already viewing shouldn't reload/reset
        # the pane (that's what caused the jump-to-bottom on every click).
        if sid and sid != self._current_source:
            self._start_tail(sid)

    def _post(self, fn, *args, **kwargs) -> bool:
        """Run ``fn`` on the UI thread from a worker; return ``False`` instead of
        raising once the app/event loop is gone (teardown), so background workers
        stop cleanly rather than surfacing a NoActiveAppError as a WorkerError.

        Only teardown-class errors (``RuntimeError`` — NoActiveAppError and
        "event loop is closed" both derive from it) are swallowed; a real bug in
        ``fn`` (Attribute/Type/Key/…) comes back as a ``_CallbackError``, so a
        worker never takes it for its own request failing (see _tail)."""
        failed: list[Exception] = []

        async def run():
            try:
                result = fn(*args, **kwargs)
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:  # noqa: BLE001 - handed to the worker below
                if isinstance(exc, RuntimeError) or (isinstance(exc, NoMatches) and not on_stack(self)):
                    raise  # teardown, or a screen popped meanwhile: call_on_screen drops them
                failed.append(exc)

        delivered = call_on_screen(self, run)
        if failed:
            raise self._CallbackError(getattr(fn, "__qualname__", repr(fn))) from failed[0]
        return delivered

    def _ask(self, fn, *args):
        """Worker thread. ``fn(*args)`` run on the UI thread, and what it returns: how a worker
        reads the view as it is now, rather than its attributes from another thread. None once
        the app is gone (see _post)."""
        answer: list = []
        self._post(lambda: answer.append(fn(*args)))
        return answer[0] if answer else None

    def _saved_shown(self, source: str | None) -> bool:
        """Whether ``source`` shows its saved console: the whole job's download is in,
        or this sub-job's was fetched first while the rest download."""
        return self._stdout_ready or (source is not None and source in self._stdout_have)

    def _console_shown(self, source: str | None) -> bool:
        """Whether ``source`` shows a saved console: the whole, or its newest chunks while it downloads."""
        return self._saved_shown(source) or (source is not None and source in self._stdout_preview)

    def _kept_console(self) -> bool:
        """A complete saved console from an earlier open of this same (finished) job."""
        if self._live() or not self._job_created_at:
            return False
        try:
            return (self._console_dir / ".complete").read_text(encoding="utf-8") == str(self._job_created_at)
        except (OSError, ValueError):  # unreadable, or not UTF-8: not a console this view kept
            return False

    def _stdout_path(self, sub_job_id: str) -> Path:
        if self._kept_console():
            return self._console_dir / sub_job_id / "stdout.log"
        root = "" if self._stage is None else self._stage.name
        return Path(root) / sub_job_id / "stdout.log"

    def _console_may_be_incomplete(self, seen: float | None, ended_at) -> bool:
        """Any thread. The job ended too recently for its console's last chunks to be uploaded for sure.

        True when ``ended_at`` (its ``state_changed_at``: when the job entered its
        terminal state) is under _KEEP_CONSOLE_AFTER_SECONDS ago, or when this view
        had the job running that recently (``seen``), whatever the timestamps say.
        """
        if seen is not None and time.monotonic() - seen < self._KEEP_CONSOLE_AFTER_SECONDS:
            return True
        ended = views.parse_ts(ended_at)
        if ended is None:
            return False
        return (datetime.now(timezone.utc) - ended).total_seconds() < self._KEEP_CONSOLE_AFTER_SECONDS

    def _keep_console(self, plan: LogScreen._Download) -> bool:
        """Worker thread. Copy the console ``plan`` downloaded into the job's cache for the next
        open; True when the copy landed (the cache may then be over its limit: see
        _hold_cache_to_limit)."""
        if not plan.root or views.parse_ts(plan.ended_at) is None or not plan.created_at:
            return False
        if self._console_may_be_incomplete(plan.seen, plan.ended_at):
            return False  # it may still be missing its last chunks: download it again next time
        lock = DownloadLock(self._console_dir)
        # Hold the console lock so a removal cannot move the job aside mid-copy, and skip
        # keeping it if the screen's lease expired meanwhile. Not waited for: whoever holds
        # it is another TUI keeping this same console, or a removal of the job, and either
        # way this view must not show its console later for it.
        if not lock.try_acquire():
            return False
        try:
            if not self._lease.held:
                return False
            return self._copy_console(plan)
        finally:
            lock.release()

    def _hold_cache_to_limit(self) -> None:
        """Worker thread. After a console was kept: the jobs used least recently go while the
        cache is over its limit. Run once the pane has the console, as the pass reads the size
        of everything in the cache first."""
        try:
            enforce_cache_limit(self._console_dir.parent.parent.parent, limit=self._cache_limit)
        except Exception:  # noqa: BLE001 - cache cleanup must never stop the view
            log_traceback("cache limit failed")

    def _copy_console(self, plan: LogScreen._Download) -> bool:
        """Whether the console is now in the job's cache."""
        tmp = None
        try:
            self._console_dir.parent.mkdir(parents=True, exist_ok=True)
            tmp = Path(tempfile.mkdtemp(prefix=".console-", dir=str(self._console_dir.parent)))
            # The preview (see _console_preview) isn't the console's: the whole is beside it.
            shutil.copytree(plan.root, tmp, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".preview"))
            (tmp / ".complete").write_text(str(plan.created_at), encoding="utf-8")
            shutil.rmtree(self._console_dir, ignore_errors=True)
            os.replace(tmp, self._console_dir)
            tmp = None
            return True
        except OSError:
            return False  # this view keeps reading its temporary copy
        finally:
            if tmp is not None:
                shutil.rmtree(tmp, ignore_errors=True)

    def _cleanup_stage(self) -> None:
        stage, owner = self._stage, self._stage_owner
        self._stage, self._stage_owner = None, None
        self._stdout_preview.clear()  # its files go with the stage
        if stage is not None:
            release_stage(stage, owner)

    def _claim_stdout(self) -> bool:
        """UI thread. One whole-job download for this view; a failure retries on the next open."""
        if not self._screen_open or self._stdout_ready or self._stdout_inflight or self._stdout_failed:
            return False
        if self._stage is None:
            self._stage, self._stage_owner = make_stage()
        self._stdout_inflight = True
        return True

    # ─── Logs tab state: tail, waiting, idle pause, restarts, scrollback ──────
    _MAX_LINES = 20_000  # lines the pane (and its re-filter buffer) keeps
    _MAX_ROWS = 200_000  # wrapped rows the Log widget may hold for those lines
    _MAX_ROWS_PER_LINE = 20
    _FILTER_DEBOUNCE_SECONDS = 0.25
    # Log reads count as activity, so pause an unattended live tail.
    _IDLE_PAUSE_SECONDS = 15 * 60
    _IDLE_CHECK_SECONDS = 5.0
    _PRE_PLACEMENT = frozenset({"pending", "queued", "creating"})
    _STARTING = frozenset({"placing", "initializing"})
    _NOT_RUNNING_YET = _PRE_PLACEMENT | _STARTING
    _NOT_ROUTED = "does not belong to session"
    _PENDING_PLACEMENT = "is pending placement"
    _EXECUTION_ERROR_RECORDS = 200  # newest error records shown from an execution log
    _EXECUTION_TAIL_RECORDS = 20  # records shown when an execution log has no errors

    def _init_log_state(self) -> None:
        # Current source lines retained for request-free repainting.
        self._entries: deque[tuple[object, str]] = deque(maxlen=self._MAX_LINES)
        self._entries_dropped = 0
        self._display_cache: dict[str, str] = {}  # raw line -> what the pane shows (see _display)
        self._export_note: str | None = None  # what s last did
        self._download_note: str | None = None  # the saved console's download progress, or its failure
        self._download_chunks: tuple[str, int, int] | None = None  # (sub-job, chunks done, of) for the bar
        self._log_bar_key: tuple | None = None  # what the bar under the pane last drew (see _render_log_bar)
        self._tail_error: str | None = None  # the live tail stopped on this error (the status line says so)
        self._tail_request: tuple[int, float] | None = None  # (tail gen, monotonic start) of the call in flight
        self._behind_note: str | None = None  # full pages: the tail is catching up
        self._export_timer = None
        self._repainting = False  # a saved console's repaint is being prepared off the UI thread
        self._paint_gen = 0  # bumped by each _paint_saved; an older paint's result is dropped
        self._saved_matched_none = False  # the saved console has lines, but none pass the filter and level
        self._saved_matched = 0  # the saved console's lines that pass them, kept or not
        self._shown_rows: list[int] = []  # wrapped rows of each _shown_lines line
        self._shown_kinds: list[str | None] = []  # the kind of each _shown_lines line (see in_context)
        self._last_day: str | None = None  # the day of the newest dated line in the pane (it has the rule)
        self._in_traceback = NO_TRACEBACK  # where a traceback stands after the newest log line (see traceback_kind)
        self._unseen = 0  # log lines written since the reader left the end (paused, or scrolled up)
        # The last jump (e, n): (what it stops at, which of them, of how many); the line is the pane's mark.
        self._jump: tuple[str, int, int] | None = None
        self._rows_total = 0
        self._shown_logs = 0  # the log lines among _shown_lines (the rest are the TUI's notes)
        self._repaint_job: LogScreen._Repaint | None = None
        self._repaint_pending: list[str] = []
        self._repaint_gen = 0
        self._pane_truncated = False  # the pane no longer holds the source's oldest lines
        self._head_missing = False  # the tail (or its cache) never had the log's first lines
        self._saved_dropped = 0  # matching saved-console lines past _MAX_LINES
        self._tail_gen = 0  # bumped by every _start_tail; an older tail's pages are dropped
        self._tail_live = False  # a live tail is polling (or held)
        self._saw_live = False
        self._replayed = False  # this tail replayed cached lines before its live pages
        self._sub_job_status: dict[str, str] = {}
        self._job_reason_details = None
        self._waiting: str | None = None  # "pending" | "starting" | "stopped"
        self._waiting_note: str | None = None
        self._wait_words: tuple[str, str] | None = None
        self._unrouted_logged: set[str] = set()  # sub-jobs whose "not routed" traceback is in the error log
        self._idle_paused = False
        self._opened_at = time.monotonic()
        self._resumed_at = 0.0
        self._idle_floor = 0.0  # the idle clock starts no earlier than this (the job wasn't running before)
        self._restart_note: str | None = None
        self._log_status_text = ""
        self._exec_logs: dict[str, tuple[list[str], list[tuple[object, list[str]]]]] = {}
        self._exec_failed: set[str] = set()
        self._exec_inflight: set[str] = set()
        self._active_seen_at: float | None = None  # monotonic time a get_job last had the job active
        self._stdout_recent = False  # the shown console was fetched moments after the job ended
        self._stdout_again = False  # the next download adds the chunks uploaded since the last one

    # The saved console's last lines the live tail's are lined up against (see _newer_tail_lines).
    _ALIGN_LINES = 3

    def _scan_saved(self, path: Path, query: str, level, worker,
                    probe: str | None = None) -> tuple[list[str], int, int, list[str], bool] | None:
        """Scan a console for matching lines, bounded per logical line."""
        kept: deque[str] = deque(maxlen=self._MAX_LINES)
        last: deque[str] = deque(maxlen=self._ALIGN_LINES)
        matched = total = 0
        found = False
        query = query.lower()
        limit = self._COPY_MAX_BYTES
        half = limit // 2
        head = tail = ""
        chars = 0
        shown = None

        def reset() -> None:
            nonlocal head, tail, chars
            head = tail = ""
            chars = 0

        def append(piece: str) -> None:
            nonlocal head, tail, chars
            chars += len(piece)
            take = min(len(piece), half - len(head))
            head += piece[:take]
            tail = (tail + piece[take:])[-half:]

        def take_segment() -> tuple[str, bool, bool]:
            truncated = chars > limit
            text = (head + tail if not truncated
                    else safe_text(head) + " …[truncated]… " + safe_text(tail))
            state = text, chars > 0, truncated
            reset()
            return state

        def emit(state: tuple[str, bool, bool]) -> bool:
            nonlocal found, matched, total
            raw, _nonempty, sanitized = state
            text = raw if sanitized else safe_text(raw)
            displayed = display_line(text)
            total += 1
            if text.strip():
                last.append(text)
            if probe is not None and not found and log_format.unstamped(text) == probe:
                found = True
            if (not query or query in displayed.lower()) and (
                    not level or entry_at_level({"_raw": text}, level)):
                kept.append(text)
                matched += 1
            return total % 4096 == 0 and worker.is_cancelled

        def consume(chunk: str) -> bool:
            nonlocal shown
            for piece in re.split(r"([\r\n])", chunk):
                if piece == "\r":
                    state = take_segment()
                    if state[1]:
                        shown = state
                elif piece == "\n":
                    state = take_segment()
                    if state[1]:
                        shown = state
                    if emit(shown or ("", False, False)):
                        return True
                    shown = None
                else:
                    append(piece)
            return worker.is_cancelled

        with path.open("rb") as handle:
            remaining = committed_size(path, handle.seek(0, 2))
            handle.seek(0)
            decoder = codecs.getincrementaldecoder("utf-8")("replace")
            while remaining:
                raw = handle.read(min(64 * 1024, remaining))
                if not raw:
                    break
                remaining -= len(raw)
                if consume(decoder.decode(raw, final=remaining == 0)):
                    return None
            if remaining and consume(decoder.decode(b"", final=True)):
                return None
        state = take_segment()
        if state[1]:
            shown = state
        if shown is not None and emit(shown):
            return None
        return list(kept), matched, total, list(last), found

    def _newer_tail_lines(self, pairs: list, last: list[str], probe_found: bool, query: str,
                          level) -> tuple[list[str], bool]:
        """Worker thread. The live tail's lines past the end of a console fetched
        moments after the job ended (its last chunks may not be uploaded yet),
        that pass the filter and level: they were on screen before the switch.
        ``pairs``: the tail cache's newest (entry, line)s, in log order.

        Returns ``(lines, gap)``. They come after the console's last lines when
        those are in the cache; when they aren't and the cache's oldest line
        isn't in the console either, the whole cache came after the console's
        end (``gap``: lines between the two may be missing)."""
        lines = [line for _, line in pairs]
        displayed = [display_line(line) for line in lines]
        after = log_format.lines_after([display_line(line) for line in last], displayed)
        gap = False
        if after is None:
            if not lines or probe_found:
                return [], False  # they overlap, but can't be lined up: add nothing
            ends = [moment for moment in map(log_format.line_moment, last) if moment is not None]
            starts = log_format.line_moment(lines[0])
            if ends and starts is not None and starts < max(ends):
                return [], False  # the tail began before the console ends: their lines just don't compare
            after, gap = lines, True
        query = query.lower()
        start = len(pairs) - len(after)
        kept = [line for (entry, line), shown in zip(pairs[start:], displayed[start:])
                if (not query or query in shown.lower()) and entry_at_level(entry, level)]
        return kept, gap

    @work(thread=True, exclusive=True, group="paint")
    def _prepare_saved(self, gen: int, source: str, path: Path, query: str, level, width: int,
                       prefixes: tuple[str, str], recent: bool, *, preview: bool = False) -> None:
        """Worker thread. Read, filter and wrap the saved console for the pane;
        _apply_saved shows it. A long console takes a while, and the UI keeps
        answering keys meanwhile."""
        worker = get_current_worker()
        try:
            console = path.is_file() and path.stat().st_size > 0
        except OSError:
            console = False
        newer: list[str] = []
        gap = False
        if console:
            # The tail's own lines, for a console that may lag behind them.
            pairs = [(entry, format_log_entry(entry))
                     for entry in self._cache.load_entries(source, limit=self._MAX_LINES)] if recent else []
            probe = log_format.unstamped(safe_text(pairs[0][1])) if pairs else None
            try:
                scanned = self._scan_saved(path, query, level, worker, probe)
            except OSError:  # removed under us (the view closed): nothing to show
                return
            if scanned is None:
                return
            lines, matched, total, last, probe_found = scanned
            if pairs:
                newer, gap = self._newer_tail_lines(pairs, last, probe_found, query, level)
        else:
            # No console for this sub-job (yet): the lines its tail cached, if any.
            entries = self._cache.load_entries(source, limit=2000)
            lines = [format_log_entry(entry) for entry in entries
                     if entry_matches(entry, query) and entry_at_level(entry, level)]
            matched, total = len(lines), len(entries)
        if worker.is_cancelled:
            return
        first, rest = prefixes
        chunks, displays, cut = log_format.wrap_lines(
            lines, width, first=first, rest=rest, max_lines=self._MAX_LINES, max_rows=self._MAX_ROWS,
            max_rows_per_line=self._MAX_ROWS_PER_LINE)
        # Their kinds and context here too, not on the UI thread: a console can be 20,000 lines.
        texts = [displays[line] for line, _, _ in chunks]
        chunks, kinds, row_kinds, day, in_block = in_context(chunks, [line_kind(text) for text in texts], texts, width)
        if worker.is_cancelled:
            return
        self._post(self._apply_saved, gen, source, console, chunks, displays, cut, width,
                   matched, total, len(lines), newer, gap, kinds=kinds, context=(row_kinds, day, in_block),
                   preview=preview)

    def _ended_job(self) -> dict:
        """This view's copy of the job with the newest status and reason (the tail
        can see the job end before the next get_job refreshes the copy)."""
        job = {**self._job, "status": self._job_status or self._job.get("status")}
        if self._job_reason is not None:
            job["reason"] = self._job_reason
        if self._job_reason_details is not None:
            job["reason_details"] = self._job_reason_details
        return job

    def _no_log_output(self) -> str:
        # A job that never got GPUs (placement_timeout and the like) wrote no logs: say why.
        if views.never_started(self._ended_job()):
            return f"No log output: {views.never_started_text(self._ended_job())}."
        return "No log output is available."

    @staticmethod
    def _span(seconds: float) -> str:
        minutes, rest = divmod(int(seconds), 60)
        return f"{minutes} min" if minutes and not rest else f"{int(seconds)} s"

    def _recent_console_note(self, *, console: bool = True, newer: int = 0) -> str:
        """The note under logs fetched moments after the job ended. ``console``: this sub-job has a saved
        console; ``newer``: the live tail's lines shown after it, which it doesn't have yet."""
        ago = self._span(self._KEEP_CONSOLE_AFTER_SECONDS)
        if console and newer:
            return (f"— the job ended less than {ago} ago, so the saved console may still be missing its last "
                    f"lines; the live tail's last {self._count(newer, 'line')} are shown after it until r "
                    "downloads them —")
        if console:
            return (f"— the job ended less than {ago} ago, so this console may still be missing its last lines; "
                    "r downloads them —")
        # No console yet: the job may have failed before it wrote one, or its
        # chunks aren't uploaded yet. Don't promise lines that may never come.
        return f"— the job ended less than {ago} ago, so its saved logs may not all be uploaded yet; r checks again —"

    def _paint_saved(self) -> None:
        """UI thread. Show the current sub-job's saved console, filtered; it is
        prepared off the UI thread, and the pane keeps what it shows until then."""
        source = self._current_source
        if not source:
            return
        # The job has ended: what its live tail waited for, or restarted from,
        # is over (the tail can notice the end before the status check does).
        self._waiting = self._waiting_note = None
        self._restart_note = None
        self._cancel_repaint()  # what a repaint on its way would show, this paint replaces
        self._paint_gen += 1
        self._repainting = True
        # Until its whole console is in, a sub-job with a preview shows that.
        preview = not self._saved_shown(source) and source in self._stdout_preview
        self._prepare_saved(self._paint_gen, source, self._stdout_preview[source] if preview else self._stdout_path(source),
                            self._filter, self._min_level, self._content_width(), self._line_prefixes(),
                            self._stdout_recent and not preview, preview=preview)
        self._render_log_status()

    def _apply_saved(self, gen: int, source: str, console: bool, chunks: list, displays: dict, cut: bool,
                     width: int, matched: int, total: int, kept: int, newer: list[str], gap: bool = False, *,
                     kinds: list | None = None, context: tuple | None = None, preview: bool = False) -> None:
        """UI thread. Show a saved console _prepare_saved read and wrapped, unless a newer paint superseded it.
        ``preview``: only its newest chunks, while the rest downloads (see _console_preview)."""
        shown = self._saved_shown(source) or (preview and source in self._stdout_preview)
        if gen != self._paint_gen or source != self._current_source or not shown or not self._screen_open:
            return
        self._repainting = False
        self._display_cache.update(displays)
        self._saved_matched = matched  # the filter's count, over the whole console (see _log_chips)
        # The TUI's own lines (shown untagged), after the console's.
        # A job that never got GPUs has no saved logs still to come.
        notes = ([self._Note(self._recent_console_note(console=console, newer=len(newer)))]
                 if self._stdout_recent and not preview and not views.never_started(self._ended_job()) else [])
        # Above a preview: its first line isn't the console's.
        head = ([self._Note(f"— the saved console's last {self._CONSOLE_PREVIEW_SECONDS // 60} minutes; "
                            "the older lines are still downloading —")] if preview else [])
        empty = False
        if console:
            self._saved_dropped = matched - kept
            extra: list[str] = []
            if newer:
                # Shown before the switch, and the console doesn't have them yet: keep them on screen.
                banner = self._Note(f"— {self._count(len(newer), 'newer line')} from the live tail, not in the "
                                    "saved console yet" + ("; lines between the two may be missing" if gap else "")
                                    + " —")
                extra = [banner, *newer]
            self._saved_matched_none = total > 0 and not matched and not newer
        else:
            self._saved_dropped = 0
            extra = self._execution_log_lines(source)
            empty = not total and not extra
            if empty and notes:
                notes.insert(0, self._Note(self._no_log_output()))
            self._saved_matched_none = total > 0 and not chunks
        if self._paused:
            # Shown once the pause ends (p).
            self._held = (head + [line for line, _, _ in chunks] + extra + notes, empty)
        else:
            self._held = None
            self._clear_pane()
            if head:
                self._write_lines(head)
            self._write_chunks(chunks, width, cut=cut, kinds=kinds, context=context)
            if extra or notes:
                self._write_lines(extra + notes)
            elif not chunks and empty:
                self._write_line(self._no_log_output())
            if width != self._content_width():
                self._wrap_width = -1  # the pane changed width meanwhile
                after_refresh(self, self._rewrap)
            if self._logview is not None:
                self._logview.scroll_end(animate=False)
        self._render_log_status()

    def _finish_download(self, ok: bool, problem: str | None = None, recent: bool = False) -> None:
        if not self._screen_open:
            self._stdout_inflight = False
            self._cleanup_stage()
            return
        self._stdout_inflight = False
        self._stdout_first = None
        self._download_note = self._download_chunks = None
        if not ok:
            self._stdout_failed = True
            if self._current_source in self._stdout_have:
                # The sub-job on screen has its console; the others' didn't come.
                self._download_note = (f"the other sub-jobs' saved consoles couldn't be downloaded "
                                       f"({problem or f'details in {_ERROR_LOG}'}); r tries again")
                self._render_log_status()
                return
            if problem:
                self._write_line(f"— {problem} —")
            else:
                self._write_line(f"— the saved console couldn't be downloaded; details in {_ERROR_LOG} —")
            self._render_log_status()
            return
        # The sub-job on screen was painted already, as soon as its console was in.
        painted = self._current_source in self._stdout_have and recent == self._stdout_recent
        self._stdout_ready = True
        self._stdout_recent = recent
        self._stdout_again = False
        self._stdout_have.clear()
        self._stdout_preview.clear()
        if painted:
            self._render_log_status()
        else:
            self._paint_saved()

    def _apply_terminal_status(self, status: str, reason: str | None) -> None:
        self._job_status = status
        if reason is not None:
            self._job_reason = reason
        self._update_subtitle()
        self._render_summary_line()  # the status badge, and the reason beside it
        self._paint_sources()
        if self._resource_active():
            self._render_resource()
        if not views.is_live(self._tab_job()):
            # Serving redraws only when it reloads, and the live refresh has stopped: one last read
            # shows it ended. Off screen, it reads again when it is next opened (on_tabbed_content_tab_activated).
            pane = self._detail_pane()
            if isinstance(pane, ServingPane) and pane.opened:
                pane.reload()

    @staticmethod
    def _sub_statuses(job: dict) -> dict[str, str]:
        statuses = {}
        for sj in job.get("sub_jobs") or []:
            if isinstance(sj, dict):
                sid = sj.get("sub_job_id") or sj.get("id")  # as _read_sources reads it
                if sid and sj.get("status"):
                    statuses[sid] = sj["status"]
        return statuses

    def _live(self) -> bool:
        return views.is_live(self._ended_job())

    def _dropping(self) -> bool:
        """A deployment being dropped: not live, but its status still changes to the end."""
        return views.is_deployment(self._job) and views.status_word(self._job_status) == "cancelling"

    def _serving_may_change(self) -> bool:
        """New drains may still arrive: the job is live or dropping, or ended before its last could upload."""
        return (self._live() or self._dropping()
                or self._console_may_be_incomplete(self._active_seen_at, self._state_changed_at))

    def _refresh_status(self) -> None:
        # The tail pauses off the Logs tab, so it can't be what notices the job ending.
        if self.app.screen is self and (self._live() or self._dropping()):
            self._fetch_status()

    @work(thread=True, exclusive=True, group="status")
    def _fetch_status(self) -> None:
        # Numbered as it is sent: a read that answers after a later one is dropped (see _take_job).
        # One the next tick's read replaced (exclusive) still counts: with a slow service every
        # read can take longer than a tick, and dropping those would drop them all.
        seq = self._next_job_read()
        try:
            job = self._client.get_job(self._job_id)
        except Exception:  # noqa: BLE001 - keep the current view; the next tick retries
            return
        if isinstance(job, dict) and job.get("status") and self._screen_open:
            self._post(self._apply_status, job, seq)

    def _apply_status(self, job: dict, seq: int) -> None:
        if not self._screen_open:
            return
        ended = self._live() and not views.is_live(job)
        # Status, reason and sub-job statuses come from this job as they are now:
        # a reason the service stopped sending (a job that left pending) is gone.
        # (_take_job also redraws the summary line from it.) An older answer than
        # the one on screen changes nothing.
        if not self._take_job(job, seq):
            return
        if ended and self._current_source:
            # Show the saved console now, as the tail would after its next error.
            self._start_tail(self._current_source, user=False)
            return
        # A Logs tab waiting for the job (or its sub-job) to start tails it once it can.
        self._resume_waiting_tail()

    def _auto_refresh_data(self) -> None:
        # Only the visible tab polls, and only while the job is live. Resources waits
        # until the status check sees the shown sub-job running: before that it has no samples.
        if self.app.screen is not self:
            return
        if not self._live():
            # A deployment's drains go on while it is dropped, and are uploaded for a while after it ends.
            pane = self._detail_pane()
            if isinstance(pane, ServingPane) and pane.opened and self._serving_may_change():
                pane.reload()
            return
        if self._resource_active() and self._resource_waiting() is None:
            self._start_resource_refresh(force=True)
        elif self._checkpoints_active():
            self._start_checkpoint_refresh(force=True)
        pane = self._detail_pane()
        if pane is not None:
            pane.refresh_live()

    def _next_job_read(self) -> int:
        """Any thread. The number of a get_job about to be sent: a read sent later has a larger one."""
        with self._job_reads_lock:
            self._job_reads += 1
            return self._job_reads

    def _take_job(self, job: dict, seq: int | None = None) -> bool:
        """UI thread. A fresh get_job replaces this view's status, reason, times and sub-job statuses.

        ``seq`` numbers the read (see _next_job_read). Reads can answer out of order (one
        that retries a slow service takes minutes), so an answer to a read sent before the
        one the view shows is dropped; so is an answer that has the job live after the view
        saw it end, which a job never comes back from. Returns whether ``job`` was taken.
        """
        if not isinstance(job, dict) or not (job.get("status") or job.get("sub_jobs")):
            return False
        if seq is not None and seq < self._job_seq:
            return False  # a read sent later has answered already
        if self._job_status and job.get("status") and views.is_live(job) and not self._live():
            return False  # an older answer, whatever its number: the job has ended
        if seq is not None:
            self._job_seq = seq
        self._sub_job_status = self._sub_statuses(job)
        self._job_reason_details = job.get("reason_details") or None
        if job.get("status") and views.is_live(job):
            self._active_seen_at = time.monotonic()  # it hadn't ended yet
        # A job that ended while the view was open has a new state_changed_at,
        # which decides whether its console is kept (_keep_console).
        self._state_changed_at = job.get("state_changed_at")
        self._job_created_at = job.get("created_at")
        status = job.get("status") or self._job_status  # (an answer without one leaves it)
        reason = job.get("reason") or ""  # omitted: the job has no reason now
        if status != self._job_status or reason != (self._job_reason or ""):
            self._apply_terminal_status(status, reason)
        # The rest of the view (the summary line's reason and reason_details) as the job is now:
        # the tail can see the job end before the status check does. The service's fields come
        # only from the fresh answer, so a reason it stopped sending disappears from the line.
        self._job = views.refresh_job(self._job, job)
        self._apply_tabs()
        if job.get("status"):
            self._job_reason = job.get("reason")
        self._render_summary_line()
        self._paint_sources()  # its sub-jobs' statuses
        return True

    def _tail_status(self, sub_job_id: str | None) -> str:
        """The status word that decides how a sub-job's tail runs.

        A placed job's sub-jobs start on their own schedule, so the sub-job's
        own status counts once the job is past placement; before that, and
        once the job has ended, the job's does.
        """
        job = views.status_word(self._job_status)
        sub_job_status = self._sub_job_status.get(sub_job_id or "")
        if not sub_job_status or job in self._PRE_PLACEMENT or not self._live():
            return job
        return views.status_word(sub_job_status)

    def _pending_note(self, sub_job_id: str) -> str:
        """The wait for placement: the job's (with its reason), or one sub-job's once the job is placed."""
        word = self._tail_status(sub_job_id)
        job_word = views.status_word(self._job_status)
        hint = f"(status checked every {self._STATUS_REFRESH_SECONDS} s · r checks now)"
        if job_word in self._PRE_PLACEMENT:
            # As the summary line puts them: "reason — reason_details".
            why = " — ".join(one_line(x).strip() for x in (self._job_reason, self._job_reason_details)
                             if x and str(x).strip())
            return (f"waiting: the job is {self._said(job_word)}" + (f" ({why})" if why else "")
                    + f"; the log tail starts by itself when the job starts {hint}")
        if word in self._PRE_PLACEMENT:
            return (f"waiting: {views.sub_job_label(sub_job_id)} is {self._said(word)}; "
                    f"its log tail starts by itself once it is placed {hint}")
        # The service refused the tail as pending placement while the status
        # says otherwise (UNKNOWN when the service couldn't read the job's state).
        return (f"waiting: the service says the job is pending placement (its status reads "
                f"{self._said(job_word)}); the log tail starts by itself when the job starts {hint}")

    def _said(self, word: str) -> str:
        """A status word as the badge says it: a deployment's in its own words (INITIALIZING, not PLACING)."""
        return views.status_word_for(self._job, word).upper()

    def _waiting_for(self, sub_job_id: str) -> tuple[str, str] | None:
        """The wait, if any, before ``sub_job_id`` can have logs: (kind, note)."""
        word = self._tail_status(sub_job_id)
        if word in self._PRE_PLACEMENT:
            return "pending", self._pending_note(sub_job_id)
        if word in self._STARTING:
            return "starting", (f"waiting: {views.sub_job_label(sub_job_id)} is {self._said(word)}; "
                                "its log appears here once it is up")
        if views.status_word(self._job_status) == "unknown":
            # The service couldn't read the job's state: its tail may not answer either. It is tried.
            return "unknown", ("waiting: the job's status is unavailable right now, so its log may not answer; "
                               "the tail keeps trying (r tries now)")
        return None

    def _set_waiting(self, kind: str, note: str) -> None:
        # Said once, on the status line: it stays in view while lines arrive,
        # and the summary line above already carries the job's reason.
        self._waiting, self._waiting_note = kind, note
        self._render_log_status()

    def _clear_waiting(self) -> None:
        if self._waiting:
            self._waiting = self._waiting_note = None
            self._render_log_status()

    def _resume_waiting_tail(self) -> None:
        """UI thread, after a fresh status: start the tail a wait was for, once it can run,
        or one that stopped on an error."""
        source = self._current_source
        if not source or self._saved_shown(source) or not self._live():
            return
        word = self._tail_status(source)
        if self._tail_error and not self._tail_live:
            # Stopped on an error. No status change may be coming to try again on
            # (a job that runs, or whose status is unavailable), so each check
            # does: while someone is watching, as reads of a running job count
            # as activity on it, and whatever the input while it starts up.
            if views.status_word(self._job_status) in self._STARTING or self._watched():
                self._start_tail(source, user=False)  # the wait, if any, comes back with it
            return
        if not self._waiting:
            return
        if self._waiting == "unknown":
            if views.status_word(self._job_status) != "unknown":
                if self._tail_live:
                    self._clear_waiting()  # its status is back, and its tail is polling
                else:
                    self._start_tail(source, user=False)
            elif not self._tail_live and self._watched():
                self._start_tail(source, user=False)  # still unknown: try again while someone watches
            return
        if self._waiting == "starting":
            if not self._tail_live:
                # Its tail stopped while it started up, when the job couldn't be
                # read to tell why: try again.
                self._start_tail(source, user=False)
            elif word in self._STARTING:
                self._refresh_waiting_note()
            else:
                self._clear_waiting()  # it is up, and its tail is polling
            return
        if word in self._PRE_PLACEMENT:
            self._refresh_waiting_note()
            return
        if self._waiting == "stopped" and word != "running":
            return  # the tail starts again once the sub-job runs
        words = self._status_words(source)
        # While the job and sub-job run, no status change is coming to try again on: each check
        # (once a minute) does, but only while someone is watching (see _watched).
        if self._wait_words == words and not (words == ("running", "running") and self._watched()):
            return  # still the statuses the service answered for: r tries now
        self._start_tail(source, user=False)

    def _status_words(self, sub_job_id: str) -> tuple[str, str]:
        return views.status_word(self._job_status), self._tail_status(sub_job_id)

    def _watched(self) -> bool:
        """Whether someone used the view in the last _IDLE_PAUSE_SECONDS. Only then do the
        status checks start a tail by themselves: an unattended view must not keep reading
        a running job."""
        return time.monotonic() - self._last_input() < self._IDLE_PAUSE_SECONDS

    def _refresh_waiting_note(self) -> None:
        """Still waiting: show the note again when it changed (a new reason, or placing → initializing)."""
        wait = self._waiting_for(self._current_source or "")
        if wait is not None and wait[0] == self._waiting and wait[1] != self._waiting_note:
            self._set_waiting(*wait)

    def _start_tail(self, sub_job_id: str, *, user: bool = True) -> None:
        """Show ``sub_job_id``: its saved console, the wait for it to start, or its tail.

        ``user`` is false when the view starts a tail by itself (a status
        check); only a user's action resumes a tail paused for lack of input.
        """
        self._current_source = sub_job_id
        self._held = None
        self._tail_gen += 1
        self._tail_live = False
        self._entries.clear()
        self._entries_dropped = 0
        self._head_missing = False
        self._export_note = None
        self._tail_error = None
        self._behind_note = None
        self._last_line_at = None  # this sub-job's own, once its lines arrive
        self._saw_live = False
        self._restart_note = None
        self._waiting = self._waiting_note = None
        self._wait_words = None
        if user:
            self._idle_paused = False
            self._resumed_at = time.monotonic()
        # A fresh tail context (new source, r, or a status change) starts
        # following the tail — otherwise a pause left over from a prior source
        # would strand this one at the top.
        self._paused = False
        self._replayed = False
        # A saved console still being prepared for the sub-job shown before is dropped when it comes.
        self._paint_gen += 1
        self._repainting = False
        self._update_subtitle()
        self._paint_sources()  # the list marks the sub-job shown
        self._clear_pane()
        if not self._stdout_ready and self._kept_console():
            self._stdout_ready = True  # finished long ago and downloaded before: replay it
        live = self._live()
        if self._stdout_ready or not live:
            self._idle_paused = False  # nothing polls now, so nothing is paused
        if self._console_shown(sub_job_id):
            self._paint_saved()
            if not self._stdout_ready and not live and self._claim_stdout():
                # The other sub-jobs' consoles, again after they failed (r).
                self._tail(sub_job_id, self._tail_gen, True, False)
            return
        download = not live and self._claim_stdout()
        if not live and not download and self._stdout_inflight:
            # The download is under way: this sub-job's console comes with it.
            self._write_line(self._download_banner(sub_job_id))
        if live:
            wait = self._waiting_for(sub_job_id)
            if wait is not None:
                self._set_waiting(*wait)
                if wait[0] == "pending":
                    # The service refuses tail-logs until the job is placed: replay
                    # the cache only, and let the status check start the tail.
                    self._tail(sub_job_id, self._tail_gen, False, False)
                    return
            else:
                self._write_line(f"— tailing {sub_job_id} —")
            self._tail_live = True
        self._render_log_status()
        self._tail(sub_job_id, self._tail_gen, download, live)

    @work(thread=True, exclusive=True, group="tail")
    def _tail(self, sub_job_id: str, gen: int, download: bool, live: bool) -> None:
        worker = get_current_worker()
        # A terminal job's zone is gone, so the operation API would error: serve
        # cache, then one saved-console download. A running/unknown job follows live.
        # A _CallbackError is a bug in this view showing what came, not a failed read:
        # no get_job is sent to tell what it means (see _view_failed).
        try:
            try:
                self._stream(worker, sub_job_id, gen, live=live)
            except self._CallbackError:
                raise
            except Exception as exc:  # noqa: BLE001 - surfaced in the UI
                if worker.is_cancelled:
                    pass
                elif live:
                    self._post(self._tail_ended, gen)
                    self._after_tail_error(worker, sub_job_id, gen, exc)
                    return
                elif not download:
                    self._report_tail_error(sub_job_id, exc, gen)
                    return
            if live:
                self._post(self._tail_ended, gen)
        except self._CallbackError as exc:
            # (The saved console replaces a terminal job's cache replay in a moment.)
            self._view_failed(sub_job_id, exc, gen, live=live, quiet=download)
        if download:  # claimed for this tail (see _claim_stdout): it runs whatever the replay met
            try:
                self._run_download(sub_job_id)
            except self._CallbackError as exc:
                self._view_failed(sub_job_id, exc, gen, live=False)

    def _tail_ended(self, gen: int) -> None:
        if gen == self._tail_gen:
            self._tail_live = False
            if self._idle_paused:
                self._idle_paused = False  # nothing is polling to resume
                self._render_log_status()

    def _stream(self, worker, sub_job_id: str, gen: int, *, live: bool) -> None:
        # Replay the local cache instantly, then (only for a live job) resume
        # the tail from the saved cursor so only new lines are fetched. Each
        # yield is a whole page (or the cache batch): it is formatted here and
        # rendered in one shot, so high-velocity logs don't saturate the UI thread.
        for kind, entries in cached_log_pages(
            self._cache,
            _TimedTail(self._client, self, gen),
            self._job_id,
            sub_job_id,
            follow=live,
            live=live,
            poll_interval=self._poll_interval,
            replay_limit=2000,
            is_cancelled=lambda: worker.is_cancelled,
            # Cancelling a worker can't interrupt time.sleep, so a cancelled tail would
            # hold its thread (and its executor slot) for a whole poll interval.
            sleep=_sleep_until_cancelled(worker),
            hold=self._tail_held,
            # A running job's tail that stops moving is restarted from the end.
            can_restart=lambda: self._tail_status(sub_job_id) == "running",
        ):
            if worker.is_cancelled:
                return
            lines = [format_log_entry(entry) for entry in entries]
            if not self._post(self._on_page, gen, kind, entries, lines):
                return

    def _on_page(self, gen: int, kind: str, entries: list, lines: list[str]) -> None:
        """UI thread. One page of the tail started as ``gen`` (or its cache replay)."""
        # Checked here, on the UI thread, so a page of an older tail, or one that
        # was in flight when the saved console replaced the pane, is dropped.
        if gen != self._tail_gen or self._console_shown(self._current_source) or not self._screen_open:
            return
        if kind == "partial":
            # Older lines of this log are neither here nor coming (the tail read only its end).
            self._head_missing = True
            self._render_log_status()
            return
        if kind == "caught-up":
            self._behind_note = None
            self._render_log_status()
            return
        if kind != "cache" and entries and self._waiting in ("starting", "unknown"):
            self._clear_waiting()  # lines are arriving: it is up
        pairs = list(zip(entries, lines))
        visible = [line for entry, line in pairs if self._visible(entry, line)]
        if kind in ("restart", "reset"):
            why = ("it had stopped moving" if kind == "restart" else
                   "the service no longer took its saved position in the log")
            self._restart_note = (f"tail restarted {time.strftime('%H:%M:%S')}: {why}, so it restarted from the "
                                  "end of the log; lines in between may be missing")
            self._write_line("— tail restarted from the end of the log; lines in between may be missing —")
        elif visible and kind == "live" and not self._saw_live:
            self._saw_live = True
            if self._replayed:  # where the replayed cache ends and new lines begin
                self._write_line("— live —")
        self._buffer(pairs)
        if kind == "cache" and entries:
            self._replayed = True
        if visible:
            self._write_lines(visible)
        if kind == "live":
            # A full page: the log runs ahead of this view, which catches up at a few reads a second.
            self._behind_note = self._catch_up_note(entries[-1]) if entries and catching_up(entries) else None
        if entries and kind != "cache":
            # The newest line's own timestamp (local time), not when the page arrived.
            clock = log_format.entry_clock(entries[-1])
            label = f"last line {clock}" if clock else f"last line received {time.strftime('%H:%M:%S')}"
            if label != self._last_line_at:
                self._last_line_at = label
                self._update_subtitle()
        self._render_log_status()

    _BEHIND_SECONDS = 5  # a tail getting full pages says how far behind it is from this lag

    @classmethod
    def _catch_up_note(cls, entry) -> str | None:
        """How far behind the log a tail that gets full pages is, by its newest line's own time;
        nothing while it keeps up (a busy log fills every page)."""
        raw = entry.get("_raw") if isinstance(entry, dict) else None
        when = views.parse_ts(str(raw).split(" ", 1)[0]) if isinstance(raw, str) else None
        if when is None:
            return None
        behind = int((datetime.now(timezone.utc) - when).total_seconds())
        if behind < cls._BEHIND_SECONDS:
            return None
        return f"catching up: the log is ahead of this view (its newest line here is {behind} s old)"

    def _buffer(self, pairs: list) -> None:
        """Keep received lines for a re-filter; the oldest go past _MAX_LINES."""
        overflow = len(self._entries) + len(pairs) - self._MAX_LINES
        if overflow > 0:
            self._entries_dropped += overflow
        self._entries.extend(pairs)

    def _visible(self, entry, line: str) -> bool:
        """Whether a received line passes the filter and level (the TUI's own notes always do)."""
        if entry is None:
            return True
        query = self._filter.lower()
        return (not query or query in line.lower()) and entry_at_level(entry, self._min_level)

    def _tail_hidden(self) -> bool:
        """Worker thread. True while the Logs tab is not on screen.

        Each tail-logs call counts as activity on a running job, so a viewer
        parked on another tab or screen must not keep polling; the saved cursor
        makes the catch-up on return cheap.
        """
        try:
            return not self._logs_active() or self.app.screen is not self
        except Exception:  # noqa: BLE001 - app shutting down; let cancellation win
            return False

    def _tail_held(self) -> bool:
        """Worker thread. True while the live tail must not poll: Logs hidden, or paused for no input."""
        return self._idle_paused or self._tail_hidden()

    def _last_input(self) -> float:
        """When the viewer last pressed a key, clicked or scrolled (monotonic seconds)."""
        try:
            seen = float(getattr(self.app, "last_input_at", 0.0) or 0.0)
        except Exception:  # noqa: BLE001 - no app (yet)
            seen = 0.0
        return max(self._opened_at, self._resumed_at, self._idle_floor, seen)

    def _idle_note(self) -> str:
        return f"paused: no input for {self._span(self._IDLE_PAUSE_SECONDS)} (press p to resume)"

    def _check_idle(self) -> None:
        """UI thread (timer). Pause a live tail that nobody has touched for _IDLE_PAUSE_SECONDS,
        and keep the status line's "no answer" note current."""
        if self._screen_open and (self._tail_request is not None or "no answer" in self._log_status_text):
            self._render_log_status()
        if self._idle_paused or not self._tail_live or not self._screen_open:
            return
        if not self._logs_active() or self.app.screen is not self:
            return  # the tail isn't polling now, and coming back to Logs is input
        if views.status_word(self._job_status) in self._NOT_RUNNING_YET:
            # Reads while the job is placed or starts up don't count as activity
            # on it, so the idle clock starts once it runs.
            self._idle_floor = time.monotonic()
            return
        if time.monotonic() - self._last_input() < self._IDLE_PAUSE_SECONDS:
            return
        self._idle_paused = True
        self._write_line(f"— {self._idle_note()} —")
        self._render_log_status()

    def _scrollback_note(self) -> str | None:
        """Say "last N lines" once the pane no longer holds the source's oldest lines."""
        saved = self._saved_shown(self._current_source)
        # A saved console is the whole log: what the tail missed doesn't count there.
        if not (self._pane_truncated or (self._head_missing and not saved) or (saved and self._saved_dropped > 0)):
            return None
        shown = self._shown_logs  # log lines, not the TUI's notes (kept as lines come and go: see _write_chunks)
        if not shown:
            return None
        full = f"{self._cli(f'download-log {shlex.quote(str(self._job_id))} --log-type stdout')} for the full log"
        filtered = bool(self._filter) or bool(self._min_level)
        # With a filter, the pane holds the newest matching lines (a live tail's
        # from the lines it had when the filter was set, and those since), so
        # the note counts only those; older matches may have left the view.
        noun = "matching line" if filtered else "line"
        return f"last {self._count(shown, noun)} · {full}"

    # A tail-logs call (with the client's own retries) unanswered this long is said on the status line.
    _NO_ANSWER_SECONDS = 20

    def _no_answer_note(self) -> str | None:
        request = self._tail_request
        if request is None or request[0] != self._tail_gen or not self._tail_live:
            return None
        waited = time.monotonic() - request[1]
        if waited < self._NO_ANSWER_SECONDS:
            return None
        return f"waiting for the service: the last log request has had no answer for {int(waited)} s"

    def _no_match_note(self) -> str | None:
        """Say so when the filter or level hides every line the pane has for this sub-job."""
        if not (self._filter or self._min_level) or self._repainting or self._repaint_job is not None:
            return None  # (the pane shows the lines before the repaint on its way)
        if self._shown_logs:
            return None
        if not (self._saved_shown(self._current_source) and self._saved_matched_none) and not any(
                entry is not None for entry, _ in self._entries):
            return None  # no lines yet: nothing is hidden
        what = " and ".join(bit for bit in (f"filter '{self._filter}'" if self._filter else "",
                                            f"level ≥{self._min_level}" if self._min_level else "") if bit)
        return f"no lines match {what} (Esc clears the filter, L changes the level)"

    @staticmethod
    def _count(n: int, noun: str) -> str:
        return f"{n:,} {noun}" + ("" if n == 1 else "s")

    def _render_log_status(self) -> None:
        """UI thread. The Logs tab's status line: shown only when it has something to say."""
        try:
            bar = self.query_one("#log-status", Static)
        except NoMatches:
            return
        idle = self._idle_note() if self._idle_paused and self._tail_live else None
        # A download's chunk count shows on the bar (see _log_live), unless the sub-job on
        # screen has its console already and the count is the other sub-jobs'.
        download = None if self._download_shown() else self._download_note
        notes = (self._job_read_note(), self._waiting_note, self._tail_error, self._no_answer_note(),
                 self._behind_note, idle, self._restart_note, download, self._no_match_note(),
                 self._scrollback_note(), self._export_note)
        bits = [bit for bit in notes if bit]
        # One row: a reason or id from the service can carry a line break or an
        # escape sequence, which must neither reach the terminal nor grow the bar.
        text = one_line("  ·  ".join(bits))
        lv = self._logview
        # Log follows new lines only from the very end. A status line that
        # appears, goes, or wraps to another row resizes the pane and leaves a
        # following view short of its end, so put it back there afterwards.
        following = lv is not None and not self._paused and lv.is_vertical_scroll_end
        changed = False
        if text != self._log_status_text:  # pages arrive every second; repaint only on a change
            self._log_status_text = text
            # Plain Content: job ids and service reasons are shown literally.
            bar.update(Content(text))
            changed = True
        if bar.display != bool(bits):
            bar.display = bool(bits)
            changed = True
        if changed and following:
            after_refresh(self, lv.scroll_end, animate=False)
        self._render_log_bar()

    # ─── the bar under the pane ──────────────────────────────────────────
    _PROGRESS_CELLS = 12  # the download's bar (see _log_live)

    def _download_shown(self) -> bool:
        """Whether the bar shows a saved-console download's progress: the sub-job on screen's own."""
        return (self._download_chunks is not None and self._stdout_inflight
                and not self._saved_shown(self._current_source))

    def _away_from_end(self) -> bool:
        """The reader isn't following the log: paused (p, or a jump), or scrolled up from its end."""
        lv = self._logview
        return self._paused or (lv is not None and bool(self._shown_lines) and not lv.is_vertical_scroll_end)

    @staticmethod
    def _bar_state(glyph: str, word: str, color: str, detail: str = "") -> Content:
        return Content.assemble((f"{glyph} {word}", f"bold {color}"), (f" · {detail}" if detail else "", theme.MUTED))

    def _log_live(self) -> Content:
        """The bar's left end: what the pane shows, and whether it follows the log.
        ``● LIVE · following``, ``‖ PAUSED · 12 new lines · p resumes``, ``✓ SAVED CONSOLE``,
        a download's progress, or the wait the status line above explains."""
        source = self._current_source
        state = self._bar_state
        if not source:
            return state("◌", "READING", theme.MUTED, "the job") if self._job_read_started is not None else Content("")
        if self._saved_shown(source):
            if self._repainting:
                return state("◌", "SAVED CONSOLE", theme.MUTED, "loading")
            held = "a newer copy came · p shows it" if self._held is not None else ""
            return state("✓", "SAVED CONSOLE", theme.ENDED, held)
        if self._download_shown():
            name, done, total = self._download_chunks
            filled = round(self._PROGRESS_CELLS * done / total) if total else 0
            return Content.assemble(("↓ DOWNLOADING", f"bold {theme.INFO}"), (f" {name} ", theme.MUTED),
                                    ("━" * filled, theme.INFO), ("━" * (self._PROGRESS_CELLS - filled), theme.FAINT),
                                    (f" {done:,}/{total:,} chunks", theme.MUTED))
        if self._stdout_inflight:
            return state("↓", "DOWNLOADING", theme.INFO, "saved console")
        if self._waiting:
            return state("○", "WAITING", theme.WARN)
        if self._tail_live:
            if self._idle_paused:
                return state("‖", "PAUSED", theme.WARN, "no input · p resumes")
            if self._away_from_end():
                new = f"{self._count(self._unseen, 'new line')} · " if self._unseen else ""
                # Paused (p, or a jump) as the subtitle says, or only scrolled up from the end.
                if self._paused:
                    return state("‖", "PAUSED", theme.WARN, f"{new}p resumes")
                return state("↑", "SCROLLED UP", theme.WARN, f"{new}p follows")
            return state("●", "LIVE", theme.OK, "catching up" if self._behind_note else "following")
        if self._tail_error:
            return state("×", "STOPPED", theme.ERR, "r retries")
        return state("◌", "CACHED", theme.MUTED, "saved on this machine")

    @staticmethod
    def _chip(text: str, color: str) -> tuple[str, str]:
        return f" {text} ", f"{color} on {theme.PANEL}"

    def _log_chips(self) -> Content:
        """The bar's right end: what the last jump (e, n) found, the level, and the filter's matches."""
        chips: list[tuple[str, str]] = []
        if self._jump is not None:
            noun, n, total = self._jump
            place = f"{n:,}" if n else "–"  # (the marked line has left the pane)
            chips.append(self._chip(f"{noun} {place}/{total:,}" if total else f"no {noun}s",
                                    theme.ERR if noun == "error" else theme.WARN))
        if self._min_level:
            chips.append(self._chip(f"≥ {self._min_level}", theme.ERR if self._min_level == "ERROR" else
                                    theme.WARN if self._min_level == "WARNING" else theme.INFO))
        if self._filter:
            if self._repainting or self._repaint_job is not None:
                chips.append(self._chip("filtering…", theme.MUTED))
            else:
                count = self._saved_matched if self._saved_shown(self._current_source) else self._shown_logs
                words = f"{count:,} match" + ("" if count == 1 else "es") if count else "no matches"
                chips.append(self._chip(words, theme.ACCENT if count else theme.WARN))
        parts: list = []
        for chip in chips:
            parts += [chip, " "]
        return Content.assemble(*parts[:-1])

    def _render_log_bar(self) -> None:
        """UI thread. Draw the bar under the pane again, when what it says changed."""
        try:
            live = self.query_one("#log-live", Static)
            chips = self.query_one("#log-chips", Static)
        except NoMatches:
            return
        shown = (self._log_live(), self._log_chips())
        key = tuple((content.plain, tuple(content.spans)) for content in shown)
        if key != self._log_bar_key:  # it is asked after every page and every scroll
            self._log_bar_key = key
            live.update(shown[0])
            chips.update(shown[1])

    def _log_scrolled(self) -> None:
        """UI thread. The pane scrolled (keys, the mouse, or following the log): back at its end,
        the lines that came meanwhile have been seen."""
        lv = self._logview
        if lv is not None and self._unseen and not self._paused and lv.is_vertical_scroll_end:
            self._unseen = 0
        self._render_log_bar()

    def _report_tail_error(self, sub_job_id: str, exc: BaseException, gen: int, *, label: str | None = None) -> None:
        """Worker thread. ``label``: what the error log calls the traceback."""
        log_traceback(label or f"tail {sub_job_id} failed")
        error = views.describe_error(exc)
        self._post(
            self._show_tail_error,
            gen,
            f"[error] tail {sub_job_id} failed: {error}  (r retries · full traceback: {_ERROR_LOG})",
            error,
        )

    def _view_failed(self, sub_job_id: str, exc: Exception, gen: int, *, live: bool, quiet: bool = False) -> None:
        """Worker thread, inside the ``except``: a callback the tail ran on the UI thread raised.

        That is a bug in this view, not a failed request, so no get_job is sent to
        tell what it means (see _after_tail_error). Its traceback goes to the error
        log, and the tail stops as on any other error (``quiet``: the error log only).
        """
        label = f"tail {sub_job_id}: a bug in the TUI while showing its log"
        if quiet:
            log_traceback(label)
            return
        try:
            if live:
                self._post(self._tail_ended, gen)
            self._report_tail_error(sub_job_id, exc.__cause__ or exc, gen, label=label)
        except self._CallbackError:
            pass  # the view can't say so either: the traceback is in the error log

    def _show_tail_error(self, gen: int, line: str, error: str = "") -> None:
        """UI thread. Only into the pane of the tail that failed, not one the viewer switched to since."""
        if gen == self._tail_gen:
            # One note: a wait's ("the tail keeps trying") no longer holds once
            # the tail has stopped. The status check that starts it again (see
            # _resume_waiting_tail) brings the wait back with it.
            self._clear_waiting()
            self._behind_note = None
            word = views.status_word(self._job_status)
            while_ = {"running": " while the job runs",
                      "unknown": " while the job's status is unavailable"}.get(word, "")
            self._tail_error = f"tail stopped: {error or 'it failed'} (r retries; so does each status check{while_})"
            self._write_line(line)
            self._render_log_status()

    def _after_tail_error(self, worker, sub_job_id: str, gen: int, exc: BaseException) -> None:
        """Worker thread. A live tail failed: one get_job decides what to show, never a retry loop.

        A job that ended switches to its saved console at once (the service
        answers tail-logs with 409 once a job is no longer running, and 404 once
        its session is gone). A job or sub-job that hasn't started yet shows the
        wait instead of an error (409 while pending placement, 400 "does not
        belong to session" for a sub-job the service isn't routing to yet), and
        the status check starts the tail when it can run. The service's own
        answer counts as much as the status word: a 409 "is pending placement"
        waits whatever the status says (UNKNOWN when the service couldn't read
        the job's state). Anything else stays an error, in place of any wait:
        r tries again, and so does the status check (see _resume_waiting_tail).
        """
        if worker.is_cancelled or not self._screen_open:
            return
        seq = self._next_job_read()
        try:
            job = self._client.get_job(self._job_id) or {}
        except Exception:  # noqa: BLE001 - a running job keeps the tail error
            job = None
        if job is None:
            if not worker.is_cancelled and self._screen_open:
                self._report_tail_error(sub_job_id, exc, gen)  # (the tail's traceback, not the read's)
            return
        if worker.is_cancelled or not self._screen_open:
            return
        # Decided on the UI thread, by the view's newest status: this answer's, unless
        # an answer to a read sent later is on screen already (see _take_job).
        step = self._ask(self._tail_failed, gen, sub_job_id, job, seq, views.describe_error(exc))
        if step is None:
            return
        what, detail = step
        if what == "error":
            self._report_tail_error(sub_job_id, exc, gen)
        elif what == "wait" and detail:
            log_traceback(detail)
        elif what == "download":
            self._run_download(detail)

    def _tail_failed(self, gen: int, sub_job_id: str, job, seq: int, text: str) -> tuple[str, str] | None:
        """UI thread. What a live tail's failure (``text``) means, now that get_job answered
        ``job`` (read ``seq``); see _after_tail_error. Returns what the tail's worker does
        next: ``("error", "")`` report it, ``("wait", label)`` log it under ``label`` (none:
        it is logged already) for the wait shown, ``("download", source)`` fetch the saved
        console (claimed here), or None."""
        if not self._screen_open:
            return None
        self._take_job(job, seq)
        if self._live():
            word = self._tail_status(sub_job_id)
            if self._NOT_ROUTED in text:
                why, label = "not-routed", f"tail {sub_job_id} waiting (not routed yet)"
                if sub_job_id in self._unrouted_logged:
                    label = ""  # the next answers would only repeat it
                self._unrouted_logged.add(sub_job_id)
            elif word in self._NOT_RUNNING_YET:
                why, label = word, f"tail {sub_job_id} waiting ({word})"
            elif self._PENDING_PLACEMENT in text:
                why, label = "pending-placement", f"tail {sub_job_id} waiting (pending placement, status {word})"
            else:
                return "error", ""
            self._wait_after_error(gen, sub_job_id, why)
            return "wait", label
        # The job ended: show its saved console now.
        if not self._claim_stdout():
            if self._console_shown(self._current_source):
                self._paint_saved()
            return None
        return "download", self._current_source or ""

    def _wait_after_error(self, gen: int, sub_job_id: str, why: str) -> None:
        """UI thread. The tail stopped because ``sub_job_id`` hasn't started: show the wait."""
        if gen != self._tail_gen or sub_job_id != self._current_source:
            return
        name = views.sub_job_label(sub_job_id)
        if why in self._PRE_PLACEMENT:
            self._set_waiting("pending", self._pending_note(sub_job_id))
        elif why == "pending-placement":
            # Tried again once the status changes (_resume_waiting_tail).
            self._wait_words = self._status_words(sub_job_id)
            self._set_waiting("pending", self._pending_note(sub_job_id))
        elif why == "not-routed":
            # Tried again once its status or the job's changes; while both run,
            # no change is coming, so each status check tries again.
            self._wait_words = job_word, word = self._status_words(sub_job_id)
            if word == "running" and job_word == "running":
                note = (f"waiting: the service isn't routing {name}'s log yet; the tail tries again each "
                        "minute (r tries now)")
            elif word == "running":
                note = (f"waiting: the service isn't routing {name}'s log yet; the tail tries again when the "
                        "job's status changes (r tries now)")
            else:
                note = (f"waiting: {name} hasn't started yet (the service isn't routing to it); "
                        f"the tail starts again once it is {views.status_word_for(self._job, 'running')}")
            self._set_waiting("stopped", note)
        else:
            self._set_waiting("stopped", f"waiting: {name} is {self._said(why)}; "
                              f"the tail starts again once it is {views.status_word_for(self._job, 'running')}")

    def _run_download(self, sub_job_id: str) -> None:
        """Worker thread. The saved-console download this view claimed (see _claim_stdout):
        _finish_download gives the claim back whatever happens, and the error log tells a bug
        in this view (a _CallbackError) from a failed download."""
        ok = False
        problem = None
        recent = False
        kept = False
        try:
            # The saved console is the whole job's output, minutes for a long job: start it
            # only once Logs is on screen, so a look at another tab doesn't pay for it.
            while self._screen_open and self._tail_hidden():
                time.sleep(0.25)
            # A client that can fetch one sub-job: the one on screen first, painted
            # as soon as it is in, then the rest (resumed: that one is done).
            by_sub_job = self._downloads_by_sub_job()
            plan = self._ask(self._download_started, sub_job_id, by_sub_job) if self._screen_open else None
            if plan is not None and self._fetch_consoles(plan):
                ok = True
                recent = self._console_may_be_incomplete(plan.seen, plan.ended_at)
                kept = self._keep_console(plan)
        except DownloadCancelled:
            pass  # the view closed; the next one resumes from what was saved
        except self._CallbackError:
            log_traceback(f"saved log {sub_job_id}: a bug in the TUI while showing it")
        except Exception as exc:  # noqa: BLE001 - keep the cache; record the traceback
            # Saved consoles are SQL on the job's Experiment: say so when the role can't read it.
            problem = views.access_hint(exc, "saved logs")
            log_traceback(f"saved log {sub_job_id} failed")
        try:
            if not self._post(self._finish_download, ok, problem, recent):
                self._cleanup_stage()
        finally:
            if kept:  # even if showing the console failed: the copy is in the cache either way
                self._hold_cache_to_limit()

    @dataclass(frozen=True)
    class _Download:
        """What a saved-console download fetches, as the view was when it started (see _download_started)."""

        first: str | None  # the sub-job on screen, fetched before the others (None: the whole job at once)
        again: bool  # only the chunks uploaded since the last download (r, moments after the job ended)
        root: str  # the view's temporary directory, downloaded into
        others: bool  # the job has other sub-jobs, fetched after ``first``
        seen: float | None  # when a get_job last had the job live (monotonic seconds)
        ended_at: object  # the job's state_changed_at
        created_at: object  # ... and its created_at (see _keep_console)

    def _fetch_consoles(self, plan: LogScreen._Download) -> bool:
        """Worker thread. Download the saved consoles ``plan`` names; False once the app went away meanwhile."""
        quiet = _quiet_download(self._client.download_stdout_logs)
        if plan.first is None:
            # r on a console fetched moments after the job ended: only the chunks uploaded since.
            results = self._client.download_stdout_logs(self._job_id, plan.root, **quiet,
                                                        **({"resume": True} if plan.again else {}))
            self._console_warnings(results, plan.root)
            return True
        # Resumed in this view's own directory: a sub-job fetched already only gets newer chunks.
        # Its newest chunks come first, shown until its whole console is in.
        preview = _download_options(self._client.download_stdout_logs, {
            "preview_seconds": self._CONSOLE_PREVIEW_SECONDS,
            "on_preview": lambda entry: self._console_preview(entry, plan.root)})
        try:
            results = self._client.download_stdout_logs(
                self._job_id, plan.root, sub_job_ids=[plan.first], resume=True,
                on_chunk=self._download_progress(first=True, others=plan.others), **quiet, **preview)
            self._console_warnings(results, plan.root)
        except NoChunksError:
            pass  # the sub-job on screen saved no console (yet): the others still come
        if not self._post(self._first_console_in, plan.first,
                          self._console_may_be_incomplete(plan.seen, plan.ended_at)):
            return False
        results = self._client.download_stdout_logs(
            self._job_id, plan.root, resume=True,
            on_chunk=self._download_progress(first=False, others=plan.others), **quiet)
        self._console_warnings(results, plan.root)
        return True

    def _console_preview(self, entry: dict, root: str) -> None:
        """Worker thread. The client's preview of the sub-job on screen, its console's newest
        chunks: kept in this view's directory (the client removes its own copy) and shown
        until the whole console is in."""
        source = entry.get("sub_job_id")
        if not source or not root or not self._screen_open:
            return
        target = Path(root) / ".preview" / source / "stdout.log"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(entry["path"], target)
        except (OSError, KeyError, TypeError):
            return  # no preview: the whole console shows once it is in
        try:
            self._post(self._show_console_preview, source, target)
        except self._CallbackError:
            # A bug in showing it must not fail the download it previews.
            log_traceback("showing a saved console's newest lines failed (a bug in the TUI)")

    def _show_console_preview(self, source: str, path: Path) -> None:
        """UI thread. A sub-job's newest saved-console lines, while the rest downloads."""
        if not self._screen_open or self._saved_shown(source):
            return
        self._stdout_preview[source] = path
        if source == self._current_source:
            self._paint_saved()

    def _console_warnings(self, results, root) -> None:
        """Worker thread. Pass a saved-console download's warnings to the UI thread."""
        warnings = _result_warnings(results, kind="stdout", job_id=self._job_id, output_dir=Path(root))
        if warnings:
            self._post(self._accept_download_warnings, "stdout", warnings)

    def _downloads_by_sub_job(self) -> bool:
        """Whether this client's download_stdout_logs takes sub_job_ids and on_chunk
        (an older client, or a test's fake, downloads the whole job at once)."""
        try:
            parameters = inspect.signature(self._client.download_stdout_logs).parameters
        except (TypeError, ValueError):
            return False
        return "sub_job_ids" in parameters and "on_chunk" in parameters

    # How often the download's chunk count is redrawn on the status line.
    _PROGRESS_SECONDS = 0.25

    def _download_started(self, sub_job_id: str, by_sub_job: bool) -> LogScreen._Download | None:
        """UI thread. A saved-console download starts: say so in the pane, and return what it
        fetches, as the view is now. ``by_sub_job``: the client fetches the sub-job on screen
        (else ``sub_job_id``) before the others; otherwise the whole job comes at once. None
        once the view has closed."""
        if not self._screen_open:
            return None
        first = (self._current_source or sub_job_id) if by_sub_job else None
        self._stdout_first = first
        self._stdout_others = first is not None and any(
            sid != first for sid in (*self._source_by_item.values(), *self._sub_job_status))
        self._write_line(self._download_banner(self._current_source))
        return self._Download(first, self._stdout_again, "" if self._stage is None else self._stage.name,
                              self._stdout_others, self._active_seen_at, self._state_changed_at,
                              self._job_created_at)

    def _download_banner(self, source: str | None) -> str:
        """UI thread. The pane's line for ``source`` while the saved consoles download."""
        first = self._stdout_first
        if first is None:
            if self._stdout_again:
                return "— downloading the saved console's newest chunks —"
            return "— downloading the saved console (the whole job; minutes for a long job) —"
        one, many = (("newest saved-console chunks",) * 2 if self._stdout_again
                     else ("saved console", "saved consoles"))
        name = views.sub_job_label(first)
        if source == first or source is None:
            if not self._stdout_others:
                return f"— downloading {name}'s {one} —"
            return f"— downloading {name}'s {one} first, then the other sub-jobs' —"
        mine = views.sub_job_label(source)
        if first not in self._stdout_have:
            return f"— downloading {name}'s {one} first, then the other sub-jobs' ({mine} among them) —"
        return f"— downloading the other sub-jobs' {many} ({mine} among them) —"

    def _download_progress(self, *, first: bool, others: bool):
        """Worker thread. The client's per-chunk callback: the chunk count on the status line, a few times a second.
        ``others``: the job has sub-jobs fetched after the first."""
        last = [0.0]

        def progress(sub_job_id: str, done: int, total: int, _bytes: int = 0) -> None:
            if not self._screen_open:
                # The view closed: the client stops at this chunk and keeps what it saved.
                raise DownloadCancelled("the saved-console view closed")
            now = time.monotonic()
            if last[0] < 0 or (done < total and now - last[0] < self._PROGRESS_SECONDS):
                return
            last[0] = now
            name = views.sub_job_label(sub_job_id)
            if first:
                text = (f"downloading {name}'s saved console: chunk {done:,} of {total:,}"
                        + (" (the other sub-jobs' come after it)" if others else ""))
            else:
                text = f"downloading the other sub-jobs' saved consoles: {name}, chunk {done:,} of {total:,}"
            try:
                self._post(self._show_download_note, text, (name, done, total))
            except self._CallbackError:
                # A bug in drawing the count: it must not fail the download, which this call is part of.
                last[0] = -1.0  # no more counts
                log_traceback("showing a saved console's download progress failed (a bug in the TUI)")

        return progress

    def _show_download_note(self, text: str | None, chunks: tuple[str, int, int] | None = None) -> None:
        """UI thread. ``chunks``: (sub-job, chunks done, of them), drawn as a bar (see _log_live)."""
        if self._screen_open and (text, chunks) != (self._download_note, self._download_chunks):
            self._download_note = text
            self._download_chunks = chunks
            self._render_log_status()

    def _first_console_in(self, source: str, recent: bool) -> None:
        """UI thread. The sub-job on screen has its saved console while the others' still come: show it."""
        if not self._screen_open:
            return
        self._stdout_have.add(source)
        self._stdout_preview.pop(source, None)  # the whole console replaces its newest lines
        self._stdout_recent = recent
        # Until the others' first chunk is counted (their listing can take a while).
        self._download_note = "downloading the other sub-jobs' saved consoles" if self._stdout_others else None
        self._download_chunks = None
        if source == self._current_source:
            self._paint_saved()
        else:
            self._render_log_status()

    def _configured_location(self) -> dict:
        """How this screen's client connects, for the copyable commands: its database
        and schema (the job's own, for a job from the account-wide list, whose
        client is bound there), endpoint and connection profile name."""
        auth = getattr(self._client, "_auth_provider", None)  # a profile client's SnowflakeProfileAuth
        found = {"database": getattr(self._client, "database", None), "schema": getattr(self._client, "schema", None),
                 "connection": getattr(auth, "connection_name", None),
                 "endpoint": getattr(self._client, "endpoint", None)}
        # Only real names: a test client's attributes may be anything.
        where = {key: value if isinstance(value, str) and value else None for key, value in found.items()}
        if where["endpoint"] == "cortex-training":
            where["endpoint"] = None
        return where

    def _cli(self, arguments: str) -> str:
        """``cortex-training [--connection C] [--database D --schema S] ARGUMENTS`` that
        reaches this job: copied from a job in another location, a bare command
        would go to the configured database and schema and not find it. Without a
        profile it connects as the TUI does (``--config FILE``, ``--host HOST`` or
        ``--base-url URL``; see ``views.sign_in_options``)."""
        where = self._configured_location()
        names = views.location_names(self._job)  # the row's own location wins
        if names is not None:
            where["database"], where["schema"] = views.sql_identifier(names[0]), views.sql_identifier(names[1])
        if not where.get("connection"):
            where.update(views.sign_in_options(getattr(self.app, "sign_in", None)))
        return " ".join(["cortex-training", *views.cli_options(**where), arguments])

    def _execution_log_lines(self, source: str) -> list[str]:
        """UI thread. What a failed job's execution log adds for a sub-job without a saved console.

        Jobs that fail while they are created or start up may have no saved
        console; their execution log (``_logs/<sub_job_id>/execution.jsonl``
        on the job's Experiment) still records the error. Empty when this
        doesn't apply or the log has nothing; the first call starts the read
        and returns a progress note. The filter and level apply to whole
        records (a record shows with its exception chain or not at all); the
        header always shows.
        """
        if views.status_word(self._job_status) != "failed":
            return []
        if source in self._exec_logs:
            notes, records = self._exec_logs[source]
            query = self._filter.lower()
            kept = [
                line
                for record, lines in records
                if (not query or any(query in line.lower() for line in lines))
                and entry_at_level(record, self._min_level)
                for line in lines
            ]
            return [*notes, *kept]
        if source not in self._exec_inflight:
            self._exec_inflight.add(source)
            self._load_execution_log(source)
        return [self._Note("— no saved console for this sub-job; reading its execution log for the error —")]

    @work(thread=True, group="execution-log")
    def _load_execution_log(self, source: str) -> None:
        failed = False
        try:
            logs = self._client.fetch_execution_logs(self._job_id, sub_job_id=source, filename="execution.jsonl")
            notes, records = self._execution_log_body(source, logs)
        except Exception as exc:  # noqa: BLE001 - shown in the pane; traceback in the error log
            failed = True
            log_traceback(f"execution log {source} failed")
            problem = views.access_hint(exc, "execution log") or (
                f"no saved console for this sub-job, and its execution log couldn't be read: "
                f"{views.describe_error(exc)} (details in {_ERROR_LOG})"
            )
            notes, records = [f"— {problem} —"], []
        self._post(self._apply_execution_log, source, notes, records, failed)

    def _execution_log_body(self, source: str, logs) -> tuple[list[str], list[tuple[object, list[str]]]]:
        """Worker thread. A sub-job's execution log for the pane: its header, then each record with its lines."""
        contents = [
            item.get("content") for item in (logs if isinstance(logs, list) else [])
            if isinstance(item, dict) and item.get("sub_job_id") == source
        ]
        records = [record for content in contents for record in log_format.execution_log_records(content)]
        if not records:
            return [], []
        name = views.sub_job_label(source)
        fetch = f"{self._cli(f'download-log {shlex.quote(str(self._job_id))}')} downloads the whole log"
        errors = [record for record in records if log_format.is_error_record(record)]
        if errors:
            shown = errors[-self._EXECUTION_ERROR_RECORDS:]
            head = (f"— no saved console for {name}: the job failed before it wrote one. The error records "
                    f"from its execution log follow ({fetch}) —")
        else:
            shown = records[-self._EXECUTION_TAIL_RECORDS:]
            head = (f"— no saved console for {name}, and its execution log has no error records; its last "
                    f"{len(shown)} record(s) follow ({fetch}) —")
        return [head], [(record, log_format.format_execution_record(record)) for record in shown]

    def _apply_execution_log(self, source: str, notes: list[str], records: list, failed: bool) -> None:
        self._exec_inflight.discard(source)
        # Its header is the TUI's own line; the records' lines are the log's.
        self._exec_logs[source] = ([self._Note(note) for note in notes], records)
        if failed:
            self._exec_failed.add(source)
        if source == self._current_source and self._saved_shown(source):
            self._paint_saved()

    def _content_width(self) -> int:
        """Usable text columns of the log pane (accounts for padding/scrollbar)."""
        lv = self._logview
        if lv is None:
            return 0
        try:
            w = lv.scrollable_content_region.width
        except Exception:  # noqa: BLE001 - not laid out yet
            w = 0
        if w <= 0:
            w = getattr(lv.size, "width", 0)
        return max(0, w)

    def _write_line(self, line: str) -> None:
        # Single lines are the TUI's own status lines, never log content.
        note = line if isinstance(line, self._Note) else self._Note(line)
        self._buffer([(None, note)])
        self._write_lines([note])
        if self._pane_truncated:
            self._render_log_status()  # it pushed a log line out: "last N lines" changes

    @staticmethod
    def _row_count(pieces: list[str]) -> int:
        # Log.write_lines splits each piece on every line boundary (not just \n).
        return sum(len(piece.splitlines()) for piece in pieces)

    @classmethod
    def _log_lines(cls, lines) -> int:
        """How many of ``lines`` are log lines, not the TUI's notes."""
        return sum(1 for line in lines if not isinstance(line, cls._Note))

    def _clear_pane(self) -> None:
        # The pane is drawn again (another sub-job, filter or level): a highlight on the
        # old lines would sit over other text, at the same line numbers. A repaint on its
        # way (see _repaint) was for what the pane showed before.
        self._cancel_repaint()
        self._reset_pane()

    def _reset_pane(self, *, keep_selection: bool = False) -> None:
        """Empty the pane and its buffers (see _clear_pane; a repaint that lands does it too)."""
        if not keep_selection:
            self.clear_selection()
        self._shown_lines = []
        self._shown_rows = []
        self._shown_kinds = []
        self._rows_total = 0
        self._shown_logs = 0
        self._pane_truncated = False
        self._last_day, self._in_traceback = None, NO_TRACEBACK
        self._unseen = 0
        self._jump = None
        if self._logview is not None:
            self._logview.clear()
            self._logview.max_lines = self._MAX_ROWS
            # Log.clear() keeps the scroll offset, and Log.write_lines(scroll_end=True)
            # follows only a view already at its end, so lines written in the same
            # callback would stay at the old offset, above the newest ones. An
            # empty pane starts at its top, which is also its end.
            self._logview.scroll_home(animate=False, immediate=True, force=True)

    def _trim_shown(self) -> int:
        """Drop the oldest shown lines past _MAX_LINES lines or _MAX_ROWS wrapped rows; how many went."""
        lines, rows = self._shown_lines, self._shown_rows
        drop = max(0, len(lines) - self._MAX_LINES)
        total = self._rows_total - sum(rows[:drop])
        while total > self._MAX_ROWS and drop < len(lines) - 1:
            total -= rows[drop]
            drop += 1
        self._rows_total = total
        if drop:
            self._shown_logs -= self._log_lines(lines[:drop])
            del lines[:drop]
            del rows[:drop]
            del self._shown_kinds[:drop]
            if not self._pane_truncated:
                self._pane_truncated = True
                self._render_log_status()
        return drop

    # How far down the pane the first dated line is looked for, to put its day's rule back.
    _RULE_LOOK_AHEAD = 50

    def _rule_the_top(self, width: int) -> None:
        """UI thread, once the oldest lines went with the day's rule above the first of them:
        put the rule back above the first dated line left, so the pane's top still says the day.
        A list insert and a check a page, and only while a long log keeps the pane full."""
        lv = self._logview
        if not isinstance(lv, _LogPane) or len(lv.lines) != self._rows_total:
            return
        row = 0
        for index, line in enumerate(self._shown_lines[:self._RULE_LOOK_AHEAD]):
            day = None if isinstance(line, self._Note) else display_day(self._display(line))
            if day:
                if lv.row_kinds[row] != DAY_KIND:
                    following = not self._paused and lv.is_vertical_scroll_end
                    lv.insert_row(row, day_rule(day, width), DAY_KIND)
                    self._shown_rows[index] += 1
                    self._rows_total += 1
                    lv.max_lines = self._rows_total
                    if following:
                        lv.scroll_end(animate=False, immediate=True, x_axis=False)
                return
            row += self._shown_rows[index]

    def _write_lines(self, lines: list[str]) -> None:
        lv = self._logview
        # A page posted by the tail can land after the screen closed; writing to its
        # detached Log starts a Textual worker that crashes the app.
        if lv is None or not lines or not self._screen_open:
            return
        if self._repaint_job is not None:
            # Shown now, and again after the lines of the repaint on its way, which
            # replaces what the pane shows when it lands (see _repaint).
            self._repaint_pending.extend(lines)
        w = self._content_width()
        # A batch is wrapped from its newest line back, only as far as the
        # pane's line and row budgets reach: a repaint of a long log would
        # otherwise wrap thousands of lines only to drop them again.
        chunks: list[tuple[str, list[str], int]] = []
        added = 0
        for line in reversed(lines):
            if len(chunks) >= self._MAX_LINES:
                break
            pieces = self._wrap(line, w)
            rows = self._row_count(pieces)
            if chunks and added + rows > self._MAX_ROWS:
                break
            chunks.append((line, pieces, rows))
            added += rows
        chunks.reverse()
        self._write_chunks(chunks, w, cut=len(chunks) < len(lines))

    def _write_chunks(self, chunks: list[tuple[str, list[str], int]], width: int, *, cut: bool = False,
                      kinds: list | None = None, context: tuple | None = None) -> None:
        """UI thread. Append lines already wrapped to ``width``: (line, its rows, row count) each.
        ``cut``: older lines of the same batch didn't fit. ``kinds`` and ``context``: each line's
        kind, and (each row's kind, the newest day, where a traceback stands), when they were
        worked out with the wrapping for an empty pane (see in_context); else here."""
        lv = self._logview
        if lv is None or not chunks or not self._screen_open:
            return
        if cut and self._shown_lines:
            # The lines on screen came before the batch's dropped ones: the pane
            # keeps one unbroken run of the newest lines.
            self._clear_pane()
        fresh = not self._shown_lines
        if context is not None and not fresh:
            # Worked out for an empty pane, which this isn't (its callers clear it first): the
            # rules it put in go, and the kinds are worked out again after the lines on screen.
            rows, at = context[0], 0
            unruled = []
            for line, pieces, count in chunks:
                ruled = count > 1 and at < len(rows) and rows[at] == DAY_KIND
                unruled.append((line, pieces[1:], count - 1) if ruled else (line, pieces, count))
                at += count
            chunks = unruled
            kinds = context = None
        if context is None:
            if kinds is None:
                kinds = [line_kind(self._display(line), note=isinstance(line, self._Note)) for line, _, _ in chunks]
            texts = [None if isinstance(line, self._Note) else self._display(line) for line, _, _ in chunks]
            chunks, kinds, row_kinds, day, in_block = in_context(chunks, kinds, texts, width, self._last_day,
                                                                 self._in_traceback)
        else:
            row_kinds, day, in_block = context
        self._last_day, self._in_traceback = day, in_block
        # The lines that come while the reader is away from the end (see _log_live); not a
        # pane drawn anew (a re-wrap while paused), whose lines were there already.
        if not fresh and (self._paused or not lv.is_vertical_scroll_end):
            self._unseen += self._log_lines(line for line, _, _ in chunks)
        # Always stick to the tail so every viewer sees the newest lines on
        # open and as they stream — cache replay and live alike. Pause is the
        # one escape hatch: it stops following so the user can scroll back.
        follow = not self._paused
        if fresh:
            self._wrap_width = width
        elif width != self._wrap_width:
            self._wrap_width = -1  # rows now wrapped to mixed widths; the next _rewrap redoes them
        wrapped = [piece for _, pieces, _ in chunks for piece in pieces]
        added = sum(rows for _, _, rows in chunks)
        self._shown_lines.extend(line for line, _, _ in chunks)
        self._shown_rows.extend(rows for _, _, rows in chunks)
        self._shown_kinds.extend(kinds)
        self._shown_logs += self._log_lines(line for line, _, _ in chunks)
        if self._shown_logs and not self._warm_up_started:
            after_refresh(self, self._warm_up)  # once these lines are drawn
        self._rows_total += added
        if cut and not self._pane_truncated:
            self._pane_truncated = True
            self._render_log_status()
        dropped = self._trim_shown()
        # The widget keeps exactly the rows of the lines kept above.
        before = len(lv.lines)
        lv.max_lines = max(1, self._rows_total)
        # One write_lines call per page keeps the UI responsive under
        # high-velocity logs (vs. a cross-thread hop + write per line).
        self._write_rows(lv, wrapped, row_kinds, scroll_end=follow)
        if follow and before + added > lv.max_lines:
            # Once full, Log drops its oldest rows and repaints only the rows past
            # its old end, which that has moved out of view, and a view already
            # at the end doesn't scroll: repaint it, or the newest lines don't show.
            lv.refresh()
        if dropped:
            self._rule_the_top(width)
        if self._jump is not None:
            self._recount_jump()  # lines came (or went): the chip's count follows them
        if len(self._display_cache) > 4 * self._MAX_LINES:
            self._prune_display_cache()

    @staticmethod
    def _write_rows(lv: Log, rows: list[str], row_kinds: list, *, scroll_end: bool) -> None:
        """Write ``rows`` to the pane, with each one's kind (see in_context)."""
        if not isinstance(lv, _LogPane):
            lv.write_lines(rows, scroll_end=scroll_end)
            return
        lv.write_rows(rows, row_kinds, scroll_end=scroll_end)

    def _prune_display_cache(self) -> None:
        """Keep the display text of the lines this view still has, and drop the rest."""
        keep = set(self._shown_lines)
        keep.update(line for _, line in self._entries)
        self._display_cache = {line: text for line, text in self._display_cache.items() if line in keep}

    def _rewrap(self) -> None:
        """Re-wrap the on-screen buffer after the pane width changes (off the UI thread: see _repaint)."""
        lv = self._logview
        if lv is None or not self._screen_open:
            return
        w = self._content_width()
        job = self._repaint_job
        if w <= 0 or w == (job.width if job is not None else self._wrap_width):
            return
        if job is not None:
            # A repaint is on its way (a new filter's, say): it is made again for this width.
            self._repaint(job.pairs, job.query, job.level, rewrap=job.rewrap, truncated=job.truncated, top=job.top,
                          keep_pending=True, mark=job.mark)
        elif not self._shown_lines:
            self._wrap_width = w  # nothing to wrap
        else:
            # A paused reader keeps their place: the line at the top of the view stays there
            # (the one a re-wrap still scrolling to it named, when two come close together).
            top = None
            if self._paused:
                top = self._pending_top if self._pending_top is not None else self._line_at_row(
                    int(lv.scroll_offset.y))
            # And the line a jump (e) marked stays marked, its place among the others too.
            mark = None
            if isinstance(lv, _LogPane) and lv.mark is not None and self._jump is not None:
                mark = (self._line_at_row(lv.mark[0]), self._jump)
            self._repaint([(None, line) for line in self._shown_lines], rewrap=True, truncated=self._pane_truncated,
                          top=top, mark=mark)

    # ─── repaints: the pane's lines filtered and wrapped off the UI thread ──
    @dataclass(frozen=True)
    class _Repaint:
        """A repaint of the log pane on its way (see _repaint)."""

        pairs: list  # (entry, line) each, oldest first
        query: str  # the filter, and the level, they must pass
        level: object
        width: int  # the pane width they are wrapped to
        rewrap: bool  # the lines on screen, at a new width (a highlight on them stays, as it always has)
        truncated: bool  # the pane doesn't hold the source's oldest lines, whatever this repaint fits
        top: int | None  # a paused reader's line (an index into pairs), back at the top of the view after
        mark: tuple | None = None  # (the line a jump marked, an index into pairs; what _jump said of it)

    def _repaint(self, pairs: list, query: str = "", level=None, *, rewrap: bool = False, truncated: bool = False,
                 top: int | None = None, keep_pending: bool = False, mark: tuple | None = None) -> None:
        """UI thread. Show the ``pairs`` ((entry, line) each, oldest first, the TUI's notes among
        them) that pass ``query`` and ``level``, filtered and wrapped off the UI thread
        (_prepare_rows) and shown by _apply_rows: the pane's 20,000 lines take a while, and keys
        keep working meanwhile. Until then the pane keeps what it shows; the lines written
        meanwhile (a page of the tail) show at once, and again after the repaint's when it
        lands (``keep_pending``: those written since the repaint this one replaces, too).
        ``rewrap``: the lines on screen, at a new width; ``truncated``: the pane doesn't hold the
        source's oldest lines, whatever this one fits; ``top``: the line of ``pairs`` a paused
        reader had at the top of the view."""
        if not keep_pending:
            self._repaint_pending = []
        self._repaint_gen += 1
        self._repaint_job = job = self._Repaint(pairs, query, level, self._content_width(), rewrap, truncated, top,
                                                mark)
        # The worker reads none of the view's state: what it needs is passed as it is now.
        self._prepare_rows(self._repaint_gen, job, self._line_prefixes(), dict(self._display_cache))

    def _cancel_repaint(self) -> None:
        """UI thread. Drop the repaint on its way, and the lines kept for after it: the pane is drawn anew."""
        if self._repaint_job is not None:
            self._repaint_job = None
            self._repaint_gen += 1
        self._repaint_pending = []

    @work(thread=True, exclusive=True, group="wrap")
    def _prepare_rows(self, gen: int, job: LogScreen._Repaint, prefixes: tuple[str, str], known: dict) -> None:
        """Worker thread. Filter and wrap ``job``'s lines for the pane; _apply_rows shows them.
        ``known``: the display cache's texts (see _display)."""
        worker = get_current_worker()
        wrapped = self._wrap_pairs(job.pairs, job.query, job.level, job.width, prefixes, known,
                                   lambda: worker.is_cancelled)
        if wrapped is not None and not worker.is_cancelled:
            self._post(self._apply_rows, gen, *wrapped)

    @classmethod
    def _wrap_pairs(cls, pairs: list, query: str, level, width: int, prefixes: tuple[str, str], known: dict,
                    cancelled) -> tuple[list, list, tuple, dict, bool] | None:
        """Any thread. The newest of ``pairs`` that pass ``query`` and ``level`` (the TUI's notes
        always do), wrapped to ``width`` from the newest back as far as the pane's line and row
        budgets reach. Returns ``(chunks, kinds, context, displays, cut)`` for _write_chunks: each
        kept line's rows and kind (see line_kind and in_context), the context they leave the pane
        in (each row's kind, the newest day, where a traceback stands), the display text of the
        log lines ``known`` lacked, and whether an older line that passes didn't fit; None once
        ``cancelled()``."""
        first, rest = prefixes
        query = query.lower()
        chunks: list[tuple[str, list[str], int]] = []
        kinds: list[str | None] = []
        texts: list[str | None] = []
        displays: dict[str, str] = {}
        total = 0
        cut = False
        for n, (entry, line) in enumerate(reversed(pairs)):
            if n % 1024 == 0 and cancelled():
                return None
            note = isinstance(line, cls._Note)
            if not note and not ((not query or query in line.lower()) and entry_at_level(entry, level)):
                continue
            if len(chunks) >= cls._MAX_LINES:
                cut = True
                break
            if note:  # as _display, _wrap and _prefixes show it
                text = safe_text(line)
                pieces = wrap_log_line(text, width, max_rows=cls._MAX_ROWS_PER_LINE)
            else:
                text = known.get(line)
                if text is None:
                    text = displays.get(line)
                    if text is None:
                        text = displays[line] = display_line(line)
                pieces = wrap_log_line(compact_stamp(text), width, first=first, rest=rest,
                                       max_rows=cls._MAX_ROWS_PER_LINE)
            rows = cls._row_count(pieces)
            if chunks and total + rows > cls._MAX_ROWS:
                cut = True
                break
            chunks.append((line, pieces, rows))
            kinds.append(line_kind(text, note=note))
            texts.append(None if note else text)
            total += rows
        chunks.reverse()
        kinds.reverse()
        texts.reverse()
        # The pane is drawn anew from these: their context starts with them (see in_context).
        chunks, kinds, row_kinds, day, in_block = in_context(chunks, kinds, texts, width)
        return chunks, kinds, (row_kinds, day, in_block), displays, cut

    def _apply_rows(self, gen: int, chunks: list, kinds: list, context: tuple, displays: dict, cut: bool) -> None:
        """UI thread. Show what _prepare_rows filtered and wrapped, then the lines written meanwhile,
        unless a newer repaint, or a clear, superseded it."""
        job = self._repaint_job
        lv = self._logview
        if gen != self._repaint_gen or job is None or lv is None or not self._screen_open:
            return
        self._repaint_job = None
        pending, self._repaint_pending = self._repaint_pending, []
        self._display_cache.update(displays)
        self._reset_pane(keep_selection=job.rewrap)
        self._wrap_width = job.width
        # One write for the whole buffer: a call per row took seconds at 20,000 lines.
        self._write_chunks(chunks, job.width, cut=cut, kinds=kinds, context=context)
        if job.truncated and not self._pane_truncated:
            self._pane_truncated = True  # older lines had left the view before this repaint
        self._write_lines(pending)
        # The pane was emptied, which leaves it at its top: back to the tail unless the reader has
        # paused to read history; then back to the line they were reading (the oldest kept, if
        # that one was dropped).
        if not self._paused:
            lv.scroll_end(animate=False)
        elif job.top is not None:
            self._pending_top = max(0, job.top - (len(job.pairs) - len(chunks)))
            after_refresh(self, self._scroll_to_pending_top)
        if job.mark is not None and job.mark[0] is not None:
            index = job.mark[0] - (len(job.pairs) - len(chunks))  # (unless the line was dropped)
            if 0 <= index < len(self._shown_lines) and isinstance(lv, _LogPane):
                lv.set_mark(self._line_rows(index))
                self._jump = job.mark[1]
                self._recount_jump()
        if job.width != self._content_width():
            after_refresh(self, self._rewrap)  # the pane changed width meanwhile
        self._render_log_status()

    def _scroll_to_pending_top(self) -> None:
        """After a re-wrap's layout: put the paused reader's line back at the top of the view."""
        top, self._pending_top = self._pending_top, None
        if top is not None and self._logview is not None:
            self._logview.scroll_to(y=sum(self._shown_rows[:top]), animate=False)

    def _copied_text(self, start: tuple[int, int] | None, end: tuple[int, int] | None) -> str:
        """UI thread (a copy). What a highlight from ``start`` to ``end`` copies: each line it
        covers whole as it was logged (what ``s`` saves: its whole time, no day's rule, no
        indent), and the highlighted part of one it covers in part (see format.copied_text)."""
        lines = self._shown_lines
        lv = self._logview
        rows = list(lv.lines) if lv is not None else []
        if len(rows) != sum(self._shown_rows):  # (never, but a copy must not fail): as drawn
            return "\n".join(rows)

        def shown(index: int) -> str:  # the text the line's rows were wrapped from (see _wrap)
            line = lines[index]
            return self._display(line) if isinstance(line, self._Note) else compact_stamp(self._display(line))

        kinds = getattr(lv, "row_kinds", [])
        rules = {row for row, kind in enumerate(kinds) if kind == DAY_KIND}
        return log_format.copied_text(rows, self._shown_rows, lambda index: lines[index], shown, start, end,
                                      indent=self._line_prefixes()[1], rules=rules)

    def _line_at_row(self, row: int) -> int | None:
        """The index in ``_shown_lines`` of the line drawn on pane row ``row``."""
        seen = 0
        for index, rows in enumerate(self._shown_rows):
            seen += rows
            if seen > row:
                return index
        return None
