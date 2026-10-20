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

"""Read-only Textual TUI for Cortex Training job logs and GPU resources.

Flow: open → a list of jobs with their status (via the SDK's list_jobs) →
pick one → its preview, then its Logs and Resources dashboard. Pass a job_id on the command line
to jump straight to that dashboard and skip the picker.

Imports ``textual`` at module load. The interface is strictly read-only and
never mutates job state.
"""

from __future__ import annotations

import asyncio
import contextvars
import gc
import os
import threading
import time
from collections import deque
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor

from textual import events
from textual import work
from textual.app import App
from textual.binding import Binding
from textual.geometry import Offset
from textual.css.query import NoMatches
from textual.notifications import Notification
from textual.notifications import Notify
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Footer
from textual.widgets._footer import FooterKey
from textual.widgets import HelpPanel
from textual.widgets import Input
from textual.widgets._toast import Toast
from textual.widgets._toast import ToastRack

from cortex_training.tui.account import JobClients
from cortex_training.tui.job_list import JobListScreen
from cortex_training.tui.logs import LogScreen
from cortex_training.tui.account import JobLister
from cortex_training.tui.format import one_line
from cortex_training.tui.clipboard import copied_note
from cortex_training.tui.clipboard import copy_local
from cortex_training.tui.clipboard import screen_dump
from cortex_training.tui.clipboard import selected_text
from cortex_training.tui.format import safe_text
from cortex_training.tui.untrusted import printable_client
from cortex_training.tui.widgets import SelectableTable
from cortex_training.tui.widgets import close_keys_panel
from cortex_training.tui.widgets import scroll_area
from cortex_training.tui.widgets import scrolling_region









class _DaemonThreadExecutor(ThreadPoolExecutor):
    """A bounded default executor whose running calls do not hold up process exit."""

    def __init__(self, max_workers: int | None = None) -> None:
        super().__init__(max_workers=max_workers)
        self._pending: deque[tuple] = deque()
        self._running = 0

    def submit(self, fn, /, *args, **kwargs) -> Future:
        future: Future = Future()
        with self._shutdown_lock:
            if self._shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")
            self._pending.append((future, contextvars.copy_context(), fn, args, kwargs))
            self._start_pending()
        return future

    def _start_pending(self) -> None:
        while self._pending and self._running < self._max_workers:
            task = self._pending.popleft()
            self._running += 1
            threading.Thread(target=self._run, args=(task,), name="cortex-tui-worker", daemon=True).start()

    def _run(self, task: tuple) -> None:
        future, context, fn, args, kwargs = task
        try:
            if future.set_running_or_notify_cancel():
                try:
                    future.set_result(context.run(fn, *args, **kwargs))
                except BaseException as exc:  # noqa: BLE001 - delivered to the awaiting worker
                    future.set_exception(exc)
        finally:
            with self._shutdown_lock:
                self._running -= 1
                self._start_pending()

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        # asyncio asks its default executor to wait during loop teardown. Running
        # network calls are daemon threads, so quitting remains immediate.
        with self._shutdown_lock:
            self._shutdown = True
            while self._pending:
                self._pending.popleft()[0].cancel()


















def _freeze_heap() -> None:
    """Exclude the stable TUI heap from later garbage collections."""
    gc.collect()
    gc.freeze()








































































class CortexTrainingLogTUI(App):
    """Entry app: show the job picker, or jump straight to a job dashboard when a
    job_id is supplied."""

    CSS = """
    #jobs { height: 1fr; }
    #listinfo { padding: 0 1; height: auto; }
    #nomatch-box { height: 1fr; scrollbar-gutter: stable; }
    #nomatch { padding: 1 2; color: $warning; height: auto; }
    #jobs > .datatable--fixed { background: $surface; }
    #sources { width: 16; border-right: solid $panel; }
    #job-tabs, #job-tabs ContentSwitcher, #logs, #resource, #logs-body {
        height: 1fr;
    }
    #logview { height: 1fr; padding: 0 1; }
    #log-status { height: auto; padding: 0 1; color: $warning; }
    #summary { color: $text-muted; padding: 0 1; }
    #logfilter { border: tall $panel; }
    #jobfilter { border: tall $panel; }
    #resource-body { height: 1fr; }
    #resource-list { width: 16; border-right: solid $panel; scrollbar-size-vertical: 1; }
    #resource-pane { height: 1fr; padding: 0 1; scrollbar-gutter: stable; scrollbar-size-vertical: 1; }
    #resource-status { color: $text-muted; height: auto; dock: top; }
    #resource-stats { height: auto; margin-top: 1; }
    #resource-charts { height: auto; margin-top: 1; text-wrap: nowrap; text-overflow: clip; }
    #summary-body { padding: 1 1; scrollbar-gutter: stable; }
    #health { height: auto; margin: 1 0; }
    #subjobs-title { height: auto; margin-top: 1; }
    #subjobs { height: auto; max-height: 12; margin: 0 1; }
    LogScreen > .screen--selection {
        background: $block-cursor-blurred-background;
        color: $block-cursor-blurred-foreground;
    }
    ToastRack { dock: top; align: right top; margin-bottom: 0; scrollbar-size-vertical: 0; }
    Toast.-information { padding: 0 1; margin-top: 0; width: auto; max-width: 80%; }
    """
    BINDINGS = [Binding("q", "quit", "Quit", priority=True)]
    ENABLE_COMMAND_PALETTE = False
    # How the client connects, set by `cortex-training tui` (never a secret), so
    # the commands the TUI shows to copy connect the same way; None when unknown
    # (the demo, tests). See views.sign_in_options.
    sign_in: dict | None = None

    def __init__(self, client, job_id=None, *, sub_job_id=None, poll_interval=1.0, now=None):
        """Every job record read through ``client`` and its per-location clients comes back printable
        (:mod:`cortex_training.tui.untrusted`)."""
        super().__init__()
        client = printable_client(client)
        self._client = client
        self._job_id = job_id
        self._sub_job_id = sub_job_id
        self._poll_interval = poll_interval
        self._now = now
        # Shared by every job list this app shows: one client per job location, which
        # list to show (a), and whether the account-wide list was available (and when to ask again).
        self._clients = JobClients(client)
        self._lister = JobLister(client)
        self._heap_frozen = False
        # A left-button press, and whether it has moved since (see on_event).
        self._press_at = None
        self._press_moved = False
        self._copy_note: Notification | None = None  # the last copy's note (see copy_out)
        self._copy_what: str | None = None  # while copy_out copies: what its note calls the text
        # This machine's clipboard takes one copy at a time, in order (see _copy_locally).
        self._copy_lock = threading.Lock()
        self._copies = 0  # copies started so far: the last one is the one the clipboard keeps
        self._press_table = False  # a press on a table, where a hand that moves a cell still clicks
        self._dragged = False  # the last press moved before its release
        self._confine: Widget | None = None  # the scrolling area a drag stays in (see _confine_drag)
        self._confine_header = False  # and the drag started on that table's header, where it stays
        self._footer_down: tuple | None = None  # (footer key, button) pressed (see _footer_click)
        self.title = "cortex training"

    def check_action(self, action: str, parameters) -> bool | None:
        if action == "quit" and isinstance(self.focused, Input):
            return False  # a letter in a search or filter
        return True

    def copy_to_clipboard(self, text: str) -> None:
        """The terminal's clipboard (OSC 52, which some terminals ignore) at once, then this machine's
        (see :mod:`cortex_training.tui.clipboard`) on a worker thread, and a note once that is done: its tool can
        take seconds (xclip waits out its timeout without ``DISPLAY``). Textual's own copies (ctrl+c in a text
        box) come here too."""
        if not text:
            return
        super().copy_to_clipboard(text)
        self._copies += 1
        self._copy_locally(text, self._copy_what or f"{len(text):,} characters", self._copies)

    @work(thread=True, group="clipboard")
    def _copy_locally(self, text: str, what: str, copy: int) -> None:
        """Worker thread: put ``text`` on this machine's clipboard, then show the note
        for it (``what`` names it). Copies take turns, in the order they were made, so
        the clipboard ends with the last; one that a newer copy replaced while it
        waited is skipped, and the newer one's note follows."""
        with self._copy_lock:
            if copy != self._copies:
                return
            local = copy_local(text)
            # Still in turn: the next copy's note can't come before this one's.
            try:
                self.call_from_thread(self._note_copy, copied_note(what, local), local)
            except RuntimeError:  # the app stopped meanwhile
                pass

    _COPY_NOTE_SECONDS = 3
    _SENT_NOTE_SECONDS = 8  # the note when only the terminal's clipboard got it: longer, with a tip

    def copy_out(self, text: str, what: str) -> None:
        """Copy ``text`` and say so once it is copied: ``what`` names it ("3 lines")."""
        text = safe_text(text)
        self._copy_what = what
        try:
            self.copy_to_clipboard(text)
        finally:
            self._copy_what = None
        if not text:
            # Nothing to copy, so no worker runs: the note is the one for a copy only
            # the terminal's clipboard took.
            self._note_copy(copied_note(what, False), False)

    def _note_copy(self, message: str, local: bool = True) -> None:
        note = Notification(message, timeout=self._COPY_NOTE_SECONDS if local else self._SENT_NOTE_SECONDS)
        # One copy's note at a time: the last one's, still showing, would stack beside
        # it. Added now, not as a message, so two copies in one burst of keys replace
        # each other too.
        previous, self._copy_note = self._copy_note, note
        try:
            if previous is not None:
                if previous in self._notifications:
                    self._unnotify(previous, refresh=False)
                self._drop_note(previous)
            self._notifications.add(note)
            self._refresh_notifications()
        except Exception:  # noqa: BLE001 - another Textual keeps its notes otherwise: as a message
            self.post_message(Notify(note))

    def _drop_note(self, note: Notification) -> None:
        """Remove what holds ``note`` on screen. Textual removes a replaced note but not
        the row it sat in (its timer would have), and each such empty row pushed the
        next note one row further down, over the screen."""
        for screen in self.screen_stack:
            for rack in screen.query(ToastRack):
                try:
                    rack.get_child_by_id(ToastRack._toast_id(note)).remove()
                except NoMatches:
                    pass

    def copy_highlight(self, screen: Screen) -> bool:
        """Copy the text the mouse highlighted on ``screen`` (in a table too: see
        :class:`SelectableTable`). False when nothing is."""
        text = selected_text(screen)
        if not text:
            return False
        lines = text.count("\n") + 1
        self.copy_out(text, f"{lines:,} lines" if lines > 1 else f"{len(text):,} characters")
        return True

    def copy_everything(self, screen: Screen, what: str = "everything on the screen") -> None:
        """Copy all ``screen`` holds, as text, with what is scrolled out of view (see
        :func:`cortex_training.tui.clipboard.screen_dump`)."""
        text = screen_dump(screen)
        self.copy_out(text, f"{what} ({text.count(chr(10)) + 1 if text else 0:,} lines)")

    def action_toggle_keys(self) -> None:
        """``?``: show the keys panel, or close it."""
        if not close_keys_panel(self.screen):
            self.action_show_help_panel()

    def validate_sub_title(self, sub_title: str) -> str:
        # The header is one row: the subtitle quotes job ids, reasons and search
        # text, which must neither break the row nor reach the terminal as an escape.
        return one_line(sub_title)

    def job_list_screen(self) -> JobListScreen:
        """A job list screen with this app's client, per-location clients and account-list state."""
        return JobListScreen(self._client, poll_interval=self._poll_interval, now=self._now, clients=self._clients,
                             lister=self._lister)

    # The heap is frozen again this long after the first list is drawn: a pause
    # of its own, once the list is on screen, rather than one with drawing it.
    _FREEZE_HEAP_AFTER_SECONDS = 1.0

    def job_list_drawn(self, screen: Screen) -> None:
        """A job list drew the jobs it loaded. The first time, freeze the heap
        soon after (see :func:`_freeze_heap`): the list's records and rows are
        most of it by then."""
        if self._heap_frozen or self.is_headless:
            return
        self._heap_frozen = True

        def freeze() -> None:
            if self.screen is screen:
                _freeze_heap()
            else:
                # A job's screen opened meanwhile: frozen, it would stay once closed.
                # The list's next load tries again.
                self._heap_frozen = False

        self.set_timer(self._FREEZE_HEAP_AFTER_SECONDS, freeze)

    def job_elsewhere(self, job) -> bool:
        """Whether ``job`` is in a database and schema other than the configured ones."""
        return self._clients.elsewhere(job)

    def on_mount(self) -> None:
        # Before any screen starts a worker: quit must not wait for them.
        loop = asyncio.get_running_loop()
        # Enough slots that a few blocked reads can't queue a new screen's first one:
        # a 2-vCPU host would otherwise allow six, and an in-flight request against a
        # slow service can't be aborted when its worker is cancelled.
        loop.set_default_executor(_DaemonThreadExecutor(max_workers=max(32, (os.cpu_count() or 1) + 4)))
        # Only in a terminal: a headless run is a test, and a test process runs many apps,
        # whose objects left in reference cycles a freeze would keep.
        if not self.is_headless:
            # What the imports made (most of the heap until a list arrives) is
            # frozen before any screen exists, so the collections while the first
            # list is read and drawn don't visit it.
            _freeze_heap()
        if self._job_id:
            # A job opened by id is in the configured database and schema.
            self.push_screen(
                LogScreen(
                    self._client,
                    self._job_id,
                    sub_job_id=self._sub_job_id,
                    poll_interval=self._poll_interval,
                )
            )
        else:
            self.push_screen(self.job_list_screen())

    async def on_event(self, event: events.Event) -> None:
        # A key, click, scroll or paste means someone is at the screen (a mouse
        # move doesn't): LogScreen pauses its live tail after a long gap.
        if isinstance(event, _INPUT_EVENTS) and not event.is_forwarded:
            self.last_input_at = time.monotonic()
        if not isinstance(event, events.MouseEvent) or event.is_forwarded:
            await super().on_event(event)
            return
        if event.button == 3 and isinstance(event, (events.MouseDown, events.MouseUp)):
            # A right-click does nothing. Textual would take it for a click:
            # one clears the highlight, and moves a table's cursor or picks an item.
            return
        self._footer_wander(event)
        if self._footer_click(event):
            return
        # Whether the pointer moved is where it went, not where a drag confined to an
        # area counts it: held past the edge straight out from the press, it would be
        # back on the press.
        pointer = event
        event = self._confine_drag(event)
        keep = not self._track_press(pointer)
        kept = self.screen.selections
        if isinstance(event, events.MouseUp) and self._dragged:
            # A drag isn't a click: Textual makes one of any press and release on one
            # widget, which toggled the header, picked a list item, switched a tab,
            # or opened or folded the job on a table's row; and it drops the highlight
            # when the button comes up where it went down, as a drag let go straight
            # out past the edge of its area does (see above).
            self._mouse_down_widget = None
            self.screen._mouse_down_offset = None
        screen = self.screen
        blank = isinstance(event, events.MouseDown) and self._on_blank(event)
        if blank:
            # A press on no text (the empty part of a list or a pane): Textual would
            # select every widget in the rows dragged over, the charts beside an
            # empty panel. It selects nothing here, and drops what was highlighted.
            screen.ALLOW_SELECT = False
            screen.clear_selection()
        try:
            await super().on_event(event)
        finally:
            if blank:
                del screen.ALLOW_SELECT
        if keep and kept and not self.screen.selections:
            self.screen.selections = kept  # a click on a note or the footer: the highlight stays

    def _footer_wander(self, event: events.MouseEvent) -> None:
        """A press on a footer key that moves two cells along it, or off its row, is a drag:
        its release runs nothing, wherever it comes back to."""
        if self._footer_down is not None and isinstance(event, events.MouseMove):
            key, button, at = self._footer_down
            if event.screen_y != at.y or abs(event.screen_x - at.x) >= SelectableTable._DRAG_CELLS:
                self._footer_down = (key, 0, at)  # no button runs it now

    def _footer_click(self, event: events.MouseEvent) -> bool:
        """A footer key runs when it is clicked: pressed and released on it, the pointer
        all but still. Textual runs it on the press, so a drag that began on q Quit quit.
        True for the press and the release there, which nothing else sees (so a click on
        the footer keeps a highlight)."""
        if isinstance(event, events.MouseDown):
            key = self._widget_at(event, FooterKey)
            self._footer_down = (key, event.button, event.screen_offset) if key is not None else None
            return key is not None
        if isinstance(event, events.MouseUp) and self._footer_down is not None:
            (key, button, at), self._footer_down = self._footer_down, None
            # As on a table's rows: a hand that moved two cells along the key, or off its row, dragged.
            still = event.screen_y == at.y and abs(event.screen_x - at.x) < SelectableTable._DRAG_CELLS
            if button == 1 and still and self._widget_at(event, FooterKey) is key:
                if getattr(key, "_disabled", False):
                    self.bell()
                else:
                    self.simulate_key(key.key)
            return True
        return False

    def _widget_at(self, event: events.MouseEvent, kinds) -> Widget | None:
        try:
            widget, _region = self.screen.get_widget_at(event.x, event.y)
        except Exception:  # noqa: BLE001 - NoWidget: nothing there
            return None
        return next((node for node in widget.ancestors_with_self if isinstance(node, kinds)), None)

    def _confine_drag(self, event: events.MouseEvent) -> events.MouseEvent:
        """A drag that starts in a scrolling area (the log, a list, the Resources pane, the
        preview) stays in it: past its edges the pointer counts as at the edge, where
        Textual scrolls the area (see :class:`FollowsPointer`). Textual would otherwise
        select every widget between the two ends, a log from its very first line, and a
        press on the log's padding took its offset from the top of the log. A table with
        no scrolling pane around it (the job list's) is an area of its own, its rows one
        and its header another (see :meth:`SelectableTable.drag_region`)."""
        if isinstance(event, events.MouseDown):
            self._confine = None
            self._confine_header = False
            if event.button in (1, 2) and not self.mouse_captured:
                widget, _offset = self.screen.get_widget_and_offset_at(event.x, event.y)
                if widget is not None:
                    self._quiet_keys_panel(widget)
                if widget is not None and widget.allow_select:
                    self._confine = scroll_area(widget)
                    self._confine_header = (isinstance(self._confine, SelectableTable)
                                            and self._confine.on_header(event.y))
        area = self._confine
        header = self._confine_header
        if isinstance(event, events.MouseUp):
            self._confine = None
            self._confine_header = False
        if area is None or not area.is_attached:
            return event
        region = area.drag_region(header=True) if header else scrolling_region(area)
        if not region:
            return event
        x = min(max(event.x, region.x), region.right - 1)
        y = min(max(event.y, region.y), region.bottom - 1)
        if (x, y) == (event.x, event.y):
            return event
        # Held off a table's header, the pointer is on it, at the top of the table, where
        # Textual would scroll the rows under it as the pointer moves up or down.
        delta_y = 0 if header else event.delta_y
        return event.__class__(event.widget, x, y, event.delta_x, delta_y, event.button, event.shift,
                               event.meta, event.ctrl, screen_x=x, screen_y=y, style=event.style)

    @staticmethod
    def _quiet_keys_panel(widget: Widget) -> None:
        """The keys panel's table can't be selected (it is drawn as one Rich table), and a
        drag from the panel selected what was beside it: the panel takes no drags."""
        panel = next((node for node in widget.ancestors_with_self if isinstance(node, HelpPanel)), None)
        if panel is not None and panel.ALLOW_SELECT:
            for node in (panel, *panel.query("*")):
                node.ALLOW_SELECT = False

    def _track_press(self, event: events.MouseEvent) -> bool:
        """Follow a press (left or middle button) to its mouse-up: whether it dragged over
        text, or was a click. False for a click on a note or the footer."""
        if event.button not in (1, 2) and not isinstance(event, events.MouseMove):
            return True
        if isinstance(event, events.MouseDown):
            widget, _offset = self.screen.get_widget_and_offset_at(event.x, event.y)
            self._press_at = event.screen_offset
            self._press_moved = self._dragged = False
            self._press_table = isinstance(widget, SelectableTable)
        elif isinstance(event, events.MouseMove):
            if self._press_at is not None and event.button in (1, 2) and self._moved_off(event.screen_offset):
                self._press_moved = True
        elif isinstance(event, events.MouseUp):
            pressed, self._press_at = self._press_at, None
            self._dragged = pressed is not None and self._press_moved
            if pressed is not None and not self._press_moved:
                if self._widget_at(event, (Footer, Toast)) is not None:  # on the footer (out of its keys) or a note
                    return False
        return True

    def _moved_off(self, at: Offset) -> bool:
        """Whether the pointer, at ``at``, has moved off the press: anywhere at all, but on
        a table only off the press's row or ``_DRAG_CELLS`` cells along it, as on a footer
        key: a hand that moves less on a row meant to click it."""
        pressed = self._press_at
        if not self._press_table:
            return at != pressed
        return at.y != pressed.y or abs(at.x - pressed.x) >= SelectableTable._DRAG_CELLS

    def press_held(self) -> bool:
        """Whether the button pressed on a table is held, and hasn't moved off the press
        (see :meth:`_moved_off`): so far a click, which highlights nothing (see
        :class:`FollowsPointer`)."""
        return self._press_at is not None and self._press_table and not self._press_moved

    def _on_blank(self, event: events.MouseEvent) -> bool:
        """Whether ``event`` (as confined) is on no text: an empty part of a list or a pane
        (not an empty line of the log, which is text with nothing on it)."""
        if self.mouse_captured:
            return False
        widget, offset = self.screen.get_widget_and_offset_at(event.x, event.y)
        return widget is not None and offset is None and widget.is_container and widget.allow_select


_INPUT_EVENTS = (events.Key, events.MouseDown, events.MouseScrollDown, events.MouseScrollUp, events.Paste)
