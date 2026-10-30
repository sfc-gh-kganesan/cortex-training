# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""The Serving tab of a deployment's job screen: its state and one row per 30 s drain."""

from __future__ import annotations

from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Callable

from rich.table import Table
from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Static
from textual.worker import get_current_worker

from cortex_training.client import NoChunksError
from cortex_training.tui import serving
from cortex_training.tui import theme
from cortex_training.tui import views
from cortex_training.tui.detail_tabs import _Pane
from cortex_training.tui.format import local_time
from cortex_training.tui.format import safe_text
from cortex_training.tui.format import short_seconds
from cortex_training.tui.widgets import _warning_text

_MAX_ROWS = 240  # two hours of drains


def _seconds(value: float | None) -> str:
    return "—" if value is None else short_seconds(value)


def _count(value: float | None) -> str:
    return "—" if value is None else f"{int(value)}"


def _kv(value: float | None) -> str:
    if value is None:
        return "—"
    return "<0.1%" if 0 < value < 0.0005 else f"{value:.1%}"  # a light load sits well under 1%


def _when(drained_at: float, tz=None) -> datetime | None:
    try:
        return datetime.fromtimestamp(drained_at, tz=tz)
    except (OverflowError, OSError, ValueError):  # finite, but past what a date holds
        return None


class ServingPane(_Pane):
    BINDINGS = [Binding("r", "refresh", "Refresh")]

    def __init__(self, client, job_id: str, root: Path, *, job: Callable[[], dict],
                 download: Callable[..., list | None], id: str = "serving-tab") -> None:
        super().__init__(client, job_id, job=job, id=id)
        self._root = Path(root)
        self._download = download
        self._reader: serving.ConsoleReader | None = None
        self._data = serving.ServingData()
        self._warnings: list = []  # the last download's, shown in the header
        self._reload_pending = False  # asked for while a load ran: one more once it finishes
        self._read = False  # a load has finished: no drain since says there is none yet
        self._chunks: tuple[int, int] | None = None  # the running download's (done, total)
        self._read_status: str | None = None  # the job's status when the last finished load began

    def compose(self) -> ComposeResult:
        yield Static("", id="serving-status")
        with VerticalScroll(id="serving-pane"):
            yield Static("", id="serving-table")

    def focus_target(self):
        return self.query_one("#serving-pane")

    # ── reading ──
    def reload(self, *, live: bool = False) -> None:
        if self._loading:
            # A load already running may have downloaded before the change this one is for (the job ending, say).
            self._reload_pending = True
            return
        self._loading = True
        self.redraw()
        self._refresh()

    @work(thread=True, exclusive=True, group="serving")
    def _refresh(self) -> None:
        worker = get_current_worker()
        job = self._job()
        sid = views.deployment_sub_job_id(job)
        finished = False

        def finish(error: str | None) -> None:
            nonlocal finished
            finished = True
            self._post(self._finish, error, job.get("status"))

        try:
            if sid is None:
                finish(None)
                return
            if self._reader is None:
                self._reader = serving.ConsoleReader(self._root / sid / "stdout.log")
            self._post(self._accept, self._reader.read())  # what is cached, while the download runs
            warnings = self._download(sub_job_id=sid, cancelled=lambda: worker.is_cancelled,
                                      progress=lambda done, total: self._post(self._show_chunks, done, total))
            if warnings is None or worker.is_cancelled:
                return
            self._post(self._accept, self._reader.read(), warnings)
            finish(None)
        except NoChunksError:  # nothing uploaded yet: a deployment that just started
            finish(None)
        except Exception as exc:  # noqa: BLE001 - said in the tab's header
            if not worker.is_cancelled:
                finish(views.access_hint(exc, "the serving metrics") or views.describe_error(exc))
        finally:
            # _finish may start the queued reload, which cancels this worker: that load is not ours to end.
            if worker.is_cancelled and not finished:
                self._post(self._cancelled)

    def _accept(self, data: serving.ServingData, warnings: list | None = None) -> None:
        self._data = data
        if warnings is not None:
            self._warnings = warnings
        self.redraw()

    def _show_chunks(self, done: int, total: int) -> None:
        self._chunks = (done, total)
        self.query_one("#serving-status", Static).update(self._status_text())

    def _finish(self, error: str | None, status: str | None) -> None:
        self._loading = False
        self._chunks = None
        self._read = True
        self._read_status = status
        self._error = error  # rows already shown stay
        if self._reload_pending:
            self._reload_pending = False
            self.reload()
        else:
            self.redraw()

    @property
    def status_changed_since_read(self) -> bool:
        """The job's status changed after the last finished load began: it may have missed the job ending."""
        return self._read and self._read_status != self._job().get("status")

    def _cancelled(self) -> None:
        self._loading = False
        self._chunks = None
        self._reload_pending = False

    # ── drawing ──
    def _first_read(self) -> bool:
        return self._loading and not self._read

    def redraw(self) -> None:
        if not self.is_attached:
            return
        self.query_one("#serving-status", Static).update(self._status_text())
        self.query_one("#serving-table", Static).update(self._table())

    def _status_text(self) -> Text:
        data, latest = self._data, self._data.latest
        text = Text(serving.serving_state(self._job(), data, reading=self._first_read()), style="bold")
        if latest is not None:
            if latest.replicas is not None:
                text.append(f"  ·  {latest.replicas} replica{'' if latest.replicas == 1 else 's'}", style=theme.MUTED)
            at = _when(latest.drained_at, timezone.utc)
            when = ""
            if at is not None and views.is_live(self._job()):
                when = views.fmt_ago(at, datetime.now(timezone.utc))
            elif at is not None and local_time(at):
                # Once a deployment has ended (or been dropped) the pane soon stops reloading: an age would go stale.
                when = f"at {local_time(at)}"
            if when:
                text.append(f"  ·  last drain {when}", style=theme.MUTED)
        if data.startup_s is not None:
            text.append(f"  ·  started in {views.fmt_duration(data.startup_s)}", style=theme.MUTED)
        if self._loading and self._chunks and self._chunks[0] < self._chunks[1]:
            text.append(f"  ·  downloading chunk {self._chunks[0]:,} of {self._chunks[1]:,}", style=theme.MUTED)
        elif self._loading:
            text.append("  ·  loading…", style=theme.MUTED)
        if self._warnings:
            text.append("\n" + " ".join(safe_text(_warning_text(self._warnings)).split()), style=theme.WARN)
        if self._error:
            text.append(f"\n[error] {self._error}  (r retries)", style=theme.ERR)
        return text

    def _table(self):
        drains = self._data.drains
        if not drains and self._first_read():
            return Text("Reading the console…", style=theme.MUTED)
        if not drains:
            return Text("No drain_metrics yet: a deployment writes one every 30 s once it serves.", style=theme.MUTED)
        start = max(0, len(drains) - _MAX_ROWS)
        caption = f"newest {_MAX_ROWS} of {len(drains)} drains" if start else None
        table = Table(box=None, pad_edge=False, header_style="bold", caption=caption,
                      caption_justify="left", caption_style=theme.MUTED)
        for name in ("Time", "req/s", "Requests", "Latency p50", "p99", "TTFT p50", "p99", "Queue p99",
                     "Run/Wait", "KV max", "Replicas"):
            table.add_column(name, justify="left" if name == "Time" else "right")
        today = datetime.now().date()
        # The date too, unless every row shown is today's: an ended deployment's rows may be days old.
        fmt = "%H:%M:%S" if all(at is None or at.date() == today
                                for at in (_when(d.drained_at) for d in drains[start:])) else "%m-%d %H:%M:%S"
        for i in range(len(drains) - 1, start - 1, -1):
            d, prev = drains[i], drains[i - 1] if i else None
            r = serving.rate(prev, d)
            errors = d.replica_errors or 0
            replicas = Text(_count(d.replicas))
            if errors > 0:
                replicas.append(f" ({errors} err)", style=theme.ERR)
            at = _when(d.drained_at)
            table.add_row(
                "—" if at is None else at.strftime(fmt),
                "—" if r is None else f"{r:.2f}",
                str(d.completed),
                _seconds(d.latency_p50), _seconds(d.latency_p99),
                _seconds(d.ttft_p50), _seconds(d.ttft_p99), _seconds(d.queue_p99),
                f"{_count(d.running)}/{_count(d.waiting)}",
                _kv(d.kv_max),
                replicas,
            )
        return table
