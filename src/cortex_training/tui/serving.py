# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""A deployment's serving numbers: the ``drain_metrics`` summary the deployment writes to the
console every 30 s (requests, latency, queue, KV cache, replicas), read from the saved console."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import json
import os
from pathlib import Path
import threading
from typing import Any
from typing import Optional

from cortex_training.tui import views
from cortex_training.tui.format import finite_number as _num
from cortex_training.tui.saved_log import committed_size

# Cheap substring checks before json.loads: most console lines are neither.
_DRAIN_MARK = '"drain_metrics"'
_STARTUP_MARK = '"job_metrics"'
# A deployment drains every 30 s. Across a longer gap the rate is unknown: a drain that failed carries
# its counts to the next, but one whose line was lost (a skipped chunk, a restart) takes its requests
# with it, and dividing what is left by the whole gap would read low.
_LONGEST_DRAIN_GAP_S = 45.0


@dataclass(frozen=True)
class Drain:
    """One 30 s drain: what completed since the previous one, and the replicas' state at its end."""

    drained_at: float
    completed: int
    latency_p50: Optional[float] = None
    latency_p99: Optional[float] = None
    ttft_p50: Optional[float] = None
    ttft_p99: Optional[float] = None
    queue_p99: Optional[float] = None
    running: Optional[float] = None
    waiting: Optional[float] = None
    kv_max: Optional[float] = None
    replicas: Optional[int] = None
    replica_errors: Optional[int] = None


@dataclass
class ServingData:
    drains: list = field(default_factory=list)  # oldest first
    startup_s: Optional[float] = None  # create-job's duration, when the console has it

    @property
    def latest(self) -> Optional[Drain]:
        return self.drains[-1] if self.drains else None


def _int(value: Any) -> Optional[int]:
    f = _num(value)
    return int(f) if f is not None else None


def _row(line: str, mark: str, msg: str) -> Optional[dict]:
    if mark not in line:
        return None
    try:
        row = json.loads(line)
    except ValueError:
        return None
    return row if isinstance(row, dict) and row.get("msg") == msg else None


def parse_line(line: str) -> Optional[Drain]:
    """The drain a console line holds; None for any other line, or a drain without its time or count."""
    row = _row(line, _DRAIN_MARK, "drain_metrics")
    if row is None:
        return None
    at, done = _num(row.get("drained_at")), _int(row.get("completed_requests"))
    if at is None or done is None:
        return None
    return Drain(
        drained_at=at, completed=done,
        latency_p50=_num(row.get("latency_s_p50")), latency_p99=_num(row.get("latency_s_p99")),
        ttft_p50=_num(row.get("ttft_s_p50")), ttft_p99=_num(row.get("ttft_s_p99")),
        queue_p99=_num(row.get("queue_s_p99")),
        running=_num(row.get("num_running_reqs_latest_sum")), waiting=_num(row.get("num_waiting_reqs_latest_sum")),
        kv_max=_num(row.get("kv_cache_usage_max")),
        replicas=_int(row.get("replica_count")), replica_errors=_int(row.get("replica_errors")),
    )


def _startup(line: str) -> Optional[float]:
    row = _row(line, _STARTUP_MARK, "job_metrics")
    if row is None or row.get("op") != "create-job" or row.get("error"):  # a failed try started nothing
        return None
    return _num(row.get("duration_s"))


class ConsoleReader:
    """A saved console (``stdout.log``) that a resumed download mostly appends to: each read parses
    the whole lines added since the last one, so while the job screen is open its console is parsed once.
    A console that was rebuilt or rewritten is read again from its start."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._offset = 0
        self._file: Optional[tuple] = None  # (st_dev, st_ino) the offset belongs to
        self._data = ServingData()
        self._lock = threading.Lock()  # a cancelled worker's read can overlap the next one's

    def read(self) -> ServingData:
        with self._lock:
            return self._read()

    def _read(self) -> ServingData:
        try:
            f = self._path.open("rb")
        except FileNotFoundError:
            return self._copy()
        with f:
            st = os.fstat(f.fileno())
            # Bytes past the resume manifest may be truncated by the next resume.
            size = committed_size(self._path, st.st_size)
            # A late chunk makes the client rebuild the console and os.replace it: a new inode.
            if size < self._offset or (st.st_dev, st.st_ino) != self._file:
                self._offset, self._data = 0, ServingData()
            self._file = (st.st_dev, st.st_ino)
            f.seek(self._offset)
            added = False
            while self._offset < size:
                raw = f.readline(size - self._offset)
                if not raw.endswith(b"\n"):
                    break  # a line cut short waits for the rest of it
                self._offset += len(raw)
                for part in raw.splitlines():
                    line = part.decode("utf-8", "replace")
                    drain = parse_line(line)
                    if drain is not None:
                        self._data.drains.append(drain)
                        added = True
                    elif self._data.startup_s is None:
                        self._data.startup_s = _startup(line)
        if added:
            self._data.drains.sort(key=lambda d: d.drained_at)  # chunks can land out of order
        return self._copy()

    def _copy(self) -> ServingData:
        return ServingData(list(self._data.drains), self._data.startup_s)


def rate(prev: Optional[Drain], cur: Drain) -> Optional[float]:
    """Requests per second over the gap since ``prev``; None for the first drain, or after a lost one."""
    if prev is None:
        return None
    gap = cur.drained_at - prev.drained_at
    return cur.completed / gap if 0 < gap <= _LONGEST_DRAIN_GAP_S else None


def serving_state(job: dict, data: ServingData, *, reading: bool = False) -> str:
    """Where a deployment stands, from its status and whether it has drained yet. ``reading``: its
    console is still being read, so no drain yet says nothing."""
    word = views.status_word(job.get("status"))
    if not views.is_live(job):
        return "dropping" if word == "cancelling" else "ended"
    if word == "unknown":
        return "serving" if data.drains else "status unknown"
    if word != "running":
        return "starting"
    if data.drains:
        return "serving"
    return "running" if reading else "starting (no metrics yet)"
