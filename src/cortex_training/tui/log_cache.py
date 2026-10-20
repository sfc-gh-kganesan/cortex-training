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

"""Client-side persistent cache for the Cortex Training log TUI.

Reopening a job should not re-retrieve logs already fetched — that is wasted
load on the (deliberately stateless) Zone Manager. The transport is already
cursor-based (``tail_logs(cursor) -> {entries, next_cursor, eof}``), so we cache,
per ``(job_id, source_id)`` on the *client*:

  * the opaque ``next_cursor`` — lets the next open resume with a delta-only fetch
  * the retrieved entries (append-only JSONL) — lets the next open repaint instantly

On reopen the TUI replays the local cache, then resumes ``tail_logs`` from the
saved cursor. Nothing is added to the server.

Pure stdlib, NO ``textual`` import, so it is unit-testable like ``format.py``.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from collections import OrderedDict
from collections import deque
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Iterator
from typing import Optional
from typing import Tuple

# Cache dir resolution mirrors cortex_training._cli._login_state_path: an explicit
# override wins, else XDG_CACHE_HOME, else ~/.cache, namespaced under the app.
_OVERRIDE_ENV = "CORTEX_TRAINING_TUI_CACHE_DIR"
_XDG_CACHE_ENV = "XDG_CACHE_HOME"
_NAMESPACE = "cortex-training"

# Backoff defaults, matching CortexTrainingClient.poll_backoff_multiplier /
# poll_max_interval. Kept module-level so a test/mock client need not carry them.
_POLL_BACKOFF_MULTIPLIER = 1.25
_POLL_MAX_INTERVAL = 6.0
_HOLD_CHECK_SECONDS = 0.25  # how soon a held tail notices it may poll again

# A stuck tail: on some service versions a log line of 256 KiB or more is never
# delivered, and every later page comes back empty with the same cursor while the
# job keeps logging; cached_pages restarts such a tail. The window shrinks while
# reads of the end bring nothing new: a quiet log then costs only a small read,
# and a window that still reaches back to the oversized line soon stops reaching it.
STALL_SECONDS = 45.0
STALL_POLLS = 3
RESTART_LINES = (200, 50, 10)
_RECENT_KEYS = 5000  # entries remembered so a restart skips lines already shown; far more than one window

# A page this full says the log runs ahead of the tail (the service's pages
# currently hold at most 2,000 entries and 256 KiB): the next read goes out
# after CATCH_UP_SECONDS instead of the poll interval, until a page isn't.
CATCH_UP_ENTRIES = 1000
CATCH_UP_BYTES = 128 * 1024
CATCH_UP_SECONDS = 0.2

_ERROR_LOG = os.path.expanduser("~/.cortex-training-errors.log")
_persist_warned = False

# Filename-safe byte set; everything else (incl. ':' '/' '%') is %XX-escaped so
# a source_id collapses to a single, reversible path component.
_SAFE = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.")
_MAX_NAME = 200


def cache_root() -> Path:
    """Base cache directory (not created here; writers mkdir lazily)."""
    override = os.environ.get(_OVERRIDE_ENV)
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get(_XDG_CACHE_ENV)
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / _NAMESPACE


def job_cache_dir(job_id: str, *, root: Optional[Path] = None) -> Path:
    """Return the shared per-job cache directory without creating it."""
    base = root if root is not None else cache_root()
    return base / "jobs" / safe_source_name(job_id)


def safe_source_name(source_id: str) -> str:
    """Encode an arbitrary source_id into one filesystem-safe path component.

    Reversible via :func:`unsafe_source_name` for ids under ``_MAX_NAME`` bytes;
    longer ids get a sha1 suffix and are not reversible (correctness comes from
    the cursors.json id->cursor map, not the filename).
    """
    out = []
    for b in source_id.encode("utf-8"):
        c = chr(b)
        out.append(c if c in _SAFE else "%%%02X" % b)
    name = "".join(out)
    if name in (".", ".."):
        name = name.replace(".", "%2E")
    if len(name) > _MAX_NAME:
        import hashlib

        digest = hashlib.sha1(source_id.encode("utf-8")).hexdigest()[:12]
        name = name[: _MAX_NAME - 13] + "-" + digest
    return name


def unsafe_source_name(name: str) -> str:
    """Inverse of :func:`safe_source_name` (for tests / introspection)."""
    raw = bytearray()
    i = 0
    while i < len(name):
        ch = name[i]
        if ch == "%" and i + 2 < len(name) + 1 and i + 3 <= len(name):
            raw.append(int(name[i + 1 : i + 3], 16))
            i += 3
        else:
            raw.append(ord(ch))
            i += 1
    return raw.decode("utf-8")


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text via same-dir temp + fsync + os.replace (atomic on one fs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".swp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class LogCache:
    """Per-job on-disk cache of log entries + resume cursors.

    Layout::

        <cache_root>/jobs/<safe(job_id)>/cursors.json
        <cache_root>/jobs/<safe(job_id)>/logs/<safe(source_id)>.jsonl
    """

    DEFAULT_MAX_BYTES = 50 * 1024 * 1024
    DEFAULT_MAX_LINES = 100_000
    # A source's file past max_bytes is cut to this share of it. Cut to just under
    # max_bytes, the next append took it over again, and every append rewrote (and
    # fsynced) the whole file; this leaves room for a fifth of the budget first.
    TRIM_TO = 0.8

    def __init__(
        self,
        job_id: str,
        *,
        root: Optional[Path] = None,
        max_bytes: int = DEFAULT_MAX_BYTES,
        max_lines: int = DEFAULT_MAX_LINES,
    ) -> None:
        self._dir = job_cache_dir(job_id, root=root)
        self._max_bytes = max_bytes
        self._max_lines = max_lines

    # ---- paths ----
    def _logs_dir(self) -> Path:
        return self._dir / "logs"

    def _jsonl_path(self, source_id: str) -> Path:
        return self._logs_dir() / (safe_source_name(source_id) + ".jsonl")

    def _cursors_path(self) -> Path:
        return self._dir / "cursors.json"

    # ---- cursors ----
    def _read_cursors(self) -> dict:
        try:
            doc = json.loads(self._cursors_path().read_text(encoding="utf-8"))
            if isinstance(doc, dict):
                return doc
        except (OSError, ValueError):
            pass
        return {"version": 1, "sources": {}}

    def get_cursor(self, source_id: str) -> Optional[str]:
        entry = self._read_cursors().get("sources", {}).get(source_id)
        if isinstance(entry, dict):
            cur = entry.get("cursor")
            return cur if isinstance(cur, str) else None
        return None

    def set_cursor(self, source_id: str, cursor: Optional[str], **fields: Any) -> None:
        doc = self._read_cursors()
        sources = doc.setdefault("sources", {})
        old = sources.get(source_id)  # its other fields (head_missing) stay
        sources[source_id] = {
            **(old if isinstance(old, dict) else {}), **fields,
            "cursor": cursor,
            "updated": datetime.now(timezone.utc).isoformat(),
        }
        doc["version"] = 1
        _atomic_write_text(self._cursors_path(), json.dumps(doc, sort_keys=True) + "\n")

    def head_missing(self, source_id: str) -> bool:
        """Whether this cache lacks the start of the source's log: its first
        fetch read only the log's last lines, or old lines were rotated out."""
        entry = self._read_cursors().get("sources", {}).get(source_id)
        return isinstance(entry, dict) and entry.get("head_missing") is True

    def set_head_missing(self, source_id: str) -> None:
        if not self.head_missing(source_id):
            self.set_cursor(source_id, self.get_cursor(source_id), head_missing=True)

    # ---- entries ----
    def append_entries(self, source_id: str, entries: Iterable[Any]) -> int:
        path = self._jsonl_path(source_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with open(path, "a", encoding="utf-8") as f:
            for e in entries:
                try:
                    line = json.dumps(e, separators=(",", ":"))
                except (TypeError, ValueError):
                    line = json.dumps({"_raw": str(e)}, separators=(",", ":"))
                f.write(line + "\n")
                n += 1
            f.flush()
        if n:
            self._rotate_if_needed(source_id)
        return n

    def load_entries(self, source_id: str, *, limit: Optional[int] = None) -> list:
        return self.load_newest(source_id, limit=limit)[0]

    def load_newest(self, source_id: str, *, limit: Optional[int] = None) -> Tuple[list, int]:
        """The newest ``limit`` cached entries (all without a limit), oldest first, and how
        many the cache holds. A torn or corrupt line is skipped and not counted; bytes that
        aren't UTF-8 (the file was damaged outside the client) read as U+FFFD."""
        path = self._jsonl_path(source_id)
        if not path.exists():
            return [], 0
        sink: Any = deque(maxlen=limit) if limit else []
        total = 0
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if limit:
                        # Only the newest ``limit`` are kept: parse them alone, at the end.
                        sink.append(line)
                        total += 1
                        continue
                    try:
                        sink.append(json.loads(line))
                        total += 1
                    except ValueError:
                        continue  # torn last line / corruption: skip
        except OSError:
            return [], 0
        if not limit:
            return list(sink), total
        out = []
        for line in sink:
            try:
                out.append(json.loads(line))
            except ValueError:
                total -= 1  # torn last line / corruption: skip
        return out, total

    def cached_sources(self) -> list:
        """Source ids that have cached log data on disk for this job. Lets the
        TUI show previously-viewed sources (and replay them) even when a live
        ``list_log_sources`` call fails."""
        d = self._logs_dir()
        if not d.exists():
            return []
        out = []
        for p in sorted(d.glob("*.jsonl")):
            try:
                out.append(unsafe_source_name(p.stem))
            except Exception:  # noqa: BLE001 - skip an undecodable (hashed) name
                continue
        return out

    # ---- maintenance ----
    def _rotate_if_needed(self, source_id: str) -> None:
        path = self._jsonl_path(source_id)
        try:
            if path.stat().st_size <= self._max_bytes:
                return
        except OSError:
            return
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except OSError:
            return
        # Keep the newest max_lines, then drop from the front until within bytes.
        kept = deque(lines, maxlen=self._max_lines)
        budget = int(self._max_bytes * self.TRIM_TO)
        total = sum(len(s.encode("utf-8")) for s in kept)
        while kept and total > budget:
            total -= len(kept[0].encode("utf-8"))
            kept.popleft()
        # The cursor is intentionally NOT touched: history is bounded but the
        # resume position is preserved, so reopen still delta-fetches correctly.
        _atomic_write_text(path, "".join(kept))
        if len(kept) < len(lines):
            try:
                self.set_head_missing(source_id)  # the oldest lines are gone
            except OSError:
                pass


# Saved consoles the TUI keeps for finished jobs (jobs/<job>/console/), all jobs
# together: past this, the ones used least recently are removed.
CONSOLE_CACHE_MAX_BYTES = 1024 ** 3
CONSOLE_MARKER = ".complete"  # written last into a kept console; its mtime is when it was last used


def _tree_bytes(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file():
                total += child.stat().st_size
        except OSError:
            continue
    return total


def touch_console(console_dir: Path) -> None:
    """Mark a kept console as just used (it is the last to be removed)."""
    try:
        os.utime(console_dir / CONSOLE_MARKER)
    except OSError:
        pass


def prune_consoles(*, root: Optional[Path] = None, keep: Optional[Path] = None,
                   max_bytes: Optional[int] = None) -> list:
    """Remove kept consoles, least recently used first, until the rest fit in
    ``max_bytes`` (default CONSOLE_CACHE_MAX_BYTES); never ``keep`` (the one
    just kept), nor one without its marker (still being written). Returns the
    removed directories."""
    max_bytes = CONSOLE_CACHE_MAX_BYTES if max_bytes is None else max_bytes
    jobs = (root if root is not None else cache_root()) / "jobs"
    kept = []
    for console in jobs.glob("*/console"):
        try:
            used = (console / CONSOLE_MARKER).stat().st_mtime
        except OSError:
            continue
        kept.append((used, _tree_bytes(console), console))
    total = sum(size for _, size, _ in kept)
    removed = []
    keep_path = keep.resolve() if keep is not None else None
    for _, size, console in sorted(kept, key=lambda item: item[0]):
        if total <= max_bytes:
            break
        if keep_path is not None and console.resolve() == keep_path:
            continue
        shutil.rmtree(console, ignore_errors=True)
        total -= size
        removed.append(console)
    return removed


def _note_persist_error(exc: BaseException) -> None:
    global _persist_warned
    if _persist_warned:
        return
    _persist_warned = True
    try:
        with open(_ERROR_LOG, "a", encoding="utf-8") as f:
            f.write("log cache persistence disabled (%s: %s)\n" % (type(exc).__name__, exc))
    except OSError:
        pass


def _safe_append(cache: LogCache, source_id: str, entries: list) -> None:
    try:
        cache.append_entries(source_id, entries)
    except OSError as exc:  # disk full / unwritable: degrade to live-only
        _note_persist_error(exc)


def _safe_set_cursor(cache: LogCache, source_id: str, cursor: Optional[str]) -> None:
    try:
        cache.set_cursor(source_id, cursor)
    except OSError as exc:
        _note_persist_error(exc)


def catching_up(entries: list) -> bool:
    """Whether a page is full enough that the log is likely ahead of the tail."""
    if len(entries) >= CATCH_UP_ENTRIES:
        return True
    size = 0
    for entry in entries:
        raw = entry.get("_raw") if isinstance(entry, dict) else None
        size += len(raw) if isinstance(raw, str) else 0
    return size >= CATCH_UP_BYTES


def _refuses_cursor(exc: BaseException) -> bool:
    """A 400 for a read with a cursor that isn't about the sub-job (``does not
    belong to session``, while it starts): the service doesn't take the cursor."""
    response = getattr(exc, "response", None)
    if getattr(response, "status_code", None) != 400:
        return False
    try:
        body = response.json()
        message = str(body.get("message") or "") if isinstance(body, dict) else ""
    except Exception:  # noqa: BLE001 - not JSON
        message = str(getattr(response, "text", "") or "")
    return "does not belong to session" not in message


def entry_key(entry: Any) -> str:
    """Identity of a log entry for dedupe.

    A live line is ``{"_raw": "<RFC3339 nanosecond timestamp> <line>"}``, so its
    text is unique; other shapes compare by their canonical JSON.
    """
    if isinstance(entry, dict) and isinstance(entry.get("_raw"), str):
        return entry["_raw"]
    try:
        return json.dumps(entry, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return repr(entry)


class _RecentKeys:
    """The keys of the last ``limit`` entries yielded, oldest dropped first."""

    def __init__(self, limit: int) -> None:
        self._keys: "OrderedDict[str, None]" = OrderedDict()
        self._limit = limit

    def add_all(self, entries: Iterable[Any]) -> None:
        for entry in entries:
            key = entry_key(entry)
            self._keys[key] = None
            self._keys.move_to_end(key)
        while len(self._keys) > self._limit:
            self._keys.popitem(last=False)

    def unseen(self, entries: Iterable[Any]) -> list:
        """The entries not yielded yet, in order and each once. Nothing is remembered."""
        out: list = []
        keys: set = set()
        for entry in entries:
            key = entry_key(entry)
            if key not in self._keys and key not in keys:
                keys.add(key)
                out.append(entry)
        return out


def cached_pages(
    cache: LogCache,
    source_id: str,
    fetch: Callable[[Optional[str]], dict],
    *,
    entries_key: str,
    follow: bool,
    poll_interval: float,
    live: bool = True,
    sleep: Callable[[float], None] = time.sleep,
    replay_limit: Optional[int] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
    hold: Optional[Callable[[], bool]] = None,
    backoff_multiplier: float = _POLL_BACKOFF_MULTIPLIER,
    max_interval: float = _POLL_MAX_INTERVAL,
    restart: Optional[Callable[[int], dict]] = None,
    can_restart: Optional[Callable[[], bool]] = None,
    clock: Callable[[], float] = time.monotonic,
    tail_window: Optional[int] = None,
) -> Iterator[Tuple[str, list]]:
    """Replay cached entries and, when live, resume ``fetch`` from its cursor.

    Batches are ``(kind, entries)``: ``cache`` or ``live`` normally; ``partial``
    when retained history has no beginning; ``reset`` after a refused cursor;
    ``caught-up`` after full pages; and ``restart`` when an empty, unmoving tail
    is recovered from the log's end. Entries are persisted before their cursor,
    so a crash may repeat but never skips them.

    ``live=False`` never fetches. ``hold`` pauses without advancing the stall
    clock. ``tail_window`` limits a cursorless first fetch. With ``restart``,
    ``STALL_SECONDS`` and ``STALL_POLLS`` trigger progressively smaller
    ``RESTART_LINES`` windows while ``can_restart`` remains true.
    """
    cancelled = is_cancelled or (lambda: False)
    held = hold or (lambda: False)
    restarting = restart is not None and live and follow
    seen = _RecentKeys(_RECENT_KEYS) if restarting else None

    cached, total = cache.load_newest(source_id, limit=replay_limit)
    cursor = cache.get_cursor(source_id)
    partial = total > len(cached) or (bool(cached) and cache.head_missing(source_id))
    if partial and not cancelled():
        yield ("partial", [])
    if cached and not cancelled():
        if seen is not None:
            seen.add_all(cached)
        yield ("cache", cached)

    if not live:
        return  # cache-only (terminal job): never touch the operation API

    delay = poll_interval
    max_interval = max(max_interval, poll_interval)
    counted: Optional[int] = None  # entries since a fetch without a cursor, until the tail catches up
    reset_cursor: Optional[str] = None  # the cursor a refused one was replaced with (refused too: give up)
    behind = False  # the last page was full: the log runs ahead of the tail
    still_since = clock()  # when the tail last moved (entries, or a new cursor)
    empty_polls = 0
    window = 0  # index into RESTART_LINES
    while True:
        if cancelled():
            return
        if held():
            sleep(_HOLD_CHECK_SECONDS)
            delay = poll_interval  # resume promptly
            still_since, empty_polls = clock(), 0
            continue
        page = None
        if seen is not None:
            if not cursor or (can_restart is not None and not can_restart()):
                # Not a stall: without a cursor each fetch tails the log's end,
                # and a job that isn't running yet isn't stuck behind a line.
                still_since, empty_polls = clock(), 0
            elif empty_polls >= STALL_POLLS and clock() - still_since >= STALL_SECONDS:
                still_since, empty_polls = clock(), 0
                end = restart(RESTART_LINES[window])
                fresh = seen.unseen(end.get(entries_key) or [])
                if not fresh:
                    # The log's newest lines were shown already: it is only quiet.
                    window = min(window + 1, len(RESTART_LINES) - 1)
                    sleep(delay)
                    continue
                page = fetch(cursor)
                if not page.get(entries_key) and page.get("next_cursor", cursor) == cursor:
                    window = 0
                    # An empty cursor would make the next poll re-read the log's tail.
                    cursor = end.get("next_cursor") or cursor
                    seen.add_all(fresh)
                    _safe_append(cache, source_id, fresh)
                    _safe_set_cursor(cache, source_id, cursor)
                    if cancelled():
                        return
                    yield ("restart", fresh)
                    delay = poll_interval
                    sleep(delay)
                    continue
                # It moved: the log was only quiet. That page replaces the restart below.
        if page is None:
            if tail_window and not cursor and counted is None:
                counted = 0  # this fetch reads only the log's last ``tail_window`` lines
            try:
                page = fetch(cursor)
            except Exception as exc:
                if not cursor or cursor == reset_cursor or not _refuses_cursor(exc):
                    raise
                # Drop it (here and in the cache, so r and the next open don't
                # send it again), and go on from the log's end.
                _safe_set_cursor(cache, source_id, None)
                end = fetch(None)
                fresh = end.get(entries_key) or []
                if seen is not None:
                    fresh = seen.unseen(fresh)
                    seen.add_all(fresh)
                cursor = reset_cursor = end.get("next_cursor") or None
                if fresh:
                    _safe_append(cache, source_id, fresh)
                _safe_set_cursor(cache, source_id, cursor)
                still_since, empty_polls, window = clock(), 0, 0
                if cancelled():
                    return
                yield ("reset", fresh)
                delay = poll_interval
                sleep(delay)
                continue
        new_cursor = page.get("next_cursor", cursor)
        entries = page.get(entries_key) or []
        if counted is not None:
            counted += len(entries)
            if counted >= (tail_window or 0):
                counted = None
                if not partial:
                    partial = True
                    try:
                        cache.set_head_missing(source_id)
                    except OSError as exc:
                        _note_persist_error(exc)
                    if cancelled():
                        return
                    yield ("partial", [])
            elif not catching_up(entries):
                counted = None  # the window held the whole log, from its first line
        if entries:
            _safe_append(cache, source_id, entries)  # entries first...
            _safe_set_cursor(cache, source_id, new_cursor)  # ...then cursor
            cursor = new_cursor
            if seen is not None:
                seen.add_all(entries)
            still_since, empty_polls, window = clock(), 0, 0
            if cancelled():
                return
            yield ("live", entries)  # whole page in one batch
            # Throttle floor: even when the source is busy, wait at least
            # poll_interval before the next request. Biases toward server
            # reliability over log freshness — caps the rate at ~1 req/s per
            # source per viewer regardless of how chatty the source is.
            # A full page means the log runs ahead of the tail: catch up at
            # a few reads a second, so the view doesn't fall further behind.
            delay = poll_interval
            behind = catching_up(entries)
            sleep(min(delay, CATCH_UP_SECONDS) if behind else delay)
            continue
        if behind:
            behind = False
            if cancelled():
                return
            yield ("caught-up", [])  # the read after full pages brought nothing: the tail is at the log's end
        if new_cursor != cursor:
            _safe_set_cursor(cache, source_id, new_cursor)
            cursor = new_cursor
            still_since, empty_polls, window = clock(), 0, 0
        else:
            empty_polls += 1
        if not follow:
            # An empty page means we've drained everything available right now.
            # Stop rather than poll forever, without relying on the server's
            # eof flag, which older deployments never set. This is the
            # "terminal job: replay + drain once, then stop" path.
            return
        sleep(delay)
        delay = min(delay * backoff_multiplier, max_interval)


def cached_log_pages(
    cache: LogCache,
    client: Any,
    job_id: str,
    sub_job_id: str,
    *,
    follow: bool,
    poll_interval: float,
    live: bool = True,
    replay_limit: Optional[int] = 2000,
    tail_lines: Optional[int] = 2000,
    is_cancelled: Optional[Callable[[], bool]] = None,
    hold: Optional[Callable[[], bool]] = None,
    can_restart: Optional[Callable[[], bool]] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Iterator[Tuple[str, list]]:
    """``cached_pages`` over ``client.tail_logs``, keyed by ``sub_job_id``.

    A cursorless first open requests only ``tail_lines``; reopens resume
    delta-only. ``can_restart`` enables stuck-tail recovery.
    """

    def fetch(cursor: Optional[str]) -> dict:
        kw: dict = {"cursor": cursor, "sub_job_id": sub_job_id}
        if not cursor and tail_lines:
            kw["max_lines"] = tail_lines  # no cursor yet: tail recent, not from start
        return client.tail_logs(job_id, **kw)

    def restart(lines: int) -> dict:
        return client.tail_logs(job_id, cursor=None, max_lines=lines, sub_job_id=sub_job_id)

    return cached_pages(
        cache,
        sub_job_id,
        fetch,
        entries_key="entries",
        follow=follow,
        poll_interval=poll_interval,
        live=live,
        replay_limit=replay_limit,
        is_cancelled=is_cancelled,
        hold=hold,
        restart=restart if can_restart is not None else None,
        can_restart=can_restart,
        tail_window=tail_lines,
        sleep=sleep,
    )
