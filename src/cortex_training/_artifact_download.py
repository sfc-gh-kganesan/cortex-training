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

"""Building blocks for downloading staged experiment artifacts.

The client downloads console and GPU metric chunks with one Snowflake ``GET``
per chunk. Each ``GET`` is mostly round-trip latency, so chunks are fetched on a
thread pool and appended to the output strictly in sorted name order. This
module holds the pieces that do not depend on the client itself: retry
classification and backoff, a reconnectable shared connection, and the
checkpointing chunk writer that keeps a resume manifest in step with the file.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import random
import re
import shutil
import tempfile
import threading
import time
import zlib
from collections.abc import Callable
from collections.abc import Iterable
from concurrent.futures import Future
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import TypeVar

import requests

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# Snowflake error numbers that mean the session or its token is gone. A fresh
# connection can succeed where retrying on the old one cannot.
_RECONNECT_ERRNOS = frozenset(
    {
        250002,  # connection is closed
        390111,  # session no longer exists
        390112,  # session expired
        390114,  # authentication token expired
        390195,  # id token invalid
    }
)
# Statement errors that are transient on the service side.
_TRANSIENT_ERRNOS = frozenset({603})  # SQL execution internal error
# File transfer errors that another attempt cannot fix. The connector raises
# these as OperationalError, which is otherwise retried.
FILE_NOT_FOUND_ERRNO = 253006  # the file was removed after it was listed
_PERMANENT_TRANSFER_ERRNOS = frozenset(
    {
        253001,  # unsupported stage storage type
        253004,  # invalid stage location
        253005,  # local path is not a directory
        FILE_NOT_FOUND_ERRNO,
        253007,  # compression not supported
        253008,  # encryption material mismatch
    }
)
_DOWNLOAD_FROM_STAGE_ERRNO = 253002
# A storage-side refusal reported through 253002; other 253002 failures are
# network errors that outlasted the connector's own retries.
_STORAGE_DENIED_RE = re.compile(
    r"\b403\b|access ?denied|forbidden|not ?authorized|authorization ?(failure|failed)",
    re.IGNORECASE,
)

RETRY = "retry"
RECONNECT = "reconnect"


class DownloadCancelled(Exception):
    """Raise from a download ``progress`` callback to stop the download.

    The download stops promptly, keeps the chunks already written so a later
    ``resume=True`` call continues from them, and re-raises this exception.
    Client telemetry records it as a cancellation rather than a failure.
    """


class NoChunksError(ValueError):
    """A sub-job named in ``sub_job_ids`` has no chunks of the requested kind.

    ``sub_job_ids`` lists the sub-jobs without chunks; ``available`` lists the
    sub-jobs that have some. It is a ``ValueError``, as before it existed.
    """

    def __init__(self, message: str, *, sub_job_ids: list[str], available: list[str]):
        super().__init__(message)
        self.sub_job_ids = sub_job_ids
        self.available = available


def is_file_not_found(exc: BaseException) -> bool:
    return getattr(exc, "errno", None) == FILE_NOT_FOUND_ERRNO


# Connector exception classes, by name. Some exist only in newer connector
# releases; a missing one is skipped rather than breaking classification.
_PERMANENT_CONNECTOR_ERRORS = ("RevocationCheckError", "NonRetryableTlsError")
_TRANSIENT_CONNECTOR_ERRORS = (
    "OperationalError",
    "InternalServerError",
    "ServiceUnavailableError",
    "GatewayTimeoutError",
    "BadGatewayError",
    "TooManyRequests",
    "RequestTimeoutError",
    "OtherHTTPRetryableError",
    "RequestExceedMaxRetryError",
)


def _connector_error_classes(names: tuple[str, ...]) -> tuple[type, ...]:
    try:
        from snowflake.connector import errors
    except ImportError:  # pragma: no cover - the connector is a dependency
        return ()
    found = (getattr(errors, name, None) for name in names)
    return tuple(cls for cls in found if isinstance(cls, type) and issubclass(cls, BaseException))


def classify_artifact_error(exc: BaseException) -> str | None:
    """Return ``RETRY``, ``RECONNECT``, or ``None`` when ``exc`` is permanent.

    Never raises: an unexpected problem while classifying counts as permanent,
    so the original error is what the caller sees.
    """
    try:
        return _classify(exc)
    except Exception:
        logger.debug("could not classify %s; treating it as permanent", type(exc), exc_info=True)
        return None


def _classify(exc: BaseException) -> str | None:
    errno = getattr(exc, "errno", None)
    if isinstance(errno, int) and errno in _RECONNECT_ERRNOS:
        return RECONNECT
    if isinstance(errno, int) and errno in _PERMANENT_TRANSFER_ERRNOS:
        return None
    if errno == _DOWNLOAD_FROM_STAGE_ERRNO:
        denied = _STORAGE_DENIED_RE.search(str(exc))
        if denied:
            logger.debug(
                "treating stage download error as permanent: message matched %r",
                denied.group(0),
            )
            return None
    permanent = _connector_error_classes(_PERMANENT_CONNECTOR_ERRORS)
    if permanent and isinstance(exc, permanent):
        return None
    transient = _connector_error_classes(_TRANSIENT_CONNECTOR_ERRORS)
    if transient and isinstance(exc, transient):
        return RETRY
    base = _connector_error_classes(("Error",))
    if base and isinstance(exc, base):
        return RETRY if isinstance(errno, int) and errno in _TRANSIENT_ERRNOS else None
    if isinstance(
        exc,
        (
            ConnectionError,
            TimeoutError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ),
    ):
        return RETRY
    return None


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded exponential backoff with jitter for artifact LIST and GET.

    ``attempts`` counts the first try. ``sleep`` is injectable for tests; by
    default the backoff waits on the cancel event so cancellation wakes it.
    """

    attempts: int = 6
    initial_delay: float = 1.0
    max_delay: float = 30.0
    sleep: Callable[[float], None] | None = None

    def delay(self, attempt: int) -> float:
        base = min(self.max_delay, self.initial_delay * (2 ** (attempt - 1)))
        return base * (0.5 + random.random() / 2)

    def pause(self, delay: float, cancel: threading.Event | None) -> bool:
        """Wait ``delay`` seconds; return True if ``cancel`` was set."""
        if self.sleep is not None:
            self.sleep(delay)
        elif cancel is not None:
            return cancel.wait(delay)
        else:
            time.sleep(delay)
        return cancel is not None and cancel.is_set()


class SessionClosed(RuntimeError):
    """The download's session was closed; a late attempt must not reopen it."""


class SharedConnection:
    """One connector connection shared by worker threads, reopened on expiry.

    The Snowflake connector allows threads to share a connection (DB-API
    ``threadsafety = 2``); each worker uses its own cursor. A replaced connection
    is closed only in :meth:`close`, because other threads may still hold
    cursors on it. ``transfer_timeout`` bounds each LIST or GET attempt made
    for this download (see :func:`run_bounded`).
    """

    def __init__(self, opener: Callable[[], Any], transfer_timeout: float | None = None):
        self.transfer_timeout = transfer_timeout
        # Counted for client telemetry: retried attempts, and attempts that ran
        # past ``transfer_timeout``.
        self.retry_count = 0
        self.transfer_timeout_count = 0
        self._opener = opener
        self._lock = threading.Lock()
        self._connection = opener()
        self._generation = 0
        self._replaced: list[Any] = []
        self._closed = False

    def note_attempt_failure(self, exc: BaseException, *, retried: bool) -> None:
        with self._lock:
            if retried:
                self.retry_count += 1
            if isinstance(exc, TransferTimeout):
                self.transfer_timeout_count += 1

    def current(self) -> tuple[Any, int]:
        with self._lock:
            if self._closed:
                raise SessionClosed("the artifact download session is closed")
            return self._connection, self._generation

    def reopen(self, generation: int) -> None:
        """Replace the connection unless another thread already did."""
        with self._lock:
            if self._closed:
                raise SessionClosed("the artifact download session is closed")
            if generation != self._generation:
                return
            fresh = self._opener()
            self._replaced.append(self._connection)
            self._connection = fresh
            self._generation += 1

    def close(self) -> None:
        with self._lock:
            self._closed = True
            replaced, self._replaced = self._replaced, []
            current = self._connection
        # A close error must not fail a finished download or hide its real error.
        for connection in [*replaced, current]:
            try:
                connection.close()
            except Exception as exc:
                # One line for the user; the traceback only at debug level.
                logger.warning("could not close the artifact download session: %s", exc)
                logger.debug("artifact download session close failure", exc_info=True)


class TransferTimeout(TimeoutError):
    """A LIST or GET attempt ran past its time limit; it is retried."""


# How long a cancelled attempt may keep running before it is abandoned. A GET
# normally finishes well inside this; waiting keeps it from refilling a temp
# directory that is about to be removed.
CANCEL_GRACE_SECONDS = 5.0


def run_bounded(
    operation: Callable[[], _T],
    *,
    timeout: float | None,
    cancel: threading.Event | None,
    description: str,
) -> _T:
    """Run ``operation`` on a helper thread and wait at most ``timeout`` seconds.

    The connector's own timeouts do not bound a stalled transfer tightly (a
    storage read may wait many minutes per request, and it retries), so an
    attempt that runs too long raises :class:`TransferTimeout` and is
    abandoned: its daemon thread keeps nothing the download still uses, since
    every attempt writes into its own temp directory. After ``cancel`` is set
    the attempt gets :data:`CANCEL_GRACE_SECONDS` to finish, then
    :class:`DownloadCancelled` is raised.
    """
    if timeout is None and cancel is None:
        return operation()
    outcome: dict[str, Any] = {}
    finished = threading.Event()

    def run() -> None:
        try:
            outcome["value"] = operation()
        except BaseException as exc:  # handed to the waiting thread
            outcome["error"] = exc
        finally:
            finished.set()

    threading.Thread(target=run, name="cortex-training-transfer", daemon=True).start()
    deadline = None if timeout is None else time.monotonic() + timeout
    cancelled_at: float | None = None
    while not finished.wait(0.2):
        now = time.monotonic()
        if cancel is not None and cancel.is_set():
            if cancelled_at is None:
                cancelled_at = now
            elif now - cancelled_at >= CANCEL_GRACE_SECONDS:
                raise DownloadCancelled(f"{description} cancelled")
        elif deadline is not None and now >= deadline:
            raise TransferTimeout(
                f"{description} did not finish within {timeout:g}s. On a slow "
                "connection, allow more time with --transfer-timeout "
                "(transfer_timeout_seconds) or fetch fewer chunks at once with "
                "--parallel (max_workers)."
            )
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def call_with_retry(
    shared: SharedConnection,
    policy: RetryPolicy,
    description: str,
    operation: Callable[[Any], _T],
    cancel: threading.Event | None = None,
) -> _T:
    """Run ``operation(connection)``, retrying transient failures.

    Once ``cancel`` is set no new attempt starts and a backoff wait ends early;
    the call then raises :class:`DownloadCancelled`.
    """
    attempt = 1
    while True:
        if cancel is not None and cancel.is_set():
            raise DownloadCancelled(f"{description} cancelled")
        connection, generation = shared.current()
        try:
            return run_bounded(
                lambda: operation(connection),
                timeout=shared.transfer_timeout,
                cancel=cancel,
                description=description,
            )
        except DownloadCancelled:
            raise
        except Exception as exc:
            if cancel is not None and cancel.is_set():
                # Failures after cancellation (for example, the temp directory
                # being removed under an in-flight GET) are not worth reporting.
                raise DownloadCancelled(f"{description} cancelled") from exc
            kind = classify_artifact_error(exc)
            retried = kind is not None and attempt < policy.attempts
            shared.note_attempt_failure(exc, retried=retried)
            if not retried:
                raise
            delay = policy.delay(attempt)
            logger.warning(
                "%s failed (%s: %s); retrying in %.1fs (attempt %d of %d)",
                description,
                type(exc).__name__,
                exc,
                delay,
                attempt + 1,
                policy.attempts,
            )
            if policy.pause(delay, cancel):
                raise DownloadCancelled(f"{description} cancelled") from exc
            attempt += 1
            if kind == RECONNECT:
                try:
                    shared.reopen(generation)
                except Exception as reopen_exc:
                    if classify_artifact_error(reopen_exc) is None or attempt >= policy.attempts:
                        raise
                    logger.warning(
                        "reopening the artifact connection failed (%s); retrying",
                        type(reopen_exc).__name__,
                    )


@dataclass(frozen=True)
class Manifest:
    """A resume manifest: chunks accounted for, in order, and committed bytes.

    ``skipped`` lists the chunks in ``names`` that were unreadable and left out
    of the file; resume does not fetch them again.
    """

    names: list[str]
    nbytes: int
    skipped: list[str]


def read_manifest(path: Path) -> Manifest | None:
    """Return the resume manifest at ``path``, if valid."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or doc.get("version") != 1:
        return None
    names = doc.get("names")
    nbytes = doc.get("bytes")
    if (
        not isinstance(names, list)
        or not names
        or not all(isinstance(name, str) and name for name in names)
        or isinstance(nbytes, bool)
        or not isinstance(nbytes, int)
        or nbytes < 0
    ):
        return None
    skipped = doc.get("skipped", [])
    if (
        not isinstance(skipped, list)
        or not all(isinstance(name, str) for name in skipped)
        or not set(skipped) <= set(names)
    ):
        return None
    return Manifest(list(names), nbytes, list(skipped))


def write_manifest(
    path: Path, names: list[str], nbytes: int, skipped: list[str] | None = None
) -> None:
    """Atomically replace the manifest at ``path``.

    ``skipped`` is written only when non-empty, so a manifest without skipped
    chunks keeps the original format.
    """
    doc: dict[str, object] = {"bytes": nbytes, "names": names, "version": 1}
    if skipped:
        doc["skipped"] = skipped
    payload = json.dumps(doc) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name.removesuffix(".json") + "-", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# Raised while decompressing a damaged or truncated gzip member.
CORRUPT_CHUNK_ERRORS = (gzip.BadGzipFile, EOFError, zlib.error)


def copy_gzip_member(downloaded: Path, output, buffer_bytes: int) -> None:
    with downloaded.open("rb") as compressed:
        with gzip.GzipFile(fileobj=compressed, mode="rb") as source:
            shutil.copyfileobj(source, output, length=buffer_bytes)


class ChunkWriter:
    """Append decompressed chunks to ``data_path`` and checkpoint a manifest.

    The manifest only ever records bytes that were fsynced first, so after a
    crash the file may be longer than the manifest says but never shorter;
    resume truncates anything past the recorded length. A chunk that fails to
    decompress is rolled back, so the file always ends on a chunk boundary.
    :meth:`abort` records every chunk that was fully appended.
    """

    def __init__(
        self,
        data_path: Path,
        manifest_path: Path,
        names: list[str],
        committed: int,
        *,
        buffer_bytes: int,
        checkpoint_chunks: int,
        checkpoint_seconds: float,
        skipped: list[str] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.data_path = data_path
        self.manifest_path = manifest_path
        self.names = list(names)
        self.skipped = list(skipped or [])
        self.offset = committed
        self._recorded = len(self.names)
        self._buffer_bytes = buffer_bytes
        self._checkpoint_chunks = max(1, checkpoint_chunks)
        self._checkpoint_seconds = checkpoint_seconds
        self._clock = clock
        self._last_checkpoint = clock()
        data_path.parent.mkdir(parents=True, exist_ok=True)
        # Owner-only, like the temp files earlier versions renamed into place.
        fd = os.open(data_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o600)
        self._output = os.fdopen(fd, "r+b")
        try:
            self._output.seek(committed)
            self._output.truncate(committed)
        except BaseException:
            self._output.close()
            raise

    def append(self, name: str, downloaded: Path) -> None:
        try:
            copy_gzip_member(downloaded, self._output, self._buffer_bytes)
        except BaseException:
            self._output.truncate(self.offset)
            self._output.seek(self.offset)
            raise
        self.offset = self._output.tell()
        self._advance(name)

    def skip(self, name: str) -> None:
        """Account for an unreadable chunk without writing it."""
        self.skipped.append(name)
        self._advance(name)

    def _advance(self, name: str) -> None:
        self.names.append(name)
        # Each checkpoint rewrites every name, so the spacing grows with the
        # list to keep total manifest writes linear in the chunk count: at
        # most a tenth of the chunks are refetched after a hard kill.
        pending = len(self.names) - self._recorded
        if pending >= max(self._checkpoint_chunks, len(self.names) // 10) or (
            pending >= len(self.names) // 50
            and self._clock() - self._last_checkpoint >= self._checkpoint_seconds
        ):
            self.checkpoint()

    def checkpoint(self) -> None:
        self._output.flush()
        os.fsync(self._output.fileno())
        if self.names and len(self.names) != self._recorded:
            write_manifest(self.manifest_path, self.names, self.offset, self.skipped)
            self._recorded = len(self.names)
        self._last_checkpoint = self._clock()

    def finish(self) -> None:
        """Make every appended chunk durable and close the file."""
        try:
            self._output.truncate(self.offset)
            self.checkpoint()
        finally:
            self._output.close()

    def abort(self) -> None:
        """Keep fully appended chunks, drop a torn one, and close the file."""
        try:
            self._output.truncate(self.offset)
            self.checkpoint()
        finally:
            self._output.close()
        if not self.names and self.offset == 0:
            self.data_path.unlink(missing_ok=True)


@dataclass(frozen=True)
class GzipPlan:
    """How one sub-job's output file is brought up to date.

    ``data_path`` is the destination itself when appending to a completed
    download, or a hidden ``.partial`` file that replaces the destination once
    every chunk is written (``promote``).
    """

    sub_job_id: str
    destination: Path
    data_path: Path
    manifest_path: Path
    chunk_paths: list[str]
    known: list[str]
    committed: int
    promote: bool
    skipped: list[str] = field(default_factory=list)

    @property
    def new_paths(self) -> list[str]:
        return self.chunk_paths[len(self.known) :]


class Background:
    """Run ``operation`` on a daemon thread; :meth:`result` waits a bounded time."""

    def __init__(self, operation: Callable[[], _T], *, name: str):
        self._outcome: list[Any] = []
        self._done = threading.Event()

        def run() -> None:
            try:
                self._outcome.append(operation())
            except Exception:
                logger.debug("%s failed", name, exc_info=True)
            finally:
                self._done.set()

        threading.Thread(target=run, name=name, daemon=True).start()

    def result(self, timeout: float, default: _T) -> _T:
        if self._done.wait(timeout) and self._outcome:
            return self._outcome[0]
        return default


class Spool:
    """A private temp directory for fetched chunks.

    After a cancel, GETs that were already running are abandoned rather than
    awaited, so :meth:`remove_after` defers removal until they finish; removing
    the directory under them would let the transfer recreate and fill it.
    """

    def __init__(self, prefix: str):
        self.path = Path(tempfile.mkdtemp(prefix=prefix))
        self._deferred = False

    def __enter__(self) -> Spool:
        return self

    def __exit__(self, *_exc: object) -> None:
        if not self._deferred:
            self.remove()

    def remove(self) -> None:
        shutil.rmtree(self.path, ignore_errors=True)

    def remove_after(self, futures: Iterable[Future[Any]]) -> None:
        """Remove the directory once every future in ``futures`` is done."""
        self._deferred = True
        remaining = [future for future in futures if not future.done()]
        if not remaining:
            self.remove()
            return
        lock = threading.Lock()
        count = [len(remaining)]

        def finished(_future: Future[Any]) -> None:
            with lock:
                count[0] -= 1
                last = count[0] == 0
            if last:
                self.remove()

        for future in remaining:
            future.add_done_callback(finished)

    @staticmethod
    def sweep(prefix: str, min_age_seconds: float) -> None:
        """Remove spool directories a killed process left in the temp directory."""
        cutoff = time.time() - min_age_seconds
        for path in Path(tempfile.gettempdir()).glob(prefix + "*"):
            try:
                if not path.is_dir() or path.is_symlink() or path.stat().st_mtime >= cutoff:
                    continue
            except OSError:
                continue
            shutil.rmtree(path, ignore_errors=True)
            if path.exists():
                logger.debug("could not remove leftover download directory %s", path)
            else:
                logger.info("removed leftover download directory %s", path)
