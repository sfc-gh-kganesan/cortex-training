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

"""One downloader per output directory, shared by the TUI and the CLI."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable

try:  # POSIX only; elsewhere the lock is in-process only.
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows
    fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class DownloadLock:
    """One downloader per output directory.

    Holds an in-process lock keyed on the directory plus, on POSIX, an
    advisory ``flock`` on a sibling ``.<name>.lock`` file so two TUI processes
    do not write the same files at once. If the file lock cannot be used at all
    (no ``fcntl``, or a filesystem without lock support) it degrades to the
    in-process lock.
    """

    _GUARD = threading.Lock()
    _LOCKS: dict[str, threading.Lock] = {}

    def __init__(self, root: Path):
        # Resolve first so ".", "./" or "a/.." name the same lock as the
        # absolute path the TUI uses.
        self._root = Path(root).expanduser().resolve()
        with self._GUARD:
            self._thread_lock = self._LOCKS.setdefault(str(self._root), threading.Lock())
        # The filesystem root has no sibling to hold a lock file; the TUI never
        # downloads there, so it gets the in-process lock only.
        self._lock_path = self._root.with_name(f".{self._root.name}.lock") if self._root.name else None
        self._handle = None
        self._held = False

    def acquire(
        self,
        keep_waiting: Callable[[], bool],
        *,
        on_wait: Callable[[], None] | None = None,
        poll_seconds: float = 0.1,
    ) -> bool:
        """Block until held; return False if ``keep_waiting()`` turns false first."""
        waited = False
        while True:
            if not keep_waiting():
                return False
            if self._thread_lock.acquire(timeout=poll_seconds):
                break
            if not waited and on_wait is not None:
                waited = True
                on_wait()
        while True:
            if self._try_file_lock():
                self._held = True
                return True
            if not waited and on_wait is not None:
                waited = True
                on_wait()
            if not keep_waiting():
                self._thread_lock.release()
                return False
            time.sleep(poll_seconds)

    @property
    def lock_path(self) -> Path | None:
        """The sibling lock file, or ``None`` for the filesystem root."""
        return self._lock_path

    def try_acquire(self) -> bool:
        """Take the lock only if it is free right now."""
        if not self._thread_lock.acquire(blocking=False):
            return False
        if self._try_file_lock():
            self._held = True
            return True
        self._thread_lock.release()
        return False

    def _try_file_lock(self) -> bool:
        if fcntl is None or self._lock_path is None:
            return True
        try:
            self._lock_path.parent.mkdir(parents=True, exist_ok=True)
            handle = open(self._lock_path, "a+b")
        except OSError as exc:
            logger.debug("download lock file unavailable (%s); using in-process lock", exc)
            return True
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return False
        except OSError as exc:
            handle.close()
            logger.debug("flock unsupported (%s); using in-process lock", exc)
            return True
        self._handle = handle
        return True

    def release(self) -> None:
        if not self._held:
            return
        self._held = False
        handle = self._handle
        self._handle = None
        if handle is not None:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            handle.close()
        self._thread_lock.release()
