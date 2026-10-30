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

"""Bounds and stale-artifact cleanup for the TUI's local job cache.

Each opened job caches its log tail (``logs/``), finished console (``console/``),
and telemetry (``metrics/``) under ``<cache_root>/jobs/<job>/``. :func:`enforce_cache_limit`
evicts least-recently-used jobs past ``DEFAULT_CACHE_LIMIT`` while skipping non-cache
entries, symlinks, leased open jobs (:class:`JobLease`), and active downloads
(:class:`DownloadLock`). The client's own training rows (``local-training/``) count toward
the limit but are never removed by it. :func:`sweep_stale` removes leftovers from
interrupted TUIs.
"""

from __future__ import annotations

import atexit
import contextlib
import heapq
import os
import re
import shutil
import socket
import stat
import tempfile
import threading
import time
import uuid
import weakref
from pathlib import Path
from typing import Optional

try:  # POSIX only; elsewhere a lease is its marker file alone, and stages aren't swept.
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows
    fcntl = None  # type: ignore[assignment]

from cortex_training import _training_log
from cortex_training._download_lock import DownloadLock
from cortex_training.tui.log_cache import cache_root
from cortex_training.tui.log_cache import unsafe_source_name

DEFAULT_CACHE_LIMIT = 5 * 1024 ** 3
# How often an open job screen refreshes its marker, and how old a marker may be before
# it stops protecting its job from a TUI on another machine (one that can't see the lock).
LEASE_HEARTBEAT_SECONDS = 5 * 60
LEASE_STALE_SECONDS = 30 * 60
# How long a closed screen's lease outlives it: its threads can't be stopped mid-request,
# and a read that answers late (every retry of a 45 s read) then writes into the job.
LEASE_GRACE_SECONDS = 5 * 60
_LEASE_WAIT_SECONDS = 2.0
_LEASE_RETRY_SECONDS = 5.0
# A half-written console copy or cursor file older than this belongs to no running TUI:
# writing one takes seconds.
_STALE_PARTIAL_SECONDS = 60 * 60
_LEGACY_STAGE_SECONDS = 7 * 24 * 3600

STAGE_PREFIX = "cortex-training-stage-"
# Cut short so a stage's or marker's name stays within a file name's 255 bytes.
_HOST = (re.sub(r"[^A-Za-z0-9.-]", "-", socket.gethostname()) or "host")[:64]
_STAGE_NAME = re.compile(re.escape(STAGE_PREFIX) + r"(?P<host>[A-Za-z0-9.-]+)\+(?P<pid>\d+)\+")
_STAGE_OWNER = ".owner"

_KEYED_JOB = re.compile(r"[0-9a-f]{16}%3A[A-Za-z0-9._%-]+")
_BARE_JOB = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
# ``stdout`` and ``.stdout.lock``: where TUIs from before the saved console moved to
# ``console`` kept it, so a job one of them cached is still the cache's to count and remove.
_JOB_ENTRIES = frozenset({"logs", "console", "metrics", "serving", "cursors.json", ".lease", ".leases",
                          ".last-used", ".metrics.lock", ".console.lock", ".serving.lock", ".cursors.lock", "stdout",
                          ".stdout.lock"})
_JOB_ENTRY_PREFIXES = (".console-", ".tmp-")
# Files the operating system leaves in a directory someone looked at: Finder's .DS_Store and
# AppleDouble ._ files, and NFS's .nfs files for one deleted while open. Counted as part of
# the job, not as a stranger's entry: one would otherwise exempt its job from the limit.
# Regular files only: a directory by such a name is someone's.
_OS_FILE = re.compile(r"\.DS_Store|\._.+|\.nfs[0-9A-Fa-f]{8,}")
_LEASE_LOCK = ".lease"
_LEASES = ".leases"
_LAST_USED = ".last-used"
_REMOVING = ".removing-"
_PASS_LOCK = ".cleanup.lock"


def _os_file(entry: os.DirEntry) -> bool:
    """Whether ``entry`` is a file the operating system left (see _OS_FILE)."""
    try:
        return bool(_OS_FILE.fullmatch(entry.name)) and entry.is_file(follow_symlinks=False)
    except OSError:
        return False


def _disk_bytes(path: Path) -> int:
    """Regular-file bytes under ``path``, ignoring symlinks and vanishing files."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            with contextlib.suppress(OSError):
                info = os.lstat(os.path.join(dirpath, name))
                if stat.S_ISREG(info.st_mode):
                    total += info.st_size
    return total


def _last_used(job_dir: Path) -> float:
    """When the job was last open: its lease's last refresh, else its newest non-lock file.

    A download can write files into a job after a screen last had it open, and lock probes
    create ``.lease`` / ``*.lock`` files, so ``.last-used`` wins and lock files are ignored,
    as are the operating system's own (Finder writes a .DS_Store when the folder is looked at).
    """
    with contextlib.suppress(OSError):
        return os.lstat(job_dir / _LAST_USED).st_mtime
    newest = 0.0
    for dirpath, _dirnames, filenames in os.walk(job_dir):
        for name in filenames:
            if name == _LEASE_LOCK or name.endswith(".lock") or _OS_FILE.fullmatch(name):
                continue
            with contextlib.suppress(OSError):
                newest = max(newest, os.lstat(os.path.join(dirpath, name)).st_mtime)
    return newest


def _jobs_dir(root: Path) -> Optional[Path]:
    """``<root>/jobs`` when it is a real directory; a link is never followed."""
    jobs = root / "jobs"
    with contextlib.suppress(OSError):
        if not jobs.is_symlink() and jobs.is_dir():
            return jobs
    return None


def _is_job_dir(path: Path) -> bool:
    """Whether ``path`` is a real TUI job directory containing only TUI-owned entries."""
    if not (_KEYED_JOB.fullmatch(path.name) or _BARE_JOB.fullmatch(path.name)):
        return False
    try:
        if path.is_symlink() or not path.is_dir():
            return False
        with os.scandir(path) as entries:
            return all(entry.name in _JOB_ENTRIES or entry.name.startswith(_JOB_ENTRY_PREFIXES)
                       or _os_file(entry) for entry in entries)
    except OSError:
        return False


def _jobs(root: Path) -> list:
    jobs = _jobs_dir(root)
    with contextlib.suppress(OSError):
        return [p for p in sorted(jobs.iterdir()) if _is_job_dir(p)] if jobs else []
    return []


def _job_label(job_dir: Path) -> str:
    """The job id a cache directory holds (its name is ``<location digest>:<job id>``, escaped)."""
    try:
        name = unsafe_source_name(job_dir.name)
    except Exception:  # noqa: BLE001 - not a name the cache writes
        return job_dir.name
    label = name.split(":", 1)[-1]
    return f"{label} (name shortened)" if len(job_dir.name) >= 200 else label


def _open_lock(path: Path) -> Optional[int]:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        return os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    except OSError:
        return None


def _same_file(fd: int, path: Path) -> bool:
    """Whether ``fd`` still refers to the file currently at ``path``."""
    try:
        return os.fstat(fd).st_ino == os.stat(path).st_ino
    except OSError:
        return False


def _try_exclusive(fd: int) -> Optional[bool]:
    """Try non-blocking ``LOCK_EX``: True taken, False held by another, None unsupported."""
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    except OSError:
        return None
    return True


class _Scheduler:
    """Single daemon thread for delayed lease retries and grace-period releases."""

    def __init__(self) -> None:
        self._heap: list = []
        self._cv = threading.Condition()
        self._thread: Optional[threading.Thread] = None
        self._seq = 0

    def after(self, delay: float, fn) -> None:
        with self._cv:
            self._seq += 1
            heapq.heappush(self._heap, (time.monotonic() + delay, self._seq, fn))
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="cortex-tui-leases", daemon=True)
                self._thread.start()
            self._cv.notify()

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._heap:
                    self._cv.wait()
                when, _seq, fn = self._heap[0]
                wait = when - time.monotonic()
                if wait > 0:
                    self._cv.wait(wait)
                    continue
                heapq.heappop(self._heap)
            with contextlib.suppress(Exception):
                fn()


_SCHEDULER = _Scheduler()


class JobLease:
    """Marks a job's cache directory as open via a shared ``.lease`` flock and a
    periodically refreshed ``.leases/<holder>`` marker."""

    def __init__(self, job_dir: Path, *, holder: Optional[str] = None) -> None:
        self._job_dir = Path(job_dir)
        self._lock_path = self._job_dir / _LEASE_LOCK
        name = holder or f"{_HOST}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self._marker = self._job_dir / _LEASES / name
        self._fd: Optional[int] = None
        self._held = False
        self._closing = False  # in the grace after its screen closed (see release_later)
        self._mutex = threading.Lock()

    @property
    def held(self) -> bool:
        """Whether the lease still protects the job (acquired, and not yet released)."""
        return self._held

    def acquire(self) -> None:
        with self._mutex:
            self._held, self._closing = True, False
        # Touch the marker before waiting on the lock so a concurrent removal checking
        # markers sees the job open immediately, and again after in case a removal moved
        # the job aside while _lock waited.
        self._touch()
        self._retry_lock()

    def _retry_lock(self) -> None:
        if not self._held or self._closing or self._fd is not None:
            return
        self._lock()
        self._touch()  # refresh in case a removal moved the directory aside before _lock
        if self._held and self._fd is None and fcntl is not None:
            _SCHEDULER.after(_LEASE_RETRY_SECONDS, self._retry_lock)

    def _lock(self) -> None:
        if fcntl is None:
            return
        deadline = time.monotonic() + _LEASE_WAIT_SECONDS
        while self._held and not self._closing:
            fd = _open_lock(self._lock_path)
            if fd is None:
                return
            try:
                fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                if time.monotonic() >= deadline:
                    return
                time.sleep(0.02)
                continue
            except OSError:
                os.close(fd)
                return
            if _same_file(fd, self._lock_path):
                with self._mutex:
                    if self._held and not self._closing and self._fd is None:
                        self._fd = fd
                        return
                os.close(fd)
                return
            os.close(fd)  # locked a file a removal just moved aside; reopen at the path
            if time.monotonic() >= deadline:
                return
            time.sleep(0.01)

    def heartbeat(self) -> None:
        """Refresh the marker and last-used timestamp, reacquiring the lock if needed."""
        if not self._held:
            return
        if self._closing:
            self._touch()
            return
        with self._mutex:
            stale = self._fd is not None and not _same_file(self._fd, self._lock_path)
            fd, self._fd = (self._fd, None) if stale else (None, self._fd)
        if fd is not None:
            os.close(fd)
        if self._fd is None:
            self._lock()
        self._touch()

    def _touch(self) -> None:
        with self._mutex:
            if not self._held:
                return
            for _attempt in range(3):
                try:
                    self._marker.parent.mkdir(parents=True, exist_ok=True)
                    self._marker.touch()
                    (self._job_dir / _LAST_USED).touch()
                    return
                except FileNotFoundError:
                    continue
                except OSError:
                    return

    def release_later(self, delay: float = LEASE_GRACE_SECONDS) -> None:
        """Release after ``delay``, or at process exit if sooner.

        Meanwhile the marker alone protects the job, refreshed now and through the grace;
        the lock's descriptor is closed at once. Kept open, a TUI that opens job after job
        held one for each screen of the last few minutes, and ran out of descriptors.
        """
        with self._mutex:
            self._closing = True
            fd, self._fd = self._fd, None
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)
        self._touch()
        _PENDING.add(self)
        _SCHEDULER.after(delay, self.release)
        if delay > LEASE_HEARTBEAT_SECONDS:
            _SCHEDULER.after(LEASE_HEARTBEAT_SECONDS, self._grace_heartbeat)

    def _grace_heartbeat(self) -> None:
        if self._held and self._closing:
            self._touch()
            _SCHEDULER.after(LEASE_HEARTBEAT_SECONDS, self._grace_heartbeat)

    def release(self) -> None:
        with self._mutex:
            if not self._held and self._fd is None:
                return
            self._held, fd, self._fd = False, self._fd, None
        _PENDING.discard(self)
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)
        _unlink(self._marker)
        with contextlib.suppress(OSError):
            self._marker.parent.rmdir()


_PENDING: set = set()


@atexit.register
def _release_pending() -> None:
    for lease in list(_PENDING):
        lease.release()


def _fresh_marker(job_dir: Path, now: float) -> bool:
    """Whether any lease, on any machine, refreshed its marker recently."""
    try:
        markers = list((Path(job_dir) / _LEASES).iterdir())
    except OSError:
        return False
    for marker in markers:
        if _OS_FILE.fullmatch(marker.name):
            continue  # Finder's or NFS's, not a lease's
        with contextlib.suppress(OSError):
            if now - marker.lstat().st_mtime < LEASE_STALE_SECONDS:
                return True
    return False


def job_in_use(job_dir: Path, *, now: Optional[float] = None) -> bool:
    """Whether a running TUI has the job open via a fresh marker or held lease lock."""
    now = time.time() if now is None else now
    if _fresh_marker(job_dir, now):
        return True
    path = Path(job_dir) / _LEASE_LOCK
    if fcntl is None or not path.exists():
        return False
    try:
        fd = os.open(path, os.O_RDWR)
    except OSError:
        return False
    try:
        return _try_exclusive(fd) is False
    finally:
        os.close(fd)


def _remove_unless_in_use(job_dir: Path, now: float) -> str:
    """``"removed"``, ``"in_use"`` or ``"failed"`` for one job.

    Under the lease and download locks the job is renamed aside before deletion, so an
    arriving screen or download never writes into a directory being deleted. Nothing is
    read between the last check for a screen's marker and the rename, so the locks are
    held for as short a time as they can be: a screen opening the job waits on them.
    """
    if _fresh_marker(job_dir, now):
        return "in_use"
    fd = None
    downloads: list = []
    try:
        if fcntl is not None:
            fd = _open_lock(job_dir / _LEASE_LOCK)
            # Unlike the pass lock, no .lease means no removal: without it exclusively,
            # removing the job could race a screen opening it.
            if fd is None or _try_exclusive(fd) is False:
                return "in_use"
        for part in ("metrics", "console", "serving"):
            path = job_dir / part
            if path.is_symlink():
                continue
            lock = DownloadLock(path)
            if not lock.try_acquire():
                return "in_use"
            downloads.append(lock)
        if _fresh_marker(job_dir, now):
            return "in_use"
        aside = job_dir.parent / f"{_REMOVING}{uuid.uuid4().hex}"
        try:
            os.rename(job_dir, aside)
        except OSError:
            return "failed"
    finally:
        for lock in downloads:
            lock.release()
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)
    shutil.rmtree(aside, ignore_errors=True)
    return "removed"


@contextlib.contextmanager
def _pass_lock(root: Path):
    """Serialize cleanup passes across TUIs so two processes don't both evict the excess.
    Yields False only while another TUI's pass holds the lock: when the lock file can't be
    created (no free inodes, a quota), the pass runs unserialized rather than free nothing."""
    if fcntl is None:
        yield True
        return
    fd = _open_lock(root / _PASS_LOCK)
    try:
        yield fd is None or _try_exclusive(fd) is not False
    finally:
        if fd is not None:
            os.close(fd)


def _local_training_bytes(root: Path) -> int:
    """What the client's own training rows under ``root`` count toward the limit.

    Only what the client writes, ``local-training/<job>/<sub_job>/training.jsonl``, read
    without following a link; a directory holding anything else, beside the files an
    operating system leaves, is someone else's and is not counted. At most the client's own
    bound on them (:data:`_training_log.MAX_TOTAL_BYTES`): past it, they would hold the
    cache over its limit whatever the TUI removed.
    """
    base = root / _training_log.DIR_NAME
    total = 0
    try:
        if base.is_symlink() or not base.is_dir():
            return 0
        with os.scandir(base) as jobs:
            for job in jobs:
                total += _local_training_job_bytes(job) or 0
    except OSError:
        pass
    return min(total, _training_log.MAX_TOTAL_BYTES)


def _local_training_job_bytes(job: os.DirEntry) -> Optional[int]:
    """The bytes of one job's rows files; None when its directory holds anything else."""
    total = 0
    try:
        if not job.is_dir(follow_symlinks=False):
            return None
        with os.scandir(job.path) as subs:
            for sub in subs:
                if _os_file(sub):
                    total += sub.stat(follow_symlinks=False).st_size
                    continue
                if not sub.is_dir(follow_symlinks=False):
                    return None
                with os.scandir(sub.path) as rows:
                    for entry in rows:
                        if not (entry.name == _training_log.FILE_NAME and entry.is_file(follow_symlinks=False)
                                or _os_file(entry)):
                            return None
                        total += entry.stat(follow_symlinks=False).st_size
    except OSError:
        return None
    return total


def telemetry_bytes(metrics_root: Path) -> int:
    """The size of a job's reconstructed telemetry files (``<sub_job>/*`` under ``metrics_root``).

    One listing per sub-job, not a walk: cheap enough to take before and after every download.
    """
    total = 0
    try:
        with os.scandir(metrics_root) as subs:
            for sub in subs:
                if not sub.is_dir(follow_symlinks=False):
                    continue
                with contextlib.suppress(OSError), os.scandir(sub.path) as files:
                    for entry in files:
                        if entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
    except OSError:
        pass
    return total


def enforce_cache_limit(root: Optional[Path] = None, limit: Optional[int] = DEFAULT_CACHE_LIMIT) -> list:
    """Remove least-recently-used jobs until the cache fits ``limit`` (None: no limit),
    skipping jobs open in a TUI or being downloaded into. Returns ``(job_id, bytes)`` pairs.

    The limit covers the cache's jobs and the client's local training rows together:
    counted apart, the real ceiling would be the two budgets added together. Those rows are
    counted but never removed here (they are how the Training metrics tab shows a step run
    from this machine before the stage has it, and the only copy of a job that does not
    save its training metrics to the stage), so it is the jobs that go."""
    if limit is None:
        return []
    root = Path(root) if root is not None else cache_root()
    if _jobs_dir(root) is None and not (root / _training_log.DIR_NAME).is_dir():
        return []
    with _pass_lock(root) as ours:
        if not ours:
            return []
        # Measured under the pass lock: what another TUI's pass removed while this one
        # waited would otherwise be counted, and a job the cache has room for removed.
        jobs = _jobs(root)
        sizes = {job: _disk_bytes(job) for job in jobs}
        total = sum(sizes.values()) + _local_training_bytes(root)
        removed = []
        if total <= limit:
            return removed
        now = time.time()
        for job_dir in sorted(jobs, key=_last_used):
            if total <= limit:
                break
            try:
                outcome = _remove_unless_in_use(job_dir, now)
            except OSError:
                outcome = "failed"
            if outcome == "removed":
                total -= sizes[job_dir]
                removed.append((_job_label(job_dir), sizes[job_dir]))
            elif not os.path.lexists(job_dir):
                # Another pass removed it (where locks don't serialize passes): its bytes
                # are gone all the same, and counting them would remove more jobs for them.
                total -= sizes[job_dir]
        return removed


def _unlink(path: Path) -> None:
    with contextlib.suppress(OSError):
        os.unlink(path)


def _stage_abandoned(stage: Path, now: float) -> bool:
    """Whether a staging directory's TUI is gone (owner lock free, or legacy > 1 week old)."""
    match = _STAGE_NAME.match(stage.name)
    if match is None:
        return now - stage.lstat().st_mtime > _LEGACY_STAGE_SECONDS
    if match.group("host") != _HOST or fcntl is None:
        return False
    try:
        fd = os.open(f"{stage}{_STAGE_OWNER}", os.O_RDWR)
    except FileNotFoundError:
        return now - stage.lstat().st_mtime > _LEGACY_STAGE_SECONDS
    except OSError:
        return False
    try:
        return _try_exclusive(fd) is True
    finally:
        os.close(fd)


def sweep_stale(root: Optional[Path] = None, *, temp_dir: Optional[str] = None,
                now: Optional[float] = None) -> list:
    """Remove abandoned staging directories, unfinished removals, stale partial writes,
    and expired lease markers. Returns the paths removed."""
    root = Path(root) if root is not None else cache_root()
    now = time.time() if now is None else now
    removed = []

    def remove(path: Path) -> None:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
        else:
            _unlink(path)
        if not os.path.lexists(path):
            removed.append(path)

    temp = Path(temp_dir if temp_dir is not None else tempfile.gettempdir())
    with contextlib.suppress(OSError):
        for stage in list(temp.iterdir()):
            if not stage.name.startswith(STAGE_PREFIX) or stage.is_symlink():
                continue
            if stage.name.endswith(_STAGE_OWNER):
                if not os.path.lexists(str(stage)[:-len(_STAGE_OWNER)]):
                    _unlink(stage)
                continue
            with contextlib.suppress(OSError):
                if stage.is_dir() and _stage_abandoned(stage, now):
                    remove(stage)
                    _unlink(Path(f"{stage}{_STAGE_OWNER}"))

    jobs = _jobs_dir(root)
    if jobs is not None:
        with contextlib.suppress(OSError):
            for entry in list(jobs.iterdir()):
                if entry.name.startswith(_REMOVING) and not entry.is_symlink():
                    remove(entry)
    for job_dir in _jobs(root):
        for dirpath, dirnames, filenames in os.walk(job_dir):
            here = Path(dirpath)
            for name in (*dirnames, *filenames):
                if name.startswith(".console-") or (name.startswith(".tmp-") and name.endswith(".swp")):
                    path = here / name
                    with contextlib.suppress(OSError):
                        if not path.is_symlink() and now - path.lstat().st_mtime > _STALE_PARTIAL_SECONDS:
                            remove(path)
            dirnames[:] = [d for d in dirnames if not d.startswith(".console-")]
        with contextlib.suppress(OSError):
            for marker in list((job_dir / _LEASES).iterdir()):
                with contextlib.suppress(OSError):
                    if now - marker.lstat().st_mtime > LEASE_STALE_SECONDS:
                        remove(marker)
            (job_dir / _LEASES).rmdir()
    return removed


def make_stage() -> tuple:
    """Create a staging directory and its locked ``.owner`` file: ``(TemporaryDirectory, fd)``."""
    stage = tempfile.TemporaryDirectory(prefix=f"{STAGE_PREFIX}{_HOST}+{os.getpid()}+")
    fd = None
    if fcntl is not None:
        try:
            fd = os.open(f"{stage.name}{_STAGE_OWNER}", os.O_RDWR | os.O_CREAT, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            if fd is not None:
                os.close(fd)
            # An unlocked .owner would read as abandoned to another TUI's sweep, which
            # would remove this stage while it is being written into. Without one the
            # stage falls back to the legacy age rule, which no live stage ever meets.
            _unlink(Path(f"{stage.name}{_STAGE_OWNER}"))
            fd = None
    if fd is not None:
        weakref.finalize(stage, _unlink, Path(f"{stage.name}{_STAGE_OWNER}"))
    return stage, fd


def release_stage(stage, fd: Optional[int]) -> None:
    """Remove a staging directory and its owner file, then drop the owner lock."""
    owner = Path(f"{stage.name}{_STAGE_OWNER}")
    stage.cleanup()
    _unlink(owner)
    if fd is not None:
        with contextlib.suppress(OSError):
            os.close(fd)
