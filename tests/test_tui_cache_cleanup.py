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

"""Keeping the TUI's local cache within bounds (tui/cache_cleanup.py)."""

import contextlib
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from cortex_training import _training_log
from cortex_training._download_lock import DownloadLock
from cortex_training.tui import cache_cleanup
from cortex_training.tui.cache_cleanup import JobLease
from cortex_training.tui.log_cache import LogCache
from cortex_training.tui.log_cache import append_error_log
from cortex_training.tui.log_cache import cached_pages
from cortex_training.tui.log_cache import job_cache_dir

MIB = 1024 * 1024
DIGEST = "0123456789abcdef"


def _job(root, name, *, mib=1, age=0.0):
    job_dir = job_cache_dir(f"{DIGEST}:{name}", root=root)
    for part in ("logs", "metrics"):
        (job_dir / part).mkdir(parents=True, exist_ok=True)
    (job_dir / "metrics" / "gpu.jsonl").write_bytes(b"x" * int(mib * MIB))
    (job_dir / "logs" / "s.jsonl").write_bytes(b"{}\n")
    _age(job_dir, age)
    return job_dir


def _age(job_dir, age):
    when = time.time() - age
    for path in (*job_dir.rglob("*"), job_dir):
        os.utime(path, (when, when), follow_symlinks=False)


def _names(removed):
    return [name for name, _size in removed]


@pytest.fixture(autouse=True)
def _release_leases():
    yield
    cache_cleanup._release_pending()


def test_only_the_cache_s_own_job_directories_are_counted_or_removed(tmp_path):
    cached = _job(tmp_path, "cached", mib=1, age=9000)
    mine = tmp_path / "jobs" / "mine"
    mine.mkdir()
    (mine / "thesis.tex").write_text("years of work")
    shaped = tmp_path / "jobs" / "0123abcd-0000-4000-8000-000000000000"
    shaped.mkdir()
    (shaped / "notes.txt").write_text("mine too")
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets" / "big.bin").write_bytes(b"x" * (3 * MIB))
    sessions = tmp_path / ".sessions"
    sessions.mkdir()
    (sessions / "notes.txt").write_text("not the cache's")

    assert cache_cleanup.enforce_cache_limit(tmp_path, limit=int(1.5 * MIB)) == []
    assert _names(cache_cleanup.enforce_cache_limit(tmp_path, limit=1)) == ["cached"]
    cache_cleanup.sweep_stale(tmp_path, temp_dir=str(tmp_path / "no-temp"))
    assert cache_cleanup.enforce_cache_limit(tmp_path, limit=1) == []
    assert not cached.exists()
    for kept in (mine / "thesis.tex", shaped / "notes.txt", tmp_path / "datasets" / "big.bin", sessions / "notes.txt"):
        assert kept.exists(), f"{kept} was removed"


def test_a_jobs_directory_that_is_a_link_is_left_alone(tmp_path):
    outside = tmp_path / "outside"
    _job(outside, "linked")
    canary = outside / "jobs" / "precious.txt"
    canary.write_text("keep")
    root = tmp_path / "cache"
    root.mkdir()
    os.symlink(outside / "jobs", root / "jobs")
    assert cache_cleanup.enforce_cache_limit(root, limit=1) == []
    cache_cleanup.sweep_stale(root, temp_dir=str(tmp_path / "no-temp"))
    assert canary.exists() and cache_cleanup._jobs(outside)


def test_nothing_outside_the_cache_is_reached_or_counted_through_a_symlink(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "precious.txt").write_text("must survive")
    big = outside / "big.bin"
    big.write_bytes(b"\0" * (5 * MIB))
    root = tmp_path / "cache"
    (root / "jobs").mkdir(parents=True)
    os.symlink(outside, root / "jobs" / f"{DIGEST}%3Alinked-job")
    real = job_cache_dir(f"{DIGEST}:real-job", root=root)
    real.mkdir(parents=True)
    os.symlink(outside, real / "metrics")
    (real / "logs").mkdir()
    os.symlink(big, real / "logs" / "link.jsonl")
    temp = tmp_path / "tmp"
    temp.mkdir()
    os.symlink(outside, temp / f"{cache_cleanup.STAGE_PREFIX}{cache_cleanup._HOST}+1+x")
    assert cache_cleanup._disk_bytes(root) == 0
    cache_cleanup.enforce_cache_limit(root, limit=1)
    cache_cleanup.sweep_stale(root, temp_dir=str(temp))
    assert (outside / "precious.txt").read_text() == "must survive" and big.exists()


def test_a_job_directory_the_tui_cannot_read_is_skipped_not_fatal(tmp_path):
    unreadable = _job(tmp_path, "unreadable", mib=1, age=9000)
    _job(tmp_path, "readable", mib=1, age=8000)
    os.chmod(unreadable, 0)
    try:
        assert _names(cache_cleanup.enforce_cache_limit(tmp_path, limit=1)) == ["readable"]
        cache_cleanup.sweep_stale(tmp_path, temp_dir=str(tmp_path / "no-temp"))
    finally:
        os.chmod(unreadable, 0o755)


def test_the_cache_keeps_under_its_limit_by_dropping_the_jobs_used_least_recently(tmp_path):
    oldest = _job(tmp_path, "oldest", mib=3, age=3000)
    older = _job(tmp_path, "older", mib=3, age=2000)
    recent = _job(tmp_path, "recent", mib=3, age=10)
    assert _names(cache_cleanup.enforce_cache_limit(tmp_path, limit=5 * MIB)) == ["oldest", "older"]
    assert not oldest.exists() and not older.exists() and recent.exists()
    assert cache_cleanup.enforce_cache_limit(tmp_path, limit=5 * MIB) == []


def test_the_last_use_a_lease_records_decides_the_order(tmp_path):
    opened_long_ago = _job(tmp_path, "opened-long-ago", mib=1, age=10)
    opened_recently = _job(tmp_path, "opened-recently", mib=1, age=9000)
    for job, age in ((opened_long_ago, 9000), (opened_recently, 10)):
        (job / ".last-used").touch()
        os.utime(job / ".last-used", (time.time() - age,) * 2)
    assert _names(cache_cleanup.enforce_cache_limit(tmp_path, limit=int(1.5 * MIB))) == ["opened-long-ago"]


def test_no_limit_removes_nothing(tmp_path):
    job = _job(tmp_path, "a", mib=2, age=9999)
    assert cache_cleanup.enforce_cache_limit(tmp_path, limit=None) == []
    assert job.exists()


def test_two_passes_at_once_remove_the_excess_once(tmp_path, monkeypatch):
    for i, age in enumerate((4000, 3000, 2000, 1000)):
        _job(tmp_path, f"j{i}", mib=1, age=age)
    entered, go = threading.Event(), threading.Event()
    real = cache_cleanup._remove_unless_in_use

    def slow(job_dir, now):
        entered.set()
        go.wait(5)
        return real(job_dir, now)

    monkeypatch.setattr(cache_cleanup, "_remove_unless_in_use", slow)
    first, limit = [], int(3.5 * MIB)
    thread = threading.Thread(target=lambda: first.extend(cache_cleanup.enforce_cache_limit(tmp_path, limit=limit)))
    thread.start()
    assert entered.wait(5)
    second = cache_cleanup.enforce_cache_limit(tmp_path, limit=limit)
    go.set()
    thread.join(5)
    assert second == [], "a second pass ran alongside the first"
    assert _names(first) == ["j0"] and len(cache_cleanup._jobs(tmp_path)) == 3


def test_a_pass_runs_without_its_lock_file_but_skips_a_job_without_its_lease(tmp_path, monkeypatch):
    # No free inodes or a quota: freeing nothing is worse than two TUIs evicting at once. A
    # job whose .lease can't be created still stays: removing it could race a screen opening it.
    old = _job(tmp_path, "old", mib=1, age=9000)
    no_lease = _job(tmp_path, "no-lease", mib=1, age=8000)
    real_open_lock = cache_cleanup._open_lock

    def open_lock(path):
        if path.name == cache_cleanup._PASS_LOCK or path.parent == no_lease:
            return None  # as when the file can't be created
        return real_open_lock(path)

    monkeypatch.setattr(cache_cleanup, "_open_lock", open_lock)
    assert _names(cache_cleanup.enforce_cache_limit(tmp_path, limit=1)) == ["old"]
    assert not old.exists() and no_lease.exists()


def _local_rows(root, job, mib, *, age=0.0):
    """A job's local training rows, laid out as the client writes them."""
    sub = root / _training_log.DIR_NAME / job / f"{job}:training:0"
    sub.mkdir(parents=True)
    rows = sub / _training_log.FILE_NAME
    rows.write_bytes(b"x" * int(mib * MIB))
    then = time.time() - age
    for path in (rows, sub, sub.parent):
        os.utime(path, (then, then))
    return sub.parent


A_MONTH_AND_A_DAY = _training_log.MAX_AGE_SECONDS + 86400


def test_the_limit_counts_the_client_s_local_training_rows(tmp_path):
    # Counted apart, the job cache and the local rows would put the real ceiling at the two
    # budgets added together. The jobs go to make the room: 2 MiB fits 4 MiB alone, not with 3 more.
    closed = _job(tmp_path, "closed", mib=2, age=9000)
    local = _local_rows(tmp_path, "a-job-run-here", mib=3)

    assert _names(cache_cleanup.enforce_cache_limit(tmp_path, limit=4 * MIB)) == ["closed"]
    assert not closed.exists() and local.exists()


def test_local_training_rows_are_never_removed_to_make_room(tmp_path):
    # They are what the Training metrics tab shows of a step run from this machine before the
    # stage has it. An open job alone over the limit once had every one of them removed, the
    # running job's included, and the cache was still exactly as far over afterwards.
    open_job = _job(tmp_path, "open", mib=4, age=10)
    local = _local_rows(tmp_path, "a-job-running-now", mib=1)
    rows = next(local.rglob(_training_log.FILE_NAME))
    before = rows.read_bytes()
    lease = JobLease(open_job)
    lease.acquire()
    try:
        for _ in range(3):  # every pass, not just the first
            assert cache_cleanup.enforce_cache_limit(tmp_path, limit=2 * MIB) == []
        assert open_job.exists()
        assert rows.read_bytes() == before, "the running job's local rows were touched"
    finally:
        lease.release()


def test_local_training_rows_are_left_to_the_client_however_old(tmp_path):
    # For a job that does not save its training metrics to the stage they are the only copy:
    # the client removes them by its own rules, on its own first write, as without a TUI.
    stale = _local_rows(tmp_path, "a-job-nobody-ran-for-a-month", mib=1, age=A_MONTH_AND_A_DAY)

    assert cache_cleanup.enforce_cache_limit(tmp_path, limit=100 * MIB) == []
    assert stale.exists(), "the TUI removed the client's rows"


def test_the_local_rows_count_at_most_the_bound_the_client_holds_them_to(tmp_path, monkeypatch):
    # Past the client's own bound (a long-lived process prunes them only once), counting them
    # whole held the cache over its limit whatever was removed, and every closed job went for it.
    monkeypatch.setattr(_training_log, "MAX_TOTAL_BYTES", 1 * MIB)
    _local_rows(tmp_path, "a-big-job", mib=3)
    closed = _job(tmp_path, "closed", mib=0.5, age=9000)

    assert cache_cleanup.enforce_cache_limit(tmp_path, limit=2 * MIB) == []
    assert closed.exists()


def test_finder_s_metadata_in_the_local_rows_does_not_take_them_out_of_the_count(tmp_path):
    local = _local_rows(tmp_path, "a-job-run-here", mib=3)
    (local / ".DS_Store").write_bytes(b"\0" * 6148)
    (local / "a-job-run-here:training:0" / "._training.jsonl").write_bytes(b"\0" * 4096)
    closed = _job(tmp_path, "closed", mib=2, age=9000)

    assert _names(cache_cleanup.enforce_cache_limit(tmp_path, limit=4 * MIB)) == ["closed"]
    assert local.exists()


def test_a_linked_local_training_directory_is_neither_counted_nor_entered(tmp_path):
    # Followed, the link put another disk's directories in reach of rmtree, a stranger's included.
    elsewhere = tmp_path / "elsewhere"
    target = elsewhere / "lt"
    _local_rows(elsewhere, "unused", mib=0)  # makes elsewhere/local-training; moved below
    (elsewhere / _training_log.DIR_NAME).rename(target)
    foreign = target / "not-the-client-s"
    foreign.mkdir()
    (foreign / "important.txt").write_text("keep me")
    stale = target / "an-old-job"
    (stale / "an-old-job:training:0").mkdir(parents=True)
    (stale / "an-old-job:training:0" / _training_log.FILE_NAME).write_bytes(b"x" * (3 * MIB))
    for path in (stale, *stale.rglob("*")):
        os.utime(path, (0, 0))
    root = tmp_path / "cache"
    root.mkdir()
    (root / _training_log.DIR_NAME).symlink_to(target, target_is_directory=True)
    closed = _job(root, "closed", mib=1, age=9000)

    assert cache_cleanup.enforce_cache_limit(root, limit=2 * MIB) == [], "bytes behind the link were counted"
    assert closed.exists()
    assert (foreign / "important.txt").read_text() == "keep me"
    assert stale.exists(), "the pass reached through the link"


def test_entries_the_client_does_not_write_are_left_alone_and_not_counted(tmp_path):
    base = tmp_path / _training_log.DIR_NAME
    base.mkdir()
    (base / "a-stray-file").write_bytes(b"x" * (4 * MIB))  # once kept the cache over its limit for good
    odd = base / "notes"
    odd.mkdir()
    (odd / "notes.txt").write_text("mine")
    # Laid out like a job's, but with a file the client never writes beside (or in place of) its rows.
    mixed = base / "a-job" / "a-job:training:0"
    mixed.mkdir(parents=True)
    (mixed / "my-analysis.csv").write_bytes(b"x" * (4 * MIB))
    for path in (odd / "notes.txt", odd, mixed / "my-analysis.csv", mixed, mixed.parent):
        os.utime(path, (0, 0))  # far past the age rule, so only the layout keeps them
    closed = _job(tmp_path, "closed", mib=1, age=9000)

    assert cache_cleanup.enforce_cache_limit(tmp_path, limit=2 * MIB) == [], "foreign bytes were counted"
    assert closed.exists()
    assert (base / "a-stray-file").exists() and (odd / "notes.txt").exists()
    assert (mixed / "my-analysis.csv").exists(), "a file the client never wrote was removed"


def test_the_local_rows_are_counted_after_the_pass_lock_is_taken(tmp_path, monkeypatch):
    # Counted before it, rows another TUI's pass removed while this one waited for the lock
    # were still in the total, and a job the cache had room for was removed in their place.
    closed = _job(tmp_path, "closed", mib=1, age=9000)
    local = _local_rows(tmp_path, "a-job-run-here", mib=3)
    real_pass_lock = cache_cleanup._pass_lock

    @contextlib.contextmanager
    def another_pass_ran_first(root):
        shutil.rmtree(local)  # what the other TUI's pass removed while this one waited
        with real_pass_lock(root) as ours:
            yield ours

    monkeypatch.setattr(cache_cleanup, "_pass_lock", another_pass_ran_first)
    assert cache_cleanup.enforce_cache_limit(tmp_path, limit=2 * MIB) == []
    assert closed.exists(), "a job was removed to make room for rows that were already gone"


def test_only_files_count_as_the_operating_system_s(tmp_path):
    # A directory by such a name, or a name only close to NFS's, is someone's: its job is
    # neither the cache's to count nor to remove.
    for name, make in (("._notes", "dir"), (".nfs-exports-backup", "dir"), (".nfsnotes", "file")):
        job = _job(tmp_path, f"with-{name}", mib=1, age=9000)
        if make == "dir":
            (job / name).mkdir()
            (job / name / "mine.txt").write_text("keep me")
        else:
            (job / name).write_text("keep me")
    nfs = _job(tmp_path, "silly-renamed", mib=1, age=9000)
    (nfs / ".nfs0123456789abcdef00000001").write_bytes(b"x")

    assert _names(cache_cleanup.enforce_cache_limit(tmp_path, limit=1)) == ["silly-renamed"]
    assert all((job / "mine.txt").exists() for job in tmp_path.glob("jobs/*/._notes"))


def test_finder_s_metadata_in_a_job_s_leases_is_not_a_lease(tmp_path):
    job = _job(tmp_path, "looked-at", mib=1, age=9000)
    (job / ".leases").mkdir()
    (job / ".leases" / ".DS_Store").write_bytes(b"\0" * 6148)

    assert not cache_cleanup.job_in_use(job)
    assert _names(cache_cleanup.enforce_cache_limit(tmp_path, limit=1)) == ["looked-at"]


def test_a_job_cached_by_a_tui_from_before_console_is_still_counted_and_removed(tmp_path):
    job = _job(tmp_path, "legacy", mib=0, age=9000)
    (job / "stdout" / "7:training:0").mkdir(parents=True)
    (job / "stdout" / "7:training:0" / "stdout.log").write_bytes(b"x" * (2 * MIB))
    (job / ".stdout.lock").touch()

    assert _names(cache_cleanup.enforce_cache_limit(tmp_path, limit=MIB)) == ["legacy"]


def test_a_job_another_pass_removed_meanwhile_counts_as_gone(tmp_path, monkeypatch):
    # Where locks don't serialize passes, the job the other pass took read as a failed
    # removal: its bytes stayed in this pass's total, and it removed more jobs for them.
    first = _job(tmp_path, "first", mib=2, age=9000)
    _job(tmp_path, "second", mib=1, age=8000)
    _job(tmp_path, "third", mib=1, age=7000)
    real_remove = cache_cleanup._remove_unless_in_use

    def another_pass_took_it(job_dir, now):
        if job_dir == first:
            shutil.rmtree(job_dir)
            return "failed"
        return real_remove(job_dir, now)

    monkeypatch.setattr(cache_cleanup, "_remove_unless_in_use", another_pass_took_it)
    assert cache_cleanup.enforce_cache_limit(tmp_path, limit=int(2.5 * MIB)) == []
    assert len(list((tmp_path / "jobs").iterdir())) == 2, "the pass removed a job for bytes already gone"


@pytest.mark.skipif(not os.path.isdir("/dev/fd"), reason="counts descriptors through /dev/fd")
def test_screens_closed_in_the_last_minutes_hold_no_descriptor(tmp_path):
    # Each closed screen held its lease's descriptor for the 5-minute grace: a TUI opening job
    # after job ran out of them (256 by default on macOS) and exited.
    before = len(os.listdir("/dev/fd"))
    leases = []
    try:
        for i in range(60):
            lease = JobLease(_job(tmp_path, f"opened-{i}", mib=0))
            lease.acquire()
            lease.release_later(3600)
            leases.append(lease)
        assert len(os.listdir("/dev/fd")) - before < 5, "each screen in its grace still holds a descriptor"
        assert all(lease.held and cache_cleanup.job_in_use(lease._job_dir) for lease in leases)
        assert cache_cleanup.enforce_cache_limit(tmp_path, limit=1) == [], "a job in its grace was removed"
    finally:
        for lease in leases:
            lease.release()


def test_a_long_grace_keeps_its_marker_fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_cleanup, "LEASE_HEARTBEAT_SECONDS", 0.1)
    job = _job(tmp_path, "closing", mib=0)
    lease = JobLease(job)
    lease.acquire()
    lease.release_later(3600)
    try:
        first = lease._marker.stat().st_mtime
        deadline = time.time() + 5
        while lease._marker.stat().st_mtime == first and time.time() < deadline:
            time.sleep(0.05)
            os.utime(lease._marker, (first, first))  # as if it had aged; the grace refreshes it
            time.sleep(0.15)
        assert lease._marker.stat().st_mtime > first, "the marker was not refreshed through the grace"
    finally:
        lease.release()


def test_a_long_host_name_still_makes_a_stage(tmp_path, monkeypatch):
    assert len(cache_cleanup._HOST) <= 64
    monkeypatch.setattr(cache_cleanup.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(cache_cleanup, "_HOST", "h" * 64)
    stage, fd = cache_cleanup.make_stage()
    cache_cleanup.release_stage(stage, fd)


def test_a_stage_s_owner_file_goes_when_its_tui_exits_without_releasing_it(tmp_path):
    code = ("import sys, tempfile; tempfile.tempdir = sys.argv[1]; "
            "from cortex_training.tui import cache_cleanup; stage, fd = cache_cleanup.make_stage()")
    subprocess.run([sys.executable, "-c", code, str(tmp_path)], check=True)
    assert not [p.name for p in tmp_path.iterdir() if p.name.startswith(cache_cleanup.STAGE_PREFIX)]


def test_a_job_s_telemetry_size_is_its_sub_jobs_files(tmp_path):
    metrics = tmp_path / "metrics"
    (metrics / "a").mkdir(parents=True)
    (metrics / "a" / "gpu.jsonl").write_bytes(b"x" * 100)
    (metrics / "a" / "training.jsonl").write_bytes(b"x" * 20)
    (metrics / "b").mkdir()
    (metrics / "b" / "gpu.jsonl").write_bytes(b"x" * 3)
    (metrics / "loose-file").write_bytes(b"x" * 1000)
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "big").write_bytes(b"x" * 5000)
    (metrics / "linked").symlink_to(tmp_path / "elsewhere", target_is_directory=True)

    assert cache_cleanup.telemetry_bytes(metrics) == 123
    assert cache_cleanup.telemetry_bytes(tmp_path / "no-such-dir") == 0


def test_finder_s_metadata_in_a_job_does_not_exempt_it_from_the_limit(tmp_path):
    # Opening the cache in Finder leaves a .DS_Store: the job then read as a stranger's
    # directory, was never counted or removed, and the cache grew past its limit unseen.
    looked_at = _job(tmp_path, "looked-at", mib=2, age=9000)
    (looked_at / ".DS_Store").write_bytes(b"\0" * 6148)
    (looked_at / "._cursors.json").write_bytes(b"\0" * 4096)
    _job(tmp_path, "recent", mib=1, age=10)

    assert _names(cache_cleanup.enforce_cache_limit(tmp_path, limit=int(1.5 * MIB))) == ["looked-at"]


def test_a_job_open_in_a_running_tui_is_never_removed(tmp_path):
    open_job = _job(tmp_path, "open", mib=3, age=5000)
    _job(tmp_path, "other", mib=3, age=100)
    JobLease(open_job).acquire()
    os.utime(open_job / ".last-used", (time.time() - 5000,) * 2)
    assert _names(cache_cleanup.enforce_cache_limit(tmp_path, limit=1 * MIB)) == ["other"]
    assert open_job.exists()


@pytest.mark.parametrize("part", ["metrics", "serving"])
def test_a_job_being_downloaded_is_never_removed(tmp_path, part):
    busy = _job(tmp_path, "busy", mib=3, age=5000)
    newer = _job(tmp_path, "newer", mib=3, age=1000)
    lock = DownloadLock(busy / part)
    assert lock.try_acquire()
    try:
        # 6 MiB over a 4 MiB limit: busy is skipped because of its download lock, so newer goes.
        assert _names(cache_cleanup.enforce_cache_limit(tmp_path, limit=4 * MIB)) == ["newer"]
        assert busy.exists() and not newer.exists()
    finally:
        lock.release()
    assert _names(cache_cleanup.enforce_cache_limit(tmp_path, limit=1 * MIB)) == ["busy"]


def test_probing_a_busy_job_s_locks_does_not_make_it_look_newer(tmp_path):
    busy = _job(tmp_path, "busy", mib=3, age=5000)
    middle = _job(tmp_path, "middle", mib=3, age=2000)
    newest = _job(tmp_path, "newest", mib=3, age=100)
    lock = DownloadLock(busy / "metrics")
    assert lock.try_acquire()
    try:
        # 9 MiB over a 7 MiB limit: probes busy (skipped), then removes middle to reach 6 MiB.
        assert _names(cache_cleanup.enforce_cache_limit(tmp_path, limit=7 * MIB)) == ["middle"]
    finally:
        lock.release()
    # busy was probed during the first pass, but its fallback age is still older than newest's.
    assert _names(cache_cleanup.enforce_cache_limit(tmp_path, limit=4 * MIB)) == ["busy"]
    assert newest.exists()


def test_a_first_download_is_protected_before_its_directory_exists(tmp_path):
    job = _job(tmp_path, "first-download", mib=1, age=5000)
    shutil.rmtree(job / "metrics")
    (job / "logs" / "s.jsonl").write_bytes(b"x" * MIB)
    lock = DownloadLock(job / "metrics")
    assert lock.try_acquire()
    try:
        assert cache_cleanup.enforce_cache_limit(tmp_path, limit=1) == []
        assert job.exists()
    finally:
        lock.release()


def test_a_removal_moves_the_job_aside_so_a_screen_opening_meanwhile_never_writes_into_it(tmp_path, monkeypatch):
    job = _job(tmp_path, "racing", mib=1)
    moved, finish = threading.Event(), threading.Event()
    real_rmtree = cache_cleanup.shutil.rmtree

    def slow_rmtree(path, *args, **kwargs):
        moved.set()
        finish.wait(5)
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(cache_cleanup.shutil, "rmtree", slow_rmtree)
    outcome = {}
    remover = threading.Thread(target=lambda: outcome.update(r=cache_cleanup._remove_unless_in_use(job, time.time())))
    remover.start()
    assert moved.wait(5)
    lease = JobLease(job)
    lease.acquire()
    (job / "logs").mkdir(parents=True, exist_ok=True)
    (job / "logs" / "new.jsonl").write_text("{}\n")
    finish.set()
    remover.join(5)
    assert outcome["r"] == "removed"
    assert (job / "logs" / "new.jsonl").exists(), "the delete reached what the new screen wrote"
    assert lease._fd is not None and cache_cleanup._same_file(lease._fd, job / ".lease")
    assert not list(job.parent.glob(".removing-*"))
    lease.release()


def test_a_lease_protects_while_held_and_not_after_release(tmp_path):
    job = _job(tmp_path, "j")
    lease = JobLease(job)
    lease.acquire()
    assert cache_cleanup.job_in_use(job)
    assert cache_cleanup.enforce_cache_limit(tmp_path, limit=1) == []
    lease.release()
    assert not cache_cleanup.job_in_use(job)
    assert not (job / ".leases").exists()


def test_a_tui_on_another_machine_is_seen_by_its_marker_whatever_its_pid(tmp_path):
    job = _job(tmp_path, "shared", mib=1, age=5000)
    marker = job / ".leases" / "other-host-4242"
    marker.parent.mkdir()
    marker.touch()
    assert cache_cleanup.job_in_use(job)
    assert cache_cleanup.enforce_cache_limit(tmp_path, limit=1) == []
    stale = time.time() - cache_cleanup.LEASE_STALE_SECONDS - 60
    os.utime(marker, (stale, stale))
    assert _names(cache_cleanup.enforce_cache_limit(tmp_path, limit=1)) == ["shared"]


def test_a_killed_tui_leaves_no_lock_and_its_marker_expires(tmp_path):
    job = _job(tmp_path, "k")
    code = ("import os, sys; from pathlib import Path; from cortex_training.tui.cache_cleanup import JobLease; "
            "JobLease(Path(sys.argv[1])).acquire(); os._exit(0)")
    subprocess.run([sys.executable, "-c", code, str(job)], check=True)
    fd = os.open(job / ".lease", os.O_RDWR)
    try:
        cache_cleanup.fcntl.flock(fd, cache_cleanup.fcntl.LOCK_EX | cache_cleanup.fcntl.LOCK_NB)
    finally:
        os.close(fd)
    assert cache_cleanup.job_in_use(job)
    later = time.time() + cache_cleanup.LEASE_STALE_SECONDS + 60
    assert not cache_cleanup.job_in_use(job, now=later)


def test_a_closed_screen_s_lease_outlives_it_for_the_grace_then_goes(tmp_path):
    job = _job(tmp_path, "closing", mib=1, age=5000)
    lease = JobLease(job)
    lease.acquire()
    lease.release_later(0.3)
    assert lease.held and cache_cleanup.job_in_use(job)
    assert cache_cleanup.enforce_cache_limit(tmp_path, limit=1) == []
    deadline = time.time() + 5
    while lease.held and time.time() < deadline:
        time.sleep(0.05)
    assert not lease.held and not cache_cleanup.job_in_use(job)
    assert _names(cache_cleanup.enforce_cache_limit(tmp_path, limit=1)) == ["closing"]


def test_leases_still_in_their_grace_are_released_when_the_process_exits(tmp_path):
    job = _job(tmp_path, "exiting")
    code = ("import sys; from pathlib import Path; from cortex_training.tui.cache_cleanup import JobLease; "
            "lease = JobLease(Path(sys.argv[1])); lease.acquire(); lease.release_later(3600)")
    subprocess.run([sys.executable, "-c", code, str(job)], check=True)
    assert not cache_cleanup.job_in_use(job), "the exited TUI's lease outlived it"


def test_a_job_reopened_in_its_grace_keeps_the_new_screen_s_marker(tmp_path):
    job = _job(tmp_path, "reopened")
    first = JobLease(job)
    first.acquire()
    first.release_later(3600)
    second = JobLease(job)
    second.acquire()
    first.release()
    assert cache_cleanup.job_in_use(job), "the reopened screen lost its protection"
    second.release()
    assert not cache_cleanup.job_in_use(job)


def test_the_sweep_removes_only_what_no_running_tui_owns(tmp_path):
    temp = tmp_path / "tmp"
    temp.mkdir()
    host = cache_cleanup._HOST

    def stage(name, *, owner=False, age=0.0):
        path = temp / name
        path.mkdir()
        if owner:
            Path(f"{path}.owner").touch()
        os.utime(path, (time.time() - age,) * 2)
        return path

    mine, mine_fd = cache_cleanup.make_stage()
    os.rename(mine.name, temp / Path(mine.name).name)
    os.rename(f"{mine.name}.owner", f"{temp / Path(mine.name).name}.owner")
    alive = temp / Path(mine.name).name
    gone = stage(f"{cache_cleanup.STAGE_PREFIX}{host}+4242+abc", owner=True)
    huge = stage(f"{cache_cleanup.STAGE_PREFIX}{host}+99999999999999999999+abc", owner=True)
    other_host = stage(f"{cache_cleanup.STAGE_PREFIX}other-machine+4242+abc", owner=True)
    legacy_old = stage(f"{cache_cleanup.STAGE_PREFIX}oldname1", age=8 * 24 * 3600)
    legacy_new = stage(f"{cache_cleanup.STAGE_PREFIX}oldname2")
    unrelated = stage("something-else")
    orphan_owner = temp / f"{cache_cleanup.STAGE_PREFIX}{host}+4242+orphan.owner"
    orphan_owner.touch()

    root = tmp_path / "cache"
    job = _job(root, "j")
    aside = job.parent / ".removing-cut-short"
    aside.mkdir()
    (aside / "f").write_text("x")
    stale_copy, fresh_copy = job / ".console-old", job / ".console-new"
    stale_tmp, fresh_tmp = job / "logs" / ".tmp-a.swp", job / "logs" / ".tmp-b.swp"
    for path in (stale_copy, fresh_copy):
        path.mkdir()
    for path in (stale_tmp, fresh_tmp):
        path.write_text("partial")
    stale_marker, fresh_marker = job / ".leases" / "gone-host-1-x", job / ".leases" / "live-host-2-y"
    stale_marker.parent.mkdir()
    for path in (stale_marker, fresh_marker):
        path.touch()
    old = time.time() - 2 * 3600
    for path in (stale_copy, stale_tmp):
        os.utime(path, (old, old))
    os.utime(stale_marker, (time.time() - cache_cleanup.LEASE_STALE_SECONDS - 60,) * 2)

    removed = set(cache_cleanup.sweep_stale(root, temp_dir=str(temp)))
    assert removed == {gone, huge, legacy_old, aside, stale_copy, stale_tmp, stale_marker}
    for kept in (alive, other_host, legacy_new, unrelated, fresh_copy, fresh_tmp, fresh_marker):
        assert kept.exists(), f"{kept} was swept"
    assert not Path(f"{gone}.owner").exists() and not orphan_owner.exists()
    os.close(mine_fd)


def test_a_stage_made_for_a_screen_is_swept_once_its_tui_is_gone(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_cleanup.tempfile, "tempdir", str(tmp_path))
    stage, fd = cache_cleanup.make_stage()
    assert cache_cleanup.sweep_stale(tmp_path / "cache") == []
    os.close(fd)
    assert [Path(stage.name)] == cache_cleanup.sweep_stale(tmp_path / "cache")
    assert not Path(f"{stage.name}.owner").exists()


def test_a_stage_whose_owner_lock_fails_is_not_swept_from_under_a_live_tui(tmp_path, monkeypatch):
    """An unlocked .owner would read as abandoned to any sweep on this host."""
    monkeypatch.setattr(cache_cleanup.tempfile, "tempdir", str(tmp_path))
    real_flock = cache_cleanup.fcntl.flock
    failed = []

    def flock(fd, operation):
        if not failed:  # only the owner lock make_stage takes, not the sweep's own probes
            failed.append(fd)
            raise OSError("this filesystem has no locks")
        return real_flock(fd, operation)

    monkeypatch.setattr(cache_cleanup.fcntl, "flock", flock)
    stage, fd = cache_cleanup.make_stage()
    assert fd is None and failed
    assert not Path(f"{stage.name}.owner").exists()
    # With no owner file the stage falls back to the legacy age rule, which a new one fails.
    assert cache_cleanup.sweep_stale(tmp_path / "cache") == []
    assert Path(stage.name).is_dir()


def test_a_job_that_cannot_be_moved_aside_is_skipped_and_the_pass_goes_on(tmp_path, monkeypatch):
    stuck = _job(tmp_path, "stuck", mib=1, age=9000)
    _job(tmp_path, "next", mib=1, age=8000)
    _job(tmp_path, "recent", mib=1, age=10)
    real_rename = cache_cleanup.os.rename

    def rename(src, dst):
        if Path(src) == stuck:
            raise PermissionError("can't move it")
        return real_rename(src, dst)

    monkeypatch.setattr(cache_cleanup.os, "rename", rename)
    assert _names(cache_cleanup.enforce_cache_limit(tmp_path, limit=int(1.5 * MIB))) == ["next", "recent"]
    assert stuck.exists()


def test_the_error_log_keeps_its_newest_lines_in_the_one_file(tmp_path):
    path = tmp_path / "errors.log"
    path.write_text("")
    inode = path.stat().st_ino
    for n in range(60):
        append_error_log(f"line {n:03d} " + "x" * 90 + "\n", path=str(path), max_bytes=1000)
    lines = path.read_text().splitlines()
    assert path.stat().st_size <= 1000 + 101
    assert lines[-1].startswith("line 059") and all(len(line) == 99 for line in lines), "a line was cut"
    assert path.stat().st_ino == inode, "the log was replaced rather than trimmed in place"
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith("errors.log")] == ["errors.log"], \
        "a second file appeared beside it"
    append_error_log("y\n", path=str(tmp_path / "no-such-dir" / "errors.log"))  # never raises


def test_a_line_written_while_another_tui_trims_the_log_is_kept(tmp_path, monkeypatch):
    import cortex_training.tui.log_cache as log_cache

    path = tmp_path / "errors.log"
    path.write_text("old history\n" * 1000)
    def slow_keep(target, keep):
        # _keep_newest with a pause between reading the newest bytes and writing them back:
        # a line appended in that pause, unlocked, is cut off by the rewrite.
        with open(target, "r+b") as f:
            start = max(0, f.seek(0, os.SEEK_END) - keep)
            f.seek(start)
            tail = f.read()
            if start:
                tail = tail[tail.find(b"\n") + 1:]
            time.sleep(0.2)
            f.seek(0)
            f.write(tail)
            f.truncate()

    monkeypatch.setattr(log_cache, "_keep_newest", slow_keep)
    writers = [threading.Thread(target=append_error_log, args=(f"new {i}\n",),
                                kwargs={"path": str(path), "max_bytes": 100}) for i in range(2)]
    for writer in writers:
        writer.start()
    for writer in writers:
        writer.join()
    text = path.read_text()
    assert "new 0\n" in text and "new 1\n" in text, "a line written during a trim was lost"


def test_a_traceback_goes_through_the_trim(tmp_path, monkeypatch):
    import cortex_training.tui.widgets as widgets

    path = tmp_path / "errors.log"
    path.write_bytes(b"x" * (6 * MIB) + b"\n")
    monkeypatch.setattr(widgets, "_ERROR_LOG", str(path))
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        widgets.log_traceback("a failure")
    assert path.stat().st_size < 6 * MIB
    assert "RuntimeError: boom" in path.read_text()
    assert not (tmp_path / "errors.log.1").exists()


def test_two_tuis_updating_cursors_of_one_job_lose_neither(tmp_path):
    def writer(source):
        cache = LogCache("shared-job", root=tmp_path)
        for n in range(150):
            cache.set_cursor(source, f"{source}-{n}")

    threads = [threading.Thread(target=writer, args=(f"sub-{i}",)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    cache = LogCache("shared-job", root=tmp_path)
    assert {f"sub-{i}": cache.get_cursor(f"sub-{i}") for i in range(4)} == {f"sub-{i}": f"sub-{i}-149" for i in range(4)}


def test_marking_a_log_s_head_missing_never_rolls_its_cursor_back(tmp_path, monkeypatch):
    advancing = LogCache("j", root=tmp_path)
    marking = LogCache("j", root=tmp_path)
    monkeypatch.setattr(marking, "head_missing", lambda source_id: False)

    def advance():
        for n in range(200):
            advancing.set_cursor("s", f"c{n}")

    def mark():
        for _ in range(200):
            marking.set_head_missing("s")

    threads = [threading.Thread(target=advance), threading.Thread(target=mark)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert advancing.get_cursor("s") == "c199"
    assert LogCache("j", root=tmp_path).head_missing("s")


def test_a_cancelled_tail_persists_nothing(tmp_path):
    cache = LogCache("late", root=tmp_path)
    cancelled = []

    def fetch(_cursor):
        cancelled.append(True)
        return {"entries": [{"msg": "late line"}], "next_cursor": "after-late"}

    pages = list(cached_pages(cache, "s", fetch, entries_key="entries", follow=True, poll_interval=0.01,
                              is_cancelled=lambda: bool(cancelled), sleep=lambda _s: None))
    assert pages == []
    assert cache.get_cursor("s") is None and cache.load_entries("s") == []


def test_a_cancelled_tail_persists_neither_an_empty_page_nor_a_missing_head(tmp_path):
    # The screen closed during the read: an empty page's new cursor, and the
    # "head missing" mark from a cursorless window, must not write the job back.
    cache = LogCache("late", root=tmp_path)
    cache.set_cursor("empty", "old")
    cancelled = []

    def fetch(cursor):
        cancelled.append(True)
        if cursor == "old":
            return {"entries": [], "next_cursor": "new"}
        return {"entries": [{"n": 1}, {"n": 2}], "next_cursor": "window-end"}

    pages = list(cached_pages(cache, "empty", fetch, entries_key="entries", follow=True, poll_interval=0.01,
                              is_cancelled=lambda: bool(cancelled), sleep=lambda _s: None))
    assert pages == [] and cache.get_cursor("empty") == "old"

    cancelled.clear()
    pages = list(cached_pages(cache, "window", fetch, entries_key="entries", follow=False, poll_interval=0.01,
                              tail_window=2, is_cancelled=lambda: bool(cancelled), sleep=lambda _s: None))
    assert pages == []
    assert cache.get_cursor("window") is None and cache.load_entries("window") == []
    assert not cache.head_missing("window")


def test_a_deployment_s_serving_folder_is_the_cache_s(tmp_path):
    job = _job(tmp_path, "deploy", mib=3, age=5000)
    (job / "serving" / "s").mkdir(parents=True)
    (job / "serving" / "s" / "stdout.log").write_bytes(b"x" * MIB)
    assert _names(cache_cleanup.enforce_cache_limit(tmp_path, limit=1 * MIB)) == ["deploy"]
