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

"""Tests for the TUI's saved-console tail reader and download lock."""

import json
import threading
import time

import pytest

from cortex_training._download_lock import DownloadLock
from cortex_training.tui.saved_log import read_saved_tail


def _write(tmp_path, data: bytes):
    path = tmp_path / "stdout.log"
    path.write_bytes(data)
    return path


def test_missing_or_empty_file_returns_none(tmp_path):
    assert read_saved_tail(tmp_path / "absent.log") is None
    assert read_saved_tail(_write(tmp_path, b"")) is None


@pytest.mark.parametrize("chunk_bytes", [1, 3, 7, 64, 1 << 20])
def test_last_lines_across_chunk_boundaries(tmp_path, chunk_bytes):
    lines = [f"line-{i:03d}" for i in range(200)]
    path = _write(tmp_path, ("\n".join(lines) + "\n").encode())
    tail = read_saved_tail(path, max_lines=25, chunk_bytes=chunk_bytes)
    assert tail.lines == lines[-25:]
    assert tail.truncated is True
    assert tail.size == path.stat().st_size


@pytest.mark.parametrize("chunk_bytes", [2, 5, 1 << 20])
def test_file_smaller_than_window_returns_every_line(tmp_path, chunk_bytes):
    path = _write(tmp_path, b"first\n\nthird\nno-newline-at-end")
    tail = read_saved_tail(path, chunk_bytes=chunk_bytes)
    assert tail.lines == ["first", "", "third", "no-newline-at-end"]
    assert tail.truncated is False


def test_window_start_drops_the_partial_first_line(tmp_path):
    path = _write(tmp_path, b"aaaaaaaaaa\nbbbb\ncccc\n")
    # 12 bytes back from the final newline starts inside "aaaaaaaaaa".
    tail = read_saved_tail(path, chunk_bytes=4, max_bytes=12)
    assert tail.lines == ["bbbb", "cccc"]
    assert tail.truncated is True


def test_window_exactly_at_line_start_keeps_that_line(tmp_path):
    path = _write(tmp_path, b"aaaa\nbbbb\ncccc\n")
    # "bbbb\ncccc" is 9 bytes, but the reader cannot see the preceding newline
    # without reading one more byte, so the line at the cut is dropped.
    assert read_saved_tail(path, max_bytes=9).lines == ["cccc"]
    assert read_saved_tail(path, max_bytes=10).lines == ["bbbb", "cccc"]


def test_filter_is_applied_and_keeps_the_newest_matches(tmp_path):
    lines = [f"{'ERROR' if i % 10 == 0 else 'info'} {i}" for i in range(100)]
    path = _write(tmp_path, ("\n".join(lines) + "\n").encode())
    tail = read_saved_tail(path, "error", max_lines=3, chunk_bytes=16)
    assert tail.lines == ["ERROR 70", "ERROR 80", "ERROR 90"]
    assert tail.truncated is True
    every = read_saved_tail(path, "error", chunk_bytes=16)
    assert every.lines == [f"ERROR {i}" for i in range(0, 100, 10)]
    assert every.truncated is False
    assert read_saved_tail(path, "no-such-text").lines == []


@pytest.mark.parametrize("chunk_bytes", [1, 3, 1 << 20])
def test_carriage_returns_show_what_a_terminal_would(tmp_path, chunk_bytes):
    # "\r" rewinds to column 0: only the last non-empty segment of a line shows,
    # and "\r\n" is an ordinary line end.
    path = _write(
        tmp_path,
        b"crlf\r\nprogress 1\rprogress 2\r\n\r\nbar 10%\rbar 50%\rbar 100%\rbad \xff byte\r\n",
    )
    tail = read_saved_tail(path, chunk_bytes=chunk_bytes)
    assert tail.lines == ["crlf", "progress 2", "", "bad � byte"]
    assert tail.truncated is False


def test_filter_matches_the_shown_segment(tmp_path):
    path = _write(tmp_path, b"step 1 loss\rstep 2 done\nother\n")
    assert read_saved_tail(path, "done").lines == ["step 2 done"]
    assert read_saved_tail(path, "loss").lines == []


def test_carriage_return_only_tail_larger_than_the_window(tmp_path):
    head = b"".join(b"line %d\n" % i for i in range(1000))
    updates = b"\r".join(b"progress %07d" % i for i in range(300000))  # ~5 MiB
    path = _write(tmp_path, head + updates + b"\n")
    tail = read_saved_tail(path, chunk_bytes=64 << 10, max_bytes=1 << 20)
    # The last progress state shows even though its line starts outside the window.
    assert tail.lines == ["progress 0299999"]
    assert tail.truncated is True
    whole = read_saved_tail(path, chunk_bytes=64 << 10)
    assert whole.lines[-2:] == ["line 999", "progress 0299999"]
    assert len(whole.lines) == 1001


def test_carriage_return_scan_is_linear(tmp_path):
    updates = b"\r".join(b"%08d" % i for i in range(2_000_000))  # ~18 MiB, one line
    path = _write(tmp_path, b"first\n" + updates)
    started = time.monotonic()
    tail = read_saved_tail(path)
    elapsed = time.monotonic() - started
    assert tail.lines == ["first", "01999999"]
    assert elapsed < 5, f"scan took {elapsed:.1f}s"


def test_single_line_longer_than_the_window_shows_its_end(tmp_path):
    path = _write(tmp_path, b"x" * 5000 + b"END\n")
    tail = read_saved_tail(path, chunk_bytes=256, max_bytes=1024, max_line_bytes=100)
    assert tail.lines == ["…" + "x" * 97 + "END"]
    assert tail.truncated is True


def test_long_line_is_capped_for_display(tmp_path):
    path = _write(tmp_path, b"short\n" + "é".encode() * 300 + b"\nlast\n")
    tail = read_saved_tail(path, chunk_bytes=7, max_line_bytes=101)
    assert tail.lines[0] == "short"
    assert tail.lines[2] == "last"
    capped = tail.lines[1]
    assert capped.startswith("…") and "�" not in capped
    assert capped == "…" + "é" * 50


def test_reads_only_bytes_the_manifest_committed(tmp_path):
    path = _write(tmp_path, b"one\ntwo\ntorn app")
    manifest = tmp_path / ".stdout.log.manifest.json"
    manifest.write_text(json.dumps({"version": 1, "names": ["a.gz"], "bytes": 8}), encoding="utf-8")
    tail = read_saved_tail(path)
    assert tail.lines == ["one", "two"]
    assert tail.size == 8
    manifest.write_text(json.dumps({"version": 2, "names": ["a.gz"], "bytes": 8}), encoding="utf-8")
    assert read_saved_tail(path).lines == ["one", "two", "torn app"]
    manifest.write_text("not json", encoding="utf-8")
    assert read_saved_tail(path).lines == ["one", "two", "torn app"]
    manifest.write_text(json.dumps({"bytes": 999}), encoding="utf-8")
    assert read_saved_tail(path).lines == ["one", "two", "torn app"]


def test_invalid_utf8_is_replaced(tmp_path):
    path = _write(tmp_path, b"bad \xff byte\n")
    assert read_saved_tail(path, chunk_bytes=3).lines == ["bad � byte"]


def test_multibyte_characters_split_across_chunks(tmp_path):
    lines = ["héllo wörld ✓"] * 5
    path = _write(tmp_path, ("\n".join(lines) + "\n").encode())
    assert read_saved_tail(path, chunk_bytes=3).lines == lines


def test_download_lock_serializes_threads(tmp_path):
    root = tmp_path / "stdout"
    first = DownloadLock(root)
    assert first.acquire(lambda: True)
    waits = []
    acquired = threading.Event()

    def contender():
        second = DownloadLock(root)
        if second.acquire(lambda: True, on_wait=lambda: waits.append(1), poll_seconds=0.01):
            acquired.set()
            second.release()

    thread = threading.Thread(target=contender)
    thread.start()
    time.sleep(0.1)
    assert not acquired.is_set()
    assert waits == [1]
    first.release()
    thread.join(timeout=5)
    assert acquired.is_set()


def test_download_lock_gives_up_when_told_to_stop(tmp_path):
    root = tmp_path / "stdout"
    holder = DownloadLock(root)
    assert holder.acquire(lambda: True)
    tries = iter([True, True, False])
    assert DownloadLock(root).acquire(lambda: next(tries), poll_seconds=0.01) is False
    holder.release()
    again = DownloadLock(root)
    assert again.acquire(lambda: True)
    again.release()


def test_download_lock_waits_on_a_file_lock_held_elsewhere(tmp_path):
    fcntl = pytest.importorskip("fcntl")
    root = tmp_path / "stdout"
    lock_path = tmp_path / ".stdout.lock"
    # A separate open file description stands in for another TUI process.
    with open(lock_path, "a+b") as other:
        fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        tries = iter([True] * 5 + [False])
        waits = []
        lock = DownloadLock(root)
        assert lock.acquire(lambda: next(tries), on_wait=lambda: waits.append(1), poll_seconds=0.01) is False
        assert waits == [1]
        fcntl.flock(other.fileno(), fcntl.LOCK_UN)
    lock = DownloadLock(root)
    assert lock.acquire(lambda: True)
    lock.release()
