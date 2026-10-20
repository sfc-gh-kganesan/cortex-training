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

"""Saved-console helpers for the TUI: a bounded tail reader.

A finished job's console can be gigabytes, so the log pane only ever reads the
end of ``stdout.log``. Downloads into the shared per-job cache are serialized
with :class:`cortex_training._download_lock.DownloadLock`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cortex_training._artifact_download import read_manifest
from cortex_training.tui.format import entry_matches

TAIL_MAX_LINES = 20000
TAIL_CHUNK_BYTES = 1 << 20
TAIL_MAX_BYTES = 64 << 20
TAIL_MAX_LINE_BYTES = 8192


@dataclass(frozen=True)
class SavedTail:
    """Matching lines from the end of a saved console, oldest first."""

    lines: list[str]
    # True when reading stopped before the start of the file.
    truncated: bool
    size: int


def committed_size(path: Path, size: int) -> int:
    """Bytes committed by a valid resume manifest, else the whole file."""
    manifest = read_manifest(path.with_name("." + path.name + ".manifest.json"))
    return manifest.nbytes if manifest is not None and manifest.nbytes <= size else size


def _decode(raw: bytes, *, cut: bool) -> str:
    if not cut:
        return raw.decode("utf-8", errors="replace")
    # The left edge may split a multi-byte character; drop its continuation bytes.
    i = 0
    while i < len(raw) and i < 3 and 0x80 <= raw[i] <= 0xBF:
        i += 1
    return "…" + raw[i:].decode("utf-8", errors="replace")


def read_saved_tail(
    path: Path,
    query: str = "",
    *,
    max_lines: int = TAIL_MAX_LINES,
    chunk_bytes: int = TAIL_CHUNK_BYTES,
    max_bytes: int = TAIL_MAX_BYTES,
    max_line_bytes: int = TAIL_MAX_LINE_BYTES,
) -> SavedTail | None:
    """Return up to ``max_lines`` lines matching ``query`` from the end of ``path``.

    Lines end at ``\n``. A ``\r`` inside a line returns the cursor to column 0,
    as in a terminal, so a line rewritten with ``\r`` (a progress bar) shows only
    its last non-empty segment and ``\r\n`` is an ordinary line end. The filter
    applies to that shown text.

    The file is scanned backwards once, in ``chunk_bytes`` steps, stopping once
    enough lines match or ``max_bytes`` have been read. Only bytes the download
    has committed are read. The line cut by the window start is dropped unless
    it is the only one; a line longer than ``max_line_bytes`` shows its last
    ``max_line_bytes`` bytes after a leading ``…``. Returns ``None`` when the
    file is missing or empty.
    """
    try:
        handle = path.open("rb")
    except OSError:
        return None
    with handle:
        size = committed_size(path, handle.seek(0, 2))
        if size == 0:
            return None
        end = size
        handle.seek(end - 1)
        if handle.read(1) == b"\n":
            end -= 1  # the final newline terminates the last line; it starts none
        kept: list[str] = []  # newest first
        capped = False
        # The \r-segment being collected (right to left), capped to its last
        # ``max_line_bytes`` bytes.
        parts: list[bytes] = []
        seg_len = 0
        shown = False  # the current line already produced its shown text

        def take(piece: bytes) -> None:
            nonlocal seg_len
            room = max_line_bytes - min(seg_len, max_line_bytes)
            if room > 0 and piece:
                parts.append(piece[-room:])
            seg_len += len(piece)

        def emit(*, cut: bool = False) -> None:
            nonlocal capped, shown, seg_len
            raw = b"".join(reversed(parts))
            parts.clear()
            text = _decode(raw, cut=cut or seg_len > max_line_bytes)
            seg_len = 0
            shown = True
            if not entry_matches(text, query):
                return
            if len(kept) >= max_lines:
                capped = True
                return
            kept.append(text)

        def drop_segment() -> None:
            nonlocal seg_len
            parts.clear()
            seg_len = 0

        pos = end
        read = 0
        while pos > 0 and not capped and read < max_bytes:
            start = max(0, pos - chunk_bytes, end - max_bytes)
            handle.seek(start)
            data = handle.read(pos - start)
            read += pos - start
            pos = start
            i = len(data)
            # Cached rfind results; each is recomputed only once ``i`` passes it,
            # so every byte is searched at most once per separator.
            nl = cr = i + 1
            while i > 0 and not capped:
                if nl >= i:
                    nl = data.rfind(b"\n", 0, i)
                if shown:
                    if nl < 0:
                        i = 0
                        break
                    i = nl
                    shown = False  # the next line to the left starts here
                    drop_segment()
                    continue
                if cr >= i:
                    cr = data.rfind(b"\r", 0, i)
                sep = max(nl, cr)
                take(data[sep + 1 : i])
                if sep < 0:
                    i = 0
                    break
                i = sep
                if seg_len:
                    emit()
                else:
                    drop_segment()
                if sep == nl:
                    if not shown:
                        emit()  # an empty line
                    shown = False
        if pos == 0 and not capped:
            if not shown:
                emit()  # the first line of the file
        elif not capped and not shown and seg_len and not kept:
            emit(cut=True)  # one line longer than the window: show its end
        kept.reverse()
        return SavedTail(lines=kept, truncated=capped or pos > 0, size=size)
