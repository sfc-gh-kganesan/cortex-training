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

"""Pure formatting helpers for the Cortex Training TUI.

Kept free of any ``textual`` import so they can be unit-tested without the
optional TUI dependency installed.
"""

from __future__ import annotations

import functools
import json
import math
import re
from datetime import datetime
from datetime import timezone
from typing import Any

from rich.cells import cell_len


def finite_number(value: Any) -> float | None:
    """``value`` as a finite float (a number or a numeric string); None for a bool, NaN,
    Infinity (which json.loads reads), an int too big for a float, or anything else."""
    if isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return f if math.isfinite(f) else None


def short_seconds(seconds: float) -> str:
    """A latency: ``420ms`` under a second, ``1.92s`` from one."""
    return f"{seconds * 1000:.0f}ms" if seconds < 0.9995 else f"{seconds:.2f}s"


def _drain_summary(entry: dict) -> str:
    """A serving drain's numbers, which its line alone hides: ``16 req · p50 50ms · 2 replicas``."""
    done = finite_number(entry.get("completed_requests"))
    if done is None:
        return ""
    parts = [f"{int(done)} req"]
    for key, label in (("latency_s_p50", "p50"), ("ttft_s_p99", "ttft p99")):
        value = finite_number(entry.get(key))
        if value is not None:
            parts.append(f"{label} {short_seconds(value)}")
    replicas, errors = finite_number(entry.get("replica_count")), finite_number(entry.get("replica_errors"))
    if replicas is not None:
        parts.append(f"{int(replicas)} replica{'' if replicas == 1 else 's'}")
    if errors is not None and errors > 0:
        parts.append(f"{int(errors)} replica error{'' if errors == 1 else 's'}")
    return " · ".join(parts)


def format_log_entry(entry: Any) -> str:
    """Render one log entry (a parsed JSONL object or ``{"_raw": ...}``)."""
    if not isinstance(entry, dict):
        return str(entry)
    if "_raw" in entry:
        text = str(entry["_raw"])
        return text + " …[truncated]" if entry.get("_truncated") else text
    ts = str(entry.get("ts", "")).strip()
    level = str(entry.get("level", "")).strip()
    logger = str(entry.get("logger", "")).strip()
    msg = entry.get("msg")
    if msg is None:
        msg = entry.get("message")
    if msg is None:
        # Unknown shape: show the raw JSON so nothing is silently dropped.
        msg = json.dumps(entry, sort_keys=True)
    if msg == "drain_metrics":
        summary = _drain_summary(entry)
        if summary:
            msg = f"{msg}  {summary}"
    parts = []
    if ts:
        parts.append(ts)
    if level:
        parts.append(f"{level:<5}")
    if logger:
        parts.append(f"{logger}:")
    parts.append(str(msg))
    line = " ".join(parts)
    if entry.get("_truncated"):
        line += " …[truncated]"
    return line


def _cells(text: str) -> int:
    return len(text) if text.isascii() else cell_len(text)


def cut_cells(text: str, room: int) -> str:
    """The longest start of ``text`` that fits in ``room`` terminal cells, never
    splitting a character; ``""`` when even the first one is wider.

    Each candidate start is measured whole, so wide characters count two cells
    and combining marks or joined emoji stay with the character before them.
    (Not ``rich.cells.chop_cells``: before rich 13.7 it returns its pieces in
    reverse order.)
    """
    if room <= 0 or not text:
        return ""
    if text.isascii() and text[:room + 1].isprintable():  # a cell per character
        return text[:room]
    # text[:fits] fits; text[:too_long] doesn't (or is all of it). Grow the
    # bound first, so a long line costs about one row's worth of measuring.
    fits, too_long = 0, min(len(text), room + 1)
    while too_long < len(text) and cell_len(text[:too_long]) <= room:
        fits, too_long = too_long, min(len(text), too_long * 2)
    if cell_len(text[:too_long]) <= room:
        return text[:too_long]
    while too_long - fits > 1:
        mid = (fits + too_long) // 2
        if cell_len(text[:mid]) <= room:
            fits = mid
        else:
            too_long = mid
    return text[:fits]


def _wrap_pieces(text: str, room: int, later: int) -> list[str]:
    """``text`` (one line) cut into rows of ``room`` cells, then ``later`` cells.

    Each step measures about one row of the text, never all that is left: a
    64 KiB line costs its length once, not once per row.
    """
    pieces: list[str] = []
    while True:
        head = cut_cells(text, room) or text[:1]  # a wide character in a one-cell row still shows
        if len(head) >= len(text):
            break  # the rest fits
        if text[len(head):len(head) + 1] == " ":
            piece, text = head.rstrip(" "), text[len(head) + 1:]  # the row ends at a word: drop that space
        else:
            space = head.rfind(" ")
            if space > 0 and head[:space].strip():  # a break between words, not inside the indentation
                piece, text = head[:space].rstrip(" "), text[space + 1:]
            else:
                piece, text = head, text[len(head):]  # one long token: break inside it
        pieces.append(piece)
        room = later
    pieces.append(text)
    return pieces


def wrap_log_line(text: Any, width: int, *, first: str = "", rest: str = "", max_rows: int = 0) -> list[str]:
    """Cell-aware word wrapping for the non-wrapping ``Log`` widget.

    ``first`` and ``rest`` prefix the first and continuation rows. Long tokens
    are hard-broken; newlines and indentation are preserved. ``max_rows``
    retains the beginning and end around an omission marker.
    """
    lines = str(text).split("\n")
    if width <= 0:
        return [first + lines[0], *(rest + line for line in lines[1:])]
    first_room = max(1, width - _cells(first))
    rest_room = max(1, width - _cells(rest))
    rows: list[str] = []
    for n, logical in enumerate(lines):
        prefix, room = (first, first_room) if n == 0 else (rest, rest_room)
        if logical.isascii() and len(logical) <= room:
            rows.append(prefix + logical)  # most lines: no wrapping, and no measuring
        else:
            pieces = _wrap_pieces(logical, room, rest_room)
            rows.append(prefix + pieces[0])
            rows.extend(rest + piece for piece in pieces[1:])
    if max_rows > 1 and len(rows) > max_rows:
        head = (max_rows - 1) // 2
        tail = max_rows - 1 - head
        hidden = len(rows) - head - tail
        rows[head:len(rows) - tail] = [rest + _more_rows_note(hidden, rest_room)]
    return rows


def _more_rows_note(hidden: int, room: int) -> str:
    """The one row that stands for a long line's ``hidden`` middle rows: the
    longest of its wordings that fits in ``room`` cells (a narrow pane loses
    the hint first, then the words, and keeps the count while it can)."""
    n = f"{hidden:,}"
    for note in (
        f"… {n} more row(s) of this line (s saves the whole log)",
        f"… {n} more row(s) (s saves the whole log)",
        f"… {n} more row(s)",
        f"… {n} more",
        f"… {n}",
    ):
        if _cells(note) <= room:
            return note
    return "…"


# ─── Text safe to draw ───────────────────────────────────────────────────────

# One rule for the text the TUI draws and didn't write (log lines, and the
# strings in the service's job records): _drawable. printable, safe_text (how
# the Logs pane draws a line) and one_line (one row) all go through it.

# A terminal escape sequence, from its ESC: CSI (colours, cursor moves), OSC
# (clipboard and title writes) up to BEL or ST, or to the end of the line when
# unterminated, DCS / SOS / PM / APC up to ST, and two-byte escapes such as
# ESC 7 or ESC ( B. A terminal acts on these instead of showing them. (The
# 8-bit forms start with a C1 control, which _CONTROL_CHARS replaces.)
_ESCAPE = re.compile(
    r"\x1b(?:\[[0-?]*[ -/]*[@-~]"
    r"|\][^\x07\x1b\n]*(?:\x07|\x1b\\)?"
    r"|[PX^_][^\x1b\n]*(?:\x1b\\)?"
    r"|[ -/]*[0-~])"
)
# What is left that a terminal would act on, or that reorders or hides the text
# around it: C0 and C1 controls other than newline, the Unicode bidi marks,
# embeddings, overrides and isolates, and the invisible zero-width space, word
# joiner and byte-order mark. Each is shown as U+FFFD. (Zero-width joiners stay
# where scripts use them; between emoji, see _JOINED_EMOJI.)
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f\u061c\u200b\u200e\u200f\u202a-\u202e\u2060\u2066-\u2069\ufeff]")
# A zero-width joiner between two emoji (a technologist, a family, a rainbow flag)
# makes one glyph where the terminal joins them and two where it doesn't (tmux and
# many terminals), and the TUI can't tell which: it counts one, a terminal that
# draws two runs the row past its columns and leaves stale cells. Without the
# joiner every terminal draws the emoji side by side, at the width counted.
_EMOJI = "\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF"
_JOINED_EMOJI = re.compile(f"(?<=[{_EMOJI}\ufe0f])\u200d(?=[{_EMOJI}])")
# Combining marks from the generic diacritic blocks kept on one character. Real
# text stacks one or two; a run of hundreds (so-called zalgo text) is drawn by
# the terminal over the rows above and below it.
MAX_COMBINING_MARKS = 4
_MARK = "[\u0300-\u036f\u1ab0-\u1aff\u1dc0-\u1dff\u20d0-\u20ff\ufe20-\ufe2f]"
_EXCESS_MARKS = re.compile(f"({_MARK}{{{MAX_COMBINING_MARKS}}}){_MARK}+")
_LINE_BREAKS = re.compile("\r\n|[\r\n\u2028\u2029]")
_TAB_CELLS = 4


def _strip_escapes(s: str) -> str:
    """``s`` without its escape sequences; an ESC that starts none shows as U+FFFD."""
    if "\x1b" not in s:  # nearly every line: a C-speed scan
        return s
    out: list[str] = []
    pos = 0
    while (at := s.find("\x1b", pos)) >= 0:
        out.append(s[pos:at])
        match = _ESCAPE.match(s, at)
        if match is None:
            out.append("\ufffd")
            pos = at + 1
        else:
            pos = match.end()
    out.append(s[pos:])
    return "".join(out)


def _newlines(s: str) -> str:
    """``s`` with each ``\\r\\n`` and Unicode line or paragraph separator as a newline."""
    return s.replace("\r\n", "\n").replace("\u2028", "\n").replace("\u2029", "\n")


def _drawable(s: str) -> str:
    """Make text inert and drawable.

    Escapes are removed; tabs and line separators normalized; controls and
    invisible characters replaced; combining marks bounded.
    """
    s = _strip_escapes(s)
    if s.isascii() and s.isprintable():  # nearly every string: nothing to do
        return s
    s = _newlines(s).replace("\t", " " * _TAB_CELLS)
    # isprintable() is false for every character _CONTROL_CHARS matches, for the
    # joiner and for a newline: the regexes run only on the strings that may need them.
    if not s.isprintable():
        s = _CONTROL_CHARS.sub("\ufffd", s)
        if "\u200d" in s:
            s = _JOINED_EMOJI.sub("", s)
    return s if s.isascii() else _EXCESS_MARKS.sub(r"\1", s)


def printable(text: Any) -> str:
    """``text`` safe to draw (see :func:`_drawable`); a string with nothing to
    replace is returned as it is."""
    s = text if isinstance(text, str) else str(text)
    return s if s.isascii() and s.isprintable() else _printable(s)


_printable = functools.lru_cache(maxsize=4096)(_drawable)  # a list render cleans each job's text several times


def safe_text(text: Any) -> str:
    """Printable log text with tabs expanded and carriage-return rewrites resolved."""
    s = _strip_escapes(str(text))  # first: an escape sequence takes no cells
    if "\r" in s or "\u2028" in s or "\u2029" in s:
        s = _newlines(s)
        if "\r" in s:
            s = "\n".join(_overwritten(line) for line in s.split("\n"))
    if "\t" in s:
        s = s.expandtabs(_TAB_CELLS)
    return _drawable(s)


def _overwritten(line: str) -> str:
    """What a terminal shows of a line that rewrites itself with carriage
    returns (a progress bar): its last non-empty part."""
    parts = [part for part in line.split("\r") if part]
    return parts[-1] if parts else ""


def one_line(text: Any) -> str:
    """``text`` safe for a one-row line, as :func:`printable` makes it, with each
    line break as `` / `` and each tab as a space.

    Escape sequences go before the line breaks do, so one left unterminated
    ends with its own line, as it does in a log line.
    """
    return printable(_LINE_BREAKS.sub(" / ", _strip_escapes(str(text))).replace("\t", " "))


# ─── Log lines as the pane shows them ────────────────────────────────────────

# The service sends a live line as "<RFC 3339 timestamp> <line>"; Ray may put
# "(Actor pid=N, ip=...)" groups in front of a JSON log record.
_STAMP = re.compile(r"(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2})(?:[.,]\d+)?(Z|[+-]\d{2}:?\d{2})?(?: |$)")
_RECORD = re.compile(r"((?:\([^()\n]*\) )*)(\{.*\})\s*", re.DOTALL)
_RECORD_TIME_FIELDS = ("ts", "time", "timestamp", "asctime")
_RECORD_LEVEL_FIELDS = ("level", "levelname", "severity")
_RECORD_LOGGER_FIELDS = ("logger", "name")


@functools.lru_cache(maxsize=4096)
def _local_time(day: str, clock: str, zone: str | None) -> str:
    """``YYYY-MM-DD HH:MM:SS`` in local time, as the rest of the TUI shows times."""
    offset = "+00:00" if zone in (None, "Z") else (zone if ":" in zone else f"{zone[:3]}:{zone[3:]}")
    try:
        moment = datetime.fromisoformat(f"{day}T{clock}{offset}")
    except ValueError:
        return f"{day} {clock}"
    return moment.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _stamp_text(match: "re.Match[str]") -> str:
    return _local_time(match.group(1), match.group(2), match.group(3))


def _record_time(record: dict) -> str:
    for field in _RECORD_TIME_FIELDS:
        value = record.get(field)
        if isinstance(value, str) and (match := _STAMP.match(value.strip() + " ")):
            return _stamp_text(match)
    return ""


def _first_field(record: dict, fields: tuple) -> str:
    for field in fields:
        value = record.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _record_lines(record: dict, when: str, actors: str) -> str:
    """A JSON log record as ``time [actors] LEVEL logger: message``, with its
    error and exception chain on the lines below, as the execution log shows them."""
    msg = record.get("msg", record.get("message"))
    head = [when or _record_time(record), actors]
    if level := _first_field(record, _RECORD_LEVEL_FIELDS):
        head.append(f"{level:<5}")
    if logger := _first_field(record, _RECORD_LOGGER_FIELDS):
        head.append(f"{logger}:")
    head.append(msg if isinstance(msg, str) else json.dumps(msg, sort_keys=True))
    if msg == "drain_metrics" and (summary := _drain_summary(record)):
        head[-1] = f"{msg}  {summary}"
    lines = [" ".join(part for part in head if part)]
    for field in _ERROR_FIELDS:
        text = _failure_text(record.get(field)).strip()
        if text and text not in lines[0]:
            lines.append(f"    {field}: {text}")
    for field in _EXCEPTION_FIELDS:
        text = _failure_text(record.get(field)).rstrip()
        if text.strip():
            lines.extend(f"    {line}" for line in text.split("\n"))
    return "\n".join(lines)


def entry_clock(entry: Any) -> str:
    """A log entry's own time as local ``HH:MM:SS``: a raw line's leading
    timestamp, or a structured entry's ``ts``; ``""`` when it has neither."""
    if isinstance(entry, dict):
        raw = entry.get("_raw")
        if isinstance(raw, str):
            match = _STAMP.match(_strip_escapes(raw[:64]))
            return _stamp_text(match)[11:] if match else ""
        return _record_time(entry)[11:]
    return ""


def display_line(text: Any) -> str:
    """Format one inert log line, localizing timestamps and expanding JSON records."""
    s = _strip_escapes(str(text))
    when = ""
    if match := _STAMP.match(s):
        when, s = _stamp_text(match), s[match.end():]
    found = _RECORD.fullmatch(s) if s.startswith(("{", "(")) else None
    if found is not None:
        try:
            record = json.loads(found.group(2))
        except ValueError:
            record = None
        if isinstance(record, dict) and ("msg" in record or "message" in record):
            # The record's own fields may carry escapes too ("\u001b" in JSON).
            return safe_text(_record_lines(record, when, found.group(1).strip()))
    s = safe_text(s)  # carriage returns resolve after the timestamp, which stays
    return f"{when} {s}" if when else s


def display_day(text: str) -> str | None:
    """The ``YYYY-MM-DD`` a display line (see :func:`display_line`) starts with, or None.

    Checked by position, not a regex: the pane asks this of every line it writes."""
    if len(text) >= 19 and text[4] == "-" and text[10] == " " and text[13] == ":" and text[16] == ":" \
            and (len(text) == 19 or text[19] == " ") and text[:4].isdigit() and text[11:13].isdigit():
        return text[:10]
    return None


def compact_stamp(text: str) -> str:
    """A display line as the pane draws it: ``HH:MM:SS``, its date on the day's rule
    (see :func:`day_rule`). ``s``, a copy, and a highlight over whole lines take each
    line as it came, its date too (see :func:`copied_text`)."""
    return text[11:] if display_day(text) else text


_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_DAY_RULE = re.compile(r"── [A-Z][a-z]{2} \d{1,2} [A-Z][a-z]{2} \d{4} ─*")


def day_rule(day: str, width: int) -> str:
    """The pane's row above the first line of ``day`` (``YYYY-MM-DD``): ``── Sun 18 Oct 2026 ───…``,
    ruled across ``width`` cells (at least its words, when the pane isn't laid out yet)."""
    try:
        moment = datetime.strptime(day, "%Y-%m-%d")
    except ValueError:
        words = day
    else:  # in English whatever the locale, as every other word the TUI shows
        words = f"{_WEEKDAYS[moment.weekday()]} {moment.day} {_MONTHS[moment.month - 1]} {moment.year}"
    head = f"── {words} "
    return head + "─" * max(2, width - len(head))


def is_day_rule(row: str) -> bool:
    """Whether a pane row is one :func:`day_rule` drew (checked cheaply first: every row is asked)."""
    return row.startswith("── ") and _DAY_RULE.fullmatch(row) is not None


def copied_text(rows: list, line_rows: list, whole, shown, start=None, end=None, *, indent: str = "",
                rules=()) -> str:
    """What a highlight over the Logs pane copies: the lines as they were logged.

    ``rows`` are the pane's rows and ``line_rows`` how many of them each line takes, in
    order; ``whole(i)`` is line ``i`` as it came (what ``s`` saves) and ``shown(i)`` the
    text its rows were wrapped from (the time without its date). ``start`` and ``end``
    are the highlight's ``(row, column)`` ends, None for the pane's first and last;
    ``rules`` the rows that are a day's rule (by the pane's row kinds, not their text).

    A line highlighted whole, from the start of its first row to the end of its last,
    is copied whole; one highlighted in part (where the highlight begins or ends) gives
    the part, as its text has it: the spaces a wrap took and the later rows' ``indent``
    left out, its own line breaks kept. A day's rule is never copied.
    """
    count = len(rows)
    if not count:
        return ""
    first_row, first_col = start if start is not None else (0, 0)
    last_row, last_col = end if end is not None else (count - 1, len(rows[-1]))
    if last_row >= count:
        last_row, last_col = count - 1, len(rows[-1])
    out: list = []
    top = 0
    for index, taken in enumerate(line_rows):
        a, b = top, top + taken - 1  # the line's rows
        top += taken
        if b < first_row:
            continue
        if a > last_row:
            break
        if taken > 1 and a in rules:
            a += 1  # its text starts under its day's rule
        if b < first_row or a > last_row:
            continue  # only the rule is highlighted
        begins = first_row < a or (first_row == a and first_col <= 0)
        ends = last_row > b or (last_row == b and last_col >= len(rows[b]))
        if begins and ends:
            out.append(str(whole(index)))
            continue
        out.append(_part_of(shown(index), rows, a, b, (first_row, first_col) if not begins else None,
                            (last_row, last_col) if not ends else None, indent))
    return "\n".join(out)


def _part_of(text: str, rows: list, a: int, b: int, start, end, indent: str) -> str:
    """The part of a line's ``text`` that rows ``a``..``b`` show from ``start`` to ``end``
    (``(row, column)``; None: from its start, or to its end). Each row is found in the
    text in turn, so the text's own spaces and line breaks come back (and the rows a very
    long line leaves out, between two that are found). A highlight that ends on a row
    that isn't in the text (the one that says how many rows are left out) takes the
    highlighted rows as shown, without their indents."""
    pieces: list = []  # (where in text it starts, or -1; the row without its indent; columns cut)
    at = 0
    for row in range(a, b + 1):
        piece, cut = rows[row], 0
        if row > a and indent and piece.startswith(indent):
            piece, cut = piece[len(indent):], len(indent)
        found = text.find(piece, at) if piece else at
        pieces.append((found, piece, cut))
        if found >= 0:
            at = found + len(piece)

    def offset(row: int, col: int) -> int | None:
        found, piece, cut = pieces[row - a]
        return None if found < 0 else found + min(len(piece), max(0, col - cut))

    lo = 0 if start is None else offset(*start)
    hi = len(text) if end is None else offset(*end)
    if lo is not None and hi is not None:
        return text[lo:hi]
    # Some row isn't in the text: the highlighted rows as they are, without their indents.
    parts = []
    for row in range(max(a, start[0] if start else a), min(b, end[0] if end else b) + 1):
        _found, piece, cut = pieces[row - a]
        from_col = max(0, start[1] - cut) if start and row == start[0] else 0
        to_col = max(0, end[1] - cut) if end and row == end[0] else len(piece)
        parts.append(piece[from_col:to_col])
    return "\n".join(parts)


def wrap_lines(lines: list, width: int, *, first: str = "", rest: str = "", max_lines: int, max_rows: int,
               max_rows_per_line: int = 0) -> tuple[list, dict, bool]:
    """Log lines as the pane shows them, wrapped from the newest back as far as
    ``max_lines`` lines and ``max_rows`` rows reach.

    Returns ``(chunks, displays, cut)``: ``(line, its rows, row count)`` per
    kept line, oldest first; each kept line's display text; and whether older
    lines didn't fit. Pure, so it runs off the UI thread.
    """
    chunks: list = []
    displays: dict = {}
    total = 0
    for line in reversed(lines):
        if len(chunks) >= max_lines:
            break
        text = displays.get(line)
        if text is None:
            text = displays[line] = display_line(line)
        rows = wrap_log_line(compact_stamp(text), width, first=first, rest=rest, max_rows=max_rows_per_line)
        if chunks and total + len(rows) > max_rows:
            break
        chunks.append((line, rows, len(rows)))
        total += len(rows)
    chunks.reverse()
    return chunks, displays, len(chunks) < len(lines)


def line_moment(text: Any) -> datetime | None:
    """When a log line was written: its leading RFC 3339 timestamp, or a JSON
    record's own ``ts`` (after any ``(Actor pid=...)`` groups); None if neither."""
    s = _strip_escapes(str(text))
    match = _STAMP.match(s)
    if match is None:
        found = _RECORD.fullmatch(s) if s.startswith(("{", "(")) else None
        if found is None:
            return None
        try:
            record = json.loads(found.group(2))
        except ValueError:
            return None
        value = next((record.get(f) for f in _RECORD_TIME_FIELDS if isinstance(record, dict)
                      and isinstance(record.get(f), str)), None)
        match = _STAMP.match(value.strip() + " ") if value else None
        if match is None:
            return None
    day, clock, zone = match.group(1), match.group(2), match.group(3)
    offset = "+00:00" if zone in (None, "Z") else (zone if ":" in zone else f"{zone[:3]}:{zone[3:]}")
    try:
        return datetime.fromisoformat(f"{day}T{clock}{offset}")
    except ValueError:
        return None


def unstamped(line: str) -> str:
    """A log line without its leading timestamp: how a live line and the same
    line in a saved console (which may not carry one) compare."""
    match = _STAMP.match(line)
    return (line[match.end():] if match else line).rstrip()


def lines_after(saved_tail: list, live: list) -> list | None:
    """The lines of ``live`` (a tail's newest lines, in log order) that come
    after ``saved_tail`` (the last lines of a saved console of the same log,
    which may lag behind the tail).

    Lines are matched on their text without a leading timestamp: live lines
    carry one, saved consoles may not. ``None`` when the two can't be lined up.
    """
    want = [unstamped(line) for line in saved_tail if str(line).strip()]
    if not want or not live:
        return None
    keys = [unstamped(str(line)) for line in live]
    k = len(want)
    for end in range(len(keys), k - 1, -1):
        if keys[end - 1] == want[-1] and keys[end - k:end] == want:
            return list(live[end:])
    return None


def source_label(source: Any) -> str:
    """Side-panel label for a sub-job source.

    A source is a sub-job: ``{"sub_job_id", "job_type"}`` (or just a sub_job_id
    string). Rendered as ``"<type> #<index>"`` — the type comes from job_type
    when present, else the middle field of the ``<uuid>:<type>:<index>`` id.
    """
    sjid = ""
    job_type = None
    if isinstance(source, dict):
        sjid = str(source.get("sub_job_id") or source.get("id") or source.get("source_id") or "")
        job_type = source.get("job_type") or source.get("type")
    else:
        sjid = str(source)
    parts = sjid.split(":")
    if not job_type and len(parts) >= 2:
        job_type = parts[1]  # <uuid>:<type>:<index>
    idx = parts[2] if len(parts) >= 3 else ""
    if job_type:
        return f"{job_type} #{idx}" if idx != "" else str(job_type)
    if isinstance(source, dict) and source.get("label"):
        return str(source["label"])
    return sjid or str(source)


_ACTIVE_STATUSES = {
    "running",
    "initializing",
    "pending",
    "placing",
    "queued",
    "creating",
}

_TERMINAL_STATUSES = {
    "failed",
    "cancelled",
    "canceled",
    "terminated",
    "completed",
    "succeeded",
    "done",
    "error",
}


def is_active_status(status: Any) -> bool:
    """Whether a job is still live, i.e. its log tail should keep following.

    Missing/unknown statuses default to active so a running-but-unclassified job
    never stops tailing; only a recognized terminal status stops the follow.
    """
    if not status:
        return True
    s = str(status).lower().removeprefix("job_state_")
    if s in _ACTIVE_STATUSES:
        return True
    if s in _TERMINAL_STATUSES:
        return False
    return True


# Times a service sends for one that was never set: Go's zero time (0001-01-01),
# the Unix epoch, and a year-9999 "never". No job ran then.
_UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_NEVER_YEAR = 9999


def parse_ts(raw: Any) -> datetime | None:
    """An ISO-8601 timestamp (nanosecond precision allowed) as an aware UTC datetime.

    A time without a zone is UTC. None when it is missing or unparseable, or
    out of range in UTC (``0001-01-01T00:00:00+01:00``): such a time can't be
    shown, sorted or compared, and one record holding it must never stop a
    screen. None too for a placeholder of a time that was never set (at or
    before the Unix epoch, or in the year 9999, Go's zero time included): it is
    shown as a dash, with no age or duration computed from it.
    """
    if not raw:
        return None
    s = str(raw).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    # Python's fromisoformat takes at most 6 fractional digits.
    s = re.sub(r"(\.\d{6})\d+", r"\1", s)
    try:
        dt = datetime.fromisoformat(s)
        dt = (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)
    except (ValueError, OverflowError):
        return None
    if dt <= _UNIX_EPOCH or dt.year >= _NEVER_YEAR:
        return None
    return dt


def local_time(dt: datetime, fmt: str = "%Y-%m-%d %H:%M") -> str:
    """``dt`` in the local zone as ``fmt``; ``""`` when the local zone puts it out of
    range (year 1 west of UTC, 9999-12-31 east of it)."""
    try:
        return dt.astimezone().strftime(fmt)
    except (OverflowError, ValueError, OSError):
        return ""


def format_created_at(raw: Any) -> str:
    """Render an ISO-8601 created_at (e.g. ``2026-06-13T18:49:54Z``) as local
    ``YYYY-MM-DD HH:MM``. Returns ``""`` for missing, unparseable or out-of-range values."""
    dt = parse_ts(raw)
    return local_time(dt) if dt else ""


# ─── Job summary (whitelisted create-job params) ─────────────────────────────

# Only the parameters a customer passes to SubJobConfig.training_job /
# sampling_job are surfaced — short label + the config key it reads.
_SUMMARY_CONFIG_KEYS = [
    ("n_gpus", "gpus"),
    ("max_seq_len", "seq"),
    ("train_batch_size", "bs"),
    ("gradient_clipping", "clip"),
]
_SUMMARY_SUBJOB_KEYS = [
    ("global_batch_size", "gbs"),
    ("dtype", "dtype"),
    ("seed", "seed"),
]


def _num(v: Any) -> str:
    """Render a value compactly: integral floats lose the .0 (n_gpus 1.0 -> 1)."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _subjob_config(sj: dict) -> dict:
    for key in ("training_config", "inference_config", "sampling_config"):
        cfg = sj.get(key)
        if isinstance(cfg, dict):
            return cfg
    return {}


def _job_config_segments(job: dict) -> list:
    """Per-sub-job static config segments (type / model / whitelisted params)."""
    segs = []
    for sj in job.get("sub_jobs") or []:
        if not isinstance(sj, dict):
            continue
        seg = []
        jt = str(sj.get("job_type") or sj.get("type") or "").lower()
        if jt:
            seg.append(jt)
        if sj.get("model_name"):
            seg.append(str(sj["model_name"]))
        cfg = _subjob_config(sj)
        for key, label in _SUMMARY_CONFIG_KEYS:
            if cfg.get(key) is not None:
                seg.append(f"{label}={_num(cfg[key])}")
        opt = cfg.get("optimizer")
        if isinstance(opt, dict):
            o = [str(opt["name"])] if opt.get("name") else []
            if opt.get("lr") is not None:
                o.append(f"lr={_num(opt['lr'])}")
            if o:
                seg.append("opt=" + "/".join(o))
        for key, label in _SUMMARY_SUBJOB_KEYS:
            if sj.get(key) is not None:
                seg.append(f"{label}={_num(sj[key])}")
        if seg:
            segs.append(" ".join(seg))
    return segs


def format_job_config(job: Any) -> str:
    """Static create-time config only (no status / reason) — for the line we
    reserve to describe the job. Shows only whitelisted params that are present,
    so it stays compact."""
    if not isinstance(job, dict):
        return ""
    return "  ·  ".join(_job_config_segments(job))


# ─── Search / filter helpers ─────────────────────────────────────────────────


def entry_matches(entry: Any, query: str) -> bool:
    """Whether a log entry's rendered text contains query (case-insensitive)."""
    if not query:
        return True
    return query.lower() in format_log_entry(entry).lower()


_LEVELS = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "WARN": 30,
    "ERROR": 40,
    "CRITICAL": 50,
    "FATAL": 50,
}


# The service sends live lines as {"_raw": "<RFC3339 ts> <line>"}, where the line
# is often a JSON log record and otherwise plain text with a level word.
_RAW_JSON_LEVEL = re.compile(r'"(?:level|levelname|severity)"\s*:\s*"([A-Za-z]+)"')
_RAW_WORD_LEVEL = re.compile(r"\b(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|FATAL)\b")


def _entry_level(entry: Any) -> str | None:
    if not isinstance(entry, dict):
        return None
    if entry.get("level"):
        return str(entry["level"])
    raw = entry.get("_raw")
    if isinstance(raw, str):
        head = raw[:512]  # a level sits near the start; don't scan KB-long payloads
        match = _RAW_JSON_LEVEL.search(head) or _RAW_WORD_LEVEL.search(head)
        if match:
            return match.group(1)
    return None


# A raw line's JSON record that carries an error or an exception chain (a value
# that isn't null, false or empty), whatever its level says.
_RAW_FAILURE = re.compile(
    r'"(?:exc|exc_info|exception|traceback|stack|stack_trace|error|error_message|err)"\s*:'
    r'(?!\s*(?:null\b|false\b|""|\[\s*\]|\{\s*\}))')


def entry_at_level(entry: Any, min_level: Any) -> bool:
    """Whether an entry is at or above min_level. A raw line's level comes from
    its JSON ``level`` field or a level word near its start; a structured entry's, from ``level``.
    Entries without a parseable level always pass, so the filter only trims
    lower-severity logs and never hides other output. A record that carries an
    error or an exception counts as ERROR, whatever its level says."""
    if not min_level:
        return True
    floor = _LEVELS.get(str(min_level).upper())
    if floor is None:
        return True
    lvl = _entry_level(entry)
    if not lvl or _LEVELS.get(str(lvl).upper(), 0) >= floor:
        return True
    if floor > _LEVELS["ERROR"] or not isinstance(entry, dict):
        return False
    if "_raw" in entry:
        raw = entry["_raw"]
        return isinstance(raw, str) and "{" in raw and _RAW_FAILURE.search(raw) is not None
    return is_error_record(entry)


# ─── Execution log (``_logs/<sub_job_id>/execution.jsonl``) ──────────────────

# Fields of an execution-log record that carry a failure: an exception chain
# (multi-line traceback text) or a one-line error message.
_EXCEPTION_FIELDS = ("exc", "exc_info", "exception", "traceback", "stack", "stack_trace")
_ERROR_FIELDS = ("error", "error_message", "err")
_FAILURE_FIELDS = frozenset((*_EXCEPTION_FIELDS, *_ERROR_FIELDS))
_HEAD_FIELDS = frozenset(("ts", "level", "logger", "_truncated"))


def execution_log_records(content: Any) -> list:
    """Parse an ``execution.jsonl`` body: one record per line, torn lines skipped.

    A line that is valid JSON but not an object is kept as ``{"_raw": line}``.
    """
    records: list = []
    for line in str(content or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        records.append(record if isinstance(record, dict) else {"_raw": line})
    return records


def is_error_record(record: Any) -> bool:
    """An error-level record, or one that carries an exception or error message."""
    if not isinstance(record, dict):
        return False
    if _LEVELS.get(str(record.get("level") or "").upper(), 0) >= _LEVELS["ERROR"]:
        return True
    return any(record.get(field) for field in (*_EXCEPTION_FIELDS, *_ERROR_FIELDS))


def _failure_text(value: Any) -> str:
    """A failure field as text: a string as it is, a list (traceback lines) joined, anything else as JSON."""
    if value is None or value is False:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(str(item).rstrip("\n") for item in value)
    try:
        return json.dumps(value, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def _record_head(record: dict) -> str:
    """The first line of a record. One without a message shows its other fields
    as JSON, but not its failure fields: those follow, indented, once."""
    if "_raw" in record or record.get("msg") is not None or record.get("message") is not None:
        return format_log_entry(record)
    rest = {key: value for key, value in record.items() if key not in _FAILURE_FIELDS}
    extra = {key: value for key, value in rest.items() if key not in _HEAD_FIELDS}
    return format_log_entry({**rest, "msg": json.dumps(extra, sort_keys=True) if extra else ""}).rstrip()


def format_execution_record(record: Any) -> list:
    """Display lines for one execution-log record: the record itself, then its
    error message and exception chain, indented under it."""
    if not isinstance(record, dict):
        return [format_log_entry(record)]
    head = _record_head(record)
    out = [head]
    for field in _ERROR_FIELDS:
        text = _failure_text(record.get(field)).strip()
        if text and text not in head:
            first, *rest = text.splitlines()
            out.append(f"    {field}: {first}")
            out.extend(f"    {line}" for line in rest)
    for field in _EXCEPTION_FIELDS:
        text = _failure_text(record.get(field)).rstrip()
        if text.strip():
            out.extend(f"    {line}" for line in text.splitlines())
    return out
