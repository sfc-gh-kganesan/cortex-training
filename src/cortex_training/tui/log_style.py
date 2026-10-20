# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Draw-time colours for plain-text log rows, tags, levels, notes, and matches."""

from __future__ import annotations

import re

from rich.highlighter import Highlighter
from rich.text import Text

MUTED = "#6b7280"
NOTE = "italic #9ca3af"
# A sub-job's colour, by its type: its tag on every row, its name in the sub-job list.
SUB_JOB_COLORS = {"training": "#22d3ee", "sampling": "#c084fc", "log_prob": "#fbbf24", "logprob": "#fbbf24"}
SUB_JOB_OTHER = "#60a5fa"
LOGGER = "#93c5fd"
KEY = "#9ca3af"
MATCH = "bold #111827 on #fbbf24"
_LEVELS = {
    "TRACE": (MUTED, None), "DEBUG": (MUTED, None), "INFO": ("#4ade80", None), "NOTICE": ("#4ade80", None),
    "WARN": ("bold #fbbf24", "#fde68a"), "WARNING": ("bold #fbbf24", "#fde68a"),
    "ERROR": ("bold #f87171", "#fca5a5"), "CRITICAL": ("bold #f87171", "#fca5a5"), "FATAL": ("bold #f87171", "#fca5a5"),
}

_TAG = re.compile(r"([A-Za-z_][\w-]*)(:\d+) │ ")  # a log line's first row: "training:0 │ "
_LATER = re.compile(r" +│ ")  # its later rows, lined up under its text
_HEAD = re.compile(r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?Z?)"
                   r"(?:\s+(TRACE|DEBUG|INFO|NOTICE|WARN(?:ING)?|ERROR|CRITICAL|FATAL)\b(?:\s+([^\s:]{1,60}:)(?=\s))?)?")
_PAIR = re.compile(r"(?<![\w./=])([A-Za-z_][\w./\-]{0,40})=(?=[^\s=])")
_PROBLEM = re.compile(r"\bTraceback \(most recent call last\):|\b[A-Z]\w*(?:Error|Exception)\b")
_NOTE = re.compile(r"— .+ —")


NOTE_KIND = "note"  # a row of one of the TUI's own notes (see LogHighlighter.row_kind)


def line_kind(text: str, *, note: bool = False) -> str | None:
    """What styles every row of a log line as its first: ``note`` for the TUI's own
    notes, else the level of a line that starts with a time and a level (``ERROR``)."""
    if note:
        return NOTE_KIND
    head = _HEAD.match(text)
    return head.group(2) if head else None


def sub_job_color(label: str) -> str:
    """The colour of a sub-job by its label or tag (``training #0``, ``sampling:1``)."""
    kind = re.split(r"[ :#]", label.strip(), maxsplit=1)[0].lower()
    return SUB_JOB_COLORS.get(kind, SUB_JOB_OTHER)


class LogHighlighter(Highlighter):
    """Style one log row; ``query`` marks case-insensitive filter matches."""

    def __init__(self) -> None:
        self.query = ""
        self.row_kind: str | None = None

    def highlight(self, text: Text) -> None:
        plain = text.plain
        if self.row_kind == NOTE_KIND or _NOTE.fullmatch(plain):
            text.stylize(NOTE)
            return
        start = 0  # where the line's own text starts, after the tag the pane adds
        if tag := _TAG.match(plain):
            text.stylize(f"bold {sub_job_color(tag.group(1))}", 0, tag.end(2))
            text.stylize(MUTED, tag.end(2), tag.end())
            start = tag.end()
        elif later := _LATER.match(plain):
            text.stylize(MUTED, 0, later.end())
            start = later.end()
            tint = _LEVELS.get(self.row_kind or "", (None, None))[1]
            if tint:  # a warning's or an error's message, on the line's later rows too
                text.stylize(tint, start, len(plain))
        body = start
        if head := _HEAD.match(plain, start):
            text.stylize(MUTED, head.start(1), head.end(1))
            level = head.group(2)
            if level:
                style, tint = _LEVELS[level]
                text.stylize(style, head.start(2), head.end(2))
                if tint:  # a warning's or an error's message, in the level's colour
                    text.stylize(tint, head.end(2), len(plain))
                if head.group(3):
                    text.stylize(LOGGER, head.start(3), head.end(3))
            start = head.end()
        for pair in _PAIR.finditer(plain, start):
            text.stylize(KEY, pair.start(), pair.end())
        for problem in _PROBLEM.finditer(plain, start):
            text.stylize("bold #f87171", problem.start(), problem.end())
        query = self.query.lower()
        lowered = plain.lower()
        if query and len(lowered) == len(plain):  # (a few characters lower-case to two)
            at = lowered.find(query, body)  # the filter matches the line, not the tag
            while at >= 0:
                text.stylize(MATCH, at, at + len(query))
                at = lowered.find(query, at + len(query))
