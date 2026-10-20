# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""TUI selection, screen dumps, and local clipboard integration.

Local tools complement OSC 52, except over SSH where they target the remote.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _commands() -> list:
    """This machine's clipboard tools, in the order to try them."""
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"):
        return []  # the clipboard here is the remote machine's, not the user's
    if sys.platform == "darwin":
        return [["pbcopy"]]
    commands = []
    if os.environ.get("WAYLAND_DISPLAY"):
        commands.append(["wl-copy"])
    if os.environ.get("DISPLAY"):
        commands += [["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]]
    return commands


def copy_local(text: str) -> bool:
    """Copy locally with a three-second tool timeout; false when unavailable."""
    target = os.environ.get("CORTEX_TRAINING_TUI_CLIPBOARD_FILE")
    if target:
        try:
            Path(target).write_text(text, encoding="utf-8")
            return True
        except OSError:
            return False
    for command in _commands():
        if shutil.which(command[0]) is None:
            continue
        try:
            # pbcopy reads its input in the locale's encoding (MacRoman without one): say UTF-8.
            subprocess.run(command, input=text.encode("utf-8"), check=True, timeout=3,
                           env={**os.environ, "LC_CTYPE": "UTF-8"},
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except (OSError, subprocess.SubprocessError):
            continue
    return False


def selected_text(screen) -> str | None:
    """Visible, still-valid highlighted text in layout order, or None."""
    parts: list[str] = []
    order = {widget: index for index, widget in enumerate(screen.query("*"))}
    for widget, selection in sorted(screen.selections.items(), key=lambda item: order.get(item[0], len(order))):
        if not widget.is_attached or not _shown(widget):
            continue
        try:
            found = widget.get_selection(selection)
        except (IndexError, ValueError):
            continue  # the selection's lines are gone from the text now shown
        if found is not None:
            parts.extend(found)
    return "".join(parts).strip("\n") or None  # (an empty widget at either end adds a bare line)


def _shown(widget) -> bool:
    """Displayed, with every container around it displayed too (not a hidden tab)."""
    node = widget
    while node is not None and hasattr(node, "display"):
        if not node.display:
            return False
        node = node.parent
    return True


def aligned(rows: list) -> list:
    """``rows`` (lists of cell texts) as lines, each column as wide as its widest cell."""
    from rich.cells import cell_len

    count = max((len(row) for row in rows), default=0)
    widths = [max((cell_len(row[i]) for row in rows if i < len(row)), default=0) for i in range(count)]
    return ["  ".join(cell + " " * (widths[i] - cell_len(cell)) for i, cell in enumerate(row)).rstrip()
            for row in rows]


def _table_text(table) -> list:
    """A table's header and every row (scrolled out of view too), in aligned columns.
    A table with a ``rows_text`` method gives its own (the job list adds whole ids)."""
    own = getattr(table, "rows_text", None)
    if callable(own):
        return own()
    rows = [[str(column.label) for column in table.ordered_columns]]
    rows += [[str(cell) for cell in table.get_row(row.key)] for row in table.ordered_rows]
    return aligned(rows)


def screen_dump(screen) -> str:
    """Visible screen content as text, including off-screen rows but no footer."""
    from textual.widgets import DataTable
    from textual.widgets import Footer
    from textual.widgets import Input
    from textual.widgets import Log
    from textual.widgets import Static

    from textual.widgets._toast import ToastRack
    from textual.widgets._tooltip import Tooltip

    lines = []  # the header widget brings the title and subtitle
    for widget in screen.query("*"):
        # Not the footer's keys, nor Textual's notes and tooltips over the screen (the
        # keys panel, when it is open, is copied).
        if not _shown(widget) or any(isinstance(node, (Footer, ToastRack, Tooltip))
                                     for node in widget.ancestors_with_self):
            continue
        if isinstance(widget, DataTable):
            lines += _table_text(widget)
        elif isinstance(widget, Log):
            lines += list(widget.lines)
        elif isinstance(widget, Input):
            if widget.value:
                lines.append(widget.value)
        elif isinstance(widget, Static):
            text = _plain(widget.render(), widget.content_region.width or 80).rstrip()
            if text.strip():
                lines += text.split("\n")
    return "\n".join(line.rstrip() for line in lines).strip("\n")


def _plain(renderable, width: int) -> str:
    """A Static's text: its content, or a Rich renderable (the keys panel's table) drawn as text."""
    from rich.console import Console
    from rich.text import Text
    from textual.content import Content

    if isinstance(renderable, (str, Content, Text)):
        return str(renderable)
    console = Console(width=max(20, width), file=io.StringIO(), color_system=None, record=True)
    console.print(renderable)
    return console.export_text()


def copied_note(what: str, local: bool) -> str:
    """What to tell the user once ``what`` (e.g. "12 lines") is copied."""
    if local:
        return f"copied {what}"
    # Only OSC 52 carried it: the terminal may have ignored it.
    return (f"sent {what} to the terminal's clipboard; if nothing pastes, hold Option "
            "(Fn in macOS Terminal) while dragging, to select with the terminal")
