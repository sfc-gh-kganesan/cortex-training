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

"""Widgets and text helpers the TUI's screens share.

Nothing here knows about a particular screen: the job list, the job's details and
the preview all build their tables, panes, footers and header lines from these.
"""

from __future__ import annotations
from cortex_training.tui.resource import ResourceData
from pathlib import Path
import time
import shlex
import hashlib
import inspect
from rich.cells import cell_len
from rich.cells import get_character_cell_size
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual import events
from textual.binding import Binding
from textual.containers import ScrollableContainer
from textual.content import Content
from textual.geometry import Offset
from textual.geometry import Region
from textual.css.query import NoMatches
from textual.dom import NoScreen
from textual.screen import Screen
from textual.scroll_view import ScrollView
from textual.selection import SELECT_ALL
from textual.selection import SelectEnd
from textual.selection import Selection
from textual.strip import Strip
from textual.widget import Widget
from textual.widgets import DataTable
from textual.widgets import Footer
from textual.widgets import Input
from textual.widgets import ListView
from textual.widgets import Log
from textual.widgets import Static
from textual.widgets import TabbedContent
from textual.widgets.data_table import RowDoesNotExist
from cortex_training.tui import views
from cortex_training.tui.clipboard import aligned
import os
import traceback


# Rich's grey62, by value: Textual's markup and styles (the preview's Content)
# don't know that name and would draw the default color instead.
ENDED_STYLE = "#9e9e9e"


def on_stack(screen: Screen) -> bool:
    """Whether ``screen`` remains on its app's stack, safe from a worker."""
    try:
        return screen in screen.app.screen_stack
    except Exception:  # noqa: BLE001 - no app: it is shutting down
        return False


def is_shown(screen: Screen) -> bool:
    """Whether ``screen`` is mounted at the top of its app's stack."""
    try:
        return screen.app.screen is screen and screen.is_mounted
    except Exception:  # noqa: BLE001 - no app: it is shutting down
        return False


# "?" on every screen: the keys panel lists every key that works where you are,
# the ones the footer has no room for too (it lists bindings the footer hides).
KEYS_BINDING = Binding("question_mark", "app.toggle_keys", "Keys", key_display="?")


def close_keys_panel(screen: Screen) -> bool:
    """UI thread. Close the keys panel (``?``, or ``^p`` → Keys) when it is open; whether it was."""
    panels = screen.query("HelpPanel")
    if not panels:
        return False
    panels.remove()
    return True


def after_refresh(screen: Screen, fn, *args, **kwargs) -> None:
    """Run after refresh only while ``screen`` remains on the stack."""

    def run():
        return fn(*args, **kwargs) if on_stack(screen) else None

    screen.call_after_refresh(run)


class JobTabs(TabbedContent):
    """Job tabs that report width changes caused by the keys panel."""

    def on_resize(self, event: events.Resize) -> None:
        self.screen._tabs_resized()


def call_on_screen(screen: Screen, fn, *args, **kwargs) -> bool:
    """Run a worker callback on the UI thread; false once the app is gone."""

    # Run even after a pop: some callbacks clean up then (a finished download's temp dir).
    async def guarded():
        try:
            result = fn(*args, **kwargs)
            if inspect.isawaitable(result):
                await result
        except NoMatches:
            if on_stack(screen):
                raise  # the screen is still live: a real bug

    try:
        app = screen.app
        if not app.is_running:
            return False
        app.call_from_thread(guarded)
        return True
    except RuntimeError:  # NoActiveAppError / closed loop: the app is shutting down
        return False


class _LogPane(Log):
    """The Logs pane, retaining each wrapped row's highlighter kind."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.row_kinds: list[str | None] = []

    def write_rows(self, rows: list[str], kinds: list[str | None], *, scroll_end: bool | None = None) -> None:
        """``rows`` as :meth:`write_lines` writes them, ``kinds`` one per row it makes."""
        self.row_kinds.extend(kinds)
        self.write_lines(rows, scroll_end=scroll_end)
        del self.row_kinds[:max(0, len(self.row_kinds) - len(self.lines))]  # as the oldest rows went

    def clear(self):
        self.row_kinds = []
        return super().clear()

    def _render_line_strip(self, y: int, rich_style):
        highlighter = self.highlighter
        if hasattr(highlighter, "row_kind"):
            highlighter.row_kind = self.row_kinds[y] if 0 <= y < len(self.row_kinds) else None
        return super()._render_line_strip(y, rich_style)


class _SourceList(ListView):
    """The sub-job list, forwarding page and end keys to the Logs pane."""

    _PANE = "#logview"
    BINDINGS = [
        Binding("pageup", "pane('page_up')", show=False),
        Binding("pagedown", "pane('page_down')", show=False),
        Binding("home", "pane('scroll_home')", show=False),
        Binding("end", "pane('scroll_end')", show=False),
    ]

    def action_pane(self, action: str) -> None:
        try:
            pane = self.screen.query_one(self._PANE)
        except NoMatches:
            return
        getattr(pane, f"action_{action}")()


def spans_content(spans) -> Content:
    """One line of ``(text, style)`` spans (as the Resources charts and stats are drawn) as Content."""
    return Content.assemble(*((text, style) if style else text for text, style in spans))


def hanging(content: Content, width: int, indent: int) -> Content:
    """Wrap one line with hanging continuation indentation."""
    wrap = getattr(content, "wrap", None)
    if wrap is None or content.cell_length <= width or width - indent < 10:
        return content
    first = wrap(width)[0]
    rest = content[len(first.plain):]
    rest = rest[len(rest.plain) - len(rest.plain.lstrip()):]  # the space the break took
    if not rest.plain:
        return first
    return Content("\n" + " " * indent).join([first, *rest.wrap(width - indent)])


def wrapped_parts(parts: list, width: int, indent: int, separator: Content) -> Content:
    """Pack parts onto hanging-indented lines, breaking between parts first."""
    lead = Content(" " * indent)
    lines: list = []
    line = None
    for part in parts:
        if line is not None and (line + separator + part).cell_length <= width:
            line = line + separator + part
            continue
        if line is not None:
            lines.append(line)
        line = lead + part
    if line is not None:
        lines.append(line)
    return Content("\n").join(hanging(line, width, indent) for line in lines)


class _FixedCell:
    """A compact fixed-width cell whose Rich ``Text`` is made only when drawn."""

    __slots__ = ("plain", "style", "spans")

    def __init__(self, plain: str, style: str = "", spans: tuple = ()) -> None:
        self.plain = plain
        self.style = style
        self.spans = spans  # ((start, end, style), ...) over plain

    @classmethod
    def assemble(cls, *parts: tuple) -> "_FixedCell":
        """A cell of ``(text, style)`` parts, one after another."""
        spans, at = [], 0
        for text, style in parts:
            if style and text:
                spans.append((at, at + len(text), style))
            at += len(text)
        return cls("".join(text for text, _style in parts), "", tuple(spans))

    def text(self) -> Text:
        text = Text(self.plain, style=self.style, overflow="ellipsis", no_wrap=True)
        for start, end, style in self.spans:
            text.stylize(style, start, end)
        return text

    def __rich_console__(self, console, options):
        return self.text().__rich_console__(console, options)

    def __str__(self) -> str:
        return self.plain


def _cell(text: str, style: str = "") -> _FixedCell:
    """A one-line printable cell, preserving spaces and never parsing markup."""
    return _FixedCell(views.printable(text).replace("\n", " "), style)


def scroll_area(widget: Widget) -> Widget | None:
    """The nearest scrolling selection area, excluding docked content."""
    table = None
    for node in widget.ancestors_with_self:
        if isinstance(node, Screen):
            return table
        if node.styles.dock in ("top", "bottom") and isinstance(node.parent, ScrollableContainer):
            return node
        if isinstance(node, DataTable):
            table = table or node
        elif isinstance(node, (ScrollView, ScrollableContainer)) and not isinstance(node, (Footer, Input)):
            return node
    return table


def scrolling_region(area: Widget) -> Region:
    """The area's scrolling content, excluding docked lines."""
    if isinstance(area, SelectableTable):
        return area.drag_region()
    region = area.scrollable_content_region
    top, bottom = region.y, region.bottom
    for child in area.children:
        if child.display and child.styles.dock == "top":
            top = max(top, child.region.bottom)
        elif child.display and child.styles.dock == "bottom":
            bottom = min(bottom, child.region.y)
    return Region(region.x, top, region.width, bottom - top) if bottom > top else region


class FollowsPointer:
    """Keep cross-widget selection under its pointer while an area scrolls."""

    def _watch__select_state(self, select_state) -> None:
        if select_state is not None and select_state.end is not None and select_state.is_attached_to_dom():
            held = getattr(self.app, "press_held", None)
            if callable(held) and held():
                self._selecting = True
                self.selections = {}
                return
            select_state = self._end_on_text(select_state)
            if select_state is None:  # no text between the pointer and the drag's start yet
                self._selecting = True
                self.selections = {}
                return
            start, end = select_state.start, select_state.end
            first, last = start.content_widget, end.content_widget
            if (first is not None and last is not None and first is not last
                    and start.content_offset is not None and end.content_offset is not None
                    and scroll_area(first) is not None and scroll_area(first) is scroll_area(last)):
                self._selecting = True
                texts = [widget for widget in scroll_area(first).query("*")
                         if widget.allow_select and not widget.is_container]
                if first in texts and last in texts:
                    head, tail = (start.content_offset, end.content_offset)
                    if texts.index(first) > texts.index(last):
                        first, last, head, tail = last, first, tail, head
                    between = texts[texts.index(first) + 1:texts.index(last)]
                    # Both ends take in the character under them, as Textual's selection in one text does.
                    self.selections = {first: Selection(head, None), **{widget: SELECT_ALL for widget in between},
                                       last: Selection(None, tail + Offset(1, 0))}
                    return
        super()._watch__select_state(select_state)

    def _end_on_text(self, state):
        """Move an endpoint on blank space to the nearest text toward its start."""
        start, end = state.start, state.end
        area = scroll_area(start.content_widget) if start.content_widget is not None else None
        if end.content_widget is not None or area is None or start.content_offset is None:
            return state
        widget = start.content_widget
        start_y = widget.content_region.y + start.content_offset.y - int(widget.scroll_offset.y)
        x, y = state.screen_offset
        step = 1 if start_y > y else -1
        region = scrolling_region(area)
        while region.y <= y < region.bottom and y != start_y + step:
            found = self._text_on_row(area, x, y)
            if found is not None:
                text, offset = found
                new = state.update_end(state.screen_offset, SelectEnd(text.parent, text, offset))
                self.set_reactive(Screen._select_state, new)
                return new
            y += step
        return None

    def _text_on_row(self, area: Widget, x: int, y: int):
        """The selectable widget and offset nearest ``(x, y)`` on that row."""
        found, offset = self.get_widget_and_offset_at(x, y)
        if found is not None and offset is not None and not found.is_container and scroll_area(found) is area:
            return found, offset
        nearest = None  # the text on the row that starts nearest to the left of the pointer
        for node in area.query("*"):
            box = node.region
            if (not node.is_container and node.allow_select and node.display and box.y <= y < box.bottom
                    and box.x <= x and (nearest is None or box.x > nearest.region.x)):
                nearest = node
        if nearest is None:
            return None
        found, offset = self.get_widget_and_offset_at(min(x, nearest.region.right - 1), y)
        return (found, offset) if found is nearest and offset is not None else None

    def _update_select(self) -> None:
        state = getattr(self, "_select_state", None)
        position = getattr(self.app, "mouse_position", None)
        if state is not None and state.end is not None and position is not None:
            widget, offset = self.get_widget_and_offset_at(position.x, position.y)
            if widget is not None:
                end = (SelectEnd(widget if isinstance(widget, Screen) else widget.parent, widget, offset)
                       if offset is not None else SelectEnd(widget, None, None))
                self.set_reactive(Screen._select_state, state.update_end(position, end))
        super()._update_select()


class SelectableTable(DataTable):
    """A DataTable with scroll-stable, character-accurate selection and copying."""

    ALLOW_SELECT = True
    _DRAG_CELLS = 2

    def render_line(self, y: int) -> Strip:
        width = self.size.width
        scroll_x, scroll_y = self.scroll_offset
        line = y if y < self._fixed_lines() else y + scroll_y
        strip = self._render_line(line, scroll_x, scroll_x + width, self.rich_style)
        return self._selectable(strip, line, scroll_x)

    def _fixed_lines(self) -> int:
        """The lines at the top that stay put as the table scrolls: the header's and any fixed rows'."""
        rows = range(min(self.fixed_rows, self.row_count))
        height = sum(self.get_row_height(self._row_locations.get_key(index)) for index in rows)
        return height + (self.header_height if self.show_header else 0)

    def _text_lines(self) -> int:
        """How many lines the table's text has: the header's and the rows' (not the blank under them)."""
        return (self.header_height if self.show_header else 0) + len(self._y_offsets)

    def _selectable(self, strip: Strip, line: int, scroll_x: int) -> Strip:
        """Add text offsets and selection highlighting to a rendered line."""
        lines = self._text_lines()
        if line >= lines:  # the blank under the rows: the empty line after them
            return strip.apply_offsets(0, lines)
        fixed, rest = self._line_parts(line)
        cut = min(cell_len(fixed), strip.cell_length)
        parts = ((strip.crop(0, cut), 0), (strip.crop(cut), len(fixed) + _chars_within(rest, scroll_x)))
        selection = self.text_selection
        span = None if selection is None else _selected_span(selection, line, fixed + rest)
        style = self._selection_style() if span is not None and span[0] < span[1] else None
        return Strip.join(
            (_draw_highlight(part, first, span, style) if style is not None else part).apply_offsets(first, line)
            for part, first in parts
        )

    def _line_parts(self, line: int) -> tuple[str, str]:
        """A rendered line's fixed and scrollable text."""
        try:
            row_key, line_in_row = self._get_offsets(line)
        except LookupError:
            return "", ""
        plain = None if line_in_row else self._plain_parts(row_key)
        return plain if plain is not None else self._drawn_parts(row_key, line_in_row)

    def _drawn_parts(self, row_key, line_in_row: int) -> tuple[str, str]:
        """Render one row line into fixed and scrollable text."""
        fixed, scrollable = self._render_line_in_row(row_key, line_in_row, self.rich_style,
                                                     self.cursor_coordinate, self.hover_coordinate)
        # The scrolling cells start with the fixed columns too, drawn again (and cut off, as here).
        skip = sum(column.get_render_width(self) for column in self.ordered_columns[:self.fixed_columns])
        rest = Strip([segment for cell in scrollable for segment in cell]).crop(skip).text
        return "".join(segment.text for cell in fixed for segment in cell), rest

    def _plain_parts(self, row_key) -> tuple[str, str] | None:
        """Fast-path one-line fitting ``_FixedCell`` rows without Rich rendering."""
        row = self.rows.get(row_key)
        if row is None or row.height != 1 or (self._labelled_row_exists and self.show_row_labels):
            return None
        try:
            cells = self.get_row(row_key)
        except (KeyError, RowDoesNotExist):  # a cell missing: what the table draws there
            return None
        pad = " " * self.cell_padding
        texts = []
        for column, cell in zip(self.ordered_columns, cells):
            if not isinstance(cell, _FixedCell) or not cell.plain.isprintable():
                return None
            room = column.get_render_width(self) - 2 * self.cell_padding
            used = cell_len(cell.plain)
            if used > room:
                return None  # cut short with "…", as Rich cuts it
            texts.append(f"{pad}{cell.plain}{' ' * (room - used)}{pad}")
        # Past the last column, the row's blank to the table's edge (as DataTable extends it).
        width = sum(column.get_render_width(self) for column in self.ordered_columns[self.fixed_columns:])
        extend = " " * max(0, self.size.width - width - self._row_label_column_width)
        return "".join(texts[:self.fixed_columns]), "".join(texts[self.fixed_columns:]) + extend

    def _selection_style(self) -> Style:
        """Selection background plus an explicitly configured foreground."""
        screen = self.screen
        style = screen.get_component_rich_style("screen--selection")
        own = screen.get_component_rich_style("screen--selection", partial=True)
        return Style(color=style.color if own.color is not None else None, bgcolor=style.bgcolor)

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        """The text under ``selection``, a line for each of the table's lines it covers."""
        lines = self._text_lines()
        start, end = selection
        first = 0 if start is None else max(0, start.y)
        last = lines - 1 if end is None else min(end.y, lines - 1)
        parts = []
        for line in range(first, last + 1):
            text = "".join(self._line_parts(line))
            span = _selected_span(selection, line, text)
            if span is not None:
                parts.append(text[span[0]:span[1]])
        return ("\n".join(parts), "\n") if parts else None

    def drag_region(self, header: bool = False) -> Region:
        """The clipped header or scrolling-row drag region."""
        region = self.scrollable_content_region
        try:
            geometry = self.screen.find_widget(self)
            region = region.intersection(geometry.region.intersection(geometry.clip))
        except Exception:  # noqa: BLE001 - not laid out
            pass
        lines = self.header_height if self.show_header else 0
        if header and lines:
            return Region(region.x, region.y, region.width, min(lines, region.height))
        if region.height <= lines:
            return region
        return Region(region.x, region.y + lines, region.width, region.height - lines)

    def on_header(self, screen_y: int) -> bool:
        """Whether screen line ``screen_y`` is the table's header."""
        top = self.scrollable_content_region.y
        return self.show_header and top <= screen_y < top + self.header_height

    def clear(self, columns: bool = False):
        self._drop_highlight()
        return super().clear(columns)

    def sort(self, *columns, key=None, reverse: bool = False):
        self._drop_highlight()
        return super().sort(*columns, key=key, reverse=reverse)

    def _drop_highlight(self) -> None:
        """The rows are about to change: a highlight in them would sit over other text."""
        try:
            screen = self.screen
        except NoScreen:
            return
        if self in screen.selections:
            screen.clear_selection()

    def on_click(self, event: events.Click) -> None:
        self.swallowed(event)

    def swallowed(self, event: events.Click) -> bool:
        """Suppress non-left clicks that DataTable would treat as row clicks."""
        if event.button == 1:
            return False
        event.prevent_default()
        event.stop()
        return True

    def rows_text(self, keys=None) -> list:
        """Aligned header and rows, adding whole ids when available."""
        whole_id = getattr(self.screen, "whole_id", None)
        extra = ["Whole ID"] if callable(whole_id) else []
        rows = [[str(column.label) for column in self.ordered_columns] + extra]
        for key in keys if keys is not None else [row.key for row in self.ordered_rows]:
            cells = [str(cell) for cell in self.get_row(key)]
            rows.append(cells + ([whole_id(key.value) or ""] if extra else []))
        return aligned(rows)


def _chars_within(text: str, cells: int) -> int:
    """Characters wholly contained in the first ``cells`` terminal cells."""
    if cells <= 0:
        return 0
    if text.isascii():
        return min(cells, len(text))
    used = 0
    for index, character in enumerate(text):
        used += get_character_cell_size(character)
        if used > cells:
            return index
    return len(text)


def _selected_span(selection: Selection, line: int, text: str) -> tuple[int, int] | None:
    """Selected character span on one line, excluding trailing blanks."""
    span = selection.get_span(line)
    if span is None:
        return None
    start, end = span
    end = len(text) if end == -1 else min(end, len(text))
    start = min(start, end)
    return start, start + len(text[start:end].rstrip())


def _draw_highlight(strip: Strip, first: int, span: tuple[int, int], style: Style) -> Strip:
    """Overlay ``style`` on a line-relative character span."""
    start, end = span
    at = first
    segments = []
    for text, own, control in strip:
        low, high = max(0, start - at), min(len(text), end - at)
        if control or low >= high:
            segments.append(Segment(text, own, control))
        else:
            if low:
                segments.append(Segment(text[:low], own))
            segments.append(Segment(text[low:high], own + style if own else style))
            if high < len(text):
                segments.append(Segment(text[high:], own))
        at += len(text)
    return Strip(segments, strip.cell_length)


_SEPARATOR = "  ·  "


def fit_line(segments: list, width: int) -> Content:
    """Fit styled segments by cutting those with minimum widths, then wrapping."""
    texts = [text for text, _style, _least in segments]
    sep = views.cell_width(_SEPARATOR)
    over = sum(views.cell_width(text) for text in texts) + sep * (len(texts) - 1) - width
    spares = [max(0, views.cell_width(text) - least) if least is not None else 0 for text, _style, least in segments]
    if 0 < over <= sum(spares):  # cutting gets it onto one row
        for i, spare in enumerate(spares):
            if over <= 0:
                break
            if spare:
                cut = min(spare, over)
                texts[i] = views.fit_cells(texts[i], views.cell_width(texts[i]) - cut)
                over -= cut
    rows: list = []
    for text, (_text, style, _least) in zip(texts, segments):
        piece = Content.styled(text, style)
        if rows and rows[-1].cell_length + sep + piece.cell_length <= width:
            rows[-1] = rows[-1] + Content(_SEPARATOR) + piece
        else:
            rows.append(piece)
    return Content("\n").join(rows)


def _fit_line(parts: list[list[tuple[str, int]]], order: list[int], width: int, variables: dict,
              *, cut: str | None = None, cut_min: int = 40) -> Content:
    """Choose progressively shorter forms until joined parts fit one line.

    ``cut`` reserves ``cut_min`` cells while forms are chosen, then takes the
    remaining room. The final line is clipped only as a last resort.
    """
    levels = [0] * len(parts)

    def build(values: dict) -> tuple[Content, int]:
        forms = [part[min(level, len(part) - 1)] for part, level in zip(parts, levels)]
        line = Content.from_markup(_SEPARATOR.join(markup for markup, _spare in forms if markup), **values)
        return line, line.cell_length + sum(spare for _markup, spare in forms)

    if width <= 0:
        return build(variables)[0]
    text = str(variables.get(cut) or "") if cut else ""
    probe = {**variables, cut: views.fit_cells(text, cut_min)} if text else variables
    cells = build(probe)[1]
    for index in order:
        if cells <= width:
            break
        levels[index] += 1
        cells = build(probe)[1]
    if text:
        used = build({**variables, cut: ""})[1]
        variables = {**variables, cut: views.fit_cells(text, max(1, width - used))}
    line = build(variables)[0]
    return line.truncate(width, ellipsis=True) if line.cell_length > width else line


class _ColoredLinks(Static):
    """A Static whose links retain their own colours."""

    @property
    def link_style(self) -> Style:
        return self.styles.link_style


class FittedFooter(Footer):
    """A footer that hides rightmost keys instead of clipping one."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Where each key ends, measured once all of them are laid out.
        self._edges: list[int] | None = None

    async def recompose(self) -> None:
        self._edges = None  # other keys: measured again
        await super().recompose()
        self.call_after_refresh(self._fit)

    def on_resize(self, event: events.Resize) -> None:
        self._fit()

    def _fit(self) -> None:
        keys = list(self.children)
        if not keys or self.region.width <= 0:
            return
        if self._edges is None or len(self._edges) != len(keys):
            if any(not key.display for key in keys):
                return  # measured only with every key shown (after a recompose)
            self._edges = [key.region.right - self.region.x for key in keys]
        room = self.region.width
        for key, edge in zip(keys, self._edges):
            if key.display != (edge <= room):
                key.display = edge <= room


class _ResourceList(_SourceList):
    """Resource views and units with pane-scrolling navigation."""

    _PANE = "#resource-pane"

    def action_cursor_down(self) -> None:
        target = self.screen._resource_down_from(self.index)
        if target is not None:
            self.index = target
            return
        super().action_cursor_down()

    def watch_index(self, old_index: int | None, new_index: int | None) -> None:
        super().watch_index(old_index, new_index)
        self.screen._resource_list_moved(new_index)

# Where an unexpected error's traceback goes, so the UI can stay one line.
_ERROR_LOG = "~/.cortex-training-errors.log"


def log_traceback(label: str) -> None:
    """Worker thread, inside an ``except``: append the traceback to the error log."""
    try:
        with open(os.path.expanduser(_ERROR_LOG), "a", encoding="utf-8") as handle:
            handle.write(f"{label}:\n{traceback.format_exc()}\n")
    except Exception:  # noqa: BLE001 - reporting an error must not raise one
        pass


_BLANK = _FixedCell("")


def _has_telemetry(data: ResourceData | None) -> bool:
    return data is not None and bool(data.kinds or data.ignored_rows_by_sub_job)


def _same_telemetry(a: ResourceData, b: ResourceData) -> bool:
    """Whether two loads hold the same snapshots (a loader reuses a sub-job's kind until new rows of it arrive)."""
    return (
        a.ignored_rows_by_sub_job == b.ignored_rows_by_sub_job
        and a.kinds.keys() == b.kinds.keys()
        and all(a.kinds[key].keys() == b.kinds[key].keys()
                and all(a.kinds[key][kind] is b.kinds[key][kind] for kind in a.kinds[key]) for key in a.kinds)
    )


def _cache_job_id(client, job_id: str) -> str:
    """Namespace a persistent cache by service endpoint and job location.

    The location is the one SQL resolves the client's identifiers to, so a job opened
    from the configured schema's list (``team_db``) and from the account-wide list,
    whose child client holds the stored names (``TEAM_DB``), share one cache.
    """
    location = views.client_location(client) or (
        getattr(client, "database", ""), getattr(client, "schema", ""))
    scope = tuple(
        value if isinstance(value, str) else ""
        for value in (
            getattr(client, "base_url", ""),
            getattr(client, "endpoint", ""),
            *location,
        )
    )
    if not any(scope):
        return job_id
    digest = hashlib.sha256("\0".join(scope).encode()).hexdigest()[:16]
    return f"{digest}:{job_id}"


_CANCEL_CHECK_SECONDS = 0.5

def _sleep_until_cancelled(worker, step: float = _CANCEL_CHECK_SECONDS):
    """A ``sleep(delay)`` for a thread worker's loop that gives up once the worker is
    cancelled, so leaving a screen frees its thread (and its executor slot) now rather
    than after a long poll interval.

    A delay no longer than ``step`` is slept in one go, as ``time.sleep`` would: the
    tail's own short waits keep their exact timing, and only a wait long enough to
    matter is broken up.
    """
    def sleep(delay: float) -> None:
        if delay <= step:
            time.sleep(delay)
            return
        deadline = time.monotonic() + delay
        while not worker.is_cancelled:
            left = deadline - time.monotonic()
            if left <= 0:
                return
            time.sleep(min(step, left))

    return sleep


# The client's own wording when a download gave up on some chunks. The TUI downloads
# into its cache, not a directory the user chose, so its retry hint names the command
# that fetches those chunks again where the user can see them.
_RESUME_HINT = "download again without resume to retry them"


def _retry_command(kind: str, job_id: str, sub_job_id: str, output_dir: Path) -> str:
    if kind == "stdout":
        command = ["cortex-training", "download-log", job_id, "--log-type", "stdout"]
    else:
        command = ["cortex-training", "download-metrics", job_id]
    command += ["--sub-job", sub_job_id, "--output-dir", str(output_dir)]
    return " ".join(shlex.quote(part) for part in command)


def _result_warnings(results, *, kind: str, job_id: str, output_dir: Path) -> dict:
    """Per-sub-job ``warnings`` from a download result list (only sub-jobs that have some).

    A download that skipped chunks says so here: without it a partial saved console or
    an incomplete telemetry file would look complete.
    """
    out: dict = {}
    for row in results if isinstance(results, list) else []:
        if not isinstance(row, dict):
            continue
        sid = row.get("sub_job_id")
        messages = row.get("warnings")
        if isinstance(sid, str) and isinstance(messages, list):
            hint = f"retry them with `{_retry_command(kind, job_id, sid, output_dir)}` (no --resume)"
            texts = [str(message).replace(_RESUME_HINT, hint) for message in messages if message]
            if texts:
                out.setdefault(sid, []).extend(texts)
    return out


def _warning_text(messages: list, limit: int = 100) -> str:
    """One line for the status: the first warning, cut, and how many more there are."""
    first = messages[0] if len(messages[0]) <= limit else messages[0][: limit - 1] + "…"
    more = f" (+{len(messages) - 1} more)" if len(messages) > 1 else ""
    return f"warning: {first}{more}"


def _quiet_download(method) -> dict:
    """Disable client stall checks when supported; the TUI reports staleness."""
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return {}
    return {"stale_after_seconds": None} if "stale_after_seconds" in parameters else {}




class _TimedTail:
    """A tail client that records when its current request began."""

    def __init__(self, client, screen, gen: int) -> None:
        self._client, self._screen, self._gen = client, screen, gen

    def tail_logs(self, *args, **kwargs):
        self._screen._tail_request = (self._gen, time.monotonic())
        try:
            return self._client.tail_logs(*args, **kwargs)
        finally:
            # Only while it is still this tail's: a cancelled older tail answering after a
            # newer one started would otherwise drop the newer request's "no answer" note.
            request = self._screen._tail_request
            if request is not None and request[0] == self._gen:
                self._screen._tail_request = None
