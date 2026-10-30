# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Incremental telemetry parsing and rendering for the Resources tab.

``ResourceLoader`` follows each manifested ``<sub_job_id>/gpu.jsonl`` file,
reusing aggregates when chunks append and rebuilding when the file changes.
Rows describe either GPUs or system nodes. Numeric ``_pct``, ``_bytes``,
``_watts`` and ``_c`` fields become metrics; memory-used percentages are
derived. Each load returns an immutable :class:`ResourceData` snapshot.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from array import array
from bisect import bisect_left
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime, timedelta, timezone
from functools import reduce
from itertools import groupby
from itertools import repeat
from operator import add
from operator import itemgetter
from operator import mul
from pathlib import Path
from statistics import median
from typing import Callable
from typing import Mapping

from cortex_training._artifact_download import read_manifest
from cortex_training.tui import views
from cortex_training.tui.theme import BACKGROUND
from cortex_training.tui.theme import CHART_HUES
from cortex_training.tui.theme import ERR
from cortex_training.tui.theme import FAINT
from cortex_training.tui.theme import GRID
from cortex_training.tui.theme import MUTED
from cortex_training.tui.theme import OK
from cortex_training.tui.theme import SURFACE
from cortex_training.tui.theme import TEXT
from cortex_training.tui.theme import WARN

RANGES: dict[str, timedelta | None] = {
    "10m": timedelta(minutes=10),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "6h": timedelta(hours=6),
    "all": None,
}
# Every job opens on this window; the stats beside it always cover the whole job too.
DEFAULT_RANGE = "10m"
_MAX_BUCKETS = 240
_MAX_SAMPLE_INTERVAL_S = 24 * 60 * 60
_MAX_FB_BYTES = float(2**63)
# Bound malformed gauges and aggregate sums.
_MAX_GAUGE = 1e18
_GIB = float(1024**3)
_MIN_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc).timestamp()
_MAX_EPOCH = datetime(9998, 12, 31, 23, 59, 59, 999999, tzinfo=timezone.utc).timestamp()
# RFC 3339 with a "T" and an explicit zone; the fraction may carry more digits than microseconds.
_TS_PARTS_RE = re.compile(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?(Z|[+-]\d\d:\d\d)")
# Raw retention and whole-job folding.
_RAW_RETENTION_S = 7 * 60 * 60.0
_FOLD_MARGIN_S = 20 * 60.0
_FOLD_CONTEXT_S = 10 * 60.0
_BIN_START_S = 60.0
_MAX_BINS = 480
_READ_BLOCK = 4 << 20
_FINGERPRINT = 64
_PEEK_BYTES = 64 * 1024
# Large first loads publish the newest window before reading older bytes.
_NEWEST_FIRST_STEP = 1 << 20
_NEWEST_FIRST_MIN_BYTES = 4 << 20
_NEWEST_FIRST_SLACK_S = 120.0
# Reporting tolerates independent worker uploads.
_REPORTING_TOLERANCE_S = 120.0
_INTERVAL_DELTAS = 64
# Min and max come from the average across units at each sampling interval; a
# finer interval than this many buckets over the raw retention is coarsened.
_MAX_NATIVE_BUCKETS = 20_000
# A row stamped further ahead of this computer's clock than this is unsupported:
# one clock-skewed row would otherwise become the newest sample, which anchors
# the windows, the freshness and the reporting units.
_MAX_AHEAD_S = 10 * 60.0
# A row without a usable sample_interval_s counts for the time since its unit's
# previous row, up to this long; a longer gap is a pause, not an interval.
_FALLBACK_INTERVAL_MAX_S = 5 * 60.0
# Bridge one missed system scrape, not a longer gap.
_BRIDGE_INTERVALS = 2.5
_NAN = float("nan")
_INF = math.inf
_OK, _SKIP, _IGNORED = 0, 1, 2
_MISSING = object()
_SECOND_CACHE: dict[str, float | None] = {}
# What json.loads runs on a document, without its wrappers: (value, end) or StopIteration.
_SCAN = json.JSONDecoder().scan_once

GPU = "gpu"
SYSTEM = "system"


@dataclass(frozen=True)
class SeriesStats:
    """Min, unit-time average, max, and latest for one scope.

    Bounds mark late binned history that may hide a more extreme value.
    """

    min: float | None = None
    avg: float | None = None
    max: float | None = None
    latest: float | None = None
    min_bound: bool = False
    max_bound: bool = False


# ─── what a load returns ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class Metric:
    """A numeric or derived metric and its stable chart scale."""

    key: str
    name: str
    unit: str
    scale: float

    @property
    def short(self) -> str:
        """The name for a narrow column: ``power``, ``GPU mem used`` (a percentage keeps its ``%``)."""
        return re.sub(r" \((GiB|W|°C)\)$", "", self.name).replace("memory", "mem")


@dataclass(frozen=True)
class Unit:
    """One GPU (``global_gpu_index``) or node (``worker_num``) of a kind."""

    id: int
    name: str
    reporting: bool
    latest_sample_ts: datetime
    latest: Mapping[str, float | None] = field(default_factory=dict, hash=False)


@dataclass(frozen=True, eq=False)
class KindView:
    """One window's charts and stats for a unit or the mean across units."""

    name: str  # the window, a key of RANGES
    start: datetime
    end: datetime
    label: str
    charts: Mapping[str, tuple[float | None, ...]] = field(default_factory=dict)
    stats: Mapping[str, SeriesStats] = field(default_factory=dict)

    def chart(self, key: str) -> tuple[float | None, ...]:
        return self.charts.get(key, ())


@dataclass(frozen=True, eq=False)
class KindResource:
    """One sub-job's GPU or system metrics, units, latest values, and views."""

    kind: str
    sub_job_id: str
    metrics: tuple[Metric, ...]
    units: tuple[Unit, ...]
    latest_sample_ts: datetime
    first_sample_ts: datetime | None = None
    sample_interval_s: float | None = None
    reported_interval_s: float | None = None
    partial: bool = False
    reporting_unit_ids: tuple[int, ...] = ()
    latest: Mapping[str, float | None] = field(default_factory=dict)
    _frames: Mapping[str, "_Frame"] = field(default_factory=dict, repr=False)
    _views: dict = field(default_factory=dict, repr=False)
    # The windows whose frame was carried from an earlier snapshot (see ResourceLoader.load's
    # ``shown``): as of their last showing, and rebuilt once shown again.
    _stale: frozenset = field(default_factory=frozenset, repr=False)
    _native: object = field(default=None, repr=False)  # its _Native, for a rebuild of the same rows

    def fresh(self, window: str | None = None) -> bool:
        """Whether ``window``'s frame, and the whole job's its stats come from, are this
        snapshot's rather than carried from an earlier one; with None, every window's."""
        if window is None:
            return not self._stale
        return window not in self._stale and "all" not in self._stale

    @property
    def unit_ids(self) -> tuple[int, ...]:
        return tuple(unit.id for unit in self.units)

    def metric(self, key: str) -> Metric | None:
        return next((metric for metric in self.metrics if metric.key == key), None)

    def view(self, window: str = DEFAULT_RANGE, unit: int | None = None) -> KindView | None:
        """One unit's window, or the mean with ``unit=None``."""
        found = self._views.get((window, unit))
        if found is None:
            frame = self._frames.get(window)
            found = None if frame is None else frame.view(self.kind, unit, self.units)
            if found is not None:
                self._views[(window, unit)] = found
        return found


@dataclass(frozen=True)
class ResourceData:
    """Each sub-job's usable telemetry and count of ignored rows."""

    kinds: dict[str, dict[str, KindResource]] = field(default_factory=dict)
    ignored_rows_by_sub_job: dict[str, int] = field(default_factory=dict)

    @property
    def ignored_rows(self) -> int:
        return sum(self.ignored_rows_by_sub_job.values())

    def kind(self, sub_job_id: str | None, kind: str) -> KindResource | None:
        return self.kinds.get(sub_job_id, {}).get(kind) if sub_job_id else None


# ─── metric names, units and scales ──────────────────────────────────────────

# A metric's unit, by its field's suffix: only a field with one is a metric (see the module docstring).
_UNIT_SUFFIXES = (("_pct", "%"), ("_bytes", "GiB"), ("_watts", "W"), ("_c", "°C"))
# A metric's field: its name is made into the metric's, which the terminal draws, so
# only plain lowercase words, and an escape sequence in a field's name never reaches it.
_METRIC_FIELD_RE = re.compile(r"[a-z0-9_]+")
# Words a metric's name spells differently from its field.
_NAME_WORDS = {"gpu": "GPU", "cpu": "CPU", "sm": "SM", "dram": "DRAM", "fb": "GPU memory", "mem": "memory"}


def is_metric_field(name: object) -> bool:
    """Whether a field is a metric: a name of ``[a-z0-9_]`` that ends in a unit suffix (``_UNIT_SUFFIXES``)."""
    return type(name) is str and _METRIC_FIELD_RE.fullmatch(name) is not None and _split_unit(name)[1] != ""


def _split_unit(field_name: str) -> tuple[str, str]:
    for suffix, unit in _UNIT_SUFFIXES:
        if field_name.endswith(suffix) and len(field_name) > len(suffix):
            return field_name[: -len(suffix)], unit
    return field_name, ""


def metric_name(field_name: str) -> str:
    """A field's name as shown: ``power_watts`` → ``power (W)``, ``fb_free_bytes`` → ``GPU memory free (GiB)``."""
    base, unit = _split_unit(field_name)
    name = " ".join(_NAME_WORDS.get(word, word) for word in base.split("_") if word) or field_name
    if unit == "%":
        return f"{name} %"
    return f"{name} ({unit})" if unit else name


def nice_ceiling(value: float | None) -> float:
    """The smallest 1, 2, 2.5 or 5 × 10^k at or above ``value`` (1 when nothing is above 0)."""
    if value is None or not value > 0 or not math.isfinite(value):
        return 1.0
    power = math.floor(math.log10(value))
    for exponent in (power - 1, power, power + 1):
        for mantissa in (1, 2, 2.5, 5):
            ceiling = mantissa * 10.0**exponent if exponent >= 0 else mantissa / 10.0 ** (-exponent)
            if ceiling >= value * (1 - 1e-12):
                return ceiling
    return 10.0 ** (power + 1)


def _compact(value: float) -> str:
    """``1000``, ``2.5``, ``0.005``, ``25k``, ``1.5M``: an axis value, as short as it reads."""
    size = abs(value)
    for step, suffix in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e4, "k")):
        if size >= step:
            scaled = value / (1e3 if suffix == "k" else step)
            return f"{scaled:g}{suffix}"
    return f"{value:g}"


def format_value(value: float | None, unit: str = "") -> str:
    """``85%``, ``1.8%``, ``87.4 GiB``, ``215 GiB``, ``117 W``, ``34 °C``, ``88``; ``—`` when unmeasured."""
    if value is None or value != value:
        return "—"
    value += 0.0  # no "-0"
    if unit == "%":
        return f"{value:.1f}%" if 0 < abs(value) < 9.95 else f"{value:.0f}%"
    if unit == "GiB":
        size = abs(value)
        return f"{value:.2f} GiB" if 0 < size < 9.995 else f"{value:.1f} GiB" if size < 99.95 else f"{value:.0f} GiB"
    text = str(int(value)) if value == int(value) and abs(value) < 1e6 else _compact(value) if abs(value) >= 1e6 \
        else f"{value:.3g}"
    return f"{text} {unit}" if unit else text


def axis_labels(metric: Metric) -> tuple[str, str, str]:
    """``metric``'s chart axis labels, top to bottom: ``100%``, ``50%``, ``0%``, or ``500 W``, ``250 W``, ``0 W``."""
    if metric.unit == "%":
        return "100%", "50%", "0%"
    top = metric.scale
    return tuple(f"{_compact(value)} {metric.unit}" if metric.unit else _compact(value)
                 for value in (top, top / 2, 0.0))


def unit_name(kind: str, unit_id: int) -> str:
    """``GPU 3`` (as the rows number it), ``node 2`` (``worker_num`` 1: nodes count from 1)."""
    spec = _KIND_SPECS[kind]
    return f"{spec.unit_word} {unit_id + spec.first_number}"


def across_text(kind: str, units: int) -> str:
    """``mean of 8 GPUs``, ``mean of 2 nodes``; ``""`` for one unit or none."""
    return f"mean of {units} {_KIND_SPECS[kind].unit_word}s" if units > 1 else ""


# ─── parsing ─────────────────────────────────────────────────────────────────


def _finite_number(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _bounded_number(value, lower: float, upper: float) -> float | None:
    result = _finite_number(value)
    return result if result is not None and lower <= result <= upper else None


def _number(value, low: float, high: float) -> float:
    """A metric's value from JSON: an int or float in [low, high], else NaN (strings are never one)."""
    if type(value) is float:
        return value if low <= value <= high else _NAN
    if type(value) is int:  # not bool; compared exactly, however large
        return float(value) if low <= value <= high else _NAN
    return _NAN


def _sample_epoch(value: object) -> float | None:
    """``sample_ts`` (RFC 3339 with a zone) as epoch seconds, or None.

    Fractions finer than a microsecond are truncated, and times outside
    1970-01-01 to 9998-12-31 are rejected. This is the hot path of a load, so
    the whole seconds of each timestamp are cached: every unit in a capture
    shares them.
    """
    if not isinstance(value, str):
        return None
    match = _TS_PARTS_RE.fullmatch(value)
    if match is None:
        return None
    base, fraction, zone = match.groups()
    key = base + zone
    seconds = _SECOND_CACHE.get(key, _MISSING)
    if seconds is _MISSING:
        try:
            seconds = datetime.fromisoformat(base + ("+00:00" if zone == "Z" else zone)).timestamp()
        except (OverflowError, ValueError, OSError):
            seconds = None
        if len(_SECOND_CACHE) > 8192:
            _SECOND_CACHE.clear()
        _SECOND_CACHE[key] = seconds
    if seconds is None:
        return None
    if fraction:
        seconds += int(fraction[:6].ljust(6, "0")) / 1_000_000
    return seconds if _MIN_EPOCH <= seconds <= _MAX_EPOCH else None


def _utc(epoch: float) -> datetime:
    return datetime.fromtimestamp(epoch, tz=timezone.utc)


def _read_manifest(path: Path) -> tuple[list[str], int] | None:
    """``(chunk names, committed bytes)`` for a complete metric file."""
    manifest = read_manifest(path.with_name("." + path.name + ".manifest.json"))
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if manifest is None or not manifest.names or size < manifest.nbytes:
        return None
    return manifest.names, manifest.nbytes


# ─── per-unit and per-kind state ─────────────────────────────────────────────


@dataclass(frozen=True)
class _Kind:
    """What sets one kind of row apart: how it is checked, its unit, its first two metrics."""

    name: str
    record_version: int
    unit_field: str
    unit_word: str
    first_number: int  # the number shown for unit 0
    headline: str  # the field charted first
    memory: str  # the derived memory used %, charted second
    used: str
    other: str  # the free bytes (GPU: used / (used + free)) or the total (system: used / total)
    other_is_free: bool
    bridge: bool  # carry a unit across one missed scrape (see _BRIDGE_INTERVALS)
    # Fields that stay the same through a job: the header names them (node_allocation_text),
    # and they get no chart or stats row. They are read even without a unit (cpu_count).
    constant: tuple[str, ...] = ()


_GPU_KIND = _Kind(GPU, 2, "global_gpu_index", "GPU", 0, "gpu_util_pct", "fb_used_pct",
                  "fb_used_bytes", "fb_free_bytes", True, False)
_SYSTEM_KIND = _Kind(SYSTEM, 3, "worker_num", "node", 1, "cpu_util_pct", "mem_used_pct",
                     "mem_used_bytes", "mem_total_bytes", False, True, ("cpu_count", "mem_total_bytes"))
_KIND_SPECS = {GPU: _GPU_KIND, SYSTEM: _SYSTEM_KIND}


class _Field:
    """One slot of a series: a metric's field, one the header names, or the derived memory used %."""

    __slots__ = ("key", "unit", "low", "high", "factor", "derived", "rank", "seen")

    def __init__(self, key: str, derived: bool = False) -> None:
        self.key = key
        self.unit = _split_unit(key)[1]
        self.derived = derived
        if self.unit == "%":
            self.low, self.high = 0.0, 100.0
        elif self.unit == "GiB":
            self.low, self.high = 0.0, _MAX_FB_BYTES
        else:
            self.low, self.high = -_MAX_GAUGE, _MAX_GAUGE
        # Values are read and summed as the rows give them, and shown in the metric's
        # unit: bytes as GiB. Scaling by a power of two is exact, so it can come last.
        self.factor = 1.0 / _GIB if self.unit == "GiB" else 1.0
        # (sample time, place in its row) where the field was first seen: the order shown.
        self.rank = (_INF, 0)
        # A value was seen that never reached the raw samples (a late row's).
        self.seen = False


class _Plan:
    """Field positions and validators for one row shape."""

    __slots__ = ("length", "get", "count", "slots", "lows", "highs", "places", "slot_set", "version", "sub_job",
                 "unit", "stamp", "interval", "memory", "ranked")


# A unit's bin (a cell): for each slot its samples' interval-weighted sum and their intervals' sum.
_SUMS = (0.0, 0.0)
# For each slot the lowest and the highest value: [low, high, low, high, ...].
_EXTREMES = (_INF, -_INF)


def _pairs(fill: tuple[float, float], slots: int, values: array | None = None) -> array:
    """``values`` (a new array when None) with a ``fill`` pair for each of ``slots`` it lacks."""
    if values is None:
        return array("d", fill) * slots
    if len(values) < 2 * slots:
        values.extend(array("d", fill) * (slots - len(values) // 2))
    return values


class _Unit:
    """One unit's aligned sample columns, latest row, and whole-job totals."""

    __slots__ = ("ts", "weight", "cols", "unsorted", "last_ts", "latest_ts", "latest_doc", "previous", "binned",
                 "blank", "totals", "extremes", "plan", "appends", "extra")

    def __init__(self, slots: int) -> None:
        self.ts = array("d")
        self.weight = array("d")
        self.cols: list[array | None] = [None] * slots
        self.unsorted = False
        self.last_ts = -_INF
        self.latest_ts = -_INF
        self.latest_doc: dict | None = None
        self.previous = _NAN
        self.binned = 0
        self.blank = 0
        self.totals = [0.0] * (2 * slots)
        self.extremes = _pairs(_EXTREMES, slots)
        self.plan: _Plan | None = None
        self.appends: list = []
        self.extra: tuple[int, ...] = ()

    def grow(self, slots: int) -> None:
        if len(self.cols) < slots:
            self.cols.extend([None] * (slots - len(self.cols)))
            self.totals.extend([0.0] * (2 * slots - len(self.totals)))
            _pairs(_EXTREMES, slots, self.extremes)

    def sort(self) -> None:
        """Put the samples in time order, in place: ``appends`` holds the columns' own appends."""
        if not self.unsorted:
            return
        order = sorted(range(len(self.ts)), key=self.ts.__getitem__)
        for column in (self.ts, self.weight, *(column for column in self.cols if column is not None)):
            column[:] = array("d", (column[i] for i in order))
        self.unsorted = False

    def drop_before(self, cutoff: float) -> None:
        self.sort()
        index = bisect_left(self.ts, cutoff)
        if index:
            for column in (self.ts, self.weight, *(column for column in self.cols if column is not None)):
                del column[:index]
            self.binned = max(0, self.binned - index)


class _Series:
    """Everything read so far from one sub-job's rows of one kind."""

    def __init__(self, sub_job_id: str, spec: _Kind = _GPU_KIND) -> None:
        self.sub_job_id = sub_job_id
        self.spec = spec
        self.record_version = spec.record_version
        self.units: dict[int, _Unit] = {}
        self.fields: list[_Field] = []
        self.slot_of: dict[str, int] = {}
        self._plans: dict[int, _Plan] = {}
        self._stamp: object = _MISSING
        self._epoch: float | None = None
        # Any accepted row, measured or not: freshness and the reporting units.
        self.first: float | None = None
        self.latest: float | None = None
        # Rows with a value to chart (an interval and a metric's reading). The
        # charts and the whole-job span cover these, so a row that carries
        # nothing to plot (a malformed or clock-skewed one) can't stretch them.
        self.data_first: float | None = None
        self.data_last: float | None = None
        self.min_interval: float | None = None
        self.reported: dict[float, int] = {}
        # Raw samples are complete from this time on (older ones live only in bins).
        self.raw_start = -math.inf
        self.bin_width = _BIN_START_S
        # bin index -> unit -> cell (see _SUMS)
        self.bins: dict[int, dict[int, array]] = {}
        # bin index -> per slot [lowest, highest] average across units at one
        # sampling interval, folded in from the raw samples before they leave the
        # retention (see trim()).
        self.extremes: dict[int, array] = {}
        # Bins holding rows that arrived after their time had been folded: those
        # rows never reached the raw samples, so their bins' extremes are bounds.
        self.inexact: set[int] = set()
        # bin index -> per slot the lowest and highest value of those rows (see _EXTREMES).
        self.late: dict[int, array] = {}
        # The newest sample time of the read in progress (see ResourceLoader._read).
        self.pass_newest = -_INF
        self.version = 0
        # Rows stamped after this (epoch seconds) are unsupported; each load sets it.
        self.horizon = _INF

    @property
    def chart_start(self) -> float | None:
        return self.data_first if self.data_first is not None else self.first

    # ── reading rows ──

    def _slot(self, key: str, derived: bool = False) -> int:
        slot = self.slot_of.get(key)
        if slot is None:
            slot = self.slot_of[key] = len(self.fields)
            self.fields.append(_Field(key, derived))
            for unit in self.units.values():
                unit.grow(len(self.fields))
        return slot

    def _replan(self, doc: dict) -> _Plan:
        spec = self.spec
        keys = list(doc)
        # The metrics' fields and the ones the header names. Strings are never metrics, so a
        # field holding one (an id, an epoch) is left out of rows of this shape; a null field
        # stays in, since a later row may measure it. The derived memory % keeps its own key:
        # a field of that name would be the same quantity.
        metric_keys = [key for key in keys if (is_metric_field(key) or key in spec.constant) and key != spec.memory
                       and (doc[key] is None or type(doc[key]) in (float, int))]
        chosen = set(metric_keys)
        order = metric_keys + [key for key in keys if key not in chosen]
        plan = _Plan()
        plan.length = len(keys)
        if len(order) > 1:
            plan.get = itemgetter(*order)
        else:
            plan.get = (lambda row, key=order[0]: (row[key],)) if order else (lambda row: ())
        plan.count = len(metric_keys)
        plan.slots = tuple(self._slot(key) for key in metric_keys)
        plan.lows = tuple(self.fields[slot].low for slot in plan.slots)
        plan.highs = tuple(self.fields[slot].high for slot in plan.slots)
        plan.places = tuple(keys.index(key) for key in metric_keys)  # each field's place in its row
        place = {key: index for index, key in enumerate(order)}
        plan.version = place.get("record_version")
        plan.sub_job = place.get("sub_job_id")
        plan.unit = place.get(spec.unit_field)
        plan.stamp = place.get("sample_ts")
        plan.interval = place.get("sample_interval_s")
        used, other = place.get(spec.used), place.get(spec.other)
        plan.memory = (used, other, self._slot(spec.memory, derived=True)) if used is not None and other is not None \
            else None
        plan.slot_set = frozenset(plan.slots + ((plan.memory[2],) if plan.memory is not None else ()))
        plan.ranked = False
        self._plans[plan.length] = plan
        return plan

    def _rank(self, plan: _Plan, ts: float) -> None:
        plan.ranked = True
        for slot, place in zip(plan.slots, plan.places):
            if (ts, place) < self.fields[slot].rank:
                self.fields[slot].rank = (ts, place)

    def _attach(self, unit: _Unit, plan: _Plan) -> None:
        """Point ``unit`` at the columns a row of ``plan``'s shape fills (and at the memory % last)."""
        unit.plan = plan
        targets = plan.slots + ((plan.memory[2],) if plan.memory is not None else ())
        unit.appends = [self._target(unit, position, slot) for position, slot in enumerate(targets)]
        unit.extra = tuple(slot for slot, column in enumerate(unit.cols)
                           if column is not None and slot not in plan.slot_set)

    @staticmethod
    def _target(unit: _Unit, position: int, slot: int) -> Callable[[float], None]:
        """Where a value of ``slot`` goes: its column, or what makes the column on the first measured value."""
        column = unit.cols[slot]
        if column is not None:
            return column.append

        def first(value: float) -> None:
            if value != value:
                return  # nothing measured yet: no column
            made = unit.cols[slot] = array("d", (_NAN,)) * len(unit.ts)
            made.append(value)
            unit.appends[position] = made.append

        return first

    def ingest(self, doc: dict) -> int:
        """Read one row of this kind (a dict); _OK, or _IGNORED for a row this client can't use."""
        plan = self._plans.get(len(doc))
        if plan is None:
            plan = self._replan(doc)
        try:
            values = plan.get(doc)
        except KeyError:  # the same length, other fields
            plan = self._replan(doc)
            values = plan.get(doc)
        if plan.version is None or plan.sub_job is None or plan.unit is None or plan.stamp is None:
            return _IGNORED
        if values[plan.version] != self.record_version or values[plan.sub_job] != self.sub_job_id:
            return _IGNORED
        unit_id = values[plan.unit]
        if type(unit_id) is not int or unit_id < 0:
            return _IGNORED
        stamp = values[plan.stamp]
        if stamp == self._stamp and type(stamp) is str:  # every GPU of a scrape shares its time
            ts = self._epoch
        else:
            ts = _sample_epoch(stamp)
            self._stamp, self._epoch = stamp, ts
        if ts is None or ts > self.horizon:
            return _IGNORED
        interval = values[plan.interval] if plan.interval is not None else None
        if type(interval) is not float or not 0.0 < interval <= _MAX_SAMPLE_INTERVAL_S:
            interval = _bounded_number(interval, 0, _MAX_SAMPLE_INTERVAL_S) or None

        unit = self.units.get(unit_id)
        if unit is None:
            unit = self.units[unit_id] = _Unit(len(self.fields))
        if interval is None:
            # No usable sample_interval_s (absent, 0 or malformed): the time since this
            # unit's row before it, when that looks like a sampling interval.
            gap = ts - unit.previous
            if 0 < gap <= _FALLBACK_INTERVAL_MAX_S:
                interval = gap
        elif interval in self.reported or len(self.reported) < 64:
            self.reported[interval] = self.reported.get(interval, 0) + 1
        unit.previous = ts
        if ts > self.pass_newest:
            self.pass_newest = ts
        if self.first is None or ts < self.first:
            self.first = ts
        if self.latest is None or ts > self.latest:
            self.latest = ts
        if ts >= unit.latest_ts:
            unit.latest_ts, unit.latest_doc = ts, doc
        if not plan.ranked:
            self._rank(plan, ts)

        if ts < self.raw_start:
            measured = self._late(unit_id, plan, values, ts, interval)
        else:
            if unit.plan is not plan:
                self._attach(unit, plan)
            appends = unit.appends
            unmeasured = 0
            # The metric values come first in ``values``; zip stops after the last of them.
            for low, high, append, value in zip(plan.lows, plan.highs, appends, values):
                if type(value) is float and low <= value <= high:
                    append(value)
                elif value is None:
                    unmeasured += 1
                    append(_NAN)
                else:
                    value = _number(value, low, high)
                    if value != value:
                        unmeasured += 1
                    append(value)
            measured = unmeasured < plan.count
            memory = plan.memory
            if memory is not None:
                used, other = values[memory[0]], values[memory[1]]
                if (type(used) is float and type(other) is float
                        and 0.0 <= used <= _MAX_FB_BYTES and 0.0 <= other <= _MAX_FB_BYTES):
                    if self.spec.other_is_free:
                        total = used + other
                        share = 100.0 * used / total if total > 0 else _NAN
                    else:
                        share = (100.0 if used >= other else 100.0 * used / other) if other > 0 else _NAN
                else:
                    share = self._memory(values, memory)
                if share == share:
                    measured = True
                appends[plan.count](share)
            for slot in unit.extra:
                unit.cols[slot].append(_NAN)
            if ts < unit.last_ts:
                unit.unsorted = True
            else:
                unit.last_ts = ts
            unit.ts.append(ts)
            if interval is None:
                unit.weight.append(_NAN)
                unit.blank += 1
            else:
                unit.weight.append(interval)
        if interval is not None:
            if self.min_interval is None or interval < self.min_interval:
                self.min_interval = interval
            if measured:
                if self.data_first is None or ts < self.data_first:
                    self.data_first = ts
                if self.data_last is None or ts > self.data_last:
                    self.data_last = ts
        return _OK

    def _memory(self, values, memory: tuple) -> float:
        """The derived memory used % of one row (NaN when it can't be told)."""
        used = _number(values[memory[0]], 0.0, _MAX_FB_BYTES)
        other = _number(values[memory[1]], 0.0, _MAX_FB_BYTES)
        if used != used or other != other:
            return _NAN
        if self.spec.other_is_free:
            total = used + other
            return 100.0 * used / total if total > 0 else _NAN
        # A node's used can read above its limit (a fallback to the host's numbers): at most 100%.
        return min(100.0, 100.0 * used / other) if other > 0 else _NAN

    def row_values(self, doc: dict | None) -> list[float | None]:
        """Each slot's value in ``doc`` (None: unmeasured), as the rows give it (bytes as bytes)."""
        out: list[float | None] = [None] * len(self.fields)
        if doc is None:
            return out
        for slot, column in enumerate(self.fields):
            if column.derived:
                continue
            value = _number(doc.get(column.key), column.low, column.high)
            out[slot] = value if value == value else None
        slot = self.slot_of.get(self.spec.memory)
        if slot is not None:
            memory = self._memory((doc.get(self.spec.used), doc.get(self.spec.other)), (0, 1, slot))
            out[slot] = memory if memory == memory else None
        return out

    def _late(self, unit_id: int, plan: _Plan, values: tuple, ts: float, interval: float | None) -> bool:
        """A row older than the raw samples: straight into its bin, whose extremes become bounds."""
        found = []
        for position, slot in enumerate(plan.slots):
            value = _number(values[position], plan.lows[position], plan.highs[position])
            if value == value:
                found.append((slot, value))
        if plan.memory is not None:
            memory = self._memory(values, plan.memory)
            if memory == memory:
                found.append((plan.memory[2], memory))
        if not found or interval is None:
            return bool(found)
        key = int(ts // self.bin_width)
        cells = self.bins.setdefault(key, {})
        slots = len(self.fields)
        cell = cells[unit_id] = _pairs(_SUMS, slots, cells.get(unit_id))
        unit = self.units[unit_id]
        totals, extremes = unit.totals, unit.extremes
        late = self.late[key] = _pairs(_EXTREMES, slots, self.late.get(key))
        for slot, value in found:
            offset = 2 * slot
            cell[offset] += value * interval
            cell[offset + 1] += interval
            totals[offset] += value * interval
            totals[offset + 1] += interval
            for store in (extremes, late):
                store[offset] = min(store[offset], value)
                store[offset + 1] = max(store[offset + 1], value)
            self.fields[slot].seen = True
        self.inexact.add(key)  # its interval was folded without it
        return True

    # ── bins ──

    def bin_pending(self) -> None:
        """Add the raw samples read since the last call to their bins (once each, before any leaves).

        Each slot's whole-job sums take the samples one at a time, in the order
        read, so they don't depend on how the file was split into reads (a refresh
        matches a full reload). A bin's sums are what its samples added to them.
        """
        width = self.bin_width
        slots = len(self.fields)

        def key_of(ts: float) -> float:
            return ts // width

        for unit_id, unit in self.units.items():
            start, stop = unit.binned, len(unit.ts)
            if start >= stop:
                continue
            ts = unit.ts
            live = [(slot, column) for slot, column in enumerate(unit.cols) if column is not None]
            _pairs(_EXTREMES, slots, unit.extremes)
            for slot, _column in live:
                low, high = _exact(unit, slot, start, stop)
                if low is None:
                    continue
                extremes = unit.extremes
                if low < extremes[2 * slot]:
                    extremes[2 * slot] = low
                if high > extremes[2 * slot + 1]:
                    extremes[2 * slot + 1] = high
            i = start
            while i < stop:
                key = int(ts[i] // width)
                if unit.unsorted:
                    j = i + 1
                    while j < stop and int(ts[j] // width) == key:
                        j += 1
                else:
                    j = bisect_left(ts, key + 1, i + 1, stop, key=key_of)
                self._bin_run(unit_id, unit, key, i, j, live, slots)
                i = j
            unit.binned = stop

    def _bin_run(self, unit_id: int, unit: _Unit, key: int, i: int, j: int, live: list, slots: int) -> None:
        """Bin samples i..j-1 of one unit, which share bin ``key``.

        Most metrics are measured on every sample, so the interval sums they share
        are folded once.
        """
        weights = unit.weight[i:j]
        keep = None
        if unit.blank and sum(weights) != sum(weights):  # some samples have no usable interval: they don't count
            keep = [k for k, w in enumerate(weights) if w > 0]
            if not keep:
                return
            weights = array("d", (weights[k] for k in keep))
        cells = self.bins.setdefault(key, {})
        cell = cells[unit_id] = _pairs(_SUMS, slots, cells.get(unit_id))
        totals = unit.totals
        folded: dict[float, float] = {}  # interval sums by the sum they start from

        def fold_weights(start: float) -> float:
            found = folded.get(start)
            if found is None:
                found = folded[start] = reduce(add, weights, start)
            return found

        for slot, column in live:
            segment = column[i:j]
            if keep is not None:
                segment = array("d", (segment[k] for k in keep))
            offset = 2 * slot
            before, before_w = totals[offset], totals[offset + 1]
            after = reduce(add, map(mul, segment, weights), before)
            if after == after:  # nothing unmeasured
                after_w = fold_weights(before_w)
            else:
                pairs = [(value * w, w) for value, w in zip(segment, weights) if value == value]
                if not pairs:
                    continue
                after = reduce(add, (part for part, _w in pairs), before)
                after_w = reduce(add, (w for _part, w in pairs), before_w)
            totals[offset], totals[offset + 1] = after, after_w
            cell[offset] += after - before
            cell[offset + 1] += after_w - before_w

    def rebin(self) -> None:
        """Double the bin width until the whole job fits in _MAX_BINS bins."""
        if not self.bins:
            return
        low, high = min(self.bins), max(self.bins)
        slots = len(self.fields)
        while high - low + 1 > _MAX_BINS:
            self.bin_width *= 2
            merged: dict[int, dict[int, array]] = {}
            for key, cells in self.bins.items():
                target = merged.setdefault(key // 2, {})
                for unit_id, cell in cells.items():
                    existing = target.get(unit_id)
                    if existing is None:
                        target[unit_id] = cell
                        continue
                    _pairs(_SUMS, slots, existing)
                    _pairs(_SUMS, slots, cell)
                    for offset in range(0, 2 * slots, 2):
                        existing[offset] += cell[offset]
                        existing[offset + 1] += cell[offset + 1]
            self.bins = merged
            # Extremes merge as extremes: a wider bin never smooths them.
            self.extremes = _merge_extremes(self.extremes, slots)
            self.late = _merge_extremes(self.late, slots)
            self.inexact = {key // 2 for key in self.inexact}
            low, high = low // 2, high // 2

    def trim(self, upto: float | None = None) -> None:
        """Keep raw samples only for the retention before the newest one (or before ``upto``).

        Before samples leave, the average across units at each sampling interval
        among them is folded into their bins' extremes, so the whole job's min and
        max stay exact once its history is older than the retention. A short
        stretch before the cutoff stays as the next fold's context: the value a
        unit last had carries into the intervals just after it.
        """
        if self.latest is None:
            return
        cutoff = self.latest - _RAW_RETENTION_S
        if upto is not None:
            cutoff = min(cutoff, upto)
        if cutoff <= self.raw_start:
            return
        self.bin_pending()
        self._fold(cutoff)
        for unit in self.units.values():
            unit.drop_before(cutoff - _FOLD_CONTEXT_S)
        self.raw_start = cutoff

    def _fold(self, right: float) -> None:
        """Fold the per-interval averages across units from ``raw_start`` to ``right`` into ``extremes``."""
        start = self.chart_start
        if self.min_interval is None or start is None:
            return
        left = max(self.raw_start, start)
        if right < left:
            return
        # One sampling interval per bucket, coarsened as _Native's are when that would
        # be more than _MAX_NATIVE_BUCKETS (a malformed row can report a tiny interval).
        width = max(self.min_interval, (right - left) / (_MAX_NATIVE_BUCKETS - 1))
        count = math.floor((right - left) / width + 1e-9) + 1
        hold_s = max(_REPORTING_TOLERANCE_S, 3 * self.min_interval)
        slots = len(self.fields)
        means, _averages, _pooled = _accumulate(self, left, right, width, count, hold_s, until=right,
                                                totals=False)
        last = count - 1
        for k in range(count):
            values = None
            for slot in range(slots):
                mean = means[slot][k]
                if mean is None:
                    continue
                if values is None:
                    key = int((right - (last - k) * width) // self.bin_width)
                    values = self.extremes[key] = _pairs(_EXTREMES, slots, self.extremes.get(key))
                if mean < values[2 * slot]:
                    values[2 * slot] = mean
                if mean > values[2 * slot + 1]:
                    values[2 * slot + 1] = mean

    def plan(self, data_first: float | None, data_last: float | None) -> None:
        """Before a first read: size the bins by rebin()'s own rule.

        A first read then ends at the width that loading the same rows a chunk at
        a time would reach.
        """
        if data_first is not None and data_last is not None:
            while int(data_last // self.bin_width) - int(data_first // self.bin_width) + 1 > _MAX_BINS:
                self.bin_width *= 2

    def charted(self) -> list[int]:
        """The slots that have a value somewhere, in the order shown (see KindResource.metrics,
        which leaves out the kind's constant fields)."""
        spec = self.spec
        seen = [slot for slot, column in enumerate(self.fields)
                if column.seen or any(unit.cols[slot] is not None for unit in self.units.values())]
        first = [self.slot_of[key] for key in (spec.headline, spec.memory) if self.slot_of.get(key) in seen]
        rest = sorted((slot for slot in seen if slot not in first), key=lambda slot: (self.fields[slot].rank, slot))
        return first + rest


# ─── views ───────────────────────────────────────────────────────────────────


def _merge_extremes(by_bin: dict[int, array], slots: int) -> dict[int, array]:
    """Per-bin extremes for bins twice as wide (see _EXTREMES)."""
    merged: dict[int, array] = {}
    for key, values in by_bin.items():
        existing = merged.get(key // 2)
        if existing is None:
            merged[key // 2] = values
            continue
        _pairs(_EXTREMES, slots, existing)
        _pairs(_EXTREMES, slots, values)
        for offset in range(0, 2 * slots, 2):
            existing[offset] = min(existing[offset], values[offset])
            existing[offset + 1] = max(existing[offset + 1], values[offset + 1])
    return merged


def _grid(left: float, latest: float, min_interval: float | None) -> tuple[float, int]:
    """(bucket width, bucket count) for a chart from ``left`` to ``latest``."""
    span = max(0.0, latest - left)
    if span <= 0:
        return 1.0, 1
    interval = min_interval or span
    width = max(span / (_MAX_BUCKETS - 1), interval)
    count = min(_MAX_BUCKETS, math.floor(span / width + 1e-9) + 1)
    return width, max(1, count)


def _mean(values) -> float | None:
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _accumulate(series: _Series, left: float, latest: float, width: float, count: int, hold_s: float,
                until: float | None = None, per_unit: dict | None = None, slots=None, totals: bool = True):
    """Bucket raw samples between ``left`` and ``latest``.

    Returns per-slot means across units, per-unit time-weighted averages, and
    pooled time-weighted averages (the latter two omitted with
    ``totals=False``). Buckets are anchored on ``latest``. A missing unit holds
    its recent value for ``hold_s``; bridge-enabled kinds also cross one short
    all-unit gap. ``per_unit`` receives each unit's bucket values.
    """
    if slots is None:
        slots = range(len(series.fields))
    slots = list(slots)
    width_s = len(slots)
    last = count - 1
    per_bucket = 1.0 / width
    sums = [[0.0] * count for _ in slots]
    counts = [[0] * count for _ in slots]
    # (first bucket, last bucket, value per slot, unit, bridge): a unit's value carried into buckets it missed.
    holds: list[tuple] = []
    averages: dict[int, list[float | None]] = {}
    pool = [0.0] * (2 * width_s)  # per slot: weighted sum, weight
    bridge_s = (_BRIDGE_INTERVALS * series.min_interval
                if series.spec.bridge and series.min_interval else 0.0)

    def reach(sample_ts: float) -> int:
        """The newest bucket a value sampled at ``sample_ts`` can be carried into."""
        behind = latest - sample_ts - hold_s
        return last if behind <= 0 else last - math.ceil(behind * per_bucket - 1e-9)

    for unit_id, unit in series.units.items():
        unit.sort()
        cols = [unit.cols[slot] if slot < len(unit.cols) else None for slot in slots]
        own = None
        if per_unit is not None:
            own = per_unit[unit_id] = [None if column is None else [None] * count for column in cols]
        ts, weight = unit.ts, unit.weight
        begin = bisect_left(ts, left - 1e-6)
        edge = bisect_left(ts, left - 0.5 * width) if last else begin
        end = max(begin, len(ts) if until is None else bisect_right(ts, until + 1e-6))
        # The sample just before the range carries into its first buckets.
        pre = edge - 1 if edge and weight[edge - 1] > 0 and ts[edge - 1] >= left - hold_s else -1
        # The samples that count (those with an interval), in order. Those in
        # [edge, begin) open the first bucket; the totals start without them.
        if unit.blank == 0:
            picked = None
            times, weights = ts[edge:end], weight[edge:end]
            opening = begin - edge
        else:
            picked = [i for i in range(edge, end) if weight[i] > 0]
            times = array("d", (ts[i] for i in picked))
            weights = array("d", (weight[i] for i in picked))
            opening = bisect_left(picked, begin)
        n = len(times)
        if last:
            buckets = [last - int((latest - t) * per_bucket + 0.5) for t in times[opening:]]  # latest >= t
            below = bisect_left(buckets, 0)
            if below:
                buckets[:below] = [0] * below
        else:
            buckets = [0] * (n - opening)
        buckets[:0] = [0] * opening
        # Runs: (bucket, first sample, after the last). One sample per bucket in
        # consecutive buckets (a regular series at its own interval) needs no list.
        single = n > 0 and buckets == list(range(buckets[0], buckets[0] + n))
        if single:
            runs = None
        else:
            runs = []
            start = 0
            for position in range(1, n):
                if buckets[position] != buckets[start]:
                    runs.append((buckets[start], start, position))
                    start = position
            if n:
                runs.append((buckets[start], start, n))

        # Where the unit's values carry (the same for every metric): (first, last
        # bucket, the run whose values carry (-1: the sample before the range), bridge).
        carried: list[tuple] = []
        held_bucket = -1 if pre >= 0 else -2
        held_ts = ts[pre] if pre >= 0 else 0.0
        source = -1
        # With one sample per consecutive bucket, only the first run can follow a gap.
        for index, (bucket, first, after) in enumerate([(buckets[0], 0, 1)] if single else runs):
            if held_bucket > -2 and bucket > held_bucket + 1:
                if bridge_s and times[first] - held_ts < bridge_s:
                    carried.append((held_bucket + 1, bucket - 1, source, True))
                else:
                    stop = min(bucket - 1, reach(held_ts))
                    if stop > held_bucket:
                        carried.append((held_bucket + 1, stop, source, False))
            if held_bucket == -2 and bucket > 0:  # its first values, with nothing before them
                carried.append((bucket - 1, bucket - 1, index, False))
            held_bucket, held_ts, source = bucket, times[after - 1], index
        if single:
            held_bucket, held_ts, source = buckets[-1], times[n - 1], n - 1
        if held_bucket > -2 and held_bucket < last:
            stop = reach(held_ts)
            if stop > held_bucket:
                carried.append((held_bucket + 1, stop, source, False))
        carried_values = [[None] * width_s for _ in carried]

        # Each run's interval sum, shared by every metric measured on all its samples.
        run_weights = None if single else [weights[a] + 0.0 if b - a == 1 else reduce(add, weights[a:b], 0.0)
                                           for _bucket, a, b in runs]
        whole_w = None  # the range's interval sum, as the slow path adds it
        mine: list[float | None] = [None] * width_s
        for j, column in enumerate(cols):
            if column is None:
                continue
            values = column[edge:end] if picked is None else array("d", (column[i] for i in picked))
            counted = None  # per run value (the values themselves when single)
            total = total_w = None
            if single:
                check = sum(values)
                if check == check:  # nothing unmeasured: C loops over the whole range
                    # One sample per bucket: each bucket's value is its sample's.
                    counted = values
                    first = buckets[0]
                    stop = first + n
                    row = sums[j]
                    row[first:stop] = map(add, row[first:stop], values)
                    tally = counts[j]
                    tally[first:stop] = map(add, tally[first:stop], repeat(1, n))
                    if own is not None:
                        own[j][first:stop] = values
                    if totals:
                        # As the slow path sums: the opening sample is taken off, then added back.
                        if whole_w is None:
                            whole_w = reduce(add, weights[opening:], 0.0 if opening else -0.0)
                        total = reduce(add, map(mul, values[opening:], weights[opening:]), 0.0 if opening else -0.0)
                        total_w = whole_w
            if counted is None:
                # A run at a time; unmeasured values (NaN) don't count.
                spans = runs if runs is not None else [(buckets[r], r, r + 1) for r in range(n)]
                counted = [None] * len(spans)
                products = list(map(mul, values, weights))
                whole = sum(products)
                clean = whole == whole  # nothing unmeasured anywhere in the range
                head = products[:opening]
                if head and sum(head) != sum(head):
                    head_w = [w for p, w in zip(head, weights[:opening]) if p == p]
                    head = [p for p in head if p == p]
                else:
                    head_w = weights[:opening]
                total = -reduce(add, head, 0.0)
                total_w = -reduce(add, head_w, 0.0)
                row, tally = sums[j], counts[j]
                mine_own = own[j] if own is not None else None
                for index, (bucket, a, b) in enumerate(spans):
                    if b - a == 1:
                        value = values[a]
                        if value != value:
                            continue
                        acc, acc_w = 0.0 + products[a], 0.0 + weights[a]
                    else:
                        segment = products[a:b]
                        measured = b - a
                        if clean or sum(segment) == sum(segment):
                            acc = reduce(add, segment, 0.0)
                            acc_w = run_weights[index] if run_weights is not None else reduce(add, weights[a:b], 0.0)
                        else:
                            acc = acc_w = 0.0
                            measured = 0
                            for part, w in zip(segment, weights[a:b]):
                                if part == part:
                                    acc += part
                                    acc_w += w
                                    measured += 1
                        if measured == 1:  # one measured sample: its own value
                            value = next(v for v in values[a:b] if v == v)
                        elif acc_w > 0:
                            value = acc / acc_w
                    if acc_w > 0:
                        counted[index] = value
                        row[bucket] += value
                        tally[bucket] += 1
                        if mine_own is not None:
                            mine_own[bucket] = value
                        total += acc
                        total_w += acc_w
                if not totals:
                    total = total_w = None
            for held, (first_bucket, stop_bucket, run, bridge) in zip(carried_values, carried):
                if run < 0:
                    value = column[pre]
                    held[j] = value if value == value else None
                else:
                    held[j] = counted[run]
            if totals and total_w is not None and total_w > 0:
                mine[j] = total / total_w
                pool[2 * j] += total
                pool[2 * j + 1] += total_w
        averages[unit_id] = mine
        for held, (first_bucket, stop_bucket, _run, bridge) in zip(carried_values, carried):
            holds.append((first_bucket, stop_bucket, held, unit_id, bridge))
    # Every unit's own samples are in, so a bucket's count says whether any unit sampled
    # it; a bridge counts too. Each unit's holds cover only buckets it skipped, so it
    # counts at most once per bucket: with its own samples, or with its one carried value.
    covered = None
    if any(hold[4] for hold in holds):
        covered = [[n > 0 for n in counts[j]] for j in range(width_s)]
        for first, stop, values, _unit_id, bridge in holds:
            if bridge:
                for j in range(width_s):
                    if values[j] is not None:
                        for k in range(first, stop + 1):
                            covered[j][k] = True
    for first, stop, values, hold_unit, _bridge in holds:
        mine_own = per_unit.get(hold_unit) if per_unit is not None else None
        for j in range(width_s):
            value = values[j]
            if value is None:
                continue
            bucket_sums, bucket_counts = sums[j], counts[j]
            inside = covered[j] if covered is not None else bucket_counts
            row = mine_own[j] if mine_own is not None else None
            for k in range(first, stop + 1):
                if inside[k]:
                    bucket_sums[k] += value
                    bucket_counts[k] += 1
                    if row is not None and row[k] is None:
                        row[k] = value
    means = [[total / n if n else None for total, n in zip(bucket_sums, bucket_counts)]
             for bucket_sums, bucket_counts in zip(sums, counts)]
    pooled = [pool[2 * j] / pool[2 * j + 1] if pool[2 * j + 1] > 0 else None for j in range(width_s)]
    return means, averages, pooled


_NO_OLDER: tuple[float | None, float | None] = (None, None)


def _stats(low: float | None, high: float | None, avg: float | None, latest: float | None, same: Callable,
           older: tuple[float | None, float | None] = _NO_OLDER) -> SeriesStats:
    """``avg``: the unit-time mean; ``older``: the lowest and highest sample of the history known only as bins.

    ``same(a, b)``: whether two values read the same at the precision shown.
    """
    # The newest capture is a point in every scope.
    lows = [value for value in (low, latest) if value is not None]
    highs = [value for value in (high, latest) if value is not None]
    shown_min = min(lows) if lows else None
    shown_max = max(highs) if highs else None
    floor, ceiling = older
    # An average across units lies between the lowest and highest samples it
    # averages, so that history can't hide a lower min than its lowest sample, or
    # a higher max than its highest.
    return SeriesStats(
        min=shown_min, avg=avg, max=shown_max, latest=latest,
        min_bound=floor is not None and shown_min is not None and floor < shown_min and not same(floor, shown_min),
        max_bound=ceiling is not None and shown_max is not None and ceiling > shown_max
        and not same(ceiling, shown_max),
    )


class _Native:
    """The average across units at each sampling interval over the raw samples.

    Buckets are anchored on the newest sample, one sampling interval wide.
    ``extremes(left)`` gives each slot's min and max over the buckets from
    ``left`` to the newest; nested scopes therefore nest their extremes.
    """

    def __init__(self, series: _Series, hold_s: float, slots: list[int]) -> None:
        self.slots = len(slots)
        self.ready = series.min_interval is not None and series.latest is not None
        self._found: dict[int, list] = {}
        if not self.ready:
            return
        self.latest = series.latest
        left = max(series.chart_start, series.raw_start)
        span = max(0.0, self.latest - left)
        self.width = max(series.min_interval, span / (_MAX_NATIVE_BUCKETS - 1)) if span > 0 else 1.0
        count = math.floor(span / self.width + 1e-9) + 1 if span > 0 else 1
        self.last = count - 1
        self.means, _averages, _pooled = _accumulate(series, left, self.latest, self.width, count, hold_s,
                                                     slots=slots, totals=False)

    def extremes(self, left: float) -> list[tuple[float | None, float | None]]:
        if not self.ready:
            return [(None, None)] * self.slots
        k = self.last - round((self.latest - left) / self.width) if self.last else 0
        k = min(self.last, max(0, k))
        found = self._found.get(k)
        if found is None:
            found = self._found[k] = []
            for means in self.means:
                values = [value for value in means[k:] if value is not None]
                found.append((min(values), max(values)) if values else (None, None))
        return found


def _merge(pair: tuple[float | None, float | None], values) -> tuple[float | None, float | None]:
    points = [value for value in (*pair, *values) if value is not None]
    return (min(points), max(points)) if points else (None, None)


def _derived_interval(series: _Series) -> float | None:
    """The median spacing of each unit's recent consecutive samples."""
    deltas: list[float] = []
    for unit in series.units.values():
        unit.sort()
        ts = unit.ts[-(_INTERVAL_DELTAS + 1):]
        deltas.extend(b - a for a, b in zip(ts, ts[1:]) if b - a > 1e-3)
    return median(deltas) if deltas else None


class _Capture:
    """The newest capture: the units reporting in it, each unit's newest values, and their average."""

    def __init__(self, series: _Series, tolerance: float) -> None:
        newest = series.latest or 0.0
        self.reporting = tuple(sorted(unit_id for unit_id, unit in series.units.items()
                                      if unit.latest_doc is not None and unit.latest_ts >= newest - tolerance))
        self.values = {unit_id: series.row_values(unit.latest_doc) for unit_id, unit in series.units.items()}
        rows = [self.values[unit_id] for unit_id in self.reporting]
        self.mean = [_mean(row[slot] for row in rows) for slot in range(len(series.fields))]


class _Frame:
    """One window of one kind, as built on the loading thread: every unit's and the mean's values."""

    __slots__ = ("name", "start", "end", "keys", "units", "charts", "stats", "unit_charts", "unit_stats")

    def view(self, kind: str, unit: int | None, units: tuple[Unit, ...]) -> KindView | None:
        if unit is None:
            # The mean of one unit is that unit's values: it goes by its name.
            label = across_text(kind, self.units) or (units[0].name if len(units) == 1 else "")
            return KindView(name=self.name, start=self.start, end=self.end, label=label, charts=self.charts,
                            stats=self.stats)
        columns = self.unit_charts.get(unit)
        if columns is None:
            return None
        charts = {key: tuple(None if value != value else value for value in column)
                  for key, column in zip(self.keys, columns) if column is not None}
        stats = dict(zip(self.keys, self.unit_stats[unit]))
        return KindView(name=self.name, start=self.start, end=self.end, label=unit_name(kind, unit), charts=charts,
                        stats=stats)


def _same_shown(unit: str) -> Callable[[float, float], bool]:
    return lambda a, b: format_value(a, unit) == format_value(b, unit)


def _exact(unit: _Unit, slot: int, begin: int, stop: int | None = None) -> tuple[float | None, float | None]:
    """The lowest and highest of one unit's samples from ``begin`` on, before ``stop`` (those with an interval)."""
    column = unit.cols[slot] if slot < len(unit.cols) else None
    if column is None or begin >= len(column):
        return None, None
    segment = column[begin:stop]
    low, high = min(segment), max(segment)  # they pass over NaN, unless it comes first
    if low == low and high == high and not unit.blank:
        return low, high
    values = [value for value, w in zip(segment, unit.weight[begin:stop]) if value == value and w > 0]
    return (min(values), max(values)) if values else (None, None)


def _scaled(value: float | None, factor: float) -> float | None:
    return value if value is None or factor == 1.0 else value * factor


def _chart(values, factor: float) -> tuple[float | None, ...]:
    return tuple(values) if factor == 1.0 else tuple(None if v is None else v * factor for v in values)


def _column(values, factor: float) -> array:
    return array("d", (_NAN if v is None else v * factor for v in values))


def _finish_frame(series: _Series, name: str, left: float, slots: list[int], capture: _Capture, means, per_unit,
                  extremes, pooled, older, unit_extremes, averages) -> _Frame:
    """A frame from its parts: its charts, and the stats of its mean and units, in each metric's unit."""
    keys = [series.fields[slot].key for slot in slots]
    shown = [_same_shown(series.fields[slot].unit) for slot in slots]
    factors = [series.fields[slot].factor for slot in slots]
    frame = _Frame()
    frame.name, frame.start, frame.end, frame.keys = name, _utc(left), _utc(series.latest), keys
    frame.charts = {key: _chart(values, factor) for key, values, factor in zip(keys, means, factors)}
    frame.unit_charts = {unit_id: [None if values is None else _column(values, factor)
                                   for values, factor in zip(columns, factors)]
                         for unit_id, columns in per_unit.items()}
    parts = [tuple(_scaled(value, factors[j]) for value in (low, high, pooled[j], capture.mean[slot]))
             + (tuple(_scaled(value, factors[j]) for value in older[j]),)
             for j, (slot, (low, high)) in enumerate(zip(slots, extremes))]
    frame.stats = {keys[j]: _stats(low, high, avg, latest, shown[j], bounds)
                   for j, (low, high, avg, latest, bounds) in enumerate(parts)}
    frame.unit_stats = {}
    for unit_id in series.units:
        latest = capture.values[unit_id]
        frame.unit_stats[unit_id] = [
            _stats(*(_scaled(value, factors[j]) for value in (low, high, averages[unit_id][j], latest[slot])), shown[j])
            for j, (slot, (low, high)) in enumerate(zip(slots, unit_extremes[unit_id]))]
    frame.units = sum(1 for unit_id in series.units if any(avg is not None for avg in averages[unit_id]))
    return frame


def _raw_frame(series: _Series, name: str, duration: timedelta | None, capture: _Capture, native: _Native,
               hold_s: float, slots: list[int]) -> _Frame:
    """A window (or a job within the raw retention) from the raw samples."""
    latest = series.latest
    first = series.chart_start
    left = first if duration is None else max(first, latest - duration.total_seconds())
    width, count = _grid(left, latest, series.min_interval)
    per_unit: dict = {}
    means, averages, pooled = _accumulate(series, left, latest, width, count, hold_s, per_unit=per_unit,
                                          slots=slots)
    unit_extremes = {}
    for unit_id, unit in series.units.items():
        begin = bisect_left(unit.ts, left - 1e-6)
        unit_extremes[unit_id] = [_exact(unit, slot, begin) for slot in slots]
    return _finish_frame(series, name, left, slots, capture, means, per_unit, native.extremes(left), pooled,
                         [_NO_OLDER] * len(slots), unit_extremes, averages)


def _binned_frame(series: _Series, name: str, capture: _Capture, native: _Native, slots: list[int]) -> _Frame:
    """The whole job from the running bins, once it is longer than the raw retention."""
    width = series.bin_width
    charts: list[list] = [[] for _ in slots]
    own: dict[int, list[list]] = {unit_id: [[] for _ in slots] for unit_id in series.units}
    # Per slot the lowest and highest average across units at one interval in the
    # history the raw samples no longer hold.
    folded = [[_INF, -_INF] for _ in slots]
    # Per slot the lowest and highest of the rows that reached that history's
    # bins after it was folded (see _Series.inexact).
    limits = [[_INF, -_INF] for _ in slots]
    # Per unit and slot its lowest and highest sample: every one of them has been binned.
    exact = {}
    for unit_id, unit in series.units.items():
        extremes = _pairs(_EXTREMES, len(series.fields), unit.extremes)
        exact[unit_id] = [[extremes[2 * slot], extremes[2 * slot + 1]] for slot in slots]
    if series.bins:
        # rebin() keeps the bins' keys within _MAX_BINS of each other, and every bin
        # holds a charted row, so no row can stretch this walk.
        for key in range(min(series.bins), max(series.bins) + 1):
            cells = series.bins.get(key, {})
            folded_here = key * width < series.raw_start
            values = series.extremes.get(key) if folded_here else None
            inexact = folded_here and key in series.inexact
            late = series.late.get(key) if inexact else None
            for j, slot in enumerate(slots):
                offset = 2 * slot
                bucket = []
                for unit_id, cell in cells.items():
                    if len(cell) > offset + 1 and cell[offset + 1] > 0:
                        value = cell[offset] / cell[offset + 1]
                        bucket.append(value)
                        own[unit_id][j].append(value)
                if late is not None and len(late) > offset + 1:
                    limits[j][0] = min(limits[j][0], late[offset])
                    limits[j][1] = max(limits[j][1], late[offset + 1])
                mean = sum(bucket) / len(bucket) if bucket else None
                charts[j].append(mean)
                if values is not None and len(values) > 2 * slot + 1:
                    folded[j][0] = min(folded[j][0], values[2 * slot])
                    folded[j][1] = max(folded[j][1], values[2 * slot + 1])
                if inexact and mean is not None:
                    # Rows that never reached the raw samples: only their bin's average is
                    # known, and its lowest and highest sample bound the true extremes.
                    folded[j][0] = min(folded[j][0], mean)
                    folded[j][1] = max(folded[j][1], mean)
            for unit_id, columns in own.items():
                for j in range(len(slots)):
                    if len(columns[j]) < len(charts[j]):
                        columns[j].append(None)
    averages = {}
    pool = [[0.0, 0.0] for _ in slots]
    for unit_id, unit in series.units.items():
        mine = []
        for j, slot in enumerate(slots):
            total, weight = unit.totals[2 * slot], unit.totals[2 * slot + 1]
            mine.append(total / weight if weight > 0 else None)
            if weight > 0:
                pool[j][0] += total
                pool[j][1] += weight
        averages[unit_id] = mine
    pooled = [total / weight if weight > 0 else None for total, weight in pool]
    # The raw samples give exact extremes for the retention, and folded ones for the
    # history before it; an inexact bin's share may be a bound, which its lowest
    # and highest samples tell.
    extremes = [_merge(pair, [value for value in folded[j] if math.isfinite(value)])
                for j, pair in enumerate(native.extremes(series.chart_start))]
    older = [tuple(value if math.isfinite(value) else None for value in pair) for pair in limits]
    unit_extremes = {unit_id: [tuple(value if math.isfinite(value) else None for value in pair) for pair in found]
                     for unit_id, found in exact.items()}
    return _finish_frame(series, name, series.chart_start, slots, capture, charts, own, extremes, pooled, older,
                         unit_extremes, averages)


def _snapshot(series: _Series, partial: bool, shown_window: str | None = None,
              previous: KindResource | None = None, same_rows: bool = False) -> KindResource:
    """Everything the UI shows of one kind, built once per change (on the loading thread).

    With ``shown_window`` and a ``previous`` snapshot, only the frames of the window on screen and of
    the whole job (the stats beside every window) are built; the others are carried from
    ``previous``, marked stale. On a job with hundreds of GPUs the long windows took seconds
    a refresh, and were rebuilt every refresh whether shown or not. With ``same_rows`` (no row
    read since ``previous``: a window was picked), only the stale frames asked for are built,
    and the rest of ``previous`` is carried as it was, fresh or not.
    """
    series.bin_pending()
    spec = series.spec
    interval = _derived_interval(series)
    # How long a unit counts as reporting without a new sample.
    hold_s = max(_REPORTING_TOLERANCE_S, 3 * (interval or series.min_interval or 0))
    capture = _Capture(series, hold_s)
    slots = series.charted()
    # The fields the header names are in each unit's newest values, but in no chart or stats row.
    shown = [slot for slot in slots if series.fields[slot].key not in spec.constant]
    reuse = same_rows and previous is not None and previous._native is not None
    native = previous._native if reuse else _Native(series, hold_s, shown)
    raw_covers_job = series.raw_start <= series.chart_start
    if shown_window is None or previous is None:
        wanted = None
    elif same_rows:
        wanted = {name for name in (shown_window, "all") if name in previous._stale}
    else:
        wanted = {shown_window, "all"}
    frames = {}
    stale = set()
    for name, duration in RANGES.items():
        carried = previous._frames.get(name) if wanted is not None and name not in wanted else None
        if carried is not None:
            frames[name] = carried
            if not same_rows or name in previous._stale:
                stale.add(name)
        else:
            frames[name] = (_raw_frame(series, name, duration, capture, native, hold_s, shown)
                            if duration is not None or raw_covers_job
                            else _binned_frame(series, name, capture, native, shown))
    metrics = []
    for slot in shown:
        column = series.fields[slot]
        if column.unit == "%":
            scale = 100.0
        else:
            peak = max((unit.extremes[2 * slot + 1] for unit in series.units.values()
                        if len(unit.extremes) > 2 * slot + 1), default=-_INF)
            if not math.isfinite(peak):  # nothing charted: the newest values
                peak = max((row[slot] for row in capture.values.values() if row[slot] is not None), default=None)
            scale = nice_ceiling(_scaled(peak, column.factor))
        name = metric_name(column.key)
        metrics.append(Metric(key=column.key, name=name, unit=column.unit, scale=scale))
    keys = [series.fields[slot].key for slot in slots]
    factors = [series.fields[slot].factor for slot in slots]
    reporting = set(capture.reporting)
    units = tuple(
        Unit(id=unit_id, name=unit_name(spec.name, unit_id), reporting=unit_id in reporting,
             latest_sample_ts=_utc(series.units[unit_id].latest_ts),
             latest={key: _scaled(capture.values[unit_id][slot], factor)
                     for key, slot, factor in zip(keys, slots, factors)})
        for unit_id in sorted(series.units)
    )
    reported = max(series.reported, key=series.reported.get) if series.reported else None
    return KindResource(
        kind=spec.name,
        sub_job_id=series.sub_job_id,
        metrics=tuple(metrics),
        units=units,
        latest_sample_ts=_utc(series.latest),
        first_sample_ts=_utc(series.first),
        sample_interval_s=interval,
        reported_interval_s=reported,
        partial=partial,
        reporting_unit_ids=capture.reporting,
        latest={key: _scaled(capture.mean[slot], factor) for key, slot, factor in zip(keys, slots, factors)},
        _frames=frames,
        _stale=frozenset(stale),
        _native=native,
    )


# ─── loading ─────────────────────────────────────────────────────────────────


def _line_start(handle, position: int) -> int:
    """The offset of the first line that starts at or after ``position``."""
    if position <= 0:
        return 0
    handle.seek(position - 1)
    scanned = position - 1
    while True:
        block = handle.read(64 * 1024)
        if not block:
            return scanned
        newline = block.find(b"\n")
        if newline >= 0:
            return scanned + newline + 1
        scanned += len(block)


def _kind_series(sub_job_id: str) -> dict[str, _Series]:
    return {kind: _Series(sub_job_id, spec) for kind, spec in _KIND_SPECS.items()}


@dataclass
class _FileState:
    series: _Series  # the GPU rows
    system: _Series  # the system rows
    ino: tuple[int, int] = (0, 0)
    names: list[str] = field(default_factory=list)
    offset: int = 0  # bytes read: the end of the last complete line
    head: bytes = b""
    tail: bytes = b""
    # Rows that could not be read (malformed, or a version, sub-job, unit or time this client can't use).
    ignored: int = 0
    # kind -> (series version, its KindResource)
    built: dict = field(default_factory=dict)

    @property
    def kinds(self) -> tuple[_Series, _Series]:
        return self.series, self.system


class ResourceLoader:
    """Single-threaded incremental reader for a job's telemetry cache."""

    def __init__(self, metrics_root: Path) -> None:
        self.metrics_root = Path(metrics_root)
        self._files: dict[str, _FileState] = {}
        self.bytes_read = 0
        self.documents_read = 0
        self.last: ResourceData | None = None

    def load(
        self,
        *,
        priority: str | None = None,
        window: timedelta | None = None,
        on_partial: Callable[[ResourceData], None] | None = None,
        shown: str | None = None,
    ) -> ResourceData:
        """Read changes, publishing a priority file's newest window first.

        ``shown``, the window on screen (a key of RANGES), builds only its frame and the whole
        job's on a change, carrying the rest from the snapshot before (see KindResource.fresh);
        None builds every window.
        """
        present: dict[str, tuple[Path, list[str], int]] = {}
        for path in _metric_files(self.metrics_root):
            manifest = _read_manifest(path)
            if manifest is not None:
                present[path.parent.name] = (path, *manifest)
        for sub_job_id in list(self._files):
            if sub_job_id not in present:
                del self._files[sub_job_id]
        horizon = time.time() + _MAX_AHEAD_S
        for sub_job_id in sorted(present, key=lambda sid: sid != priority):
            path, names, committed = present[sub_job_id]
            partial = on_partial if sub_job_id == priority else None
            try:
                self._update(sub_job_id, path, names, committed, window or RANGES[DEFAULT_RANGE], partial, horizon)
            except OSError:
                self._files.pop(sub_job_id, None)
            except BaseException:
                # A partial failed update must restart to avoid double counting.
                self._files.pop(sub_job_id, None)
                raise
        data = self._data(set(), shown)
        self.last = data
        return data

    def preview(self, sub_job_id: str, path: Path) -> ResourceData:
        """One sub-job's telemetry from a file of its newest rows alone (``download_metrics``'
        preview), marked partial: the rest of its history is still downloading. The loader's
        own state is untouched, so the sub-job's file is read as ever once it lands."""
        fresh = _kind_series(sub_job_id)
        state = _FileState(series=fresh[GPU], system=fresh[SYSTEM])
        horizon = time.time() + _MAX_AHEAD_S
        with Path(path).open("rb") as handle:
            size = os.fstat(handle.fileno()).st_size
            spans = self._peek(handle, size, sub_job_id, horizon)
            for series in state.kinds:
                series.horizon = horizon
                series.plan(*spans[series.spec.name])
            self._read(state, handle, 0, size)
        found: dict[str, KindResource] = {}
        for series in state.kinds:
            series.trim()
            if series.latest is not None:
                found[series.spec.name] = _snapshot(series, True)
        return ResourceData(kinds={sub_job_id: found} if found else {},
                            ignored_rows_by_sub_job={sub_job_id: state.ignored} if state.ignored else {})

    def _data(self, partial: set[str], shown: str | None = None) -> ResourceData:
        ignored: dict[str, int] = {}
        kinds: dict[str, dict[str, KindResource]] = {}
        for sub_job_id in sorted(self._files):
            state = self._files[sub_job_id]
            if state.ignored:
                ignored[sub_job_id] = state.ignored
            found: dict[str, KindResource] = {}
            for series in state.kinds:
                if series.latest is None:
                    continue
                kind = series.spec.name
                if sub_job_id in partial:
                    resource = _snapshot(series, True)
                else:
                    built = state.built.get(kind)
                    if built is None or built[0] != series.version or not built[1].fresh(shown):
                        # The first snapshot builds every window, so none is ever missing.
                        resource = _snapshot(series, False, shown, built[1] if built else None,
                                             same_rows=built is not None and built[0] == series.version)
                        state.built[kind] = (series.version, resource)
                    else:
                        resource = built[1]
                found[kind] = resource
            if found:
                kinds[sub_job_id] = found
        return ResourceData(kinds=kinds, ignored_rows_by_sub_job=ignored)

    def _update(self, sub_job_id: str, path: Path, names: list[str], committed: int, window: timedelta,
                on_partial: Callable[[ResourceData], None] | None, horizon: float) -> None:
        with path.open("rb") as handle:
            stat = os.fstat(handle.fileno())
            ino = (stat.st_dev, stat.st_ino)
            state = self._files.get(sub_job_id)
            if state is not None and self._extends(state, handle, ino, names, committed):
                if committed > state.offset:
                    for series in state.kinds:
                        series.horizon = horizon
                    state.offset = self._read(state, handle, state.offset, committed)
                    for series in state.kinds:
                        series.trim()
                    state.tail = self._fingerprint(handle, state.offset)
                state.names = list(names)
                return
            # New, rebuilt or rewritten: read it from the start.
            fresh = _kind_series(sub_job_id)
            state = self._files[sub_job_id] = _FileState(series=fresh[GPU], system=fresh[SYSTEM], ino=ino,
                                                         names=list(names))
            spans = self._peek(handle, committed, sub_job_id, horizon)
            for series in state.kinds:
                series.horizon = horizon
                series.plan(*spans[series.spec.name])
            if on_partial is not None and committed >= _NEWEST_FIRST_MIN_BYTES:
                end, rest = self._read_newest(state, handle, committed, window)
                if rest > 0:
                    on_partial(self._data({sub_job_id}))
                    self._read(state, handle, 0, rest)
            else:
                end = self._read(state, handle, 0, committed)
            state.offset = end
            for series in state.kinds:
                series.trim()
            state.head = self._fingerprint(handle, min(_FINGERPRINT, end))
            state.tail = self._fingerprint(handle, end)

    def _read_newest(self, state: _FileState, handle, committed: int, window: timedelta) -> tuple[int, int]:
        """Read slices back from the end until ``window`` is covered.

        The GPU rows decide while any have been read; a file whose newest slices
        hold only system rows is judged by those. Returns (the offset after the
        last complete line, the start of what is left).
        """
        position, step = committed, _NEWEST_FIRST_STEP
        end: int | None = None
        while position > 0:
            begin = _line_start(handle, position - step) if position > step else 0
            if begin >= position:  # one line longer than the step
                step *= 2
                continue
            stop = self._read(state, handle, begin, position, fold=False)
            if end is None:
                end = stop
            position = begin
            series = state.series if state.series.first is not None else state.system
            if series.first is not None and series.first <= (
                series.latest - window.total_seconds() - _NEWEST_FIRST_SLACK_S
            ):
                break
            step *= 2
        return (end or 0), position

    def _extends(self, state: _FileState, handle, ino: tuple[int, int], names: list[str], committed: int) -> bool:
        """Whether this is the file read before, with bytes only appended."""
        if ino != state.ino or committed < state.offset or names[: len(state.names)] != state.names:
            return False
        return (
            self._fingerprint(handle, len(state.head)) == state.head
            and self._fingerprint(handle, state.offset) == state.tail
        )

    @staticmethod
    def _fingerprint(handle, end: int) -> bytes:
        start = max(0, end - _FINGERPRINT)
        handle.seek(start)
        return handle.read(end - start)

    @staticmethod
    def _peek(handle, committed: int, sub_job_id: str, horizon: float) -> dict[str, tuple[float | None, float | None]]:
        """Per kind (first charted, newest charted) sample time, from the two ends of the file.

        Scratch series ingest the rows there, so they are judged exactly as the
        read will judge them: a file that ends in rows of the other kind still
        sizes each kind's bins from its own newest charted row.
        """

        def scan(lines) -> dict[str, _Series]:
            scratch = _kind_series(sub_job_id)
            for series in scratch.values():
                series.horizon = horizon
            for raw in lines:
                try:
                    doc = json.loads(raw)
                except (RecursionError, UnicodeDecodeError, ValueError):
                    continue
                if type(doc) is not dict:
                    continue
                kind = doc.get("kind")
                series = scratch[GPU] if kind is None or kind == GPU else scratch[SYSTEM] if kind == SYSTEM else None
                if series is not None:
                    series.ingest(doc)
            return scratch

        handle.seek(0)
        head = scan(handle.read(min(committed, _PEEK_BYTES)).split(b"\n")[:-1])
        start = max(0, committed - _PEEK_BYTES)
        handle.seek(start)
        lines = handle.read(committed - start).split(b"\n")[:-1]
        tail = scan(lines[1:] if start else lines)  # the first may be the end of an earlier line
        return {kind: (head[kind].data_first, tail[kind].data_last) for kind in _KIND_SPECS}

    def _read(self, state: _FileState, handle, start: int, end: int, *, fold: bool = True) -> int:
        """Ingest the complete lines in [start, end); return the offset after the last one.

        Rows in the file run oldest first, so after each block the samples well
        behind the newest row read so far leave the raw retention (``fold``):
        their per-interval extremes go into the bins (see :meth:`_Series.trim`).
        Each kind folds on its own newest row, since the kinds interleave in the
        file. A newest-first slice (``fold=False``) is followed by older rows, so
        it folds nothing.
        """
        gpu, system = state.series, state.system
        for series in (gpu, system):
            series.pass_newest = -_INF
            series._plans = {}  # row shapes looked at afresh: each read can move a field's first sighting earlier
        gpu_ingest, system_ingest = gpu.ingest, system.ingest
        loads, scan = json.loads, _SCAN
        handle.seek(start)
        position = start
        remaining = end - start
        carry = b""
        documents = 0
        gpu_rows = system_rows = 0
        ignored = 0
        while remaining > 0:
            block = handle.read(min(remaining, _READ_BLOCK))
            if not block:
                break
            remaining -= len(block)
            self.bytes_read += len(block)
            if carry:
                block = carry + block
            cut = block.rfind(b"\n") + 1
            carry = block[cut:]  # an unterminated line is not complete yet
            lines: list = []
            text = False
            if cut:
                body = block[:cut - 1]
                # Decoded once for the block: a str parses faster than bytes, and in ASCII
                # each line's characters are its bytes, which the offsets count.
                text = body.isascii()
                lines = body.decode("ascii").split("\n") if text else body.split(b"\n")
            for raw in lines:
                position += len(raw) + 1
                if not raw or raw.isspace():
                    continue
                documents += 1
                try:
                    # JSON's own scanner, called directly; anything it doesn't read to the
                    # end of the line (spaces around it, more after it) json.loads judges.
                    try:
                        doc, stop = scan(raw, 0) if text else (None, -1)
                    except StopIteration:
                        stop = -1
                    if stop != len(raw):
                        doc = loads(raw)
                except (RecursionError, UnicodeDecodeError, ValueError):
                    ignored += 1
                    continue
                if type(doc) is not dict:
                    ignored += 1
                    continue
                kind = doc.get("kind")
                if kind is None or kind == GPU:
                    gpu_rows += 1
                    if gpu_ingest(doc) == _IGNORED:
                        ignored += 1
                elif kind == SYSTEM:
                    system_rows += 1
                    if system_ingest(doc) == _IGNORED:
                        ignored += 1
                # Any other kind is one this client doesn't read: skipped, not counted.
            for series in (gpu, system):
                series.bin_pending()
                series.rebin()
                if fold:
                    series.trim(upto=series.pass_newest - _FOLD_MARGIN_S)
        self.documents_read += documents
        state.ignored += ignored
        if gpu_rows:
            gpu.version += 1
        if system_rows:
            system.version += 1
        return position


def load_resource_data(metrics_root: Path) -> ResourceData:
    """One full read of a telemetry cache directory, keeping no state."""
    return ResourceLoader(metrics_root).load()


def cached_metric_sources(metrics_root: Path) -> list[str]:
    return [path.parent.name for path in _metric_files(metrics_root) if _read_manifest(path) is not None]


def _printable_name(name: str) -> bool:
    """Whether ``name`` is safe to draw as it is: no control, bidi or zero-width
    character (``str.isprintable``), and nothing :func:`views.printable` changes."""
    return name.isprintable() and views.printable(name) == name


def _metric_files(metrics_root: Path) -> list[Path]:
    """Each ``<sub_job_id>/gpu.jsonl`` under ``metrics_root``, in name order.

    The directory's name is the sub-job's id, which the TUI draws, and it comes
    from the stage's listing. One that isn't printable as it is (an escape
    sequence, say) is no sub-job's id, so it is passed over rather than shown.
    """
    try:
        paths = sorted(metrics_root.glob("*/gpu.jsonl"))
    except OSError:
        return []
    return [path for path in paths if _printable_name(path.parent.name)]


_LOADERS: OrderedDict[str, ResourceLoader] = OrderedDict()
_LOADERS_GUARD = threading.Lock()
_MAX_LOADERS = 4


def resource_loader(metrics_root: Path) -> ResourceLoader:
    """The shared loader for a cache directory, so reopening a job reuses what was read.

    Loaders for the few most recently used directories are kept.
    """
    key = str(Path(metrics_root).resolve())
    with _LOADERS_GUARD:
        loader = _LOADERS.pop(key, None) or ResourceLoader(Path(metrics_root))
        _LOADERS[key] = loader
        while len(_LOADERS) > _MAX_LOADERS:
            _LOADERS.popitem(last=False)
        return loader


# ─── GPU type and allocation ─────────────────────────────────────────────────
# What a sub-job adds to its job's GPU request is views.requested_gpus's rule, as in the job list and the preview.


@dataclass(frozen=True)
class GpuAllocation:
    """What a sub-job asked for: the job's GPU type and the sub-job's GPUs."""

    hardware: str | None = None
    # The sub-job's own GPUs (None: its config doesn't say).
    requested: int | None = None
    # ``multiplex_job_id``: the sub-job whose GPUs this one runs on.
    shared_with: str | None = None
    log_prob: bool = False


def gpu_allocation(job: dict | None, sub_job_id: str | None) -> GpuAllocation:
    """What ``sub_job_id`` of ``job`` asked for, read as the job list and the preview read it."""
    job = job if isinstance(job, dict) else {}
    hardware = views.collapse(job.get("hardware")) or None
    sub_job = next(
        (sj for sj in views.sub_jobs(job) if sub_job_id and (sj.get("sub_job_id") or sj.get("id")) == sub_job_id),
        None,
    )
    if sub_job is None:
        return GpuAllocation(hardware=hardware)
    kind = views.sub_job_kind(sub_job)
    shared = views.multiplex_target(sub_job)
    count = views.requested_gpus(sub_job)
    # requested_gpus reads a missing or unusable n_gpus as 0: unknown here.
    requested = 0 if kind == "log_prob" or shared else (count if count > 0 else None)
    return GpuAllocation(hardware=hardware, requested=requested, shared_with=shared or None,
                         log_prob=kind == "log_prob")


def allocation_text(allocation: GpuAllocation, reporting: int | None = None, seen: int | None = None, *,
                    hardware: bool = True) -> str:
    """``H200  ·  GPUs: 8 requested, 8 reporting`` for the Resources header (``hardware=False``: without the GPU type)."""
    if allocation.log_prob:
        requested = "0 requested (log_prob)"  # the sub-job id's spelling, as the list and Logs use
    elif allocation.shared_with:
        requested = "0 requested (multiplexed)"
    elif allocation.requested is None:
        requested = "requested count unknown"
    else:
        requested = f"{allocation.requested} requested"
    text = f"GPUs: {requested}"
    if hardware:
        text = f"{allocation.hardware or 'GPU type unknown'}  ·  {text}"
    if reporting is not None:
        text += f", {reporting} reporting"
        if seen is not None and seen != reporting:
            text += f" ({seen} seen)"
    return text


def no_metrics_text(allocation: GpuAllocation, *, waiting: str | None = None, who: str = "this sub-job",
                    kind: str = GPU) -> str:
    """Why a sub-job's GPU (or ``kind=SYSTEM``: CPU and memory) view is empty once its telemetry has loaded.

    ``waiting``: the status word while the job or sub-job (``who``) has no
    samples yet, such as ``placing``, or ``running`` before its first samples
    arrive; they are still to come.
    """
    what = "GPU metrics" if kind == GPU else "CPU or memory metrics"
    if allocation.shared_with:
        # Only the sub-job that owns the GPUs reports them, covering everything that runs on them.
        return (f"no {what} for this sub-job: it runs on the GPUs of {allocation.shared_with}, "
                "whose metrics include it")
    if waiting == "running":  # running, but its first samples haven't come yet
        return f"no {what} yet: {who} is RUNNING; the first samples arrive about a minute after it starts"
    if waiting:
        return (f"no {what} yet: {who} is {waiting.upper()} (the first samples arrive about a minute "
                "after it starts running)")
    if kind == SYSTEM:
        # Most deployments collect none: the usual case, said quietly.
        return "CPU and memory aren't collected for this sub-job"
    return "no GPU metrics for this sub-job"


def headline_metric(resource: KindResource) -> Metric | None:
    """The metric the Resources panel lists beside each unit: GPU util % (GPU) or CPU util % (system).

    None when that field is null in every row (the panel then shows no value).
    """
    return resource.metric(_KIND_SPECS[resource.kind].headline)


def _value_span(values: list[float], unit: str) -> str:
    """``88``, ``860 GiB``; ``44–88`` and ``430–860 GiB`` when the nodes differ."""
    low, high = min(values), max(values)
    if format_value(low, unit) == format_value(high, unit):
        return format_value(high, unit)
    return f"{format_value(low)}–{format_value(high, unit)}"


def node_allocation_text(resource: KindResource, reporting: int | None = None, *, sizes: bool = True) -> str:
    """``nodes: 2 reporting  ·  88 CPUs  ·  860 GiB each`` for the System view's header.

    ``reporting`` overrides the nodes in the newest capture (0 for a running
    sub-job that stopped sending samples); ``(3 seen)`` follows it when other
    nodes reported before. The CPU count and memory come from each node's newest
    ``cpu_count`` and ``mem_total_bytes`` (a range when the nodes differ);
    ``sizes=False`` leaves them out.
    """
    seen = len(resource.units)
    reporting = len(resource.reporting_unit_ids) if reporting is None else reporting
    text = f"nodes: {reporting} reporting" + (f" ({seen} seen)" if seen != reporting else "")
    if not sizes:
        return text
    parts = [text]
    for key, unit, word in (("cpu_count", "", " CPUs"), ("mem_total_bytes", "GiB", "")):
        found = [value for value in (node.latest.get(key) for node in resource.units) if value is not None]
        if found:
            parts.append(_value_span(found, unit) + word)
    if len(parts) > 1:
        parts[-1] += " each"
    return "  ·  ".join(parts)


def skipped_rows_text(count: int, *, short: bool = False) -> str:
    """``3 rows skipped (malformed or unsupported)``: rows the loader could not use (``short``: without the why)."""
    text = f"{count} {'row' if count == 1 else 'rows'} skipped"
    return text if short else f"{text} (malformed or unsupported)"


# ─── text ────────────────────────────────────────────────────────────────────

_BLOCKS = "▁▂▃▄▅▆▇█"


def _fit_columns(values: tuple[float | None, ...], width: int) -> list[float | None]:
    """``values`` across ``width`` columns, oldest first.

    With more values than columns, a column is the mean of those it covers (None where
    none has one). With fewer, each value is as many columns wide, give or take one where
    they don't divide evenly, so the values span the ``width`` they are given. In a chart,
    that is the part of the x axis their times cover (see :func:`_chart_spans`): all of
    it for a window they cover, only the left part for a job shorter than the window.
    """
    if width <= 0 or not values:
        return []
    if len(values) <= width:
        return [values[column * len(values) // width] for column in range(width)]
    collapsed: list[float | None] = []
    for column in range(width):
        start = math.floor(column * len(values) / width)
        end = max(start + 1, math.floor((column + 1) * len(values) / width))
        present = [value for value in values[start:end] if value is not None]
        collapsed.append(sum(present) / len(present) if present else None)
    return collapsed


def level_glyph(value: float | None, scale: float = 100.0) -> str:
    """One cell, a bar as tall as ``value`` of ``scale`` (``▁`` to ``█``); blank when unmeasured."""
    if value is None:
        return " "
    return _BLOCKS[min(len(_BLOCKS) - 1, max(0, math.ceil(value / scale * len(_BLOCKS)) - 1))]


# ─── health ──────────────────────────────────────────────────────────────────
# How a value of a known metric reads at a glance: "ok", "warn" or "err", by the first
# bound it reaches, (bound, health) in order; a metric that isn't listed is not judged
# (power has no bound without the GPU's power limit, which the rows don't carry). Low
# GPU util is a warning, not an error: a job between steps or waiting on its sampler is
# briefly idle. GPU memory near full is a warning only: vLLM reserves most of it on
# purpose. A node short of memory is the one an OOM kill ends.
_HEALTH_BOUNDS: dict[tuple[str, str], tuple[tuple[Callable[[float], bool], str], ...]] = {
    (GPU, "gpu_util_pct"): ((lambda v: v < 50, "warn"),),
    (GPU, "fb_used_pct"): ((lambda v: v >= 95, "warn"),),
    (GPU, "gpu_temp_c"): ((lambda v: v >= 90, "err"), (lambda v: v >= 85, "warn")),
    (SYSTEM, "cpu_util_pct"): ((lambda v: v >= 95, "warn"),),
    (SYSTEM, "mem_used_pct"): ((lambda v: v >= 97, "err"), (lambda v: v >= 90, "warn")),
}
_SEVERITY = {None: 0, "ok": 1, "warn": 2, "err": 3}
HEALTH_COLORS = {"ok": OK, "warn": WARN, "err": ERR, None: TEXT}


def metric_health(kind: str, key: str, value: float | None) -> str | None:
    """``"ok"``, ``"warn"`` or ``"err"`` for ``value`` of metric ``key`` (see :data:`_HEALTH_BOUNDS`).

    None when unmeasured, or for a metric this view doesn't judge.
    """
    bounds = _HEALTH_BOUNDS.get((kind, key))
    if bounds is None or value is None or value != value:
        return None
    return next((health for reached, health in bounds if reached(value)), "ok")


def worst_health(healths) -> str | None:
    """The most severe of ``healths`` (``err`` over ``warn`` over ``ok``); None for none of them."""
    return max(healths, key=_SEVERITY.__getitem__, default=None)


_EIGHTHS_WIDE = "▏▎▍▌▋▊▉█"
METER_TRACK = GRID  # seen under a list's highlight, never read as a value


def meter_spans(value: float | None, scale: float, cells: int, color: str | None = None,
                track: str = METER_TRACK) -> list[tuple[str, str]]:
    """A horizontal meter ``cells`` wide, filled to ``value`` of ``scale`` in eighths of a cell.

    With a ``color``, the fill is in it over a ``track`` background that runs the
    meter's whole width (so the empty part shows, under a list's highlight too);
    without one, the empty part is ``·``. A measured value above 0 fills at least an
    eighth. Unmeasured, the meter is all track.
    """
    if cells <= 0:
        return []
    filled = 0
    if value is not None and value == value and scale > 0:
        filled = min(cells * 8, max(1 if value > 0 else 0, round(value / scale * cells * 8)))
    full, part = divmod(filled, 8)
    bar = "█" * full + (_EIGHTHS_WIDE[part - 1] if part else "")
    rest = cells - full - (1 if part else 0)
    if color is None:
        return [(bar + "·" * rest, "")]
    spans = [(bar, f"{color} on {track}")] if bar else []
    if rest:
        spans.append((" " * rest, f"on {track}"))
    return spans


# ─── the KPI strip ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Kpi:
    """One tile of the strip over the stats: a label, a value, a detail and how healthy it reads."""

    label: str
    value: str
    detail: str = ""
    health: str | None = None


def _kw(watts: float) -> str:
    return f"{watts / 1000:.1f} kW" if watts >= 1000 else format_value(watts, "W")


def kpi_tiles(resource: KindResource, view: KindView, *, unit: int | None = None, requested: int | None = None,
              reporting: int | None = None, stale: bool = False) -> list[Kpi]:
    """The strip's tiles for ``view``: utilization, memory, power, temperature and reporting units.

    ``view`` is ``unit``'s (None: the mean across the units). Only the metrics the
    rows have get a tile; a value is the view's latest (the
    mean across the units in the newest capture, or the unit's own), as the chart
    titles show it. For the mean, a tile is as healthy as its least healthy unit
    (one hot GPU is enough to warn), and its detail says how the units spread
    (``84–95%``, ``max 69 °C``, ``5.1 kW total``). The last tile counts the units
    reporting against ``requested`` (None: unknown); ``stale`` (a running sub-job
    that stopped sending samples) leaves the values unjudged and the count an error.
    """
    spec = _KIND_SPECS[resource.kind]
    word = spec.unit_word
    mean = unit is None
    units = [one for one in resource.units if one.reporting] or list(resource.units)
    if not mean:
        units = [one for one in resource.units if one.id == unit]

    def latest(key: str) -> float | None:
        stats = view.stats.get(key)
        return stats.latest if stats is not None else None

    def each(key: str) -> list[float]:
        return [value for value in (one.latest.get(key) for one in units) if value is not None]

    def health(key: str, value: float | None) -> str | None:
        if stale:
            return None
        found = [value, *(each(key) if mean else ())]
        return worst_health([metric_health(resource.kind, key, one) for one in found])

    tiles = []
    util = resource.metric(spec.headline)
    if util is not None:
        value = latest(util.key)
        found = each(util.key) if mean else []
        detail = ""
        if len(found) > 1 and format_value(min(found), util.unit) != format_value(max(found), util.unit):
            low = format_value(min(found), util.unit).removesuffix(util.unit).rstrip()
            detail = f"{low}–{format_value(max(found), util.unit)}"
        tiles.append(Kpi(f"{word} util" if resource.kind == GPU else "CPU util", format_value(value, util.unit),
                         detail, health(util.key, value)))
    used, share = resource.metric(spec.used), resource.metric(spec.memory)
    if used is not None or share is not None:
        percent = latest(share.key) if share is not None else None
        value = format_value(percent, "%")
        if used is not None and latest(used.key) is not None:
            amount = latest(used.key)
            if spec.other_is_free:
                other = latest(spec.other)
                total = None if other is None else amount + other
            else:  # a node's total stays the same, so only the header and the units' newest rows have it
                source = resource.latest if mean else (units[0].latest if units else {})
                total = source.get(spec.other)
            value = format_value(amount, "GiB")
            if total:
                value = f"{value.removesuffix(' GiB')} / {format_value(total, 'GiB')}"
        tiles.append(Kpi("GPU memory" if resource.kind == GPU else "memory", value,
                         format_value(percent, "%") if used is not None and percent is not None else "",
                         health(share.key, percent) if share is not None else None))
    power = resource.metric("power_watts")
    if power is not None:
        found = each(power.key) if mean else []
        tiles.append(Kpi("power", format_value(latest(power.key), power.unit),
                         f"{_kw(sum(found))} total" if len(found) > 1 else "", health(power.key, latest(power.key))))
    temp = resource.metric("gpu_temp_c")
    if temp is not None:
        found = each(temp.key) if mean else []
        tiles.append(Kpi("temp", format_value(latest(temp.key), temp.unit),
                         f"max {format_value(max(found), temp.unit)}" if len(found) > 1 else "",
                         health(temp.key, latest(temp.key))))
    shown = len(resource.reporting_unit_ids) if reporting is None else reporting
    if stale or shown == 0:
        count = "err"
    elif requested and shown < requested:
        count = "warn"
    else:
        count = "ok"
    tiles.append(Kpi(f"{word}s reporting", f"{shown}/{requested}" if requested else str(shown),
                     "samples stopped" if stale else "", count))
    return tiles


_TILE_GAP = 1  # cells between two tiles
_TILE_EDGE = "▎"


def _tile_width(tile: Kpi, detail: bool) -> int:
    second = views.cell_width(tile.value) + (2 + views.cell_width(tile.detail) if detail and tile.detail else 0)
    return 2 + max(views.cell_width(tile.label), second) + 1


def kpi_strip_spans(tiles: list[Kpi], width: int, *, styled: bool = True,
                    dim: bool = False) -> list[list[tuple[str, str]]]:
    """The tiles as lines of ``(text, style)`` spans: two lines a row of tiles, as wide as ``width``.

    Each tile is a card: an edge in its health's colour, its label above, its value
    (bold, in that colour) and its detail beside it. The tiles share one row,
    stretched to fill it; where they don't fit, the details go from the right, then the tiles flow
    onto more rows. ``dim`` (the values are old): every value not judged is faint.
    ``styled=False``: plain text.
    """
    if not tiles or width <= 0:
        return []
    # The details go one at a time from the right until the tiles fit one row.
    keep = [bool(tile.detail) for tile in tiles]
    for index in reversed(range(len(tiles))):
        if sum(_tile_width(tile, kept) for tile, kept in zip(tiles, keep)) + _TILE_GAP * (len(tiles) - 1) <= width:
            break
        keep[index] = False
    sizes = [_tile_width(tile, kept) for tile, kept in zip(tiles, keep)]
    rows: list[list[int]] = [[]]
    used = 0
    for index, size in enumerate(sizes):
        if rows[-1] and used + _TILE_GAP + size > width:
            rows.append([])
            used = 0
        used += size + (_TILE_GAP if rows[-1] else 0)
        rows[-1].append(index)
    card = f"on {SURFACE}" if styled else ""
    lines = []
    for row in rows:
        spare = max(0, width - sum(sizes[index] for index in row) - _TILE_GAP * (len(row) - 1))
        top: list[tuple[str, str]] = []
        bottom: list[tuple[str, str]] = []
        for place, index in enumerate(row):
            tile = tiles[index]
            # The room the row has left, shared out: the strip spans the pane.
            inner = max(1, min(width, sizes[index] + spare // len(row) + (1 if place < spare % len(row) else 0)) - 3)
            color = FAINT if dim and tile.health is None else HEALTH_COLORS.get(tile.health, TEXT)
            if place:
                top.append((" " * _TILE_GAP, ""))
                bottom.append((" " * _TILE_GAP, ""))
            edge = (f"{_TILE_EDGE} ", f"{color} {card}".strip() if styled else "")
            label = views.fit_cells(tile.label, inner)
            top += [edge, (f"{label:<{inner}} ", f"{MUTED} {card}" if styled else "")]
            value = views.fit_cells(tile.value, inner)
            rest = inner - views.cell_width(value)
            note = views.fit_cells(tile.detail, rest - 2) if keep[index] and rest > 3 else ""
            bottom += [edge, (value, f"bold {color} {card}" if styled else "")]
            tail = (f"  {note}" if note else "").ljust(rest) + " "
            bottom.append((tail, f"{FAINT if dim else MUTED} {card}" if styled else ""))
        lines += [top, bottom]
    return lines


def _eighths(value: float | None, height: int, scale: float) -> int | None:
    """How high ``value`` of ``scale`` reaches in a ``height``-row chart, in eighths of a row.

    At least one eighth for a measured value (a 0 still shows, unlike a gap), at most
    the chart's top; None for a column without a sample.
    """
    if value is None:
        return None
    return min(height * 8, max(1, math.ceil(value / scale * height * 8)))


def _time_labels(start: datetime, end: datetime, chart_width: int) -> str:
    """The times under a chart's ``chart_width`` columns: start, middle and end, or fewer where they don't fit.

    No label repeats another: a span under two minutes shows seconds, and a
    label that would still read the same as its neighbour is left out. A span
    of a day or more dates its labels where they fit.
    """
    span = end - start
    dated = span >= timedelta(days=1) and chart_width >= 2 * len("10-13 20:59") + 1
    precise = span < timedelta(minutes=2)
    times = (start, start + span / 2, end)
    shape = "%m-%d %H:%M" if dated else "%H:%M:%S" if precise else "%H:%M"
    first, middle, last = (moment.astimezone().strftime(shape) for moment in times)
    size = len(last)
    if chart_width >= 3 * size + 2 and len({first, middle, last}) == 3:
        left_gap = (chart_width - 3 * size) // 2
        right_gap = chart_width - 3 * size - left_gap
        return first + (" " * left_gap) + middle + (" " * right_gap) + last
    if chart_width >= 2 * size + 1 and first != last:
        return first + (" " * (chart_width - 2 * size)) + last
    # Too few columns for two labels: the end's time alone, under the right edge.
    label = end.astimezone().strftime("%m-%d %H:%M" if dated else "%H:%M")
    return label.rjust(chart_width) if chart_width >= len(label) else ""


# The frame, labels and the 50% gridline, quieter than the series: the theme's own tones.
CHART_AXIS = FAINT
CHART_GRID = GRID
CHART_MUTED = MUTED
_GRID = "┈"
# The baseline under a column without a sample: a gap reads as missing data, never as a 0.
_GAP_AXIS = "╌"
# Right of the newest column, on the row its value reaches: where the series is now.
_NOW = "◂"
# The fill under a series' edge, as a share of its hue over the background: faint at the
# baseline and a little stronger toward the top, so a steady 90% reads as a lit edge
# over a shadow rather than as a solid block of colour.
_FILL_SHARE = (0.14, 0.32)


def chart_hue(index: int) -> str:
    """The colour of the ``index``-th chart (see :data:`CHART_HUES`)."""
    return CHART_HUES[index % len(CHART_HUES)]


def mix(hue: str, share: float, base: str = BACKGROUND) -> str:
    """``hue`` blended over ``base`` (both ``#rrggbb``): ``share`` 1 is the hue itself, 0 is ``base``."""
    share = max(0.0, min(1.0, share))
    return "#" + "".join(
        f"{round(int(hue[i:i + 2], 16) * share + int(base[i:i + 2], 16) * (1 - share)):02x}" for i in (1, 3, 5)
    )


def _columns_between(start: datetime, end: datetime, x_axis: tuple[datetime, datetime],
                     columns: int) -> tuple[int, int]:
    """The first and last of a chart's ``columns`` that ``start`` to ``end`` falls in, on an x axis
    from ``x_axis[0]`` to ``x_axis[1]``, each column an equal share of its time."""
    low, high = x_axis
    total = (high - low).total_seconds()
    if total <= 0:
        return 0, columns - 1

    def column(moment: datetime) -> int:
        return min(columns - 1, max(0, math.floor((moment - low).total_seconds() / total * columns)))

    first = column(start)
    return first, max(first, column(end))


def _chart_spans(values, start: datetime, end: datetime, width: int, height: int, labels: tuple[str, str, str],
                 scale: float, hue: str | None,
                 x_axis: tuple[datetime, datetime] | None = None) -> list[list[tuple[str, str]]]:
    """The lines of an area chart of ``values`` from 0 to ``scale``, ``height`` rows tall,
    its axis labelled ``labels``: each line a list of ``(text, style)`` spans.

    The x axis runs between the times ``x_axis`` gives, the values' own ``start`` and
    ``end`` by default. The values span their share of its columns (see
    :func:`_fit_columns`), and its time before or after them is blank over an unbroken
    baseline. Each column rises to its value in eighths of a row. With a ``hue``, the
    cell where a column ends (its top edge) is in the hue and the cells under it in a
    faint fill of it (see :data:`_FILL_SHARE`), so the eye follows the series' line; the
    frame is muted. Without one, every style is empty and the glyphs are the same. A
    column without a sample stays blank down to the baseline, which is broken under it
    (:data:`_GAP_AXIS`). The middle label's row has a dotted gridline where the series
    stays under it. In the column right of the newest one, :data:`_NOW` marks the row it
    reaches; there is none when the newest column is a gap (nothing is known about now).
    """
    if width <= 0:
        return [[("", "")]]
    axis_style, grid_style = (CHART_AXIS, CHART_GRID) if hue else ("", "")
    axis = max(len(label) for label in labels)
    if width < axis + 4:
        return [[("".join(level_glyph(value, scale) for value in _fit_columns(values, width)), hue or "")]]
    height = max(1, height)
    inner = max(1, width - axis - 3)  # the newest column's marker follows it
    x_axis = x_axis or (start, end)
    first, last = _columns_between(start, end, x_axis, inner)
    before, after = first, inner - 1 - last
    columns = _fit_columns(values, last - first + 1) or [None] * (last - first + 1)
    reach = [_eighths(value, height, scale) for value in columns]
    newest = reach[-1] if reach else None
    now_level = None if newest is None else (newest - 1) // 8
    low, high = _FILL_SHARE
    top, middle, bottom = labels
    middle_row = height // 2
    lines = []
    for index in range(height):
        if index == 0:
            mark = f"{top:>{axis}} ┤"
        elif index == middle_row:
            mark = f"{middle:>{axis}} ┤"
        else:
            mark = " " * axis + " │"
        level = height - 1 - index
        grid = index == middle_row and height > 2
        # The fill is in its colour on its colour: a terminal that leaves a gap between
        # rows of █ shows no stripes.
        shade = mix(hue, low + (high - low) * (level + 1) / height) if hue else ""
        styles = {"edge": hue or "", "fill": f"{shade} on {shade}" if hue else "", "grid": grid_style, "gap": "",
                  # The gridline goes on through the fill, a shade lighter than it.
                  "under": f"{mix(hue, 0.5)} on {shade}" if hue else ""}
        cells = []
        for eighths in reach:
            fill = -1 if eighths is None else eighths - 8 * level
            if fill > 8:
                cells.append((_GRID, "under") if grid and hue else ("█", "fill"))
            elif fill > 0:
                cells.append((_BLOCKS[fill - 1], "edge"))
            else:
                cells.append((_GRID, "grid") if grid else (" ", "gap"))
        # Runs of edge, fill and gridline, each in its own style.
        runs = groupby(cells, itemgetter(1))
        # The axis's time before and after the values: blank, the gridline running on through it.
        empty = (lambda count: (_GRID * count, grid_style)) if grid else (lambda count: (" " * count, ""))
        line = [(mark, axis_style)]
        if before:
            line.append(empty(before))
        line += [("".join(cell for cell, _kind in run), styles[kind]) for kind, run in runs]
        line.append((_NOW, hue or "") if level == now_level else empty(1) if after else (" ", ""))
        if after:
            line.append(empty(after))
        lines.append(line)
    baseline = "".join("─" if eighths is not None else _GAP_AXIS for eighths in reach)
    lines.append([(f"{bottom:>{axis}} └{'─' * before}{baseline}─{'─' * after}", axis_style)])
    # The times span the marker's column too: the axis's end is under the chart's right edge.
    lines.append([((" " * (axis + 2) + _time_labels(*x_axis, inner + 1)).rstrip(), axis_style)])
    return lines


def _x_axis(view: KindView, complete: bool) -> tuple[datetime, datetime]:
    """The times ``view``'s charts run between: its window, ending at its newest sample. A job
    shorter than the window has the window from its first sample instead, so its samples start
    at the left; a window whose older samples are still loading (not ``complete``) keeps ending
    at its newest, the older ones to come at the left."""
    duration = RANGES.get(view.name)
    if duration is None or view.end - view.start >= duration:
        return view.start, view.end
    if complete:
        return view.start, view.start + duration
    return view.end - duration, view.end


def metric_chart_spans(view: KindView, metric: Metric, width: int, *, height: int = 2,
                       hue: str | None = None, complete: bool = True) -> list[list[tuple[str, str]]]:
    """``metric``'s chart in ``view`` as lines of ``(text, style)`` spans (see :func:`_chart_spans`), its axis
    labelled by :func:`axis_labels`: with an even ``height`` the middle label's row tops out at half the scale.
    The x axis spans the window (see :func:`_x_axis`; ``complete`` as :func:`window_complete` says)."""
    return _chart_spans(view.chart(metric.key), view.start, view.end, width, height, axis_labels(metric),
                        metric.scale, hue, x_axis=_x_axis(view, complete))


def has_metric_samples(view: KindView, key: str) -> bool:
    return any(value is not None for value in view.chart(key))


def format_interval(seconds: float | None) -> str:
    """``5 s``, ``2.5 s``, ``1m 30s``; ``—`` when unknown."""
    if seconds is None or not math.isfinite(seconds) or seconds <= 0:
        return "—"
    if seconds < 10 and abs(seconds - round(seconds)) >= 0.05:
        return f"{seconds:.1f} s"
    if seconds < 60:
        return f"{round(seconds)} s"
    minutes, rest = divmod(round(seconds), 60)
    return f"{minutes}m {rest:02d}s" if rest else f"{minutes}m"


def format_age(seconds: float) -> str:
    seconds = int(max(0, seconds))
    return f"{seconds} s" if seconds < 60 else views.fmt_duration(seconds)


def scrape_text(sub_job: KindResource) -> str:
    """``scraped every 5 s``: how often the sampler reads the sub-job's GPUs or nodes (the rows'
    spacing, else the interval they report)."""
    return f"scraped every {format_interval(sub_job.sample_interval_s or sub_job.reported_interval_s)}"


def freshness_text(sub_job: KindResource, now: datetime, *, stale: bool = False,
                   clock: bool = True, what: str = "GPU samples") -> str:
    """``newest 40 s ago (12:34:56)``; ``stale``: the sub-job should be sending samples and isn't.

    A stale one reads ``no new GPU samples for 41m 56s; the windows end at 12:34:56``
    (``what`` names the samples): the windows and the latest values are measured
    back from that sample. ``clock=False`` leaves the sample's time out: ``newest 40 s ago``.
    """
    age = max(0.0, (now - sub_job.latest_sample_ts).total_seconds())
    when = sub_job.latest_sample_ts.astimezone().strftime("%H:%M:%S" if age < 86400 else "%Y-%m-%d %H:%M")
    if stale:
        text = f"no new {what} for {format_age(age)}"
        return f"{text}; the windows end at {when}" if clock else text
    text = f"newest {format_age(age)} ago"
    return f"{text} ({when})" if clock else text


def age_slack(sub_job: KindResource, now: datetime) -> int:
    """Cells the age in :func:`freshness_text` can still grow by (``5 s`` to ``59m 59s``).

    A header keeps that much room, so the age ticking past a minute (or an
    hour) never makes the line longer than the width it was fitted to.
    """
    age = max(0.0, (now - sub_job.latest_sample_ts).total_seconds())
    return max(0, len("59m 59s") - len(format_age(age)))


def window_label(name: str) -> str:
    return "whole job" if name == "all" else f"last {name}"


_STAT_NAMES = ("min", "avg", "max", "latest")
# In place of the stats a sub-job's older samples are needed for: while they download,
# and once a download has ended without them (it failed; the header says why).
LOADING_OLDER = "loading older samples…"
OLDER_NOT_LOADED = "older samples not loaded"


def window_complete(sub_job: KindResource, view: KindView) -> bool:
    """Whether every sample of ``view``'s window has been read (False while older data loads)."""
    if not sub_job.partial:
        return True
    duration = RANGES.get(view.name)
    return duration is not None and sub_job.first_sample_ts is not None and (
        sub_job.first_sample_ts <= view.end - duration
    )


def metric_stats_spans(view: KindView, whole: KindView | None, metrics, *, width: int, partial: bool = False,
                       window_partial: bool = False, missing: str = LOADING_OLDER) -> list[list[tuple[str, str]]]:
    """The stats block as lines of ``(text, style)`` spans: each metric's name in its
    chart's colour (:func:`chart_hue`), the latest values bold, the column heads muted
    and right-aligned over their values, and a rule between the window and the whole job."""
    metrics = list(metrics)
    window_partial = window_partial or (partial and view.name == "all")
    scopes = [(window_label(view.name), None if window_partial else view)]
    if view.name != "all" and whole is not None:
        scopes.append(("whole job", None if partial else whole))

    def cells(scope: KindView | None, metric: Metric) -> list[str] | None:
        if scope is None:
            return None
        stats = scope.stats.get(metric.key)
        if stats is None:
            return ["—"] * 4
        found = [format_value(value, metric.unit) for value in (stats.min, stats.avg, stats.max, stats.latest)]
        if stats.min_bound:
            found[0] = "≤" + found[0]
        if stats.max_bound:
            found[2] = "≥" + found[2]
        return found

    table = [[cells(scope, metric) for _name, scope in scopes] for metric in metrics]
    cell = max([len(name) for name in _STAT_NAMES]
               + [len(text) for row in table for found in row if found for text in found]) + 1
    block = 4 * cell
    missing_cell = missing if len(missing) <= block else missing[: block - 1] + "…"

    def scope_cells(found: list[str] | None) -> list[tuple[str, str]]:
        if found is None:
            return [(f"{missing_cell:>{block}}", f"italic {CHART_MUTED}")]
        # The latest value stands out: it is what the panel and the chart titles show. Each
        # cell's padding is a span of its own, so its digits start where the column says.
        spans = []
        for index, text in enumerate(found):
            spans += [(" " * (cell - len(text)), ""), (text, f"bold {TEXT}" if index == 3 else "")]
        return spans

    label = view.label
    # Each scope says what its own values are: a GPU that stopped reporting is in the whole
    # job's mean, not the window's.
    labels = [label, *((whole.label,) if len(scopes) > 1 else ())]
    titles = [(name, f" · {own}" if own else "") for (name, _scope), own in zip(scopes, labels)]
    if any(len(name + own) > block for name, own in titles):  # both carry it, or neither
        titles = [(name, "") for name, _scope in scopes]
    header = "".join(f"{name:>{cell}}" for name in _STAT_NAMES)
    for names in ([metric.name for metric in metrics], [metric.short for metric in metrics]):
        name_width = max([len(name) for name in names] + [0]) + 2
        if width >= name_width + (block + 2) * len(scopes):
            # A table: the metric's name, then each scope's four columns, its title over
            # them and its column heads right-aligned over its values; a rule parts the scopes.
            title_line: list[tuple[str, str]] = [(" " * name_width, "")]
            head_line: list[tuple[str, str]] = [(f"{'metric':<{name_width}}", CHART_AXIS)]
            for position, (name, own) in enumerate(titles):
                lead = (" │", CHART_GRID) if position else ("  ", "")
                gap = block - len(name) - len(own)
                title_line += [lead, (" " * (gap // 2), ""), (name, "bold"), (own, CHART_MUTED),
                               (" " * (gap - gap // 2), "")]
                head_line.append(lead)
                for head in _STAT_NAMES:
                    head_line += [(" " * (cell - len(head)), ""), (head, CHART_AXIS)]
            lines = [title_line, head_line]
            for index, (name, row) in enumerate(zip(names, table)):
                line = [(f"{name:<{name_width}}", chart_hue(index))]
                for position, found in enumerate(row):
                    line += [(" │", CHART_GRID) if position else ("  ", ""), *scope_cells(found)]
                lines.append(line)
            return lines
    # Narrow: a row per metric and scope; the first line says what the values are. The
    # scopes go by their keys (10m, job) before the metrics' names are cut.
    names = [name for name, _scope in scopes] if len(scopes) > 1 else [""]
    longest = max(len(metric.short) for metric in metrics) + 1 if metrics else 0
    if len(scopes) > 1 and width - (max(len(name) for name in names) + 1) - block < min(longest, 12):
        names = [view.name, "job"]
    scope_width = max(len(name) for name in names) + 1 if len(scopes) > 1 else 0
    name_width = max(4, width - scope_width - block)
    lines = [[(views.fit_cells(label, name_width + scope_width).ljust(name_width + scope_width), "bold"),
              (header, CHART_AXIS)]]
    for index, (metric, row) in enumerate(zip(metrics, table)):
        for position, (name, found) in enumerate(zip(names, row)):
            shown = views.fit_cells(metric.short, name_width - 1) if position == 0 else ""
            lines.append([(f"{shown:<{name_width}}", chart_hue(index)), (f"{name:<{scope_width}}", CHART_MUTED),
                          *scope_cells(found)])
    return lines
