# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import cortex_training.tui.resource as resource_module

BUSY = "gpu_util_pct"


def load_resource_data(root):
    """A load, each sub-job's GPU kind read as busy % per GPU: the shapes these tests were first written against."""
    data = (loader := resource_module.ResourceLoader(root)).load()
    sub_jobs = {sub_job_id: SimpleNamespace(views={name: _view(kinds["gpu"], name) for name in resource_module.RANGES})
                for sub_job_id, kinds in data.kinds.items() if "gpu" in kinds}
    return SimpleNamespace(sub_jobs=sub_jobs, ignored_rows=data.ignored_rows,
                           ignored_rows_by_sub_job=data.ignored_rows_by_sub_job, loader=loader)


def _view(resource, name):
    mean = resource.view(name)
    rows = [SimpleNamespace(global_gpu_index=gpu_id, current_util_pct=unit.latest.get(BUSY),
                            window_avg_util_pct=_avg(resource.view(name, gpu_id)),
                            fb_used_bytes=unit.latest.get("fb_used_bytes"))
            for gpu_id, unit in zip(resource.unit_ids, resource.units)]
    return SimpleNamespace(start=mean.start, end=mean.end, chart=mean.chart(BUSY), rows=rows,
                           average_util_pct=_avg(mean), mean=mean, metric=resource.metric(BUSY), gpu_count=len(rows),
                           fb_gpu_count=sum(row.fb_used_bytes is not None for row in rows))


def _avg(view):
    return getattr(view.stats.get(BUSY), "avg", None)


def render_chart(view, width):
    lines = resource_module.metric_chart_spans(view.mean, view.metric, width, height=1)
    return "\n".join("".join(text for text, _style in line) for line in lines)


def parse_sample_ts(value):
    epoch = resource_module._sample_epoch(value)
    return None if epoch is None else datetime.fromtimestamp(epoch, tz=timezone.utc)


def _row(
    timestamp: datetime,
    gpu: int,
    util: float | None,
    *,
    interval: float = 5.0,
    local_gpu: int = 0,
    kind: str | None = None,
    version: int = 2,
) -> dict:
    row = {
        "record_version": version,
        "sub_job_id": "ignored-wire-id",
        "sample_ts": timestamp.isoformat().replace("+00:00", "Z"),
        "worker_num": gpu,
        "local_gpu_index": local_gpu,
        "global_gpu_index": gpu,
        "gpu_util_pct": util,
        "fb_used_bytes": float((gpu + 1) * 1024**3),
        "fb_free_bytes": float(8 * 1024**3),
        "sample_interval_s": interval,
    }
    if kind is not None:
        row["kind"] = kind
    return row


def _write_metrics(
    root: Path,
    sub_job_id: str,
    rows: list[dict | str],
    *,
    committed: int | None = None,
    manifest: bool = True,
) -> Path:
    path = root / sub_job_id / "gpu.jsonl"
    path.parent.mkdir(parents=True)
    normalized = []
    for item in rows:
        if isinstance(item, dict):
            item = {**item, "sub_job_id": sub_job_id}
        normalized.append(item)
    payload = "".join((item if isinstance(item, str) else json.dumps(item)) + "\n" for item in normalized).encode()
    path.write_bytes(payload)
    if manifest:
        size = len(payload) if committed is None else committed
        path.with_name(".gpu.jsonl.manifest.json").write_text(
            json.dumps({"version": 1, "names": ["chunk.gz"], "bytes": size}) + "\n",
            encoding="utf-8",
        )
    return path


def test_parse_rfc3339_nano_on_python_floor():
    parsed = parse_sample_ts("2026-01-01T00:00:00.123456789Z")
    assert parsed == datetime(2026, 1, 1, 0, 0, 0, 123456, tzinfo=timezone.utc)
    for fraction, microsecond in [
        ("1", 100000),
        ("12", 120000),
        ("1234", 123400),
        ("12345", 123450),
    ]:
        parsed = parse_sample_ts(f"2026-01-01T00:00:00.{fraction}Z")
        assert parsed is not None
        assert parsed.microsecond == microsecond
    assert parse_sample_ts("2026-01-01T00:00:00Z") is not None
    assert parse_sample_ts("2026-01-01T01:00:00+01:00") == datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert parse_sample_ts("not-a-time") is None
    assert parse_sample_ts("2026-01-01 00:00:00Z") is None
    assert parse_sample_ts("2026-01-01T00:00:00") is None  # no zone
    assert parse_sample_ts("0001-01-01T00:00:00+05:00") is None
    assert parse_sample_ts("9999-12-31T23:59:59-05:00") is None


def test_redacted_v2_fixture_preserves_zero(tmp_path):
    fixture = Path(__file__).parent / "fixtures" / "gpu_metrics_v2.jsonl"
    root = tmp_path / "metrics"
    path = root / "JOB_ID:training:0" / "gpu.jsonl"
    path.parent.mkdir(parents=True)
    path.write_bytes(fixture.read_bytes())
    path.with_name(".gpu.jsonl.manifest.json").write_text(
        json.dumps({"version": 1, "names": ["chunk.gz"], "bytes": path.stat().st_size})
    )

    data = load_resource_data(root)
    view = data.sub_jobs["JOB_ID:training:0"].views["all"]
    assert view.rows[0].current_util_pct == 0.0
    assert view.average_util_pct == 0.0
    assert data.ignored_rows == 0


def test_mixed_system_rows_are_silent_but_bad_gpu_rows_warn(tmp_path):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    system = {
        "record_version": 3,
        "kind": "system",
        "sample_ts": now.isoformat(),
        "worker_num": 0,
        "cpu_util_pct": 25,
    }
    unknown = {"record_version": 3, "kind": "future", "sample_ts": now.isoformat()}
    missing_identity = _row(now, 0, 50)
    missing_identity.pop("global_gpu_index")
    unsupported = _row(now, 0, 50, version=1)
    _write_metrics(
        tmp_path,
        "job:training:0",
        [system, unknown, _row(now, 0, 0), missing_identity, unsupported, "{bad json"],
    )

    data = load_resource_data(tmp_path)
    assert data.ignored_rows == 3
    assert data.sub_jobs["job:training:0"].views["all"].rows[0].current_util_pct == 0


def test_extreme_timestamp_rows_are_ignored_while_valid_rows_render(tmp_path):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    too_early = _row(now, 0, 10)
    too_early["sample_ts"] = "0001-01-01T00:00:00Z"
    overflowing = _row(now, 1, 20)
    overflowing["sample_ts"] = "9999-12-31T23:59:59-05:00"
    _write_metrics(
        tmp_path,
        "job:training:0",
        [too_early, overflowing, _row(now, 2, 30)],
    )

    data = load_resource_data(tmp_path)
    view = data.sub_jobs["job:training:0"].views["all"]
    assert [row.global_gpu_index for row in view.rows] == [2]
    assert data.ignored_rows == 2
    assert render_chart(view, 30)


def test_kind_gpu_and_invalid_values_keep_latest_without_weighting(tmp_path):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    missing_interval = _row(now, 0, float("nan"), kind="gpu")
    missing_interval["sample_interval_s"] = None
    later = _row(now + timedelta(seconds=5), 0, float("inf"), kind="gpu")
    later["fb_used_bytes"] = float("inf")
    later["sample_interval_s"] = -1
    _write_metrics(tmp_path, "job:training:0", [missing_interval, later])

    view = load_resource_data(tmp_path).sub_jobs["job:training:0"].views["all"]
    assert view.rows[0].current_util_pct is None
    assert view.rows[0].fb_used_bytes is None
    assert view.rows[0].window_avg_util_pct is None
    assert view.average_util_pct is None


def test_overflowing_number_and_deep_json_are_ignored_without_crashing(tmp_path):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    overflow = _row(now, 0, 10)
    overflow["gpu_util_pct"] = 10**400
    overflow["fb_used_bytes"] = 1e308
    oversized_interval = _row(now, 1, 50, interval=1e308)
    deeply_nested = "[" * 1100 + "0" + "]" * 1100
    _write_metrics(
        tmp_path,
        "job:training:0",
        [overflow, oversized_interval, deeply_nested],
    )

    data = load_resource_data(tmp_path)
    view = data.sub_jobs["job:training:0"].views["all"]
    assert view.rows[0].current_util_pct is None
    assert view.rows[0].fb_used_bytes is None
    assert view.rows[1].current_util_pct == 50
    assert view.rows[1].window_avg_util_pct is None
    assert data.ignored_rows == 1
    chart = render_chart(view, 30)
    assert all(len(line) <= 30 for line in chart.splitlines())


def test_gpu_kind_still_requires_v2_and_matching_sub_job(tmp_path):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    wrong_version = _row(now, 0, 10, kind="gpu", version=1)
    wrong_sub_job = _row(now, 1, 20, kind="gpu")
    _write_metrics(tmp_path, "job:training:0", [wrong_version, wrong_sub_job])
    path = tmp_path / "job:training:0" / "gpu.jsonl"
    docs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    docs[1]["sub_job_id"] = "different:training:0"
    payload = "".join(json.dumps(doc) + "\n" for doc in docs)
    path.write_text(payload, encoding="utf-8")
    path.with_name(".gpu.jsonl.manifest.json").write_text(
        json.dumps({"version": 1, "names": ["chunk.gz"], "bytes": path.stat().st_size}),
        encoding="utf-8",
    )

    data = load_resource_data(tmp_path)
    assert data.sub_jobs == {}
    assert data.ignored_rows_by_sub_job == {"job:training:0": 2}


def test_unsupported_rows_are_tracked_per_sub_job_without_valid_samples(tmp_path):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    _write_metrics(tmp_path, "job:training:0", [_row(now, 0, 10, version=1)])
    _write_metrics(tmp_path, "job:sampling:0", [_row(now, 0, 20), "{bad json"])

    data = load_resource_data(tmp_path)
    assert "job:training:0" not in data.sub_jobs
    assert data.ignored_rows_by_sub_job == {
        "job:training:0": 1,
        "job:sampling:0": 1,
    }
    assert data.ignored_rows == 2


def test_identity_is_global_gpu_within_sub_job(tmp_path):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    _write_metrics(
        tmp_path,
        "job:training:0",
        [_row(now, 0, 10, local_gpu=0), _row(now, 1, 90, local_gpu=0)],
    )
    _write_metrics(tmp_path, "job:sampling:0", [_row(now, 0, 30, local_gpu=0)])

    data = load_resource_data(tmp_path)
    training = data.sub_jobs["job:training:0"].views["all"]
    sampling = data.sub_jobs["job:sampling:0"].views["all"]
    assert [row.global_gpu_index for row in training.rows] == [0, 1]
    assert [row.current_util_pct for row in training.rows] == [10, 90]
    assert [row.current_util_pct for row in sampling.rows] == [30]


def test_regular_five_second_series_is_contiguous_and_includes_latest(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [_row(start + timedelta(seconds=5 * i), 0, 10) for i in range(61)]
    _write_metrics(tmp_path, "job:training:0", rows)

    view = load_resource_data(tmp_path).sub_jobs["job:training:0"].views["10m"]
    assert len(view.chart) == 61
    assert all(value == 10 for value in view.chart)
    assert view.end == start + timedelta(minutes=5)


def test_regular_series_with_scrape_jitter_stays_contiguous(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [
        _row(
            start + timedelta(seconds=5 * i, milliseconds=40 if i % 2 == 0 else -40),
            0,
            90,
        )
        for i in range(61)
    ]
    _write_metrics(tmp_path, "job:training:0", rows)

    view = load_resource_data(tmp_path).sub_jobs["job:training:0"].views["10m"]
    assert len(view.chart) == 61
    assert all(value == 90 for value in view.chart)


def test_ranges_are_per_sub_job_bounded_and_preserve_gaps(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    _write_metrics(
        tmp_path,
        "job:training:0",
        [_row(start + timedelta(seconds=5 * i), 0, 10) for i in range(1000)],
    )
    _write_metrics(
        tmp_path,
        "job:sampling:0",
        [_row(start, 0, 20), _row(start + timedelta(seconds=10), 0, 20)],
    )

    data = load_resource_data(tmp_path)
    training = data.sub_jobs["job:training:0"].views["all"]
    sampling = data.sub_jobs["job:sampling:0"].views["10m"]
    assert len(training.chart) == 240
    assert sampling.start == start
    assert sampling.end == start + timedelta(seconds=10)
    assert sampling.chart == (20, None, 20)


def test_summary_weights_gpu_time_for_gpus_with_unequal_sample_counts(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [_row(start + timedelta(seconds=5 * i), 0, 0) for i in range(10)]
    rows.append(_row(start + timedelta(seconds=45), 1, 100))
    _write_metrics(tmp_path, "job:training:0", rows)

    view = load_resource_data(tmp_path).sub_jobs["job:training:0"].views["all"]
    # GPU 1's one sample against GPU 0's ten counts for the time it reported; the newest interval averages both.
    assert abs(view.average_util_pct - 100 / 11) < 1e-9
    assert view.chart[-1] == 50


def test_sample_interval_weights_each_gpu_average(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    _write_metrics(
        tmp_path,
        "job:training:0",
        [
            _row(start, 0, 0, interval=5),
            _row(start + timedelta(seconds=5), 0, 100, interval=15),
        ],
    )

    view = load_resource_data(tmp_path).sub_jobs["job:training:0"].views["all"]
    assert view.rows[0].window_avg_util_pct == 75
    assert view.average_util_pct == 75


def test_loading_reads_each_row_once_for_all_ranges(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    _write_metrics(
        tmp_path,
        "job:training:0",
        [_row(start + timedelta(seconds=5 * i), 0, 10) for i in range(10)],
    )
    data = load_resource_data(tmp_path)
    assert set(data.sub_jobs["job:training:0"].views) == {"10m", "30m", "1h", "6h", "all"}
    assert data.loader.documents_read == 10
    assert data.loader.bytes_read == (tmp_path / "job:training:0" / "gpu.jsonl").stat().st_size


def test_latest_cells_do_not_backfill_and_window_can_be_empty_for_gpu(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    older = _row(start, 0, 30)
    older["fb_used_bytes"] = 123.0
    # GPU 1's newest sample is 20 minutes in, so GPU 0 has none in the shortest (10m) window.
    latest = _row(start + timedelta(minutes=20), 1, 80)
    gpu_zero_latest = _row(start + timedelta(minutes=1), 0, None)
    gpu_zero_latest["fb_used_bytes"] = None
    _write_metrics(tmp_path, "job:training:0", [older, gpu_zero_latest, latest])

    view = load_resource_data(tmp_path).sub_jobs["job:training:0"].views["10m"]
    row0 = view.rows[0]
    assert row0.current_util_pct is None
    assert row0.window_avg_util_pct is None
    assert row0.fb_used_bytes is None
    assert view.fb_gpu_count == 1
    assert view.gpu_count == 2


def test_manifest_bounds_cache_reads(tmp_path):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    path = _write_metrics(tmp_path, "job:training:0", [_row(now, 0, 10)])
    first_size = path.stat().st_size
    with path.open("ab") as handle:
        handle.write(json.dumps(_row(now, 1, 90)).encode() + b"\n")
    path.with_name(".gpu.jsonl.manifest.json").write_text(
        json.dumps({"version": 1, "names": ["chunk.gz"], "bytes": first_size})
    )
    data = load_resource_data(tmp_path)
    assert [row.global_gpu_index for row in data.sub_jobs["job:training:0"].views["all"].rows] == [0]

    path.with_name(".gpu.jsonl.manifest.json").unlink()
    assert load_resource_data(tmp_path).sub_jobs == {}
    path.with_name(".gpu.jsonl.manifest.json").write_text(
        json.dumps({"version": 1, "names": ["chunk.gz"], "bytes": path.stat().st_size + 1})
    )
    assert load_resource_data(tmp_path).sub_jobs == {}


def test_unterminated_committed_tail_is_not_warned(tmp_path):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    path = _write_metrics(tmp_path, "job:training:0", [_row(now, 0, 10)])
    with path.open("ab") as handle:
        handle.write(b"\n  \n")
        handle.write(b'{"record_version":2')
    path.with_name(".gpu.jsonl.manifest.json").write_text(
        json.dumps({"version": 1, "names": ["chunk.gz"], "bytes": path.stat().st_size})
    )
    data = load_resource_data(tmp_path)
    assert data.ignored_rows == 0
    assert "job:training:0" in data.sub_jobs


def test_chart_uses_fixed_scale_labels_and_fits_narrow_width(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [_row(start + timedelta(seconds=5 * i), 0, 10) for i in range(61)]
    _write_metrics(tmp_path, "job:training:0", rows)
    view = load_resource_data(tmp_path).sub_jobs["job:training:0"].views["all"]

    chart = render_chart(view, 30)
    first, axis, labels = chart.splitlines()
    # A column per bucket, then the marker of the newest value's row.
    assert first.endswith("◂")
    assert set(first.removeprefix("100% ┤").removesuffix("◂")) == {"▁"}
    assert len(first) <= 30
    assert len(axis) <= 30
    assert "100%" in first and "0%" in axis
    assert view.start.astimezone().strftime("%H:%M") in labels
    assert view.end.astimezone().strftime("%H:%M") in labels

    narrow = render_chart(view, 20)
    assert all(len(line) <= 20 for line in narrow.splitlines())
    assert view.end.astimezone().strftime("%H:%M") in narrow.splitlines()[-1]
    for width in range(1, 7):
        assert all(len(line) <= width for line in render_chart(view, width).splitlines())

    wide = render_chart(view, 120).splitlines()
    assert all(len(line) == 120 for line in wide)
    assert wide[0].index("▁") == len("100% ┤")  # 61 buckets across 113 columns start at the axis
    assert wide[0].index("▁") == wide[2].index(view.start.astimezone().strftime("%H:%M"))
    assert wide[2].endswith(view.end.astimezone().strftime("%H:%M"))


def test_a_job_shorter_than_its_window_charts_from_the_left_on_the_window_s_scale(tmp_path):
    # 2m40s of 5 s samples, as a job cancelled early has: "last 10m" runs ten minutes from its first sample,
    # the samples on the left of its 113 columns, and the whole job spans them all.
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    _write_metrics(tmp_path, "job:training:0", [_row(start + timedelta(seconds=5 * i), 0, 10) for i in range(33)])
    views = load_resource_data(tmp_path).sub_jobs["job:training:0"].views
    row, axis, times = render_chart(views["10m"], 120).splitlines()
    assert row == "100% ┤" + "▁" * 31 + "◂" + " " * 82  # 160 s of 600 s: the first 31 columns
    assert axis == "  0% └" + "─" * 114
    clock = [moment.astimezone().strftime("%H:%M") for moment in (start, start + timedelta(minutes=10))]
    assert [times.split()[0], times.split()[-1]] == clock
    assert render_chart(views["all"], 120).splitlines()[0] == "100% ┤" + "▁" * 113 + "◂"


def test_only_gauges_named_with_a_unit_are_metrics():
    for name in ("gpu_util_pct", "fb_free_bytes", "power_watts", "gpu_temp_c", "sm_active_pct", "mem_used_bytes"):
        assert resource_module.is_metric_field(name), name
    # Ids, counts, durations and counters: averaged over a window as gauges, they would chart nonsense.
    for name in ("pid", "uptime_s", "cpu_count", "seq", "seq_epoch", "worker_num", "global_gpu_index",
                 "record_version", "sample_interval_s", "xid_errors_total", "ecc_dbe_volatile_total",
                 "ecc_sbe_count", "_pct", None, 7):
        assert not resource_module.is_metric_field(name), name
    # A field's name is drawn as the metric's: only plain lowercase words, never an escape sequence.
    for name in ("\x1b]52;c;aGk=\x07_pct", "util\x1b[2J_pct", "GPU_util_pct", "gpu util_pct", "gpu-util_pct",
                 "gpu_util_pct\n", "gpü_util_pct", "\u202egpu_util_pct"):
        assert not resource_module.is_metric_field(name), repr(name)


def test_unitless_fields_get_no_chart_and_the_header_still_names_cpus_and_memory(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(3):
        at = start + timedelta(seconds=5 * i)
        gpu = _row(at, 0, 40)
        gpu.update(power_watts=300.0, pid=99, xid_errors_total=0, ecc_sbe_volatile_total=2 * i)
        gpu["\x1b]0;retitled\x07_pct"] = 50.0
        rows.append(gpu)
        rows.append({
            "kind": "system", "record_version": 3, "sample_ts": at.isoformat().replace("+00:00", "Z"),
            "worker_num": 0, "seq": i, "seq_epoch": "e", "sample_interval_s": 5.0, "cpu_util_pct": 25.0,
            "cpu_count": 88.0, "mem_used_bytes": float(86 * 1024**3), "mem_total_bytes": float(860 * 1024**3),
            "pid": 4242 + i, "uptime_s": 60.0 + 5 * i, "ctx_switches_total": 1000 * i,
        })
    _write_metrics(tmp_path, "job:training:0", rows)

    data = resource_module.load_resource_data(tmp_path)
    assert data.ignored_rows == 0
    gpu = data.kind("job:training:0", resource_module.GPU)
    system = data.kind("job:training:0", resource_module.SYSTEM)
    assert [metric.key for metric in gpu.metrics] == [
        "gpu_util_pct", "fb_used_pct", "fb_used_bytes", "fb_free_bytes", "power_watts"]
    # The memory used % is derived; the CPUs and the memory limit only go in the header.
    assert [metric.key for metric in system.metrics] == ["cpu_util_pct", "mem_used_pct", "mem_used_bytes"]
    for resource in (gpu, system):
        keys = {metric.key for metric in resource.metrics}
        for unit in (None, 0):
            view = resource.view("all", unit)
            assert set(view.charts) == set(view.stats) == keys
    assert system.view("10m").stats["mem_used_pct"].latest == 10.0
    assert resource_module.node_allocation_text(system) == "nodes: 1 reporting  ·  88 CPUs  ·  860 GiB each"


def test_sub_job_directories_with_unprintable_names_are_passed_over(tmp_path):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    _write_metrics(tmp_path, "job:training:0", [_row(now, 0, 10)])
    # A sub-job's id is its directory's name, drawn on the terminal: an escape sequence, a
    # line break, a bidi override or a pile of combining marks is no sub-job's id.
    for name in ("job:training:1\x1b]52;c;aGk=\x07", "job:\x1b[2Jtraining:2", "job:training:3\n",
                 "job\u202e:training:4", "job:training:5" + "\u0301" * 10):
        _write_metrics(tmp_path, name, [_row(now, 0, 20), "{bad json"])

    assert resource_module.cached_metric_sources(tmp_path) == ["job:training:0"]
    data = resource_module.load_resource_data(tmp_path)
    assert list(data.kinds) == ["job:training:0"]
    assert data.ignored_rows_by_sub_job == {}


def test_history_past_seven_hours_is_binned_and_keeps_its_span(tmp_path):
    # A long job's samples are binned so the whole-job view stays bounded: kept one by
    # one, twelve hours would be hundreds of thousands of points. The bins must still
    # cover the job's real span and keep its extremes.
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    minutes, step = 12 * 60, 60.0  # twelve hours at a sample a minute, past the 7-hour mark
    rows = [_row(start + timedelta(seconds=minute * step), 0,
                 100.0 if minute == 300 else float(minute % 50), interval=step)
            for minute in range(minutes)]
    assert minutes > resource_module._MAX_BINS, "the fixture must exceed the bin cap"
    _write_metrics(tmp_path, "job:training:0", rows)

    resource = resource_module.ResourceLoader(tmp_path).load().kinds["job:training:0"]["gpu"]
    whole = resource.view("all")
    stats = whole.stats[BUSY]
    # Binned, not dropped: the peak and the floor survive, and the span reaches the end.
    assert stats.max == 100.0, "binning lost the job's peak"
    assert stats.min == 0.0
    span = (whole.end - whole.start).total_seconds()
    assert span >= (minutes - 2) * step, f"the binned span is short: {span}s"
    # Bounded: the chart a whole-job view draws comes from at most _MAX_BINS points.
    points = whole.chart(BUSY)
    assert 0 < len(points) <= resource_module._MAX_BINS, len(points)
    # The last ten minutes are still read from the samples themselves, unbinned.
    recent = resource.view("10m")
    assert recent.stats[BUSY].latest == stats.latest


# ─── rendering ───────────────────────────────────────────────────────────────

_LABELS = ("100%", "50%", "0%")
_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _chart(values, *, width=20, height=4, hue="#4ade80"):
    return resource_module._chart_spans(values, _T0, _T0 + timedelta(minutes=10), width, height, _LABELS, 100.0, hue)


def _text(line) -> str:
    return "".join(text for text, _style in line)


def test_chart_draws_a_lit_edge_over_a_faint_fill():
    lines = _chart([90.0] * 13)  # width 20 = 4 label cells, " ┤", 13 columns and the marker
    rows = [_text(line) for line in lines[:4]]
    # 90% of four rows is 29 of 32 eighths: three full rows, then 5/8 of the top row.
    assert rows[0] == "100% ┤" + "▅" * 13 + "◂"
    assert rows[1].endswith("█" * 13 + " ") and rows[3].endswith("█" * 13 + " ")
    styles = {style for line in lines[:4] for text, style in line if "█" in text}
    # The fill is the hue faded over the background (in its own colour on its own colour).
    assert styles and all(style.split(" on ")[0] == style.split(" on ")[1] for style in styles)
    assert all(style.split(" on ")[0] != "#4ade80" for style in styles)
    assert ("▅" * 13, "#4ade80") in lines[0]
    # The gridline at 50% runs on through the fill, a shade lighter than it.
    assert "┈" * 13 in rows[2]


def test_chart_leaves_a_gap_blank_and_breaks_its_baseline():
    values = [50.0, None, None, 50.0, 0.0]
    lines = _chart(values, width=12, height=2, hue=None)
    *rows, axis, _times = (_text(line) for line in lines)
    columns = [row[6:11] for row in rows]
    # A missing bucket is blank to the baseline, unlike a measured 0, which shows an eighth.
    assert [column[1:3] for column in columns] == ["  ", "  "]
    assert columns[-1][4] == "▁"
    assert axis.endswith("─╌╌──" + "─")
    # The newest column has a value: the marker is on its row.
    assert rows[-1].endswith("◂")


def test_chart_has_no_now_marker_when_the_newest_bucket_is_a_gap():
    lines = _chart([40.0, 40.0, None], width=12, height=2, hue=None)
    assert not any("◂" in _text(line) for line in lines)


def test_a_window_of_fewer_buckets_than_columns_spans_the_chart():
    # A short window at a long scrape interval, its samples covering it: each bucket is as many of the 33
    # columns wide, from the axis on, so nothing is blank between the axis and the bars.
    row, axis, times = (_text(line) for line in _chart([10.0, 50.0, 90.0], width=40, height=1, hue=None))
    assert row == "100% ┤" + "▁" * 11 + "▄" * 11 + "█" * 11 + "◂"
    assert axis == "  0% └" + "─" * 34
    # The times span it too: the window's start under the first column, its end under the marker.
    assert times.startswith(" " * 6 + _T0.astimezone().strftime("%H:%M")) and len(times) == 40
    # Where they don't divide evenly, some buckets are a column wider.
    row = _text(_chart([10.0, 50.0, 90.0, 50.0], width=40, height=1, hue=None)[0])
    assert row == "100% ┤" + "▁" * 9 + "▄" * 8 + "█" * 8 + "▄" * 8 + "◂"


def _clock(moment) -> str:
    return moment.astimezone().strftime("%H:%M")


def test_a_window_longer_than_its_samples_keeps_its_time_scale():
    # Five minutes of samples under "last 10m", over 33 columns: half of them. A job that short starts at the
    # left, the rest of its window blank after the marker; a window whose older samples are still loading ends
    # at the right, where its newest samples are, with the older ones to come before them.
    view = resource_module.KindView(name="10m", start=_T0, end=_T0 + timedelta(minutes=5), label="",
                                    charts={BUSY: (10.0,) * 61})
    metric = resource_module.Metric(key=BUSY, name="GPU util %", unit="%", scale=100.0)

    def chart(complete):
        return [_text(line) for line in resource_module.metric_chart_spans(view, metric, 40, height=1,
                                                                           complete=complete)]

    row, axis, times = chart(True)
    assert row == "100% ┤" + "▁" * 17 + "◂" + " " * 16
    assert axis == "  0% └" + "─" * 34  # unbroken past the newest sample: no gap, just time to come
    assert times.split() == [_clock(_T0), _clock(_T0 + timedelta(minutes=5)), _clock(_T0 + timedelta(minutes=10))]
    row, axis, times = chart(False)
    assert row == "100% ┤" + " " * 16 + "▁" * 17 + "◂"
    assert axis == "  0% └" + "─" * 34
    assert times.split() == [_clock(_T0 - timedelta(minutes=5)), _clock(_T0), _clock(_T0 + timedelta(minutes=5))]
    # The gridline runs on through the blank time, past the marker.
    rows = [_text(line) for line in resource_module.metric_chart_spans(view, metric, 40, height=4)]
    assert rows[2] == " 50% ┤" + "┈" * 34


def test_mix_blends_a_hue_over_the_background():
    assert resource_module.mix("#ffffff", 1.0, "#000000") == "#ffffff"
    assert resource_module.mix("#ffffff", 0.0, "#000000") == "#000000"
    assert resource_module.mix("#ff0000", 0.5, "#000000") == "#800000"
    assert resource_module.mix("#ff0000", 2.0, "#000000") == "#ff0000"  # clamped


def test_meter_fills_in_eighths_of_a_cell_on_a_track():
    meter = resource_module.meter_spans
    assert meter(50.0, 100.0, 4) == [("██··", "")]
    assert meter(90.0, 100.0, 4) == [("███▋", "")]  # 28.8 of 32 eighths
    assert meter(0.5, 100.0, 4) == [("▏···", "")]  # above 0 shows
    assert meter(0.0, 100.0, 4) == [("····", "")]
    assert meter(None, 100.0, 4) == [("····", "")]
    assert meter(250.0, 100.0, 4) == [("████", "")]
    styled = meter(50.0, 100.0, 4, "#3fb950", track="#30363d")
    assert styled == [("██", "#3fb950 on #30363d"), ("  ", "on #30363d")]
    assert meter(50.0, 100.0, 0) == []


def test_health_bounds_judge_only_the_metrics_they_know():
    health = resource_module.metric_health
    gpu, system = resource_module.GPU, resource_module.SYSTEM
    assert health(gpu, "gpu_util_pct", 90.0) == "ok"
    assert health(gpu, "gpu_util_pct", 20.0) == "warn"
    assert health(gpu, "gpu_temp_c", 70.0) == "ok"
    assert health(gpu, "gpu_temp_c", 86.0) == "warn"
    assert health(gpu, "gpu_temp_c", 91.0) == "err"
    assert health(gpu, "fb_used_pct", 96.0) == "warn"
    assert health(system, "mem_used_pct", 98.0) == "err"
    assert health(system, "cpu_util_pct", 20.0) == "ok"
    assert health(gpu, "power_watts", 700.0) is None
    assert health(gpu, "gpu_util_pct", None) is None
    assert resource_module.worst_health(["ok", None, "err", "warn"]) == "err"
    assert resource_module.worst_health([]) is None


def _gpu_resource(tmp_path, utils, temps):
    rows = []
    for i in range(3):
        at = _T0 + timedelta(seconds=5 * i)
        for gpu, (util, temp) in enumerate(zip(utils, temps)):
            row = _row(at, gpu, util)
            row.update(power_watts=500.0 + 10 * gpu, gpu_temp_c=temp)
            rows.append(row)
    _write_metrics(tmp_path, "job:training:0", rows)
    return resource_module.ResourceLoader(tmp_path).load().kind("job:training:0", resource_module.GPU)


def test_kpi_tiles_show_what_the_rows_have_and_the_least_healthy_unit(tmp_path):
    resource = _gpu_resource(tmp_path, [90.0, 30.0], [60.0, 88.0])
    view = resource.view("10m")
    tiles = {tile.label: tile for tile in resource_module.kpi_tiles(resource, view, requested=4)}
    assert list(tiles) == ["GPU util", "GPU memory", "power", "temp", "GPUs reporting"]
    assert tiles["GPU util"].value == "60%" and tiles["GPU util"].detail == "30–90%"
    assert tiles["GPU util"].health == "warn"  # one GPU at 30%
    # GPU 0 uses 1 GiB and GPU 1 2 GiB, each with 8 GiB free: a mean of 1.5 of 9.5 GiB.
    assert tiles["GPU memory"].value == "1.50 / 9.50 GiB" and tiles["GPU memory"].detail == "16%"
    assert tiles["power"].value == "505 W" and tiles["power"].detail == "1.0 kW total"
    assert tiles["power"].health is None
    assert tiles["temp"].detail == "max 88 °C" and tiles["temp"].health == "warn"
    assert (tiles["GPUs reporting"].value, tiles["GPUs reporting"].health) == ("2/4", "warn")

    one = {tile.label: tile for tile in resource_module.kpi_tiles(resource, resource.view("10m", 0), unit=0,
                                                                  requested=2)}
    assert (one["GPU util"].value, one["GPU util"].detail, one["GPU util"].health) == ("90%", "", "ok")
    assert one["GPUs reporting"].health == "ok"

    stale = {tile.label: tile for tile in resource_module.kpi_tiles(resource, view, requested=2, reporting=0,
                                                                    stale=True)}
    assert stale["GPU util"].health is None and stale["temp"].health is None
    assert (stale["GPUs reporting"].value, stale["GPUs reporting"].health) == ("0/2", "err")


def test_kpi_strip_fills_its_width_and_drops_details_before_wrapping():
    tiles = [resource_module.Kpi("GPU util", "90%", "84–95%", "ok"), resource_module.Kpi("power", "632 W", "5.1 kW")]
    for width in (40, 80):
        lines = resource_module.kpi_strip_spans(tiles, width)
        assert len(lines) == 2
        assert all(sum(len(text) for text, _style in line) == width for line in lines)
    wide = [_text(line) for line in resource_module.kpi_strip_spans(tiles, 80, styled=False)]
    assert "GPU util" in wide[0] and "84–95%" in wide[1] and "5.1 kW" in wide[1]
    # Too narrow for both details: the right one goes first.
    snug = [_text(line) for line in resource_module.kpi_strip_spans(tiles, 30, styled=False)]
    assert "84–95%" in snug[1] and "5.1 kW" not in snug[1]
    # Too narrow for the tiles side by side: a row each.
    assert len(resource_module.kpi_strip_spans(tiles, 12)) == 4
    edge = resource_module.kpi_strip_spans(tiles, 80)[1][0]
    assert edge == ("▎ ", f"{resource_module.HEALTH_COLORS['ok']} on {resource_module.SURFACE}")


def test_stats_table_heads_line_up_with_their_columns(tmp_path):
    resource = _gpu_resource(tmp_path, [90.0, 30.0], [60.0, 88.0])
    view, whole = resource.view("10m"), resource.view("all")
    lines = [_text(line) for line in resource_module.metric_stats_spans(view, whole, resource.metrics, width=160)]
    title, head, *rows = lines
    assert head.startswith("metric")
    assert "last 10m" in title and "whole job" in title
    # Each head ends in the column where its values end.
    for name in ("min", "avg", "max", "latest"):
        end = head.index(name) + len(name)
        assert all(row[end - 1] != " " for row in rows), name
    assert head.count("│") == 1 and all(row.count("│") == 1 for row in rows)
    assert len({len(line) for line in lines}) == 1


def test_a_preview_is_its_sub_job_s_newest_rows_marked_partial_and_leaves_the_loader_alone(tmp_path):
    # download_metrics' preview: a sub-job's newest half hour alone, drawn while the rest downloads.
    sub = "job:training:0"
    now = datetime.now(timezone.utc).replace(microsecond=0)
    rows = [{**_row(now - timedelta(minutes=back), gpu, 50.0, interval=60.0), "sub_job_id": sub}
            for back in range(120, -1, -1) for gpu in range(2)]
    _write_metrics(tmp_path, sub, rows)  # the whole history, as the download will commit it
    preview = tmp_path / "preview.jsonl"
    preview.write_text("".join(json.dumps(row) + "\n" for row in rows[-2 * 33:]))  # its newest 32 minutes
    loader = resource_module.ResourceLoader(tmp_path)
    shown = loader.preview(sub, preview).kind(sub, "gpu")
    assert shown.partial and shown.latest_sample_ts == now and shown.first_sample_ts == now - timedelta(minutes=32)
    # The windows it covers are whole; a longer one waits for the older samples.
    assert resource_module.window_complete(shown, shown.view("10m"))
    assert resource_module.window_complete(shown, shown.view("30m"))
    assert not resource_module.window_complete(shown, shown.view("1h"))
    # The loader's own state is untouched: the file it reads next is read whole.
    full = loader.load().kind(sub, "gpu")
    assert not full.partial and full.first_sample_ts == now - timedelta(minutes=120)


# ─── a refresh builds only the window on screen (and the whole job's) ────────

def _same(x, y) -> bool:
    """Equal, with NaN equal to NaN: two builds of one view can't be compared with == alone."""
    import math
    if isinstance(x, float) and isinstance(y, float):
        return x == y or (math.isnan(x) and math.isnan(y))
    if isinstance(x, (tuple, list)) and isinstance(y, (tuple, list)):
        return len(x) == len(y) and all(_same(a, b) for a, b in zip(x, y))
    if isinstance(x, dict) and isinstance(y, dict):
        return x.keys() == y.keys() and all(_same(x[key], y[key]) for key in x)
    if hasattr(x, "__dataclass_fields__") and type(x) is type(y):
        return all(_same(getattr(x, name), getattr(y, name)) for name in x.__dataclass_fields__)
    return x == y


SUB = "7:training:0"


def _scrapes(start, first, count, gpus=4):
    return [_row(start + timedelta(seconds=5 * k), gpu, float((k * 7 + gpu * 13) % 100))
            for k in range(first, first + count) for gpu in range(gpus)]


def _append(root, rows, chunks):
    path = root / SUB / "gpu.jsonl"
    with path.open("a", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps({**row, "sub_job_id": SUB}) + "\n")
    path.with_name(".gpu.jsonl.manifest.json").write_text(
        json.dumps({"version": 1, "names": ["chunk.gz"] + [f"chunk{n}.gz" for n in range(1, chunks)],
                    "bytes": path.stat().st_size}) + "\n", encoding="utf-8")


def _two_hours(tmp_path):
    start = datetime(2026, 10, 20, 10, 0, tzinfo=timezone.utc)
    _write_metrics(tmp_path, SUB, _scrapes(start, 0, 1440))  # two hours at 5 s
    return start


def test_a_refresh_builds_only_the_window_on_screen_and_the_whole_job_s(tmp_path):
    # Every window was rebuilt on every refresh: on a job with hundreds of GPUs the long ones
    # took seconds, whichever was on screen.
    start = _two_hours(tmp_path)
    loader = resource_module.ResourceLoader(tmp_path)
    assert loader.load(shown="10m").kinds[SUB]["gpu"].fresh(), "a first build leaves a window missing"

    _append(tmp_path, _scrapes(start, 1440, 6), chunks=2)
    gpu = loader.load(shown="10m").kinds[SUB]["gpu"]

    assert gpu.fresh("10m") and not gpu.fresh()
    assert gpu._stale == {"30m", "1h", "6h"}, "the window on screen or the whole job's was carried"
    whole = resource_module.ResourceLoader(tmp_path).load().kinds[SUB]["gpu"]  # every window, from scratch
    for window in ("10m", "all"):  # the chart, and the stats beside it
        for unit in (None, *gpu.unit_ids):
            assert _same(gpu.view(window, unit), whole.view(window, unit)), (window, unit)


def test_a_window_carried_from_before_is_rebuilt_once_shown(tmp_path):
    start = _two_hours(tmp_path)
    loader = resource_module.ResourceLoader(tmp_path)
    loader.load(shown="10m")
    _append(tmp_path, _scrapes(start, 1440, 6), chunks=2)
    loader.load(shown="10m")

    gpu = loader.load(shown="6h").kinds[SUB]["gpu"]  # nothing new to read: the window was picked

    assert gpu.fresh("6h")
    whole = resource_module.ResourceLoader(tmp_path).load().kinds[SUB]["gpu"]
    for unit in (None, *gpu.unit_ids):
        assert _same(gpu.view("6h", unit), whole.view("6h", unit)), unit


def test_without_a_window_named_every_window_is_built(tmp_path):
    start = _two_hours(tmp_path)
    loader = resource_module.ResourceLoader(tmp_path)
    loader.load()
    _append(tmp_path, _scrapes(start, 1440, 6), chunks=2)
    assert loader.load().kinds[SUB]["gpu"].fresh()


def test_picking_a_window_with_no_new_rows_builds_only_that_window(tmp_path, monkeypatch):
    # The whole job's frame and the extremes were rebuilt too on every pick: seconds a press
    # of w on a job with hundreds of GPUs, for frames already up to date.
    start = _two_hours(tmp_path)
    loader = resource_module.ResourceLoader(tmp_path)
    loader.load(shown="10m")
    _append(tmp_path, _scrapes(start, 1440, 6), chunks=2)
    loader.load(shown="10m")
    built, natives = [], []
    real_raw, real_native = resource_module._raw_frame, resource_module._Native
    monkeypatch.setattr(resource_module, "_raw_frame", lambda series, name, *a, **k: (
        built.append((series.spec.name, name)), real_raw(series, name, *a, **k))[1])
    monkeypatch.setattr(resource_module, "_Native", lambda *a, **k: (natives.append(1), real_native(*a, **k))[1])

    gpu = loader.load(shown="1h").kinds[SUB]["gpu"]

    assert built == [("gpu", "1h")] and not natives, "frames already up to date were built again"
    assert gpu.fresh("1h") and gpu.fresh("10m") and gpu._stale == {"30m", "6h"}
    monkeypatch.setattr(resource_module, "_raw_frame", real_raw)
    monkeypatch.setattr(resource_module, "_Native", real_native)
    whole = resource_module.ResourceLoader(tmp_path).load().kinds[SUB]["gpu"]
    for window in ("1h", "10m", "all"):
        for unit in (None, *gpu.unit_ids):
            assert _same(gpu.view(window, unit), whole.view(window, unit)), (window, unit)
