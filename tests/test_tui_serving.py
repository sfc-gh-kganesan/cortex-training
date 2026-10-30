# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""A deployment's serving numbers, read from its console's drain_metrics lines (``tui/serving.py``)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

pytest.importorskip("textual")

from textual.widgets import Static  # noqa: E402
from textual.widgets import TabbedContent  # noqa: E402

from cortex_training import AccountJobListUnavailable  # noqa: E402
from cortex_training.client import NoChunksError  # noqa: E402
from cortex_training.tui import serving  # noqa: E402
from cortex_training.tui.app import CortexTrainingLogTUI  # noqa: E402
from cortex_training.tui.logs import LogScreen  # noqa: E402


def _drain(at, done, **extra) -> str:
    row = {"ts": "2026-10-26T19:46:24+00:00", "level": "INFO", "msg": "drain_metrics", "op": "drain-metrics",
           "drained_at": at, "completed_requests": done, "latency_s_p50": 0.05, "latency_s_p99": 0.2,
           "ttft_s_p50": 0.01, "ttft_s_p99": 0.03, "queue_s_p99": 0.0, "num_running_reqs_latest_sum": 1.0,
           "num_waiting_reqs_latest_sum": 0.0, "kv_cache_usage_max": 0.25, "replica_count": 1,
           "replica_errors": 0, "stdout_summarized": True, **extra}
    return json.dumps(row)


STARTUP = json.dumps({"msg": "job_metrics", "op": "create-job", "duration_s": 135.9, "error": None})


def test_parse_line_reads_a_drain_and_skips_everything_else():
    d = serving.parse_line(_drain(100.0, 16))
    assert d.drained_at == 100.0 and d.completed == 16 and d.latency_p99 == 0.2 and d.replicas == 1
    assert serving.parse_line("plain text with drain_metrics in it") is None
    assert serving.parse_line(json.dumps({"msg": "something_else", "drained_at": 1})) is None
    assert serving.parse_line(_drain(None, 3)) is None  # no time: no row
    idle = serving.parse_line(_drain(100.0, 0, latency_s_p50=None, latency_s_p99=None, kv_cache_usage_max=None))
    assert idle.completed == 0 and idle.latency_p50 is None and idle.kv_max is None
    assert serving.parse_line(_drain(100.0, 1, latency_s_p99="NaN")).latency_p99 is None


REAL_CONSOLE = Path(__file__).parent / "fixtures" / "deployment_console.log"  # trimmed from a real deployment


def test_the_real_console_lines_parse(tmp_path):
    path = tmp_path / "stdout.log"
    path.write_bytes(REAL_CONSOLE.read_bytes())
    data = serving.ConsoleReader(path).read()
    assert data.startup_s == 135.96204860392027
    assert [(d.drained_at, d.completed) for d in data.drains] == [
        (1793043984.3675478, 0), (1793044854.6796148, 1), (1793044944.7124422, 0)]
    idle, busy = data.drains[0], data.drains[1]
    assert idle.latency_p50 is None and idle.kv_max is None and idle.replicas == 1 and idle.running == 0.0
    assert busy.latency_p50 == 0.6584606170654297 and busy.ttft_p99 == 0.0229952335357666
    assert busy.queue_p99 == 0.004767656326293945
    assert busy.kv_max == 2.6320984404804015e-05 and busy.replicas == 1 and busy.replica_errors == 0


def _failed_create(line: str) -> str:
    """The real create-job row, as a deployment writes one that raised (``Type: message``)."""
    return json.dumps({**json.loads(line), "duration_s": 2.0, "error": "RuntimeError: CUDA out of memory"})


def test_a_failed_create_job_is_no_startup_time(tmp_path):
    created = next(ln for ln in REAL_CONSOLE.read_text().splitlines() if '"job_metrics"' in ln)
    path = tmp_path / "stdout.log"
    path.write_text(_failed_create(created) + "\n")
    reader = serving.ConsoleReader(path)
    assert reader.read().startup_s is None
    with path.open("a") as f:
        f.write(created + "\n")  # the retry that started it
    assert reader.read().startup_s == 135.96204860392027


def test_rate_divides_by_the_actual_gap():
    a, b, c, d = (serving.parse_line(_drain(t, n)) for t, n in ((0.0, 0), (30.0, 15), (80.0, 20), (130.0, 30)))
    assert serving.rate(None, a) is None
    assert serving.rate(a, b) == 0.5
    assert serving.rate(b, c) is None and serving.rate(c, d) is None  # a drain lost in between: no rate
    assert serving.rate(b, b) is None
    late = serving.parse_line(_drain(70.0, 20))  # a drain a little late is still one drain
    assert serving.rate(b, late) == 0.5


def test_reader_reads_only_what_was_appended(tmp_path):
    path = tmp_path / "stdout.log"
    reader = serving.ConsoleReader(path)
    assert reader.read().drains == []  # no file yet
    path.write_text(STARTUP + "\n" + _drain(0.0, 1) + "\n")
    first = reader.read()
    assert [d.completed for d in first.drains] == [1] and first.startup_s == 135.9
    with path.open("a") as f:
        f.write("INFO not json\n" + _drain(30.0, 2) + "\n")
    assert [d.completed for d in reader.read().drains] == [1, 2]
    assert reader._offset == path.stat().st_size


def test_a_partial_last_line_waits_for_the_next_read(tmp_path):
    path = tmp_path / "stdout.log"
    line = _drain(0.0, 4)
    path.write_text(line[:20])
    reader = serving.ConsoleReader(path)
    assert reader.read().drains == []
    path.write_text(line + "\n")
    assert [d.completed for d in reader.read().drains] == [4]


def test_a_rewritten_file_is_read_again_from_its_start(tmp_path):
    path = tmp_path / "stdout.log"
    path.write_text(_drain(0.0, 1) + "\n" + _drain(30.0, 2) + "\n")
    reader = serving.ConsoleReader(path)
    reader.read()
    path.write_text(_drain(0.0, 9) + "\n")
    assert [d.completed for d in reader.read().drains] == [9]


def test_a_console_rebuilt_in_place_is_read_again_from_its_start(tmp_path):
    # A late chunk makes the client rebuild the console and os.replace it, usually larger.
    path = tmp_path / "stdout.log"
    path.write_text(_drain(100.0, 1) + "\n" + _drain(160.0, 3) + "\n")
    reader = serving.ConsoleReader(path)
    reader.read()
    rebuilt = tmp_path / "stdout.log.partial"
    rebuilt.write_text("".join(_drain(t, n) + "\n" for t, n in ((100.0, 1), (130.0, 2), (160.0, 3), (190.0, 4))))
    rebuilt.replace(path)
    drains = reader.read().drains
    assert [(d.drained_at, d.completed) for d in drains] == [(100.0, 1), (130.0, 2), (160.0, 3), (190.0, 4)]
    assert sum(d.completed for d in drains) == 10


def test_bytes_past_the_committed_size_wait_for_the_next_read(tmp_path, monkeypatch):
    path = tmp_path / "stdout.log"
    first = _drain(0.0, 1) + "\n"
    path.write_text(first + _drain(30.0, 2) + "\n")
    monkeypatch.setattr(serving, "committed_size", lambda p, size: len(first))
    reader = serving.ConsoleReader(path)
    assert [d.completed for d in reader.read().drains] == [1]
    monkeypatch.setattr(serving, "committed_size", lambda p, size: size)
    assert [d.completed for d in reader.read().drains] == [1, 2]


def test_a_long_new_tail_is_read_a_line_at_a_time(tmp_path):
    import tracemalloc

    path = tmp_path / "stdout.log"
    filler = "INFO a console line that is not a drain " + "x" * 60 + "\n"
    path.write_text(filler * 80_000 + _drain(0.0, 5) + "\n")  # about 8 MB, new since the last read
    reader = serving.ConsoleReader(path)
    tracemalloc.start()
    try:
        drains = reader.read().drains
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert [d.completed for d in drains] == [5]
    assert peak < 1_000_000, peak  # not the whole tail held at once


def test_a_drain_time_past_what_a_date_holds_shows_a_dash():
    from cortex_training.tui.serving_view import ServingPane

    pane = ServingPane(MagicMock(), "8", Path("."), job=lambda: _deployment(), download=lambda **_: [])
    pane._data = serving.ServingData([serving.parse_line(_drain(1e20, 2))])
    assert "last drain" not in str(pane._status_text())
    table = pane._table()
    assert next(iter(table.columns[0].cells)) == "—"


def test_only_a_positive_replica_error_count_is_shown():
    from cortex_training.tui.serving_view import ServingPane

    pane = ServingPane(MagicMock(), "8", Path("."), job=lambda: _deployment(), download=lambda **_: [])
    pane._data = serving.ServingData([serving.parse_line(_drain(1.0, 2, replica_errors=-1)),
                                      serving.parse_line(_drain(31.0, 3, replica_errors=2))])
    assert [str(cell) for cell in pane._table().columns[-1].cells] == ["1 (2 err)", "1"]


def test_serving_rows_carry_their_date_unless_all_are_today_s():
    import time
    from datetime import datetime
    from datetime import timedelta

    from cortex_training.tui.serving_view import ServingPane

    now = datetime.now().replace(microsecond=0)
    if now.date() != (now + timedelta(seconds=5)).date():
        time.sleep(6)  # the pane reads the clock too: not across midnight
        now = datetime.now().replace(microsecond=0)
    pane = ServingPane(MagicMock(), "8", Path("."), job=lambda: _deployment(), download=lambda **_: [])
    pane._data = serving.ServingData([serving.parse_line(_drain(now.timestamp(), 2))])
    assert list(pane._table().columns[0].cells) == [now.strftime("%H:%M:%S")]
    two_days_ago = now.timestamp() - 2 * 86400
    pane._data = serving.ServingData([serving.parse_line(_drain(two_days_ago, 1)),
                                      serving.parse_line(_drain(now.timestamp(), 2))])
    assert list(pane._table().columns[0].cells) == [
        now.strftime("%m-%d %H:%M:%S"), datetime.fromtimestamp(two_days_ago).strftime("%m-%d %H:%M:%S")]


def test_an_ended_deployment_s_last_drain_reads_as_a_time_not_an_age(monkeypatch):
    from datetime import datetime

    from cortex_training.tui.serving_view import ServingPane

    drained_at = datetime(2026, 10, 26, 19, 46).timestamp()
    job = {"value": _deployment()}
    pane = ServingPane(MagicMock(), "8", Path("."), job=lambda: job["value"], download=lambda **_: [])
    pane._data = serving.ServingData([serving.parse_line(_drain(drained_at, 2))])
    assert " ago" in str(pane._status_text())
    job["value"] = _deployment("CANCELLED")  # the pane soon stops reloading: an age would go stale
    assert "last drain at 2026-10-26 19:46" in str(pane._status_text())
    monkeypatch.setattr("cortex_training.tui.serving_view.local_time", lambda dt: "")  # out of the local range
    assert "last drain" not in str(pane._status_text())


def test_a_long_run_shows_its_newest_drains_and_says_so():
    from cortex_training.tui.serving_view import _MAX_ROWS
    from cortex_training.tui.serving_view import ServingPane

    pane = ServingPane(MagicMock(), "8", Path("."), job=lambda: _deployment(), download=lambda **_: [])
    pane._data = serving.ServingData([serving.parse_line(_drain(30.0 * i, i)) for i in range(_MAX_ROWS)])
    assert pane._table().caption is None
    pane._data = serving.ServingData([serving.parse_line(_drain(30.0 * i, i)) for i in range(_MAX_ROWS + 5)])
    table = pane._table()
    assert table.row_count == _MAX_ROWS and table.caption == f"newest {_MAX_ROWS} of {_MAX_ROWS + 5} drains"
    assert next(iter(table.columns[2].cells)) == str(_MAX_ROWS + 4)  # newest first


def test_serving_state():
    none, some = serving.ServingData(), serving.ServingData([serving.parse_line(_drain(0.0, 1))])
    assert serving.serving_state({"status": "PLACING"}, none) == "starting"
    assert serving.serving_state({"status": "RUNNING"}, none) == "starting (no metrics yet)"
    assert serving.serving_state({"status": "RUNNING"}, some) == "serving"
    assert serving.serving_state({"status": "CANCELLED"}, some) == "ended"
    assert serving.serving_state({"status": "CANCELLING"}, some) == "dropping"  # its badge says DROPPING
    # UNKNOWN is live but not known to run: its drains say it serves, and nothing else says it starts.
    assert serving.serving_state({"status": "UNKNOWN"}, some) == "serving"
    assert serving.serving_state({"status": "UNKNOWN"}, none) == "status unknown"


def test_a_first_load_says_it_reads_the_console_rather_than_no_metrics():
    from cortex_training.tui.serving_view import ServingPane

    pane = ServingPane(MagicMock(), "8", Path("."), job=lambda: _deployment(), download=lambda **_: [])
    pane._loading = True
    status, table = str(pane._status_text()), str(pane._table())
    assert "no metrics" not in status and "No drain_metrics" not in table, (status, table)
    assert "Reading the console" in table
    pane._finish(None, "RUNNING")
    assert "no metrics yet" in str(pane._status_text()) and "No drain_metrics yet" in str(pane._table())
    pane._loading = True  # a later reload still finds none: the header stays as it was
    assert "no metrics yet" in str(pane._status_text()) and "No drain_metrics yet" in str(pane._table())


DEPLOY = "8:sampling:0"


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "cache"))
    import cortex_training.tui.widgets as widgets_module

    monkeypatch.setattr(widgets_module, "_ERROR_LOG", str(tmp_path / "errors.log"))


def _deployment(status="RUNNING"):
    return {"job_id": "8", "status": status,
            "sub_jobs": [{"sub_job_id": DEPLOY, "job_type": "sampling", "status": status,
                          "inference_config": {"deployment_class": "EVALUATION", "num_replicas": 1.0}}]}


def _console(lines):
    """A ``download_stdout_logs`` that saves ``lines`` as the sub-job's console."""

    def download(job_id, output_dir, *, resume=False, sub_job_ids=None, **_kwargs):
        sid = list(sub_job_ids)[0]
        path = Path(output_dir) / sid / "stdout.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(line + "\n" for line in lines))
        return [{"sub_job_id": sid, "saved_path": str(path)}]

    return download


def _client(job, lines=()):
    c = MagicMock()
    c.model_deployment_base_url = "https://HOST/api/v2/databases/DB/schemas/PUBLIC/cortex-training/inference/v1"
    c.list_jobs_page.side_effect = AccountJobListUnavailable("not deployed", reason="not_deployed")
    c.list_jobs.return_value = []
    c.get_job.return_value = job
    c.tail_logs.return_value = {"entries": [], "next_cursor": "", "eof": True}
    c.download_stdout_logs.side_effect = _console(list(lines))
    return c


async def _wait(pilot, predicate, tries=300):
    for _ in range(tries):
        await pilot.pause(0.01)
        if predicate():
            return True
    return False


async def _settle(app, pilot):
    app.workers.cancel_all()
    for _ in range(5):
        await pilot.pause()


def _text(screen, selector) -> str:
    """What the widget shows, as text (a Rich Table's str() is its repr)."""
    widget = screen.query_one(selector, Static)
    return "\n".join(strip.text for strip in widget.render_lines(widget.content_region.reset_offset))


def _visible_tabs(screen) -> list:
    tabs = screen.query_one("#job-tabs", TabbedContent)
    return [pane.id for pane in tabs.query("TabPane") if not tabs.get_tab(pane.id).has_class("-hidden")]


async def _open(app, pilot) -> LogScreen:
    assert await _wait(pilot, lambda: isinstance(app.screen, LogScreen) and app.screen._sources_ready)
    return app.screen


def test_open_by_id_switches_to_deployment_tabs():
    c = _client(_deployment(), [_drain(0.0, 0), _drain(30.0, 15)])

    async def run():
        app = CortexTrainingLogTUI(c, "8", poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            screen = await _open(app, pilot)
            assert await _wait(pilot, lambda: _visible_tabs(screen) == ["serving", "logs", "resource"])
            assert await _wait(pilot, lambda: "● READY" in str(screen.query_one("#summary", Static).render()))
            assert not c.download_stdout_logs.called  # lazy: only once the tab is shown
            await pilot.press("v")
            assert await _wait(pilot, lambda: "serving" in _text(screen, "#serving-status"))
            assert c.download_stdout_logs.call_args.kwargs["sub_job_ids"] == [DEPLOY]
            assert c.download_stdout_logs.call_args.kwargs["resume"] is True
            table = _text(screen, "#serving-table")
            rows = [line for line in table.splitlines() if ":" in line]  # the drain rows start with a time
            # Time, req/s, Requests, ...: newest first, 15 requests over 30 s; the first drain has no rate.
            assert rows[0].split()[2:4] == ["0.50", "15"]  # 1970 rows: the date comes first
            assert rows[1].split()[2:4] == ["—", "0"]
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_training_job_keeps_its_tabs():
    job = {"job_id": "7", "status": "RUNNING", "sub_jobs": [{"sub_job_id": "7:training:0", "job_type": "training"}]}
    c = _client(job)

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            screen = await _open(app, pilot)
            assert await _wait(pilot, lambda: _visible_tabs(screen) == ["logs", "resource", "training", "checkpoints"])
            await _settle(app, pilot)

    asyncio.run(run())


def test_hidden_tab_keys_do_nothing():
    c = _client(_deployment())

    async def run():
        app = CortexTrainingLogTUI(c, "8", poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            screen = await _open(app, pilot)
            assert await _wait(pilot, lambda: "serving" in _visible_tabs(screen))
            for key in ("m", "c"):
                await pilot.press(key)
                await pilot.pause(0.05)
                assert screen.query_one("#job-tabs", TabbedContent).active == "logs"
            await pilot.press("t")
            assert await _wait(pilot, lambda: screen._active_tab == "resource")
            await pilot.press("t")
            assert await _wait(pilot, lambda: screen._active_tab == "serving")  # round to the first visible tab
            await _settle(app, pilot)

    asyncio.run(run())


def test_no_console_yet_is_not_an_error_and_a_failure_says_so():
    c = _client(_deployment())
    c.download_stdout_logs.side_effect = NoChunksError("no chunks yet", sub_job_ids=[DEPLOY], available=[])

    async def run():
        app = CortexTrainingLogTUI(c, "8", poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            screen = await _open(app, pilot)
            assert await _wait(pilot, lambda: "serving" in _visible_tabs(screen))
            await pilot.press("v")
            pane = screen.query_one("#serving-tab")
            assert await _wait(pilot, lambda: c.download_stdout_logs.called and not pane._loading)
            status = _text(screen, "#serving-status")
            assert "starting (no metrics yet)" in status and "[error]" not in status
            c.download_stdout_logs.side_effect = RuntimeError("boom")
            await pilot.press("r")
            assert await _wait(pilot, lambda: "boom" in _text(screen, "#serving-status"))
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_download_s_warning_shows_in_the_serving_header():
    c = _client(_deployment(), [_drain(0.0, 3)])
    saved = c.download_stdout_logs.side_effect

    def download(job_id, output_dir, **kwargs):
        rows = saved(job_id, output_dir, **kwargs)
        return [{**rows[0], "warnings": ["skipped 2 chunks that failed to download"]}]

    c.download_stdout_logs.side_effect = download

    async def run():
        app = CortexTrainingLogTUI(c, "8", poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            screen = await _open(app, pilot)
            assert await _wait(pilot, lambda: "serving" in _visible_tabs(screen))
            await pilot.press("v")
            pane = screen.query_one("#serving-tab")
            assert await _wait(pilot, lambda: c.download_stdout_logs.called and not pane._loading)
            assert "warning: skipped 2 chunks" in _text(screen, "#serving-status")
            await _settle(app, pilot)

    asyncio.run(run())


def test_an_ended_deployment_reads_ended_with_its_rows():
    c = _client(_deployment("CANCELLED"), [_drain(0.0, 3)])

    async def run():
        app = CortexTrainingLogTUI(c, "8", poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            screen = await _open(app, pilot)
            assert await _wait(pilot, lambda: "serving" in _visible_tabs(screen))
            await pilot.press("v")
            assert await _wait(pilot, lambda: "ended" in _text(screen, "#serving-status"))
            rows = [line for line in _text(screen, "#serving-table").splitlines() if ":" in line]
            assert len(rows) == 1 and rows[0].split()[3] == "3"  # date, time, req/s, Requests
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_deployment_that_ends_on_the_serving_tab_reads_ended_with_its_last_drains():
    c = _client(_deployment(), [_drain(0.0, 3)])

    async def run():
        app = CortexTrainingLogTUI(c, "8", poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            screen = await _open(app, pilot)
            assert await _wait(pilot, lambda: "serving" in _visible_tabs(screen))
            await pilot.press("v")
            assert await _wait(pilot, lambda: _text(screen, "#serving-status").startswith("serving"))
            c.download_stdout_logs.side_effect = _console([_drain(0.0, 3), _drain(30.0, 6)])  # uploaded at the end
            c.get_job.return_value = _deployment("CANCELLED")
            screen._fetch_status()  # the 60 s status check
            assert await _wait(pilot, lambda: _text(screen, "#serving-status").startswith("ended"))
            assert await _wait(pilot, lambda: len([ln for ln in _text(screen, "#serving-table").splitlines()
                                                   if ":" in ln]) == 2)
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_cancelled_serving_download_stops_at_its_next_chunk():
    c = _client(_deployment(), [_drain(0.0, 3)])

    async def run():
        app = CortexTrainingLogTUI(c, "8", poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            screen = await _open(app, pilot)
            assert await _wait(pilot, lambda: "serving" in _visible_tabs(screen))
            chunks, closed = [], [False]

            def download(job_id, output_dir, *, on_chunk, **_kwargs):
                for done in (1, 2):
                    chunks.append(done)
                    closed[0] = True  # cancelled, or the screen closed, while the first chunk came in
                    on_chunk(DEPLOY, done, 2, 0)
                return [{"sub_job_id": DEPLOY}]

            c.download_stdout_logs.side_effect = download
            assert screen._download_serving_console(sub_job_id=DEPLOY, cancelled=lambda: closed[0]) is None
            assert chunks == [1]
            chunks.clear()
            closed[0] = False
            with patch.object(LogScreen, "_screen_open", new_callable=PropertyMock, side_effect=lambda: not closed[0]):
                assert screen._download_serving_console(sub_job_id=DEPLOY, cancelled=lambda: False) is None
            assert chunks == [1]  # the screen closed mid-download: stopped at its next chunk too
            await _settle(app, pilot)

    asyncio.run(run())


def test_the_serving_header_counts_the_download_s_chunks_while_it_runs():
    import threading

    c = _client(_deployment(), [_drain(0.0, 3)])
    saved, go = c.download_stdout_logs.side_effect, threading.Event()

    def download(job_id, output_dir, *, on_chunk, **kwargs):
        on_chunk(DEPLOY, 1, 40, 0)
        go.wait(5)
        on_chunk(DEPLOY, 40, 40, 0)
        return saved(job_id, output_dir, **kwargs)

    c.download_stdout_logs.side_effect = download

    async def run():
        app = CortexTrainingLogTUI(c, "8", poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            screen = await _open(app, pilot)
            assert await _wait(pilot, lambda: "serving" in _visible_tabs(screen))
            await pilot.press("v")
            assert await _wait(pilot, lambda: "downloading chunk 1 of 40" in _text(screen, "#serving-status"))
            go.set()
            assert await _wait(pilot, lambda: _text(screen, "#serving-status").startswith("serving"))
            assert "chunk" not in _text(screen, "#serving-status")
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_broken_progress_count_does_not_fail_the_serving_download():
    c = _client(_deployment(), [_drain(0.0, 3)])
    saved = c.download_stdout_logs.side_effect

    def download(job_id, output_dir, *, on_chunk, **kwargs):
        for done in (1, 2):
            on_chunk(DEPLOY, done, 2, 0)
        return saved(job_id, output_dir, **kwargs)

    async def run():
        app = CortexTrainingLogTUI(c, "8", poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            screen = await _open(app, pilot)
            assert await _wait(pilot, lambda: "serving" in _visible_tabs(screen))
            c.download_stdout_logs.side_effect = download
            calls = []

            def progress(done, total):
                calls.append((done, total))
                raise ValueError("a bug in drawing it")

            assert screen._download_serving_console(sub_job_id=DEPLOY, cancelled=lambda: False,
                                                    progress=progress) == []
            assert calls == [(1, 2)]  # counted once, then no more
            await _settle(app, pilot)

    asyncio.run(run())


@pytest.mark.parametrize("long_after", [False, True])
def test_serving_reads_the_end_s_late_drains_when_it_is_opened_again(long_after):
    c = _client(_deployment(), [_drain(0.0, 3)])

    def serving_reads(pane):
        return sum(call.kwargs.get("sub_job_ids") == [DEPLOY] and call.args[1] == pane._root
                   for call in c.download_stdout_logs.call_args_list)

    async def run():
        app = CortexTrainingLogTUI(c, "8", poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            screen = await _open(app, pilot)
            assert await _wait(pilot, lambda: "serving" in _visible_tabs(screen))
            await pilot.press("v")
            assert await _wait(pilot, lambda: _text(screen, "#serving-status").startswith("serving"))
            await pilot.press("l")
            calls = c.download_stdout_logs.call_count
            pane = screen.query_one("#serving-tab")
            c.get_job.return_value = _deployment("CANCELLED")
            screen._fetch_status()
            assert await _wait(pilot, lambda: screen._job_status == "CANCELLED")
            await _settle(app, pilot)
            assert not any(call.kwargs.get("sub_job_ids") == [DEPLOY] and call.args[1] == pane._root
                           for call in c.download_stdout_logs.call_args_list[calls:])  # not off screen
            c.download_stdout_logs.side_effect = _console([_drain(0.0, 3), _drain(30.0, 6)])  # uploaded late
            if long_after:  # back past the upload window: the end is still read once
                screen._active_seen_at = None
                screen._state_changed_at = "2020-01-01T00:00:00Z"
            await pilot.press("v")
            assert await _wait(pilot, lambda: len([ln for ln in _text(screen, "#serving-table").splitlines()
                                                   if ":" in ln]) == 2)
            assert _text(screen, "#serving-status").startswith("ended")
            if long_after:
                assert await _wait(pilot, lambda: not pane._loading)
                reads = serving_reads(pane)
                await pilot.press("l")
                await pilot.press("v")
                await _wait(pilot, lambda: serving_reads(pane) > reads, tries=50)
                assert serving_reads(pane) == reads  # and only once
            await _settle(app, pilot)

    asyncio.run(run())


def test_serving_reads_again_when_opened_again_only_while_drains_may_arrive():
    def serving_reads(c, pane):
        return sum(call.kwargs.get("sub_job_ids") == [DEPLOY] and call.args[1] == pane._root
                   for call in c.download_stdout_logs.call_args_list)

    async def run(job):
        c = _client(job, [_drain(0.0, 3)])
        app = CortexTrainingLogTUI(c, "8", poll_interval=60)  # no poll: only opening the tab reads
        async with app.run_test(size=(140, 40)) as pilot:
            screen = await _open(app, pilot)
            assert await _wait(pilot, lambda: "serving" in _visible_tabs(screen))
            await pilot.press("v")
            pane = screen.query_one("#serving-tab")
            assert await _wait(pilot, lambda: serving_reads(c, pane) == 1 and not pane._loading)
            await pilot.press("l")
            await pilot.press("v")
            await _wait(pilot, lambda: serving_reads(c, pane) > 1, tries=50)
            reads = serving_reads(c, pane)
            await _settle(app, pilot)
            return reads

    assert asyncio.run(run(_deployment())) == 2
    long_ago = {**_deployment("CANCELLED"), "state_changed_at": "2020-01-01T00:00:00Z"}
    assert asyncio.run(run(long_ago)) == 1  # no drain can still arrive


def test_a_dropping_deployment_is_checked_until_it_is_dropped():
    c = _client(_deployment(), [_drain(0.0, 3)])

    async def run():
        app = CortexTrainingLogTUI(c, "8", poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            screen = await _open(app, pilot)
            assert await _wait(pilot, lambda: "serving" in _visible_tabs(screen))
            await pilot.press("v")
            assert await _wait(pilot, lambda: _text(screen, "#serving-status").startswith("serving"))
            c.get_job.return_value = _deployment("CANCELLING")
            screen._fetch_status()
            assert await _wait(pilot, lambda: _text(screen, "#serving-status").startswith("dropping"))
            screen._active_seen_at = None
            screen._state_changed_at = "2020-01-01T00:00:00Z"  # a drop that hangs: still drained
            pane = screen.query_one("#serving-tab")
            reload = pane.reload
            pane.reload = MagicMock()
            screen._auto_refresh_data()
            assert pane.reload.called
            pane.reload = reload
            c.get_job.return_value = _deployment("CANCELLED")
            screen._refresh_status()  # the status check still runs: CANCELLING is not the end
            assert await _wait(pilot, lambda: _text(screen, "#serving-status").startswith("ended"))
            await _settle(app, pilot)

    asyncio.run(run())


def test_the_live_tick_reads_serving_again():
    c = _client(_deployment(), [_drain(0.0, 3)])

    def serving_reads(pane):
        return sum(call.kwargs.get("sub_job_ids") == [DEPLOY] and call.args[1] == pane._root
                   for call in c.download_stdout_logs.call_args_list)

    async def run():
        app = CortexTrainingLogTUI(c, "8", poll_interval=60)  # no poll: only the tick below reads again
        async with app.run_test(size=(140, 40)) as pilot:
            screen = await _open(app, pilot)
            assert await _wait(pilot, lambda: "serving" in _visible_tabs(screen))
            await pilot.press("v")
            pane = screen.query_one("#serving-tab")
            assert await _wait(pilot, lambda: serving_reads(pane) == 1 and not pane._loading)
            assert screen._live()
            screen._auto_refresh_data()  # what the 30-second timer calls
            assert await _wait(pilot, lambda: serving_reads(pane) == 2)
            await _settle(app, pilot)

    asyncio.run(run())


def test_serving_reads_again_while_the_last_drains_may_still_upload():
    c = _client(_deployment(), [_drain(0.0, 3)])

    async def run():
        app = CortexTrainingLogTUI(c, "8", poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            screen = await _open(app, pilot)
            assert await _wait(pilot, lambda: "serving" in _visible_tabs(screen))
            await pilot.press("v")
            assert await _wait(pilot, lambda: _text(screen, "#serving-status").startswith("serving"))
            c.get_job.return_value = _deployment("CANCELLED")
            screen._fetch_status()
            assert await _wait(pilot, lambda: _text(screen, "#serving-status").startswith("ended"))
            pane = screen.query_one("#serving-tab")
            assert await _wait(pilot, lambda: not pane._loading)
            c.download_stdout_logs.side_effect = _console([_drain(0.0, 3), _drain(30.0, 6)])  # a chunk uploaded late
            screen._auto_refresh_data()  # the 30 s tick, minutes after the end
            assert await _wait(pilot, lambda: len([ln for ln in _text(screen, "#serving-table").splitlines()
                                                   if ":" in ln]) == 2)
            screen._active_seen_at = None
            screen._state_changed_at = "2020-01-01T00:00:00Z"  # long ended: nothing more can come
            pane.reload = MagicMock()
            screen._auto_refresh_data()
            assert not pane.reload.called
            await _settle(app, pilot)

    asyncio.run(run())


def test_an_answer_without_sub_jobs_keeps_a_deployment_s_tabs():
    c = _client(_deployment(), [_drain(0.0, 3)])

    async def run():
        app = CortexTrainingLogTUI(c, "8", poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            screen = await _open(app, pilot)
            assert await _wait(pilot, lambda: "serving" in _visible_tabs(screen))
            await pilot.press("v")
            c.get_job.return_value = {"job_id": "8", "status": "UNKNOWN", "reason_details": "no answer"}
            screen._fetch_status()
            assert await _wait(pilot, lambda: screen._job_status == "UNKNOWN")
            await pilot.pause(0.05)
            assert "serving" in _visible_tabs(screen) and "training" not in _visible_tabs(screen)
            assert screen._active_tab == "serving"
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_tiny_kv_cache_use_reads_as_under_a_tenth_of_a_percent():
    from cortex_training.tui.serving_view import _kv

    assert _kv(2.6e-05) == "<0.1%" and _kv(0.0) == "0.0%" and _kv(0.123) == "12.3%" and _kv(None) == "—"


def test_a_training_job_that_ends_never_reads_a_console_for_serving():
    job = {"job_id": "7", "status": "RUNNING", "sub_jobs": [{"sub_job_id": "7:training:0", "job_type": "training"}]}
    c = _client(job)

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            screen = await _open(app, pilot)
            pane = screen.query_one("#serving-tab")
            pane.reload = MagicMock(wraps=pane.reload)
            c.get_job.return_value = {**job, "status": "CANCELLED"}
            screen._fetch_status()
            assert await _wait(pilot, lambda: screen._job_status == "CANCELLED")
            await pilot.pause(0.1)
            assert not pane.opened and not pane.reload.called
            # The Logs tab saves the ended job's console; nothing downloads into Serving's directory.
            assert not [call for call in c.download_stdout_logs.call_args_list
                        if str(screen._serving_root) in map(str, call.args)], c.download_stdout_logs.call_args_list
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_reload_asked_for_while_one_runs_runs_once_after_it():
    import threading

    c = _client(_deployment(), [_drain(0.0, 3)])
    gate, calls = threading.Event(), []
    lines = [[_drain(0.0, 3)], [_drain(0.0, 3), _drain(30.0, 6)]]

    def download(job_id, output_dir, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            gate.wait(5)
        return _console(lines[min(len(calls), 2) - 1])(job_id, output_dir, **kwargs)

    c.download_stdout_logs.side_effect = download

    async def run():
        app = CortexTrainingLogTUI(c, "8", poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            screen = await _open(app, pilot)
            assert await _wait(pilot, lambda: "serving" in _visible_tabs(screen))
            await pilot.press("v")
            pane = screen.query_one("#serving-tab")
            assert await _wait(pilot, lambda: len(calls) == 1 and pane._loading)
            pane.reload()  # the 30 s refresh, say, while the first load runs
            gate.set()
            assert await _wait(pilot, lambda: len(calls) == 2 and not pane._loading)
            assert await _wait(pilot, lambda: len([ln for ln in _text(screen, "#serving-table").splitlines()
                                                   if ":" in ln]) == 2)
            await pilot.pause(0.2)
            assert len(calls) == 2  # once, not a loop
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_queued_reload_still_loads_after_the_load_before_it_ends():
    # The queued reload starts from the first load's _finish, which cancels that (exclusive) worker:
    # its cleanup must not mark the queued one as done, or the next tick starts a third and cancels it.
    import threading

    c = _client(_deployment(), [_drain(0.0, 3)])
    gates, calls = [threading.Event(), threading.Event()], []

    def download(job_id, output_dir, **kwargs):
        calls.append(1)
        gates[min(len(calls), 2) - 1].wait(5)
        return _console([_drain(0.0, 3), _drain(30.0, 6)])(job_id, output_dir, **kwargs)

    c.download_stdout_logs.side_effect = download

    async def run():
        app = CortexTrainingLogTUI(c, "8", poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            screen = await _open(app, pilot)
            assert await _wait(pilot, lambda: "serving" in _visible_tabs(screen))
            await pilot.press("v")
            pane = screen.query_one("#serving-tab")
            assert await _wait(pilot, lambda: len(calls) == 1 and pane._loading)
            pane.reload()
            gates[0].set()
            assert await _wait(pilot, lambda: len(calls) == 2)
            await pilot.pause(0.2)
            assert pane._loading and "loading…" in _text(screen, "#serving-status")
            pane.reload()  # r, while the queued load runs: queued again, not a third download now
            await pilot.pause(0.2)
            assert len(calls) == 2
            gates[1].set()
            assert await _wait(pilot, lambda: len(calls) == 3 and not pane._loading)
            assert await _wait(pilot, lambda: len([ln for ln in _text(screen, "#serving-table").splitlines()
                                                   if ":" in ln]) == 2)
            await _settle(app, pilot)

    asyncio.run(run())


def test_two_reads_at_once_take_turns(tmp_path, monkeypatch):
    import threading

    path = tmp_path / "stdout.log"
    path.write_text("".join(_drain(float(t), 1) + "\n" for t in range(50)))
    reader = serving.ConsoleReader(path)
    real, inside, overlapped = serving.parse_line, [0], []

    def slow_parse(line):
        inside[0] += 1
        overlapped.append(inside[0] > 1)
        threading.Event().wait(0.001)
        inside[0] -= 1
        return real(line)

    monkeypatch.setattr(serving, "parse_line", slow_parse)
    threads = [threading.Thread(target=reader.read) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not any(overlapped)
    assert [d.drained_at for d in reader.read().drains] == [float(t) for t in range(50)]


def test_logs_and_serving_read_a_drain_s_numbers_alike():
    from cortex_training.tui.format import format_log_entry

    line = _drain(100.0, "16", latency_s_p50="0.05", ttft_s_p99="0.215", replica_count="2", replica_errors="1")
    d = serving.parse_line(line)
    assert (d.completed, d.latency_p50, d.ttft_p99, d.replicas, d.replica_errors) == (16, 0.05, 0.215, 2, 1)
    assert format_log_entry(json.loads(line)).endswith(
        "drain_metrics  16 req · p50 50ms · ttft p99 215ms · 2 replicas · 1 replica error")


def test_one_number_helper_refuses_what_is_not_a_finite_number():
    from cortex_training.tui import format as fmt

    assert fmt.finite_number(2) == 2.0 and fmt.finite_number("0.5") == 0.5
    for bad in (None, True, False, float("nan"), float("inf"), "NaN", "-Infinity", "1e400", 10 ** 400, "", "x", [1]):
        assert fmt.finite_number(bad) is None, bad
