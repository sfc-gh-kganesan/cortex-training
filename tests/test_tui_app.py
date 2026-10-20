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

"""Smoke tests for the Textual app (job picker -> log screen). Skipped unless
the optional ``textual`` extra is installed (``pip install 'cortex-training[tui]'``)."""

import asyncio
import glob
import inspect
import json
import os
import re
import threading
import time
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("textual")

from textual.widgets import Input  # noqa: E402
from textual.widgets import DataTable  # noqa: E402
from textual.widgets import Label  # noqa: E402
from textual.widgets import Static  # noqa: E402
from textual.widgets import TabbedContent  # noqa: E402

from cortex_training import AccountJobListUnavailable  # noqa: E402
from cortex_training import CortexTrainingClient  # noqa: E402
from cortex_training.client import DownloadCancelled  # noqa: E402
from cortex_training.tui import __main__ as tui_main  # noqa: E402
from cortex_training.tui.app import CortexTrainingLogTUI  # noqa: E402
from cortex_training.tui.job_list import JobListScreen  # noqa: E402
from cortex_training.tui.logs import LogScreen  # noqa: E402
from cortex_training.tui.demo import DemoClient  # noqa: E402
from cortex_training.tui import views  # noqa: E402
from cortex_training.tui.untrusted import printable_client  # noqa: E402
from cortex_training.tui.format import wrap_log_line  # noqa: E402
from cortex_training.tui.resource import RANGES  # noqa: E402
from cortex_training.tui.screens import JobSummaryScreen  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    # The saved console and caches live under the per-job cache dir; keep every
    # test (including those that do not set it themselves) out of ~/.cache.
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "isolated-cache"))
    import cortex_training.tui.widgets as app_module

    # ... and out of ~/.cortex-training-errors.log: a test that reads the log finds it in tmp_path.
    monkeypatch.setattr(app_module, "_ERROR_LOG", str(tmp_path / "errors.log"))


def _client():
    c = MagicMock()
    # The account-wide list isn't deployed (what accounts get today), so the job
    # list shows list_jobs, the configured schema's jobs.
    c.list_jobs_page.side_effect = AccountJobListUnavailable("not deployed", reason="not_deployed")
    c.list_jobs.return_value = [
        {"job_id": "7", "status": "RUNNING", "sub_jobs": [{"job_type": "training"}]},
        {"job_id": "old", "status": "FAILED"},
    ]
    # Sources come from get_job().sub_jobs now (one per sub-job). RUNNING keeps
    # the live tail path; terminal-job behavior is covered by overriding get_job.
    c.get_job.return_value = {
        "status": "RUNNING",
        "sub_jobs": [
            {"sub_job_id": "7:training:0", "job_type": "training"},
            {"sub_job_id": "7:sampling:0", "job_type": "sampling"},
        ],
    }
    # Finite results so the tail worker doesn't loop forever.
    c.tail_logs.return_value = {"entries": [], "next_cursor": "", "eof": True}
    return c


async def _wait(pilot, app, predicate, tries=100):
    for _ in range(tries):
        await pilot.pause()
        if predicate():
            return True
    return False


async def _settle(app, pilot):
    """Stop background workers and drain before ``run_test`` tears down, so a
    thread worker (ours, or Textual's internal Log resize worker) can't call into
    a half-destroyed app and surface a NoActiveAppError as a WorkerError. Without
    this the live-tail/resize workers make these tests flaky."""
    try:
        app.workers.cancel_all()
    except Exception:  # noqa: BLE001
        pass
    for _ in range(5):
        await pilot.pause()


async def _run_with_job_id():
    app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot, app, lambda: isinstance(app.screen, LogScreen) and len(app.screen.query("#sources ListItem")) == 2
        )
        assert ok, "log screen / sources did not load"
        screen = app.screen
        assert screen._source_by_item["src-0"] == "7:training:0"
        assert screen._source_by_item["src-1"] == "7:sampling:0"
        await _settle(app, pilot)


async def _run_with_picker():
    app = CortexTrainingLogTUI(_client(), poll_interval=0.01)  # no job_id -> picker
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot, app, lambda: isinstance(app.screen, JobListScreen) and len(app.screen._job_by_item) >= 2
        )
        assert ok, "job picker did not load jobs"
        screen = app.screen
        # active job (RUNNING) sorted first
        assert screen.whole_id("job-0") == "7"
        assert screen.whole_id("job-1") == "old"
        assert re.search(r"last refreshed \d{2}:\d{2}:\d{2}", app.sub_title or "")
        await _settle(app, pilot)


def test_app_with_job_id_opens_logs():
    asyncio.run(_run_with_job_id())


def test_app_without_job_id_shows_job_picker():
    asyncio.run(_run_with_picker())


def test_account_list_keeps_equal_ids_from_two_schemas():
    async def run():
        c = _client()
        c.database, c.schema = "DB", "HOME"
        c.list_jobs_page.side_effect = None
        c.list_jobs_page.return_value = {
            "jobs": [
                {"job_id": "same", "status": "FAILED", "database_name": "DB", "schema_name": "ONE"},
                {"job_id": "same", "status": "FAILED", "database_name": "DB", "schema_name": "TWO"},
            ], "next_page_token": None}
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        app._lister.account_wide = True
        async with app.run_test() as pilot:
            assert await _wait(pilot, app, lambda: len(getattr(app.screen, "_job_by_item", {})) == 2)
            screen = app.screen
            assert [screen.whole_id(key) for key in screen._job_by_item] == ["same", "same"]
            assert {job["schema_name"] for job in screen._jobs_by_key.values()} == {"ONE", "TWO"}
            await _settle(app, pilot)

    asyncio.run(run())


def test_location_scopes_cache_identity_and_unknown_not_found_is_historical():
    from cortex_training.tui.widgets import _cache_job_id

    class Client:
        base_url = "https://ACCOUNT.snowflakecomputing.com"
        endpoint = "cortex-training"
        database = "DB"

        def __init__(self, schema):
            self.schema = schema

    assert _cache_job_id(Client("ONE"), "same") != _cache_job_id(Client("TWO"), "same")
    job = {"job_id": "missing", "status": "UNKNOWN",
           "reason_details": "the job was not found in the Control Plane"}
    screen = LogScreen(_client(), "missing", job=job)
    assert not screen._live()
    screen._job_created_at = "created"
    screen._console_dir.mkdir(parents=True)
    (screen._console_dir / ".complete").write_text("created")
    assert screen._kept_console()
    entry = {"_raw": "step 1 loss\rstep 2 done"}
    assert screen._newer_tail_lines([(entry, entry["_raw"])], [], False, "loss", None)[0] == []


def test_copied_job_commands_keep_a_custom_endpoint():
    from cortex_training.tui.views import cli_options

    c = _client()
    c.database, c.schema, c.endpoint, c._auth_provider = "DB", "SCHEMA", "custom-endpoint", None
    command = " ".join(cli_options(**LogScreen(c, "7")._configured_location()))
    assert "--database DB" in command and "--schema SCHEMA" in command
    assert "--endpoint custom-endpoint" in command


def _listed_units(screen) -> int:
    """The GPUs (or nodes) the Resources panel lists, without ALL."""
    return sum(1 for entry in screen._resource_entries if entry[0] == "unit" and entry[1] is not None)


async def _run_resource_tab(tmp_path):
    c = _client()

    def download(job_id, output_dir, *, resume=False, **_kwargs):
        assert job_id == "7"
        assert resume is True
        _write_gpu_metrics(output_dir, "7:training:0", gpu_count=2)
        _write_gpu_metrics(output_dir, "7:sampling:0", gpu_count=1)
        return []

    c.download_metrics.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(130, 36)) as pilot:  # the header beside the panel fits it whole
        ok = await _wait(
            pilot,
            app,
            lambda: isinstance(app.screen, LogScreen) and app.screen._current_source == "7:training:0",
        )
        assert ok
        tabs = app.screen.query_one("#job-tabs", TabbedContent)
        assert tabs.active == "logs"
        assert {pane.id for pane in app.screen.query("TabPane")} == {"logs", "resource"}
        c.download_metrics.assert_not_called()

        tabs.active = "resource"
        ok = await _wait(
            pilot,
            app,
            lambda: c.download_metrics.call_count == 1
            and _listed_units(app.screen) == 2
            and not app.screen._resource_loading,
        )
        assert ok, "Resource did not load two GPU rows"
        status = str(app.screen.query_one("#resource-status", Static).render())
        assert "2 reporting" in status
        assert "training #0" in status  # named by type and index
        calls = c.download_metrics.call_count

        # w cycles the GPU window; the digits do nothing on the job detail screen.
        for _ in range(len(RANGES)):
            if app.screen._resource_range == "all":
                break
            await pilot.press("w")
        ok = await _wait(
            pilot,
            app,
            lambda: app.screen._resource_range == "all"
            # Named by type and index; at this width the window may show as its key (all).
            and re.match(r"training #0 (\(1/2\)|\(1 of 2\))  ·  (whole job|all)  ·",
                         str(app.screen.query_one("#resource-status", Static).render())),
        )
        assert ok
        await pilot.press("w")
        ok = await _wait(pilot, app, lambda: app.screen._resource_range == "10m")
        assert ok
        assert c.download_metrics.call_count == calls
        app.screen._content_width = lambda: 10
        app.screen._write_line("x" * 20)
        assert len(app.screen._logview._lines) > len(app.screen._shown_lines)
        app.screen.action_refresh_sources()
        ok = await _wait(
            pilot,
            app,
            lambda: c.download_metrics.call_count == calls + 1 and not app.screen._resource_loading,
        )
        assert ok, "r did not refresh metrics on Resource"
        tabs.active = "logs"
        app.screen._content_width = lambda: 200
        ok = await _wait(
            pilot,
            app,
            lambda: app.screen._active_tab == "logs" and app.screen._wrap_width == 200,
        )
        assert ok
        assert len(app.screen._logview._lines) == len(app.screen._shown_lines)
        downloads = c.download_metrics.call_count
        app.screen.action_refresh_sources()
        await pilot.pause()
        assert c.download_metrics.call_count == downloads
        app.screen._start_tail("7:sampling:0")
        tabs.active = "resource"
        ok = await _wait(
            pilot,
            app,
            lambda: _listed_units(app.screen) == 1
            and "sampling #0" in str(app.screen.query_one("#resource-status", Static).render()),
        )
        assert ok, "Resource did not follow the Logs-selected sub-job"
        assert c.download_metrics.call_count == downloads
        tabs.active = "logs"
        await _wait(pilot, app, lambda: app.screen._active_tab == "logs")
        previous_range = app.screen._resource_range
        app.screen.action_resource_range("all")
        assert app.screen._resource_range == previous_range
        await pilot.press("t")
        ok = await _wait(pilot, app, lambda: tabs.active == "resource")
        assert ok
        await pilot.press("t")
        ok = await _wait(pilot, app, lambda: tabs.active == "logs")
        assert ok
        await _settle(app, pilot)


def test_resource_tab_is_lazy_and_renders_multiple_gpus(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    asyncio.run(_run_resource_tab(tmp_path))


async def _run_resource_cold_cache_failure():
    c = _client()
    c.download_metrics.side_effect = RuntimeError("metrics [/]")
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(100, 36)) as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._current_source is not None)
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        ok = await _wait(
            pilot,
            app,
            lambda: c.download_metrics.called and not app.screen._resource_loading,
        )
        assert ok
        status = str(app.screen.query_one("#resource-status", Static).render())
        assert "GPU metrics unavailable: RuntimeError: metrics [/]" in status
        assert _listed_units(app.screen) == 0
        await _settle(app, pilot)


def test_resource_cold_cache_download_failure_is_visible(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    asyncio.run(_run_resource_cold_cache_failure())


async def _run_resource_parser_failure_resets_loading():
    c = _client()
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(100, 36)) as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._current_source is not None)
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        ok = await _wait(
            pilot,
            app,
            lambda: not app.screen._resource_loading
            and "OverflowError: bad metric"
            in str(app.screen.query_one("#resource-status", Static).render()),
        )
        assert ok
        status = str(app.screen.query_one("#resource-status", Static).render())
        assert "GPU metrics unavailable: OverflowError: bad metric" in status
        await _settle(app, pilot)


def test_resource_parser_failure_does_not_leave_loading_stuck(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    import cortex_training.tui.resource_view as app_module

    loader = MagicMock()
    loader.last = None  # nothing read yet
    loader.load.side_effect = OverflowError("bad metric")
    monkeypatch.setattr(app_module, "resource_loader", lambda root: loader)
    asyncio.run(_run_resource_parser_failure_resets_loading())


def _assert_header_fits(screen):
    """The Resources header is its two lines, neither wider than the tab: nothing under it moves."""
    from rich.cells import cell_len

    status = screen.query_one("#resource-status", Static)
    lines = str(status.render()).split("\n")
    assert len(lines) == 2, lines
    assert all(cell_len(line) <= status.content_size.width for line in lines), (status.content_size.width, lines)


async def _run_resource_partial_download_failure():
    c = _client()

    def download(job_id, output_dir, *, resume, **_kwargs):
        _write_gpu_metrics(output_dir, "7:training:0", gpu_count=2)
        raise RuntimeError("later sub-job failed")

    c.download_metrics.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(130, 36)) as pilot:  # the header beside the panel fits it whole
        ok = await _wait(pilot, app, lambda: app.screen._current_source is not None)
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        ok = await _wait(
            pilot,
            app,
            lambda: not app.screen._resource_loading
            and _listed_units(app.screen) == 2,
        )
        assert ok
        status = str(app.screen.query_one("#resource-status", Static).render())
        # The line makes room for the error (100 columns here: "cached · " gives way), never wrapped.
        assert "refresh failed: RuntimeError: later sub-job failed" in status
        _assert_header_fits(app.screen)
        await _settle(app, pilot)


def test_resource_partial_download_is_reparsed_after_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    asyncio.run(_run_resource_partial_download_failure())


async def _run_resource_partial_other_sub_job_retries():
    c = _client()
    calls = 0

    def download(job_id, output_dir, *, resume, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            _write_gpu_metrics(output_dir, "7:sampling:0", gpu_count=1)
            raise RuntimeError("training metrics failed")
        _write_gpu_metrics(output_dir, "7:training:0", gpu_count=2)
        return []

    c.download_metrics.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(100, 36)) as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._current_source == "7:training:0")
        assert ok
        tabs = app.screen.query_one("#job-tabs", TabbedContent)
        tabs.active = "resource"
        ok = await _wait(pilot, app, lambda: calls == 1 and not app.screen._resource_loading)
        assert ok
        status = str(app.screen.query_one("#resource-status", Static).render())
        assert "GPU metrics unavailable: RuntimeError: training metrics failed" in status
        assert "no GPU metrics" not in status
        tabs.active = "logs"
        await _wait(pilot, app, lambda: app.screen._logs_active())
        tabs.active = "resource"
        ok = await _wait(
            pilot,
            app,
            lambda: calls == 2
            and not app.screen._resource_loading
            and _listed_units(app.screen) == 2,
        )
        assert ok, "re-entering Resource did not retry after a partial failure for another sub-job"
        await _settle(app, pilot)


def test_resource_partial_other_sub_job_failure_retries_on_reentry(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    asyncio.run(_run_resource_partial_other_sub_job_retries())


async def _run_resource_cached_failure():
    c = _client()
    c.download_metrics.side_effect = RuntimeError("refresh unavailable")
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(130, 36)) as pilot:  # the header beside the panel fits it whole
        ok = await _wait(pilot, app, lambda: app.screen._current_source is not None)
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        ok = await _wait(
            pilot,
            app,
            lambda: not app.screen._resource_loading
            and _listed_units(app.screen) == 2,
        )
        assert ok
        status = str(app.screen.query_one("#resource-status", Static).render())
        assert "refresh failed: RuntimeError: refresh unavailable" in status  # on the line, not wrapped
        _assert_header_fits(app.screen)
        app.screen.action_resource_range("all")
        status = str(app.screen.query_one("#resource-status", Static).render())
        assert "refresh failed: RuntimeError: refresh unavailable" in status
        _assert_header_fits(app.screen)
        assert _listed_units(app.screen) == 2
        await _settle(app, pilot)


def test_resource_failed_refresh_keeps_cached_rows_and_error(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    from cortex_training.tui.log_cache import job_cache_dir

    _write_gpu_metrics(job_cache_dir("7") / "metrics", "7:training:0", gpu_count=2)
    asyncio.run(_run_resource_cached_failure())


async def _run_resource_unsupported_warning():
    c = _client()

    def download(job_id, output_dir, *, resume, **_kwargs):
        path = _write_gpu_metrics(output_dir, "7:training:0", gpu_count=1)
        row = json.loads(path.read_text(encoding="utf-8"))
        row["record_version"] = 1
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        path.with_name(".gpu.jsonl.manifest.json").write_text(
            json.dumps({"version": 1, "names": ["chunk.gz"], "bytes": path.stat().st_size}),
            encoding="utf-8",
        )
        return []

    c.download_metrics.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(100, 36)) as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._current_source is not None)
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        ok = await _wait(pilot, app, lambda: not app.screen._resource_loading)
        assert ok
        status = str(app.screen.query_one("#resource-status", Static).render())
        assert "training #0" in status  # named by type and index
        assert "no usable GPU metrics" in status
        assert "1 row skipped (malformed or unsupported)" in status
        await _settle(app, pilot)


def test_resource_all_unsupported_rows_show_warning(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    asyncio.run(_run_resource_unsupported_warning())


async def _run_resource_narrow_no_samples():
    c = _client()

    def download(job_id, output_dir, *, resume, **_kwargs):
        path = _write_gpu_metrics(output_dir, "7:training:0", gpu_count=1)
        row = json.loads(path.read_text(encoding="utf-8"))
        row["gpu_util_pct"] = None
        payload = json.dumps(row) + "\n"
        path.write_text(payload, encoding="utf-8")
        path.with_name(".gpu.jsonl.manifest.json").write_text(
            json.dumps({"version": 1, "names": ["chunk.gz"], "bytes": path.stat().st_size}),
            encoding="utf-8",
        )
        return []

    c.download_metrics.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(60, 30)) as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._current_source is not None)
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        ok = await _wait(
            pilot,
            app,
            lambda: not app.screen._resource_loading
            and _listed_units(app.screen) == 1,
        )
        assert ok
        # A metric without samples has no chart and no stats row: the memory % still charts.
        charts = str(app.screen.query_one("#resource-charts", Static).render())
        assert "GPU util %" not in charts and "GPU memory used %" in charts
        width = app.screen.query_one("#resource-charts", Static).content_size.width
        assert all(len(line) <= width for line in charts.splitlines())
        calls = c.download_metrics.call_count
        app.screen.on_resize(None)
        await pilot.pause()
        assert c.download_metrics.call_count == calls
        await _settle(app, pilot)


def test_resource_narrow_table_no_samples_and_resize_is_local(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    asyncio.run(_run_resource_narrow_no_samples())


async def _run_resource_chart_fits_content_width():
    c = _client()

    def download(job_id, output_dir, *, resume, **_kwargs):
        _write_gpu_metrics(output_dir, "7:training:0", gpu_count=1)
        return []

    c.download_metrics.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(60, 30)) as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._current_source is not None)
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        ok = await _wait(
            pilot,
            app,
            lambda: not app.screen._resource_loading
            and _listed_units(app.screen) == 1,
        )
        assert ok
        chart = app.screen.query_one("#resource-charts", Static)
        lines = str(chart.render()).splitlines()
        assert lines
        # The pane's padding is outside the charts: they are drawn to the width they have.
        pane = app.screen.query_one("#resource-pane")
        assert pane.region.width - chart.content_region.width == 3  # its padding and the scrollbar's gutter
        assert chart.content_size.width == chart.content_region.width
        assert all(len(line) <= chart.content_size.width for line in lines)
        await _settle(app, pilot)


def test_resource_chart_accounts_for_widget_padding(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    asyncio.run(_run_resource_chart_fits_content_width())


async def _run_resource_ignores_refresh_while_busy(started, release):
    c = _client()

    def download(job_id, output_dir, *, resume, **_kwargs):
        started.set()
        assert release.wait(timeout=5)
        _write_gpu_metrics(output_dir, "7:training:0")
        return []

    c.download_metrics.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(100, 36)) as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._current_source is not None)
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        ok = await _wait(pilot, app, started.is_set)
        assert ok
        app.screen.action_refresh_sources()
        app.screen.action_refresh_sources()
        await pilot.pause()
        assert c.download_metrics.call_count == 1
        release.set()
        ok = await _wait(pilot, app, lambda: not app.screen._resource_loading)
        assert ok
        await _settle(app, pilot)


def test_resource_ignores_refresh_while_busy(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    started = threading.Event()
    release = threading.Event()
    try:
        asyncio.run(_run_resource_ignores_refresh_while_busy(started, release))
    finally:
        release.set()


async def _run_resource_reopen_waits_for_lock(started, release):
    c = _client()
    calls = 0

    def download(job_id, output_dir, *, resume, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            assert release.wait(timeout=5)
        _write_gpu_metrics(output_dir, "7:training:0")
        return []

    c.download_metrics.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(100, 36)) as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._current_source is not None)
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        ok = await _wait(pilot, app, started.is_set)
        assert ok
        app.pop_screen()
        await pilot.pause()
        app.push_screen(LogScreen(c, "7", poll_interval=0.01))
        ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen) and app.screen._current_source is not None)
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        for _ in range(5):
            await pilot.pause()
        assert calls == 1
        release.set()
        ok = await _wait(
            pilot,
            app,
            lambda: calls == 2
            and not app.screen._resource_loading
            and _listed_units(app.screen) == 2,
        )
        assert ok, "reopened Resource did not continue after the prior download released the lock"
        await _settle(app, pilot)


def test_resource_reopen_waits_for_inflight_download(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    started = threading.Event()
    release = threading.Event()
    try:
        asyncio.run(_run_resource_reopen_waits_for_lock(started, release))
    finally:
        release.set()


async def _run_terminal_resource_default():
    c = _client()
    c.get_job.return_value = _terminal_job()
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(100, 36)) as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._sources_ready)
        assert ok
        assert app.screen._resource_range == "10m"  # every job opens on the last 10 minutes
        await _settle(app, pilot)


def test_direct_terminal_job_resource_defaults_to_ten_minutes():
    asyncio.run(_run_terminal_resource_default())


async def _run_active_to_terminal_resource_default():
    app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
    async with app.run_test(size=(100, 36)) as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._sources_ready)
        assert ok
        assert app.screen._resource_range == "10m"
        # The window never moves under the user when the job ends.
        app.screen._apply_terminal_status("COMPLETED", None)
        assert app.screen._resource_range == "10m"
        app.screen._resource_range = "30m"
        app.screen._apply_terminal_status("FAILED", None)
        assert app.screen._resource_range == "30m"
        await _settle(app, pilot)


def test_active_job_becoming_terminal_keeps_the_resource_window():
    asyncio.run(_run_active_to_terminal_resource_default())


async def _run_resource_escape_returns_to_parent(*, direct):
    c = _client()
    app = CortexTrainingLogTUI(c, "7" if direct else None, poll_interval=0.01)
    async with app.run_test(size=(100, 36)) as pilot:
        if direct:
            ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen))
        else:
            ok = await _wait(pilot, app, lambda: isinstance(app.screen, JobListScreen))
            assert ok
            app.push_screen(LogScreen(c, "7", poll_interval=0.01))
            ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen))
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        ok = await _wait(pilot, app, lambda: app.screen._resource_active())
        assert ok
        await pilot.press("escape")
        ok = await _wait(
            pilot,
            app,
            lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item,
        )
        assert ok
        if not direct:
            assert isinstance(app.screen, JobListScreen)
        await _settle(app, pilot)


def test_resource_escape_returns_from_direct_job():
    asyncio.run(_run_resource_escape_returns_to_parent(direct=True))


def test_resource_escape_returns_to_picker():
    asyncio.run(_run_resource_escape_returns_to_parent(direct=False))


async def _run_resource_guards(tmp_path):
    c = _client()
    c.download_metrics.side_effect = lambda job_id, output_dir, *, resume, **_options: (
        _write_gpu_metrics(output_dir, "7:training:0"),
        [],
    )[1]
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(100, 36)) as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._current_source is not None)
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        ok = await _wait(pilot, app, lambda: not app.screen._resource_loading)
        assert ok
        screen = app.screen
        table = screen.query_one("#resource-list")
        screen.set_focus(table)
        width = screen._sources_width
        level = screen._min_level
        paused = screen._paused
        copied = []
        app.copy_to_clipboard = copied.append
        screen.action_filter()
        screen.action_cycle_level()
        screen.action_toggle_pause()
        screen.action_shrink_sources()
        screen.action_grow_sources()
        screen.action_save_log()
        for _ in range(5):
            await pilot.pause()
        assert screen.focused is table
        assert screen._sources_width == width
        assert screen._min_level == level
        assert screen._paused == paused
        assert copied == []
        assert not list(tmp_path.glob("cortex-training-*.log"))
        await _settle(app, pilot)


def test_log_only_actions_are_guarded_on_resource(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("HOME", str(tmp_path))
    asyncio.run(_run_resource_guards(tmp_path))


async def _run_sub_job_scope(sub_job_id):
    c = _client()
    app = CortexTrainingLogTUI(c, "7", sub_job_id=sub_job_id, poll_interval=0.01)
    async with app.run_test(size=(100, 36)) as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._sources_ready)
        assert ok
        if sub_job_id == "7:sampling:0":
            assert list(app.screen._source_by_item.values()) == ["7:sampling:0"]
            assert app.screen._current_source == "7:sampling:0"
        else:
            assert app.screen._source_by_item == {}
            assert app.screen._current_source is None
            app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
            for _ in range(5):
                await pilot.pause()
            c.download_metrics.assert_not_called()
            labels = [str(item.query_one(Label).render()) for item in app.screen.query("#sources ListItem")]
            assert any("sub-job not found" in label for label in labels)
        await _settle(app, pilot)


def test_sub_job_id_scopes_source_list():
    asyncio.run(_run_sub_job_scope("7:sampling:0"))


def test_unmatched_sub_job_does_not_download_metrics():
    asyncio.run(_run_sub_job_scope("7:missing:0"))


async def _run_refresh_picker():
    c = _client()
    jobs = c.list_jobs.return_value
    refresh_started = threading.Event()
    finish_refresh = threading.Event()
    calls = 0

    def list_jobs(**_bounds):
        nonlocal calls
        calls += 1
        if calls > 1:
            refresh_started.set()
            finish_refresh.wait(timeout=5)
        return jobs

    c.list_jobs.side_effect = list_jobs
    app = CortexTrainingLogTUI(c, poll_interval=0.01)
    try:
        async with app.run_test() as pilot:
            ok = await _wait(
                pilot,
                app,
                lambda: isinstance(app.screen, JobListScreen)
                and len(app.screen._job_by_item) == 2,
            )
            assert ok, "initial jobs did not load"
            previous_refresh = app.screen._last_refreshed
            # Refresh re-populates with the same item IDs (job-0, job-1). Must
            # show that work is underway without discarding the last success.
            app.screen.action_refresh()
            ok = await _wait(pilot, app, refresh_started.is_set)
            assert ok, "refresh did not start"
            assert "refreshing…" in (app.sub_title or "")
            assert f"last refreshed {previous_refresh}" in (app.sub_title or "")
            finish_refresh.set()
            ok = await _wait(
                pilot,
                app,
                lambda: "refreshing…" not in (app.sub_title or "")
                and len(app.screen._job_by_item) == 2,
            )
            assert ok, "refresh did not finish cleanly"
            await _settle(app, pilot)
    finally:
        finish_refresh.set()


async def _run_refresh_sources():
    app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot, app, lambda: isinstance(app.screen, LogScreen) and len(app.screen.query("#sources ListItem")) == 2
        )
        assert ok, "initial sources did not load"
        app.screen.action_refresh_sources()
        ok = await _wait(pilot, app, lambda: len(app.screen.query("#sources ListItem")) == 2)
        assert ok, "source refresh changed item count (DuplicateIds regression)"
        await _settle(app, pilot)


def test_picker_refresh_no_duplicate_ids():
    asyncio.run(_run_refresh_picker())


def test_source_refresh_no_duplicate_ids():
    asyncio.run(_run_refresh_sources())


async def _run_empty_source_refresh_clears_selection():
    c = _client()
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._current_source is not None)
        assert ok
        # The service now says the job has no sub-jobs (an empty answer would say
        # nothing, and keep the sub-jobs the view knows).
        c.get_job.return_value = {"job_id": "7", "status": "RUNNING", "sub_jobs": []}
        app.screen.action_refresh_sources()
        ok = await _wait(
            pilot,
            app,
            lambda: app.screen._current_source is None
            and len(app.screen.query("#sources ListItem")) == 0,
        )
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        await pilot.pause()
        c.download_metrics.assert_not_called()
        status = str(app.screen.query_one("#resource-status", Static).render())
        assert "this job has no sub-jobs" in status
        await _settle(app, pilot)


def test_empty_source_refresh_clears_resource_selection(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    asyncio.run(_run_empty_source_refresh_clears_selection())


async def _run_offline_fallback():
    c = _client()
    c.get_job.side_effect = RuntimeError("network down")  # can't fetch sub-jobs
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot,
            app,
            lambda: isinstance(app.screen, LogScreen)
            and "7:training:0" in (app.screen._source_by_item or {}).values(),
        )
        assert ok, "cached sub-job not shown after get_job failure"
        await _settle(app, pilot)


def test_offline_shows_cached_sources(tmp_path, monkeypatch):
    # Seed a cached sub-job for job "7", then make get_job fail: the source list
    # must fall back to the cached sub-job so logs stay reachable.
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    from cortex_training.tui.log_cache import LogCache

    LogCache("7").append_entries("7:training:0", [{"_raw": "cached line"}])
    asyncio.run(_run_offline_fallback())


def test_offline_shows_metrics_only_cached_sources(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    from cortex_training.tui.log_cache import job_cache_dir

    _write_gpu_metrics(job_cache_dir("7") / "metrics", "7:training:0", gpu_count=2)
    asyncio.run(_run_offline_fallback())


async def _run_terminal_cache_only():
    c = _client()
    c.get_job.return_value = {  # terminal -> zone gone; metadata still lists sub-jobs
        "status": "FAILED",
        "sub_jobs": [{"sub_job_id": "7:training:0", "job_type": "training"}],
    }
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot,
            app,
            lambda: isinstance(app.screen, LogScreen)
            and "7:training:0" in (app.screen._source_by_item or {}).values(),
        )
        assert ok, "sub-job source not shown for terminal job"
        # A terminal job is served from cache only — no tail-logs operation call.
        c.tail_logs.assert_not_called()
        await _settle(app, pilot)


def test_terminal_job_is_cache_only(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    from cortex_training.tui.log_cache import LogCache

    LogCache("7").append_entries("7:training:0", [{"_raw": "cached line"}])
    asyncio.run(_run_terminal_cache_only())


async def _run_resize_sources():
    app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen))
        assert ok, "log screen did not open"
        s = app.screen
        w0 = s._sources_width
        s.action_grow_sources()
        assert s._sources_width == w0 + s._SOURCES_STEP
        s.action_shrink_sources()
        s.action_shrink_sources()
        assert s._sources_width == max(s._SOURCES_MIN, w0 - s._SOURCES_STEP)
        for _ in range(50):
            s.action_shrink_sources()
        assert s._sources_width == s._SOURCES_MIN  # clamps low
        for _ in range(50):
            s.action_grow_sources()
        # Clamps high: the log keeps its minimum width (an 80-column test screen).
        assert s._sources_width == s._sources_cap() == app.screen.size.width - s._LOG_MIN_WIDTH
        await _settle(app, pilot)


def test_sources_panel_resize():
    asyncio.run(_run_resize_sources())


async def _run_picker_filter():
    app = CortexTrainingLogTUI(_client(), poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot, app, lambda: isinstance(app.screen, JobListScreen) and len(app.screen._job_by_item) >= 2
        )
        assert ok, "jobs did not load"
        s = app.screen
        s.query_one("#jobfilter", Input).value = "old"  # fires Input.Changed → re-render
        ok = await _wait(
            pilot,
            app,
            lambda: [s.whole_id(key) for key in s._job_by_item] == ["old"],
        )
        assert ok, f"filter did not narrow the list: {s._job_by_item}"
        await _settle(app, pilot)


def test_picker_filter():
    asyncio.run(_run_picker_filter())


def test_a_search_builds_each_hidden_older_job_s_search_text_once(monkeypatch):
    # A search that matches no job of the last six months names the older jobs it
    # matches. Their search text is kept as the shown jobs' is, not built again on
    # every keystroke.
    from collections import Counter

    from cortex_training.tui import views

    now = datetime(2026, 10, 1, tzinfo=timezone.utc)
    c = _client()
    c.list_jobs.return_value = [{"job_id": "7", "status": "RUNNING", "created_at": now.isoformat()}] + [
        {"job_id": f"old-{i}", "status": "FAILED",
         "created_at": (now - timedelta(days=400 + i)).isoformat()}
        for i in range(3)
    ]
    built = Counter()
    search_haystack = views.search_haystack

    def counted(job):
        built[views.job_id_of(job)] += 1
        return search_haystack(job)

    monkeypatch.setattr(views, "search_haystack", counted)

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01, now=now)
        async with app.run_test(size=(140, 40)) as pilot:
            ok = await _wait(pilot, app, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
            assert ok, "jobs did not load"
            screen = app.screen
            for text in ("o", "ol", "old", "old-"):
                screen.query_one("#jobfilter", Input).value = text
                ok = await _wait(pilot, app, lambda: f"“{text}”" in (screen._nomatch or ""))
                assert ok, f"no message for {text!r}: {screen._nomatch!r}"
            assert {jid: built[jid] for jid in ("old-0", "old-1", "old-2")} == {"old-0": 1, "old-1": 1, "old-2": 1}
            await _settle(app, pilot)

    asyncio.run(run())


def test_job_table_replace_rows_shows_exactly_those_rows():
    # replace_rows fills the table through DataTable's own clear() and add_row():
    # the rows, their keys and the cursor must come out as the caller asked.
    from textual.app import App

    from cortex_training.tui.widgets import _cell
    from cortex_training.tui.job_list import _JobTable

    class Tables(App):
        CSS = "DataTable { height: 8; }"

        def compose(self):
            yield _JobTable(id="fast", cursor_type="row", fixed_columns=1)
            yield _JobTable(id="own", cursor_type="row", fixed_columns=1)

    def rows(count, tag):
        return [(f"{tag}-{i}", (_cell(f"{tag} {i}"), _cell(f"{i * 7}"))) for i in range(count)]

    def state(table):
        return (table.row_count, [row.key.value for row in table.ordered_rows],
                [[str(cell) for cell in table.get_row_at(i)] for i in range(table.row_count)],
                table.cursor_coordinate, table.virtual_size,
                [table.render_line(y).text for y in range(table.size.height)])

    async def run():
        app = Tables()
        async with app.run_test(size=(40, 20)) as pilot:
            fast, own = app.query_one("#fast", _JobTable), app.query_one("#own", _JobTable)
            for table in (fast, own):
                table.add_column("Job ID", width=8, key="Job ID")
                table.add_column("N", width=4, key="N")
            # Two tables filled the same way hold the same rows, however often refilled.
            for plan in (rows(30, "a"), rows(5, "b"), [], rows(12, "c")):
                fast.replace_rows(plan)
                own.replace_rows(plan)
                await pilot.pause()
                assert state(fast) == state(own)
                assert [row.key.value for row in fast.ordered_rows] == [key for key, _cells in plan]
                assert fast.row_count == len(plan)
            fast.move_cursor(row=9)
            own.move_cursor(row=9)
            fast.replace_rows(rows(20, "d"))
            own.replace_rows(rows(20, "d"))
            await pilot.pause()
            assert state(fast) == state(own)
            assert [str(cell) for cell in fast.get_row_at(0)] == ["d 0", "0"]

    asyncio.run(run())


async def _run_logscreen_controls():
    app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen) and app.screen._logview is not None)
        assert ok, "log screen did not open"
        s = app.screen
        # Following is managed manually (write-time), so the widget's blanket
        # auto_scroll stays off; pause just toggles the follow flag.
        assert s._logview.auto_scroll is False
        s.action_toggle_pause()
        assert s._paused is True
        s.action_toggle_pause()
        assert s._paused is False
        assert s._min_level is None
        s.action_cycle_level()
        assert s._min_level == "INFO"
        s.action_cycle_level()
        assert s._min_level == "WARNING"
        s.action_cycle_level()
        assert s._min_level == "ERROR"
        s.action_cycle_level()
        assert s._min_level is None
        await _settle(app, pilot)


def test_logscreen_controls():
    asyncio.run(_run_logscreen_controls())


async def _run_always_tails_bottom():
    app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot,
            app,
            lambda: isinstance(app.screen, LogScreen)
            and app.screen._current_source is not None
            and len(app.screen.query("#sources ListItem")) == 2,
        )
        assert ok, "log screen / initial tail did not start"
        s = app.screen
        lv = s._logview
        # Writing past the viewport keeps the view pinned to the tail — every
        # viewer lands on the newest lines, no manual scroll needed.
        for i in range(200):
            s._write_line(f"line {i}")
        await pilot.pause()
        assert lv.is_vertical_scroll_end, "log should stick to the bottom"
        # A new live line keeps us at the bottom.
        s._write_line("newest live line")
        await pilot.pause()
        assert lv.is_vertical_scroll_end, "live tail should keep following the bottom"
        # Pause is the escape hatch: scroll up to read history and new lines
        # must not yank the user back down.
        s.action_toggle_pause()
        assert s._paused is True
        lv.scroll_home(animate=False)
        await pilot.pause()
        s._write_line("live while paused")
        await pilot.pause()
        assert not lv.is_vertical_scroll_end, "paused tail must not follow"
        # Un-pausing snaps back to the tail.
        s.action_toggle_pause()
        await pilot.pause()
        assert lv.is_vertical_scroll_end, "un-pause should jump back to the bottom"
        await _settle(app, pilot)


def test_log_always_tails_bottom():
    asyncio.run(_run_always_tails_bottom())


async def _run_replay_lands_at_bottom():
    c = _client()
    c.get_job.return_value = {  # terminal → cache-only replay, no live poll
        "status": "FAILED",
        "sub_jobs": [{"sub_job_id": "7:training:0", "job_type": "training"}],
    }
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot, app, lambda: isinstance(app.screen, LogScreen) and app.screen._current_source == "7:training:0"
        )
        assert ok, "log screen / cache replay did not start"
        s = app.screen
        lv = s._logview
        # The cached backlog is far taller than the viewport — exactly the
        # reopen-a-long-running-job case. The old code wrote the replay with
        # scroll_end=False and stranded the viewer at the oldest line; the fix
        # lands them on the newest. Wait for the replay, then assert the tail.
        ok = await _wait(pilot, app, lambda: any("cache line 299" in ln for ln in s._shown_lines))
        assert ok, "cache replay did not render"
        await pilot.pause()
        assert lv.is_vertical_scroll_end, "cache replay must land at the tail, not the top"
        await _settle(app, pilot)


def test_cache_replay_lands_at_bottom(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    from cortex_training.tui.log_cache import LogCache

    LogCache("7").append_entries("7:training:0", [{"_raw": f"cache line {i}"} for i in range(300)])
    asyncio.run(_run_replay_lands_at_bottom())


async def _run_source_switch_resets_pause():
    app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot,
            app,
            lambda: isinstance(app.screen, LogScreen)
            and app.screen._current_source is not None
            and len(app.screen.query("#sources ListItem")) == 2,
        )
        assert ok, "log screen / initial tail did not start"
        s = app.screen
        lv = s._logview
        # Pause on the first source (the user is reading history).
        s.action_toggle_pause()
        assert s._paused is True
        # Switching to another source is a fresh follow context: a stale pause
        # must clear so the new source lands on — and follows — its own tail
        # instead of being stranded at the top.
        other = s._source_by_item["src-1"]
        assert other != s._current_source
        s._start_tail(other)
        await pilot.pause()
        assert s._paused is False, "switching sources must clear a stale pause"
        assert s._current_source == other
        for i in range(200):
            s._write_line(f"line {i}")
        await pilot.pause()
        assert lv.is_vertical_scroll_end, "new source should follow its tail"
        await _settle(app, pilot)


def test_source_switch_resets_pause():
    asyncio.run(_run_source_switch_resets_pause())


async def _run_last_updated():
    c = _client()
    calls = {"n": 0}

    def tail(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"entries": [{"_raw": "live line"}], "next_cursor": "C1", "eof": False}
        return {"entries": [], "next_cursor": "C1", "eof": False}

    c.tail_logs.side_effect = tail
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        # Wait for the subtitle to pick up the freshness stamp from a live batch: a line
        # without a timestamp of its own is dated by when it arrived.
        ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen)
                         and "last line received" in (app.sub_title or ""))
        assert ok, "subtitle never showed a last-line stamp"
        assert app.screen._last_line_at is not None
        assert "read-only" not in app.sub_title  # dropped from the subtitle
        assert "RUNNING" in app.sub_title  # job status now on this line
        # The summary line is reserved for static config — no status word there.
        from textual.widgets import Static

        summary = str(app.screen.query_one("#summary", Static).render())
        assert "RUNNING" not in summary
        await _settle(app, pilot)


def test_subtitle_last_updated():
    asyncio.run(_run_last_updated())


async def _run_export():
    app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen) and app.screen._logview is not None)
        assert ok, "log screen did not open"
        s = app.screen
        s._current_source = "head:server"
        s.action_save_log()  # background worker writes the file
        ok = await _wait(
            pilot,
            app,
            lambda: bool(glob.glob(os.path.expanduser("~/cortex-training-7-*.log"))),
        )
        assert ok, "export file was not written"
        content = open(
            glob.glob(os.path.expanduser("~/cortex-training-7-*.log"))[0],
            encoding="utf-8",
        ).read()
        assert "alpha" in content and "beta" in content
        await _settle(app, pilot)


def test_log_export(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))  # so ~/ writes into tmp
    from cortex_training.tui.log_cache import LogCache

    LogCache("7").append_entries("head:server", [{"_raw": "alpha"}, {"_raw": "beta"}])
    asyncio.run(_run_export())


async def _run_wrap_and_select():
    from textual.widgets import Log

    app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        # Wait until the initial source is selected (its tail started + banner
        # written) so _start_tail won't reset _shown_lines under us.
        ok = await _wait(
            pilot,
            app,
            lambda: isinstance(app.screen, LogScreen)
            and app.screen._logview is not None
            and app.screen._current_source is not None
            and len(app.screen.query("#sources ListItem")) == 2,
        )
        assert ok, "log screen / initial tail did not start"
        s = app.screen
        # The log pane is a Log (not RichLog): it participates in Textual's
        # native text selection, and the screen can copy the selection.
        assert isinstance(s._logview, Log)
        assert s._logview.ALLOW_SELECT is True
        assert hasattr(s, "action_copy_text")  # ctrl+c → copy selection

        # Wrapping: a long line is split to the pane width on write. The pane
        # may prefix the line with its sub-job tag, which wraps with it.
        s._content_width = lambda: 10
        before = len(s._logview._lines)
        s._write_line("x" * 60)  # 60 / 10 = 6 visual lines, plus any tag
        assert len(s._logview._lines) - before == len(wrap_log_line(s._display("x" * 60), 10))
        assert s._shown_lines[-1] == "x" * 60  # buffer keeps the unwrapped, untagged line

        # Re-wrap on width change: widening collapses each buffered line to one. The
        # lines are wrapped off the UI thread, so the pane changes a moment later.
        s._content_width = lambda: 200
        s._rewrap()
        ok = await _wait(pilot, app, lambda: s._repaint_job is None and s._wrap_width == 200)
        assert ok, "the re-wrap did not land"
        assert len(s._logview._lines) == len(s._shown_lines)
        await _settle(app, pilot)


def test_log_wrap_and_selectable():
    asyncio.run(_run_wrap_and_select())


async def _pause_until(pilot, predicate, seconds=3.0):
    for _ in range(int(seconds / 0.05)):
        if predicate():
            return True
        await pilot.pause(0.05)
    return predicate()


async def _settled_pane(pilot, screen):
    """Let a re-wrap the screen started while it opened (its sources panel was sized) land."""
    await pilot.pause(2 * LogScreen._REWRAP_SETTLE_SECONDS)
    assert await _pause_until(pilot, lambda: screen._repaint_job is None)


async def _open_live_tail(app, pilot):
    ok = await _wait(
        pilot,
        app,
        lambda: isinstance(app.screen, LogScreen)
        and app.screen._current_source == "7:training:0"
        and len(app.screen.query("#sources ListItem")) == 2,
    )
    assert ok, "log screen / initial tail did not start"
    await _settled_pane(pilot, app.screen)
    return app.screen


def _live_page(screen, lines):
    """Hand ``screen`` one page of its live tail, as the tail's worker does."""
    screen._on_page(screen._tail_gen, "live", [{"_raw": line} for line in lines], list(lines))


def test_rewrap_and_refilter_run_off_the_ui_thread_and_keep_lines_that_come_meanwhile(monkeypatch):
    async def run():
        started, release = threading.Event(), threading.Event()
        original = LogScreen._wrap_pairs.__func__

        def slow(cls, *args):
            started.set()
            assert release.wait(timeout=5)
            return original(cls, *args)

        app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
        async with app.run_test() as pilot:
            s = await _open_live_tail(app, pilot)
            _live_page(s, [f"step-{i:02d} loss=0.5 " + "x" * 40 for i in range(30)])
            await _settled_pane(pilot, s)
            monkeypatch.setattr(LogScreen, "_wrap_pairs", classmethod(slow))
            rows = list(s._logview.lines)
            s._content_width = lambda: 30
            s._rewrap()  # returns at once: the lines are wrapped by a worker
            assert await _pause_until(pilot, started.is_set)
            assert s._repaint_job is not None and list(s._logview.lines) == rows
            _live_page(s, ["came during the repaint"])  # shown at once, and kept after it
            assert s._shown_lines[-1] == "came during the repaint" and s._logview.lines[:len(rows)] == rows
            release.set()
            assert await _pause_until(pilot, lambda: s._repaint_job is None and s._wrap_width == 30)
            assert s._shown_lines[-1] == "came during the repaint"
            assert s._shown_lines.count("came during the repaint") == 1
            assert sum(s._shown_rows) == len(s._logview.lines)
            assert all(len(row) <= 30 for row in s._logview.lines)
            # A filter repaints off the UI thread too, from the lines the tail received.
            started.clear()
            release.clear()
            s.query_one("#logfilter", Input).value = "step-07"
            assert await _pause_until(pilot, started.is_set)
            assert any("step-03" in line for line in s._shown_lines)  # the old lines, until it lands
            release.set()
            assert await _pause_until(pilot, lambda: s._repaint_job is None
                                      and not any("step-03" in line for line in s._shown_lines))
            assert [line for line in s._shown_lines if "step-" in line] == ["step-07 loss=0.5 " + "x" * 40]
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_drag_resize_wraps_the_log_once_for_the_width_it_ends_on(monkeypatch):
    async def run():
        widths = []
        original = LogScreen._wrap_pairs.__func__

        def counted(cls, pairs, query, level, width, *rest):
            widths.append(width)
            return original(cls, pairs, query, level, width, *rest)

        app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
        async with app.run_test() as pilot:
            s = await _open_live_tail(app, pilot)
            _live_page(s, [f"line {i} " + "y" * 60 for i in range(20)])
            await _settled_pane(pilot, s)
            monkeypatch.setattr(LogScreen, "_wrap_pairs", classmethod(counted))
            for width in (50, 45, 40, 35):
                s._content_width = lambda width=width: width
                LogScreen.on_resize(s, None)  # as Textual calls it, for each size of a drag
            assert await _pause_until(pilot, lambda: s._wrap_width == 35 and s._repaint_job is None)
            assert widths == [35]
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_log_line_that_reads_like_a_note_is_still_a_log_line():
    from cortex_training.tui.log_style import NOTE_KIND

    async def run():
        app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
        async with app.run_test() as pilot:
            s = await _open_live_tail(app, pilot)
            logs = s._shown_logs
            _live_page(s, ["— live —"])  # a log line with the text of the TUI's own note
            assert s._shown_logs == logs + 1
            assert s._logview.lines[-1].startswith("training:0 │ ")  # tagged as the sub-job's
            assert s._logview.row_kinds[-1] != NOTE_KIND
            s._write_line("— live —")  # the TUI's own note
            assert s._shown_logs == logs + 1
            assert s._logview.lines[-1] == "— live —" and s._logview.row_kinds[-1] == NOTE_KIND
            await _settle(app, pilot)

    asyncio.run(run())


def test_the_scrollback_count_follows_the_lines_as_they_come_and_go(monkeypatch):
    monkeypatch.setattr(LogScreen, "_MAX_LINES", 10)

    async def run():
        app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
        async with app.run_test() as pilot:
            s = await _open_live_tail(app, pilot)

            def logs():
                return sum(not isinstance(line, LogScreen._Note) for line in s._shown_lines)

            _live_page(s, [f"line {i}" for i in range(6)])
            s._write_line("— a note —")
            _live_page(s, [f"line {i}" for i in range(6, 12)])  # the oldest go past _MAX_LINES
            assert s._pane_truncated and s._shown_logs == logs() == 9
            assert s._scrollback_note().startswith("last 9 lines · ")
            s._content_width = lambda: 25
            s._rewrap()
            assert await _pause_until(pilot, lambda: s._repaint_job is None and s._wrap_width == 25)
            assert s._shown_logs == logs() == 9 and s._scrollback_note().startswith("last 9 lines · ")
            s._clear_pane()
            assert s._shown_logs == 0
            await _settle(app, pilot)

    asyncio.run(run())


def test_counting_a_saved_console_streams_it_and_counts_its_lines_as_before(tmp_path):
    block = 1 << 20  # what _count_lines reads at a time
    payloads = [
        b"",
        b"one",
        b"one\n",
        b"one\r\ntwo\rthree\n\nfive",
        "vt\x0bff\x0cfs\x1cgs\x1drs\x1enel\x85ls\N{LINE SEPARATOR}ps\N{PARAGRAPH SEPARATOR}end".encode(),
        b"bad \xff\xfe bytes\n\x85 alone",  # not UTF-8: replaced, as the pane decodes it
        b"x" * (block - 1) + b"\r\nlast",  # one \r\n across the first block's end
        b"x" * (block - 1) + b"\r",
        b"x" * (block - 1) + "\N{LINE SEPARATOR}after".encode(),  # a break of three bytes across it
    ]
    for n, payload in enumerate(payloads):
        path = tmp_path / f"console-{n}.log"
        path.write_bytes(payload)
        before = len(payload.decode("utf-8", errors="replace").splitlines())  # what the save counted
        assert LogScreen._count_lines(str(path)) == before, payload[-16:]


def test_save_writes_the_sub_job_shown_when_s_was_pressed(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from cortex_training.tui.log_cache import LogCache

    LogCache("7").append_entries("head:server", [{"_raw": "alpha"}, {"_raw": "beta"}])

    async def run():
        app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
        async with app.run_test() as pilot:
            s = await _open_live_tail(app, pilot)
            s._current_source = "head:server"
            s.action_save_log()
            s._current_source = "7:sampling:0"  # the view moves on before the worker runs
            saved = tmp_path / "cortex-training-7-head-server.log"
            assert await _pause_until(pilot, lambda: s._export_note is not None)
            assert saved.read_text(encoding="utf-8") == "alpha\nbeta\n"
            assert s._export_note == f"saved 2 lines to {saved}"
            await _settle(app, pilot)

    asyncio.run(run())


def _write_stdout(output_dir, sub_job_id, text):
    path = Path(output_dir) / sub_job_id / "stdout.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_gpu_metrics(output_dir, sub_job_id, *, gpu_count=2):
    path = Path(output_dir) / sub_job_id / "gpu.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    # Recent samples: the job is RUNNING, and a running job's samples are fresh.
    newest = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(seconds=30)
    for gpu in range(gpu_count):
        rows.append(
            {
                "record_version": 2,
                "sub_job_id": sub_job_id,
                "sample_ts": (newest + timedelta(seconds=gpu)).strftime("%Y-%m-%dT%H:%M:%S") + ".123456789Z",
                "worker_num": gpu,
                "local_gpu_index": 0,
                "global_gpu_index": gpu,
                "gpu_util_pct": float(gpu * 80),
                "fb_used_bytes": float((gpu + 1) * 1024**3),
                "fb_free_bytes": float(8 * 1024**3),
                "sample_interval_s": 5.0,
            }
        )
    payload = "".join(json.dumps(row) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    path.with_name(".gpu.jsonl.manifest.json").write_text(
        json.dumps({"version": 1, "names": ["chunk.gz"], "bytes": path.stat().st_size}),
        encoding="utf-8",
    )
    return path


def _terminal_job():
    return {
        "status": "FAILED",
        "sub_jobs": [
            {"sub_job_id": "7:training:0", "job_type": "training"},
            {"sub_job_id": "7:sampling:0", "job_type": "sampling"},
        ],
    }


async def _run_terminal_shows_saved_file():
    c = _client()
    c.get_job.return_value = _terminal_job()

    def download(job_id, output_dir, **kwargs):
        _write_stdout(output_dir, "7:training:0", "saved line\n")
        return []

    c.download_stdout_logs.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: "saved line" in app.screen._shown_lines)
        assert ok, "saved console did not replace the pane"
        c.tail_logs.assert_not_called()
        await _settle(app, pilot)


def test_terminal_open_shows_downloaded_file(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    asyncio.run(_run_terminal_shows_saved_file())


async def _run_keeps_cache_when_download_fails():
    c = _client()
    c.get_job.return_value = _terminal_job()
    c.download_stdout_logs.side_effect = RuntimeError("unavailable")
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot,
            app,
            lambda: "cached line" in app.screen._shown_lines and not app.screen._stdout_inflight,
        )
        assert ok, "cache line missing after a failed download"
        assert "No log output is available." not in app.screen._shown_lines
        downloads = c.download_stdout_logs.call_count
        app.screen._refilter()
        for _ in range(5):
            await pilot.pause()
        assert c.download_stdout_logs.call_count == downloads
        await _settle(app, pilot)


def test_failed_download_keeps_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    from cortex_training.tui.log_cache import LogCache

    LogCache("7").append_entries("7:training:0", [{"_raw": "cached line"}])
    asyncio.run(_run_keeps_cache_when_download_fails())


async def _run_empty_file_keeps_cache():
    c = _client()
    c.get_job.return_value = _terminal_job()

    def download(job_id, output_dir, **kwargs):
        _write_stdout(output_dir, "7:training:0", "")
        return []

    c.download_stdout_logs.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot,
            app,
            lambda: app.screen._stdout_ready and "cached line" in app.screen._shown_lines,
        )
        assert ok, "empty saved file did not leave the cache on screen"
        await _settle(app, pilot)


def test_empty_saved_file_keeps_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    from cortex_training.tui.log_cache import LogCache

    LogCache("7").append_entries("7:training:0", [{"_raw": "cached line"}])
    asyncio.run(_run_empty_file_keeps_cache())


async def _run_hidden_filter_shows_no_note(release, started):
    c = _client()
    c.get_job.return_value = _terminal_job()

    def download(job_id, output_dir, **kwargs):
        started.set()
        assert release.wait(timeout=5)
        _write_stdout(output_dir, "7:training:0", "")
        return []

    c.download_stdout_logs.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: "cached line" in app.screen._shown_lines and started.is_set())
        assert ok, "cache was not shown before the filter"
        app.screen._filter = "no-such-line"
        release.set()
        ok = await _wait(pilot, app, lambda: app.screen._stdout_ready)
        assert ok, "download did not finish"
        for _ in range(5):
            await pilot.pause()
        assert "cached line" not in app.screen._shown_lines
        assert "No log output is available." not in app.screen._shown_lines
        await _settle(app, pilot)


def test_filter_that_hides_cache_shows_no_empty_note(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    from cortex_training.tui.log_cache import LogCache

    LogCache("7").append_entries("7:training:0", [{"_raw": "cached line"}])
    release = threading.Event()
    try:
        asyncio.run(_run_hidden_filter_shows_no_note(release, threading.Event()))
    finally:
        release.set()


async def _run_running_tail_error_does_not_download():
    c = _client()
    c.tail_logs.side_effect = RuntimeError("zone down")
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: any(line.startswith("[error] tail") for line in app.screen._shown_lines))
        assert ok, "running-job tail error was not shown"
        c.download_stdout_logs.assert_not_called()
        await _settle(app, pilot)


def test_running_tail_error_does_not_download():
    asyncio.run(_run_running_tail_error_does_not_download())


async def _run_tail_error_after_finish_shows_file():
    c = _client()
    calls = {"get": 0, "tail": 0}

    def get_job(_job_id):
        calls["get"] += 1
        if calls["get"] == 1:
            return {
                "status": "RUNNING",
                "sub_jobs": [{"sub_job_id": "7:training:0", "job_type": "training"}],
            }
        return {"status": "FAILED", "reason": "cancelled", "sub_jobs": []}

    def tail_logs(*_args, **_kwargs):
        calls["tail"] += 1
        raise RuntimeError("zone gone")

    def download(job_id, output_dir, **kwargs):
        _write_stdout(output_dir, "7:training:0", "after finish\n")
        return []

    c.get_job.side_effect = get_job
    c.tail_logs.side_effect = tail_logs
    c.download_stdout_logs.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: "after finish" in app.screen._shown_lines)
        assert ok, "terminal tail error did not show the saved console"
        assert app.screen._job_status == "FAILED"
        tails = calls["tail"]
        app.screen._refilter()
        await _wait(pilot, app, lambda: "after finish" in app.screen._shown_lines)
        assert calls["tail"] == tails
        await _settle(app, pilot)


def test_tail_error_after_terminal_status_shows_file_once():
    asyncio.run(_run_tail_error_after_finish_shows_file())


def test_a_slow_job_read_never_brings_an_ended_job_back():
    started, release = threading.Event(), threading.Event()
    c = _client()
    reads = {"n": 0}

    def get_job(_job_id):
        reads["n"] += 1
        if reads["n"] == 1:  # the view's first read, still retrying a slow service
            started.set()
            assert release.wait(timeout=5)
            return {"status": "RUNNING", "sub_jobs": [
                {"sub_job_id": "7:training:0", "job_type": "training", "status": "RUNNING"}]}
        return {"status": "FAILED", "sub_jobs": []}

    c.get_job.side_effect = get_job

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test() as pilot:
            ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen) and started.is_set())
            assert ok
            s = app.screen
            # A read sent later (the status check) answers first: the job has failed.
            s._apply_status({"status": "FAILED", "reason": "sub_job_failed", "sub_jobs": [
                {"sub_job_id": "7:training:0", "job_type": "training", "status": "FAILED"}]}, s._next_job_read())
            assert s._job_status == "FAILED" and app.sub_title.startswith("FAILED")
            release.set()  # ... and then the first read answers, with the job still running
            ok = await _wait(pilot, app, lambda: s._current_source == "7:training:0")
            assert ok
            for _ in range(10):
                await pilot.pause()
            assert s._job_status == "FAILED" and app.sub_title.startswith("FAILED")
            assert s._sub_job_status == {"7:training:0": "FAILED"}
            assert not s._tail_live
            c.tail_logs.assert_not_called()  # never a tail of the dead job: its saved console instead
            # However it is numbered, an answer that has the ended job live is an older copy.
            assert not s._take_job({"status": "RUNNING", "sub_jobs": []}, s._next_job_read())
            assert s._job_status == "FAILED"
            await _settle(app, pilot)

    try:
        asyncio.run(run())
    finally:
        release.set()


def test_a_bug_in_showing_a_page_sends_no_job_read(tmp_path, monkeypatch):
    errors = tmp_path / "errors.log"  # where the fixture puts the error log
    c = _client()
    c.tail_logs.side_effect = lambda *a, **k: {"entries": [{"_raw": "a line"}], "next_cursor": "C1", "eof": False}
    on_page = LogScreen._on_page

    def broken(self, gen, kind, entries, lines):
        if kind == "live" and entries:
            raise TypeError("a bug in drawing the page")
        return on_page(self, gen, kind, entries, lines)

    monkeypatch.setattr(LogScreen, "_on_page", broken)

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test() as pilot:
            ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen)
                             and any(line.startswith("[error] tail") for line in app.screen._shown_lines))
            assert ok, "the tail did not stop"
            s = app.screen
            # Said as before, but the job isn't read again to tell what a failed request meant.
            assert s._shown_lines[-1].startswith(
                "[error] tail 7:training:0 failed: TypeError: a bug in drawing the page")
            assert s._tail_error.startswith("tail stopped: TypeError: a bug in drawing the page")
            assert not s._tail_live
            for _ in range(10):
                await pilot.pause()
            assert c.get_job.call_count == 1  # the view's own first read only
            logged = errors.read_text(encoding="utf-8")
            assert "a bug in the TUI while showing its log" in logged and "TypeError: a bug in drawing" in logged
            await _settle(app, pilot)

    asyncio.run(run())


async def _run_terminal_download_failure_keeps_cache():
    c = _client()
    calls = {"get": 0}

    def get_job(_job_id):
        calls["get"] += 1
        if calls["get"] == 1:
            return {
                "status": "RUNNING",
                "sub_jobs": [{"sub_job_id": "7:training:0", "job_type": "training"}],
            }
        return {"status": "FAILED", "sub_jobs": []}

    c.get_job.side_effect = get_job
    c.tail_logs.side_effect = RuntimeError("zone gone")
    c.download_stdout_logs.side_effect = RuntimeError("unavailable")
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot,
            app,
            lambda: c.download_stdout_logs.called and "cached line" in app.screen._shown_lines,
        )
        assert ok, "failed saved-console download did not leave the cache"
        for _ in range(5):
            await pilot.pause()
        assert not any(line.startswith("[error] tail") for line in app.screen._shown_lines)
        await _settle(app, pilot)


def test_terminal_download_failure_keeps_cache_without_tail_error(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    from cortex_training.tui.log_cache import LogCache

    LogCache("7").append_entries("7:training:0", [{"_raw": "cached line"}])
    asyncio.run(_run_terminal_download_failure_keeps_cache())


async def _run_healthy_tail_does_not_download():
    c = _client()
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._current_source == "7:training:0")
        assert ok, "live tail did not start"
        await _settle(app, pilot)
        c.download_stdout_logs.assert_not_called()


def test_healthy_running_tail_does_not_download():
    asyncio.run(_run_healthy_tail_does_not_download())


async def _run_filter_during_download_once(release, started, calls):
    c = _client()
    c.get_job.return_value = _terminal_job()

    def download(job_id, output_dir, **kwargs):
        calls.append(output_dir)
        started.set()
        assert release.wait(timeout=5)
        _write_stdout(output_dir, "7:training:0", "from file\n")
        return []

    c.download_stdout_logs.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, started.is_set)
        assert ok, "download did not start"
        app.screen._refilter()
        assert len(calls) == 1
        ok = await _wait(pilot, app, lambda: "cached line" in app.screen._shown_lines)
        assert ok, "cache replay did not paint during the download"
        release.set()
        ok = await _wait(pilot, app, lambda: "from file" in app.screen._shown_lines)
        assert ok, "finished download did not replace the current source"
        assert "cached line" not in app.screen._shown_lines
        for _ in range(5):
            await pilot.pause()
        assert "cached line" not in app.screen._shown_lines
        assert len(calls) == 1
        await _settle(app, pilot)


def test_filter_during_download_does_not_start_another(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    from cortex_training.tui.log_cache import LogCache

    LogCache("7").append_entries("7:training:0", [{"_raw": "cached line"}])
    release = threading.Event()
    started = threading.Event()
    try:
        asyncio.run(_run_filter_during_download_once(release, started, []))
    finally:
        release.set()


async def _run_switch_source_does_not_download_again():
    c = _client()
    c.get_job.return_value = _terminal_job()
    calls = []

    def download(job_id, output_dir, **kwargs):
        calls.append(1)
        _write_stdout(output_dir, "7:training:0", "training line\n")
        return []

    c.download_stdout_logs.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._stdout_ready)
        assert ok, "first download did not finish"
        app.screen._start_tail("7:sampling:0")
        await _wait(pilot, app, lambda: "sampling cache" in app.screen._shown_lines)
        assert len(calls) == 1
        await _settle(app, pilot)


def test_source_switch_after_download_does_not_download_again(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    from cortex_training.tui.log_cache import LogCache

    LogCache("7").append_entries("7:sampling:0", [{"_raw": "sampling cache"}])
    asyncio.run(_run_switch_source_does_not_download_again())


async def _run_missing_sibling_shows_note():
    c = _client()
    c.get_job.return_value = _terminal_job()

    def download(job_id, output_dir, **kwargs):
        _write_stdout(output_dir, "7:training:0", "training line\n")
        return []

    c.download_stdout_logs.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._stdout_ready)
        assert ok, "first download did not finish"
        app.screen._start_tail("7:sampling:0")
        ok = await _wait(pilot, app, lambda: "No log output is available." in app.screen._shown_lines)
        assert ok, "empty sibling did not show the finished-view note"
        assert "training line" not in app.screen._shown_lines
        assert c.download_stdout_logs.call_count == 1
        await _settle(app, pilot)


def test_missing_sibling_after_download_shows_empty_note():
    asyncio.run(_run_missing_sibling_shows_note())


async def _run_save_exports_file(tmp_path):
    c = _client()
    c.get_job.return_value = _terminal_job()

    payload = b"file body\r\nsecond\r"

    def download(job_id, output_dir, **kwargs):
        path = _write_stdout(output_dir, "7:training:0", "")
        path.write_bytes(payload)
        return []

    c.download_stdout_logs.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: "file body" in app.screen._shown_lines)
        assert ok, "saved console was not shown"
        app.screen._filter = "file"
        copied: list[str] = []
        app.copy_to_clipboard = copied.append
        app.screen.action_copy_text()  # nothing highlighted: the whole log
        ok = await _wait(pilot, app, lambda: bool(copied))
        assert ok, "copy did not export the saved console"
        # Copied without control characters (pasted elsewhere, they would act there); s keeps the bytes.
        assert copied == ["file body\nsecond"]
        home = tmp_path / "home"
        home.mkdir()
        import cortex_training.tui.widgets as app_module

        original = app_module.os.path.expanduser
        app_module.os.path.expanduser = lambda path: str(home / os.path.basename(path))
        try:
            app.screen.action_save_log()
            saved = home / "cortex-training-7-training-0.log"  # <job8>-<type>-<index>, the job id once
            ok = await _wait(pilot, app, saved.is_file)
            assert ok, "save did not write the console file"
            assert saved.read_bytes() == b"file body\r\nsecond\r"
        finally:
            app_module.os.path.expanduser = original

        # A whole-log copy reads only the bounded tail, including when one line
        # alone is larger than the clipboard limit.
        app.screen._stdout_path("7:training:0").write_bytes(b"x" * (app.screen._COPY_MAX_BYTES + 100))
        copied.clear()
        app.screen.action_copy_text()
        assert await _wait(pilot, app, lambda: bool(copied))
        assert copied[0].startswith("…")
        assert len(copied[0].encode("utf-8")) <= app.screen._COPY_MAX_BYTES
        await _settle(app, pilot)


def test_save_exports_saved_console_not_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "cache"))
    from cortex_training.tui.log_cache import LogCache

    LogCache("7").append_entries("7:training:0", [{"_raw": "cached only"}])
    asyncio.run(_run_save_exports_file(tmp_path))


async def _run_pause_holds_replacement(release, started):
    c = _client()
    c.get_job.return_value = _terminal_job()

    def download(job_id, output_dir, **kwargs):
        started.set()
        assert release.wait(timeout=5)
        _write_stdout(output_dir, "7:training:0", "held line\n")
        return []

    c.download_stdout_logs.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: "cached line" in app.screen._shown_lines and started.is_set())
        assert ok, "cache was not on screen before the file arrived"
        app.screen.action_toggle_pause()
        release.set()
        ok = await _wait(pilot, app, lambda: app.screen._held is not None)
        assert ok, "paused view did not hold the saved console"
        assert "held line" not in app.screen._shown_lines
        app.screen.action_toggle_pause()
        ok = await _wait(pilot, app, lambda: "held line" in app.screen._shown_lines)
        assert ok, "unpause did not show the saved console"
        app.screen.action_toggle_pause()
        app.screen._held = (["stale line"], False)
        app.screen._start_tail("7:training:0")
        assert app.screen._held is None
        assert "stale line" not in app.screen._shown_lines
        ok = await _wait(pilot, app, lambda: "held line" in app.screen._shown_lines)
        assert ok, "re-tail did not repaint the saved console"
        assert "stale line" not in app.screen._shown_lines
        await _settle(app, pilot)


def test_pause_holds_saved_console_until_unpause(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    from cortex_training.tui.log_cache import LogCache

    LogCache("7").append_entries("7:training:0", [{"_raw": "cached line"}])
    release = threading.Event()
    try:
        asyncio.run(_run_pause_holds_replacement(release, threading.Event()))
    finally:
        release.set()


async def _run_leave_during_download(release, started, dirs):
    c = _client()
    c.get_job.return_value = _terminal_job()

    def download(job_id, output_dir, **kwargs):
        dirs.append(output_dir)
        started.set()
        assert release.wait(timeout=5)
        _write_stdout(output_dir, "7:training:0", "too late\n")
        return []

    c.download_stdout_logs.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, started.is_set)
        assert ok, "download did not start"
        app.pop_screen()
        await pilot.pause()
        release.set()
        ok = await _wait(pilot, app, lambda: dirs and not Path(dirs[0]).exists())
        assert ok, "temp dir was not removed after leaving"
        assert not any("too late" in line for line in getattr(app.screen, "_shown_lines", []))
        await _settle(app, pilot)


def test_leave_during_download_removes_temp_dir():
    release = threading.Event()
    try:
        asyncio.run(_run_leave_during_download(release, threading.Event(), []))
    finally:
        release.set()


async def _run_failure_then_retry():
    c = _client()
    c.get_job.return_value = _terminal_job()
    calls = []

    def download(job_id, output_dir, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            exc = RuntimeError("stage unavailable")
            raise exc
        _write_stdout(output_dir, "7:training:0", "after retry\n")
        return []

    c.download_stdout_logs.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot,
            app,
            lambda: any("saved console couldn't be downloaded" in line for line in app.screen._shown_lines),
        )
        assert ok, "download failure was not shown in the pane"
        app.screen._refilter()
        for _ in range(5):
            await pilot.pause()
        assert len(calls) == 1, "a filter change retried the failed download"
        app.screen.action_refresh_sources()
        ok = await _wait(pilot, app, lambda: "after retry" in app.screen._shown_lines)
        assert ok, "r did not retry the saved-console download"
        assert len(calls) == 2
        await _settle(app, pilot)


def test_download_failure_shows_error_and_r_retries(monkeypatch, tmp_path):
    import cortex_training.tui.widgets as app_module

    monkeypatch.setattr(app_module, "_ERROR_LOG", str(tmp_path / "errors.log"))
    asyncio.run(_run_failure_then_retry())


async def _run_saved_console_read_off_ui_thread(seen):
    real = LogScreen._scan_saved

    def spy(self, path, query, *rest):
        seen.append((threading.current_thread() is threading.main_thread(), query))
        return real(self, path, query, *rest)

    LogScreen._scan_saved = spy
    try:
        c = _client()
        c.get_job.return_value = _terminal_job()

        def download(job_id, output_dir, **kwargs):
            _write_stdout(output_dir, "7:training:0", "step loss\r" * 20_000 + "step 2 done\nbeta\n")
            return []

        c.download_stdout_logs.side_effect = download
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test() as pilot:
            ok = await _wait(pilot, app, lambda: "beta" in app.screen._shown_lines)
            assert ok, "saved console was not shown"
            app.screen._filter = "done"
            app.screen._refilter()
            ok = await _wait(
                pilot,
                app,
                lambda: "step 2 done" in app.screen._shown_lines
                and not any("loss" in line or "beta" in line for line in app.screen._shown_lines),
            )
            assert ok, "filter was not applied to the saved console"
            await _settle(app, pilot)
    finally:
        LogScreen._scan_saved = real


def test_saved_console_is_read_off_the_ui_thread():
    seen = []
    asyncio.run(_run_saved_console_read_off_ui_thread(seen))
    assert seen and not any(on_main for on_main, _ in seen)
    assert "done" in [query for _, query in seen]


async def _run_progress_status_and_cancel(state):
    c = _client()
    c.get_job.return_value = _terminal_job()

    class _Downloads:
        # A real signature: unknown keyword arguments would fail the call.
        def download_stdout_logs(self, job_id, output_dir, *, resume=False, on_chunk=None, sub_job_ids=None):
            if sub_job_ids is not None:
                return []
            state["resume"] = resume
            try:
                for i in range(1, 10000):
                    on_chunk("7:training:0", i, 10000, i * 1024)
                    if i == 1:
                        state["started"].set()
                    time.sleep(0.005)
            except BaseException as exc:  # noqa: BLE001 - the TUI's cancel signal
                state["raised"] = type(exc)
                raise
            return []

    c.download_stdout_logs = _Downloads().download_stdout_logs
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: re.search(r"chunk [\d,]+ of 10,000", app.screen._download_note or ""))
        assert ok, "download progress did not reach the status line"
        app.pop_screen()
        for _ in range(5):
            await pilot.pause()
        await _settle(app, pilot)
    for _ in range(200):
        if "raised" in state:
            break
        time.sleep(0.01)


def test_progress_updates_status_and_leaving_cancels_download(monkeypatch, tmp_path):
    import cortex_training.tui.widgets as app_module

    monkeypatch.setattr(app_module, "_ERROR_LOG", str(tmp_path / "errors.log"))
    state = {"started": threading.Event()}
    asyncio.run(_run_progress_status_and_cancel(state))
    assert state["resume"] is True
    assert issubclass(state.get("raised"), DownloadCancelled), "download was not cancelled"
    assert not (tmp_path / "errors.log").exists(), "a cancel was logged as a failure"


async def _run_selected_sub_job_first(release, calls):
    c = _client()
    c.get_job.return_value = _terminal_job()

    class _Downloads:
        def download_stdout_logs(
            self, job_id, output_dir, *, resume=False, on_chunk=None, sub_job_ids=None, **_kwargs
        ):
            calls.append(None if sub_job_ids is None else list(sub_job_ids))
            if sub_job_ids is not None:
                for sid in sub_job_ids:
                    _write_stdout(output_dir, sid, f"{sid} saved\n")
                return []
            assert release.wait(timeout=5)
            _write_stdout(output_dir, "7:sampling:0", "7:sampling:0 saved\n")
            return []

    c.download_stdout_logs = _Downloads().download_stdout_logs
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: "7:training:0 saved" in app.screen._shown_lines)
        assert ok, "selected sub-job was not shown before the rest finished"
        ok = await _wait(pilot, app, lambda: len(calls) == 2)
        assert ok, "the rest of the job was not downloaded"
        assert calls == [["7:training:0"], None]
        assert app.screen._stdout_inflight
        release.set()
        ok = await _wait(pilot, app, lambda: app.screen._stdout_ready)
        assert ok, "whole-job download did not finish"
        app.screen._start_tail("7:sampling:0")
        ok = await _wait(pilot, app, lambda: "7:sampling:0 saved" in app.screen._shown_lines)
        assert ok, "sibling sub-job was not shown after the full download"
        await _settle(app, pilot)


def test_selected_sub_job_downloads_and_paints_first():
    release = threading.Event()
    calls = []
    try:
        asyncio.run(_run_selected_sub_job_first(release, calls))
    finally:
        release.set()


async def _run_reopen_while_download_blocks(state):
    c = _client()
    c.get_job.return_value = _terminal_job()

    class _Downloads:
        # The first download blocks (like a retry backoff that still polls the
        # callback) until the TUI cancels it; later ones finish at once.
        def download_stdout_logs(self, job_id, output_dir, *, resume=False, on_chunk=None, sub_job_ids=None):
            state["calls"] += 1
            if state["calls"] > 1:
                if sub_job_ids:  # the reopened view's sub-job on screen; the rest adds nothing
                    _write_stdout(output_dir, "7:training:0", "second view\n")
                return []
            state["started"].set()
            try:
                while True:
                    on_chunk("7:training:0", 0, 10, 0)
                    time.sleep(0.02)
            except BaseException as exc:  # noqa: BLE001 - the TUI's cancel signal
                state["raised"] = type(exc)
                raise

    c.download_stdout_logs = _Downloads().download_stdout_logs
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, state["started"].is_set)
        assert ok, "first download did not start"
        app.pop_screen()
        await pilot.pause()
        app.push_screen(LogScreen(c, "7", poll_interval=0.01))
        ok = await _wait(
            pilot,
            app,
            lambda: isinstance(app.screen, LogScreen) and "second view" in app.screen._shown_lines,
            tries=300,
        )
        assert ok, "reopened view stayed waiting on the cancelled download"
        await _settle(app, pilot)


def test_reopen_does_not_wait_forever_on_a_blocked_download(monkeypatch, tmp_path):
    import cortex_training.tui.widgets as app_module

    monkeypatch.setattr(app_module, "_ERROR_LOG", str(tmp_path / "errors.log"))
    state = {"calls": 0, "started": threading.Event()}
    asyncio.run(_run_reopen_while_download_blocks(state))
    assert issubclass(state["raised"], DownloadCancelled), "download was not cancelled"
    assert state["calls"] == 3  # the blocked one, then the reopened view's sub-job and the rest
    assert not (tmp_path / "errors.log").exists()


async def _run_quit_during_download(state):
    c = _client()
    c.get_job.return_value = _terminal_job()

    class _Downloads:
        def download_stdout_logs(self, job_id, output_dir, *, resume=False, on_chunk=None, sub_job_ids=None):
            state["started"].set()
            try:
                while True:
                    on_chunk("7:training:0", 0, 10, 0)
                    time.sleep(0.02)
            finally:
                state["returned"].set()

    c.download_stdout_logs = _Downloads().download_stdout_logs
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, state["started"].is_set)
        assert ok, "download did not start"
        await app.action_quit()
        await pilot.pause()


def test_quit_during_download_stops_it():
    state = {"started": threading.Event(), "returned": threading.Event()}
    asyncio.run(_run_quit_during_download(state))
    assert state["returned"].wait(timeout=5), "download kept running after quit"


async def _run_job_detail_tabs():
    c = _client()
    c.download_metrics.return_value = []
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(140, 40)) as pilot:
        ok = await _wait(
            pilot,
            app,
            lambda: isinstance(app.screen, LogScreen) and app.screen._current_source == "7:training:0",
        )
        assert ok
        screen = app.screen
        tabs = screen.query_one("#job-tabs", TabbedContent)
        # The Resources tab is composed (its panel and pane), with nothing in it until it is shown.
        assert screen.query_one("#resource-list") is not None and screen.query_one("#resource-pane") is not None
        assert screen._resource_entries == []

        await pilot.press("g")
        ok = await _wait(pilot, app, lambda: tabs.active == "resource" and c.download_metrics.call_count == 1)
        assert ok
        assert screen.query_one("#footer").display  # the footer stays on every tab
        await pilot.press("5")  # a digit picks no window (w does)
        await pilot.pause()
        assert tabs.active == "resource" and screen._resource_range != "all"

        # Esc on a tab other than Logs goes back, even with a log filter set.
        await pilot.press("l")
        ok = await _wait(pilot, app, lambda: tabs.active == "logs")
        assert ok
        screen.query_one("#logfilter", Input).value = "x"
        await pilot.press("g")
        ok = await _wait(pilot, app, lambda: tabs.active == "resource")
        assert ok
        await pilot.press("escape")
        ok = await _wait(pilot, app, lambda: app.screen is not screen)
        assert ok
        await _settle(app, pilot)


def test_job_detail_tabs_load_lazily():
    asyncio.run(_run_job_detail_tabs())


def test_job_list_refreshes_on_its_own(monkeypatch):
    monkeypatch.setattr(JobListScreen, "_REFRESH_SECONDS", 0.05)
    c = _client()

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test() as pilot:
            for _ in range(100):
                await pilot.pause(0.05)
                if c.list_jobs.call_count >= 3:
                    break
            assert c.list_jobs.call_count >= 3, "the job list never refreshed on its own"
            await _settle(app, pilot)

    asyncio.run(run())


def _notes(app) -> list:
    return [note.message for note in app._notifications]


async def _run_highlighting_copies_nothing_until_ctrl_c():
    from textual import events

    from cortex_training.tui.clipboard import selected_text

    app = CortexTrainingLogTUI(_client(), poll_interval=0.01)
    copied: list[str] = []
    app.copy_to_clipboard = copied.append

    async def mouse(cls, x, button=1):  # as a terminal sends it: the button held through a drag
        app.post_message(cls(None, x, 1, 0, 0, button, False, False, False, screen_x=x, screen_y=1))
        await pilot.pause()

    async with app.run_test() as pilot:
        assert await _wait(pilot, app, lambda: len(getattr(app.screen, "_job_by_item", {})) >= 2)
        for cls, x in ((events.MouseDown, 4), (events.MouseMove, 20), (events.MouseUp, 20)):
            await mouse(cls, x)
        highlighted = selected_text(app.screen)
        for cls in (events.MouseDown, events.MouseUp):
            await mouse(cls, 30, button=3)  # a right-click
        await pilot.pause(0.5)  # long enough for a copy made on mouse-up to land
        assert highlighted and copied == []
        await pilot.press("ctrl+c")
        assert await _wait(pilot, app, lambda: copied) and copied == [highlighted]
        await _settle(app, pilot)


def test_highlighting_copies_nothing_until_ctrl_c():
    asyncio.run(_run_highlighting_copies_nothing_until_ctrl_c())


# ─── text selection in tables ─────────────────────────────────────────────────


async def _demo_list(app, pilot):
    """The demo's job list, drawn: the screen and its table."""
    assert await _wait(pilot, app, lambda: isinstance(app.screen, JobListScreen)
                       and len(app.screen._job_by_item) >= 2, tries=400)
    await pilot.pause(0.1)
    return app.screen, app.screen.query_one("#jobs", DataTable)


def _pointer(app, pilot):
    """``mouse(kind, x, y, button=1)``: a mouse event as a terminal sends it, the button held
    through a drag and each move's delta from the last event (Textual scrolls a highlight's
    area only for a pointer that moves up or down)."""
    last = {}

    async def mouse(kind, x, y, button=1):
        was_x, was_y = last.get("at", (x, y))
        last["at"] = (x, y)
        app.post_message(kind(None, x, y, x - was_x, y - was_y, button, False, False, False, screen_x=x, screen_y=y))
        await pilot.pause()

    return mouse


async def _drag(mouse, start, end, via=()):
    from textual import events

    await mouse(events.MouseDown, *start)
    for x, y in (*via, end):
        await mouse(events.MouseMove, x, y)
    await mouse(events.MouseUp, *end)


async def _click(mouse, x, y, moved_to=None):
    from textual import events

    await mouse(events.MouseDown, x, y)
    if moved_to is not None:
        await mouse(events.MouseMove, *moved_to)
    await mouse(events.MouseUp, *(moved_to or (x, y)))


def _screen_lines(app) -> list:
    return [strip.text for strip in app.screen._compositor.render_strips()]


def _highlighted_columns(app, y, table) -> list:
    """The columns of screen line ``y`` drawn with the selection's background (these lines
    have a cell for each character)."""
    from textual.filter import monochrome_style

    selection_style = table._selection_style()
    backgrounds = {selection_style.bgcolor, monochrome_style(selection_style).bgcolor}
    columns, x = [], 0
    for segment in app.screen._compositor.render_strips()[y]:
        for _character in segment.text:
            if segment.style is not None and segment.style.bgcolor in backgrounds:
                columns.append(x)
            x += 1
    return columns


def _first_job_line(lines) -> int:
    return next(y for y, line in enumerate(lines) if line[4:6] in ("▸ ", "▾ "))


async def _copied(app, pilot, copied):
    await pilot.press("ctrl+c")
    assert await _wait(pilot, app, lambda: copied)
    return copied.pop()


async def _run_a_drag_in_a_job_list_cell_highlights_and_copies_just_its_text():
    from cortex_training.tui.clipboard import selected_text

    app = CortexTrainingLogTUI(DemoClient(), poll_interval=0.01)
    copied: list[str] = []
    app.copy_to_clipboard = copied.append
    async with app.run_test(size=(130, 40)) as pilot:
        screen, table = await _demo_list(app, pilot)
        mouse = _pointer(app, pilot)
        lines = _screen_lines(app)
        y = _first_job_line(lines)
        cursor = table.cursor_row
        # Inside the job id: just the characters dragged over, drawn highlighted, and no click.
        await _drag(mouse, (7, y), (14, y), via=[(10, y)])
        assert selected_text(screen) == lines[y][7:15]
        assert _highlighted_columns(app, y, table) == list(range(7, 15))
        assert table.cursor_row == cursor and app.screen is screen
        await pilot.pause(0.3)
        assert copied == []  # highlighting copies nothing by itself
        assert await _copied(app, pilot, copied) == lines[y][7:15]  # not the column names, not a whole row
        # Sideways to the end: the fixed Job ID column still selects its own text, a scrolled one its own.
        table.scroll_to(x=table.max_scroll_x, animate=False)
        await pilot.pause(0.1)
        lines = _screen_lines(app)
        await _drag(mouse, (10, y + 3), (20, y + 3), via=[(15, y + 3)])
        assert await _copied(app, pilot, copied) == lines[y + 3][10:21]
        assert _highlighted_columns(app, y + 3, table) == list(range(10, 21))
        x = lines[table.region.y].index("Created")  # the column names, over the column scrolled into view
        await _drag(mouse, (x, y + 5), (x + 10, y + 5), via=[(x + 5, y + 5)])
        assert await _copied(app, pilot, copied) == lines[y + 5][x:x + 11]
        assert _highlighted_columns(app, y + 5, table) == list(range(x, x + 11))
        await _settle(app, pilot)


def test_a_drag_in_a_job_list_cell_highlights_and_copies_just_its_text():
    asyncio.run(_run_a_drag_in_a_job_list_cell_highlights_and_copies_just_its_text())


async def _run_a_click_on_the_sort_s_name_or_arrow_changes_it():
    app = CortexTrainingLogTUI(DemoClient(), poll_interval=0.01)
    async with app.run_test(size=(130, 40)) as pilot:
        screen, _table = await _demo_list(app, pilot)
        mouse = _pointer(app, pilot)
        y = screen.query_one("#listinfo", Static).region.y
        sort, descending = screen._sort, screen._descending
        line = _screen_lines(app)[y]
        await _click(mouse, line.index(f"sort: {sort}") + len("sort: ") + 1, y)  # the name: the next sort, as s
        assert await _pause_until(pilot, lambda: screen._sort != sort)
        assert screen._descending == descending
        line = _screen_lines(app)[y]
        await _click(mouse, max(line.rfind("↓"), line.rfind("↑")), y)  # the arrow: the other way, as o
        assert await _pause_until(pilot, lambda: screen._descending != descending)
        await _settle(app, pilot)


def test_a_click_on_the_sort_s_name_or_arrow_changes_it():
    asyncio.run(_run_a_click_on_the_sort_s_name_or_arrow_changes_it())


async def _run_a_click_on_the_resources_header_s_sub_job_or_window_switches_it():
    client = DemoClient()
    job = next(j for j in client.list_jobs() if j["status"] == "RUNNING" and len(j["sub_jobs"]) >= 2)
    app = CortexTrainingLogTUI(client, job["job_id"], poll_interval=0.01)
    async with app.run_test(size=(150, 40)) as pilot:
        assert await _pause_until(pilot, lambda: isinstance(app.screen, LogScreen) and app.screen._current_source)
        screen = app.screen
        screen.query_one("#job-tabs", TabbedContent).active = "resource"
        assert await _pause_until(pilot, lambda: screen._resource_data.kinds.get(screen._resource_source())
                                  and not screen._resource_loading, seconds=10.0)
        mouse = _pointer(app, pilot)
        status = screen.query_one("#resource-status", Static).content_region

        def spot(part: int) -> int:
            """Where to click on the header's part: 0 the sub-job (``training #0 (1 of 2)``), 1 the window."""
            line = _screen_lines(app)[status.y][status.x:]
            return status.x + line.index(line.split("  ·  ")[part]) + 1

        source, window = screen._resource_source(), screen._resource_range
        await _click(mouse, spot(0), status.y)  # the sub-job: the next one, as space
        assert await _pause_until(pilot, lambda: screen._resource_source() != source)
        assert screen._resource_range == window
        await _click(mouse, spot(1), status.y)  # the window: the next one, as w
        assert await _pause_until(pilot, lambda: screen._resource_range != window)
        await _settle(app, pilot)


def test_a_click_on_the_resources_header_s_sub_job_or_window_switches_it():
    asyncio.run(_run_a_click_on_the_resources_header_s_sub_job_or_window_switches_it())


async def _run_a_drag_across_job_list_rows_copies_that_span():
    app = CortexTrainingLogTUI(DemoClient(), poll_interval=0.01)
    copied: list[str] = []
    app.copy_to_clipboard = copied.append
    async with app.run_test(size=(200, 40)) as pilot:  # every column in view
        screen, table = await _demo_list(app, pilot)
        mouse = _pointer(app, pilot)
        width = table.scrollable_content_region.width
        lines = [line[:width] for line in _screen_lines(app)]
        y = _first_job_line(lines)
        await _drag(mouse, (20, y), (60, y + 3), via=[(30, y + 1), (50, y + 2)])
        assert await _copied(app, pilot, copied) == "\n".join(
            [lines[y][20:].rstrip(), lines[y + 1].rstrip(), lines[y + 2].rstrip(), lines[y + 3][:61].rstrip()])
        # Drawn over the same characters (a line's trailing blank isn't copied, nor drawn).
        assert _highlighted_columns(app, y, table) == list(range(20, len(lines[y].rstrip())))
        assert _highlighted_columns(app, y + 1, table) == list(range(len(lines[y + 1].rstrip())))
        assert _highlighted_columns(app, y + 3, table) == list(range(61))
        assert _highlighted_columns(app, y - 1, table) == [] and _highlighted_columns(app, y + 4, table) == []
        # Held below the list, the drag stays in its rows: nothing else is highlighted.
        await _drag(mouse, (10, y + 4), (10, 38), via=[(10, 30), (10, 36)])
        assert set(screen.selections) == {table}
        # A drag that starts on the column names stays on them.
        names = table.region.y
        await _drag(mouse, (4, names), (40, names + 7), via=[(20, names + 3)])
        selection = screen.selections[table]
        assert (selection.start.y, selection.end.y) == (0, 0)
        assert await _copied(app, pilot, copied) == lines[names][4:41].rstrip()
        # A change to the rows (e: every job unfolded) drops a highlight in them.
        await _drag(mouse, (8, y), (16, y), via=[(12, y)])
        assert screen.selections
        await pilot.press("e")
        assert await _wait(pilot, app, lambda: not screen.selections)
        await _settle(app, pilot)


def test_a_drag_across_job_list_rows_copies_that_span():
    asyncio.run(_run_a_drag_across_job_list_rows_copies_that_span())


async def _run_job_list_clicks_still_click_with_text_selection():
    app = CortexTrainingLogTUI(DemoClient(), poll_interval=0.01)
    async with app.run_test(size=(130, 40)) as pilot:
        screen, table = await _demo_list(app, pilot)
        mouse = _pointer(app, pilot)
        lines = _screen_lines(app)
        y = _first_job_line(lines)
        await _drag(mouse, (8, y), (16, y), via=[(12, y)])
        assert screen.selections
        rows_top = table.region.y + 1  # the first row's line, under the column names
        # A click moves the cursor and drops the highlight; so does one whose hand moved a cell.
        await _click(mouse, 60, y + 5)
        assert table.cursor_row == y + 5 - rows_top and not screen.selections
        await _click(mouse, 60, y + 7, moved_to=(61, y + 7))
        assert table.cursor_row == y + 7 - rows_top and not screen.selections
        # A drag isn't a click: the cursor stays where it was.
        await _drag(mouse, (60, y + 9), (70, y + 11), via=[(65, y + 10)])
        assert table.cursor_row == y + 7 - rows_top and screen.selections
        # Nor is one let go straight below the table, where the pointer counts as on the
        # row it was pressed on: the highlight stays too.
        bottom = table.scrollable_content_region.bottom - 1
        await _drag(mouse, (60, bottom), (60, 38))
        assert table.cursor_row == y + 7 - rows_top and screen.selections and app.screen is screen
        table.scroll_to(y=0, animate=False)  # where Textual scrolled it while the drag was held below
        await pilot.pause()
        # A click on a job's ▾ folds it.
        rows = table.row_count
        assert lines[y][4] == "▾"
        await _click(mouse, 4, y)
        assert await _wait(pilot, app, lambda: table.row_count < rows)
        assert _screen_lines(app)[y][4] == "▸"
        # A click on the cursor's row opens the job's preview.
        await _click(mouse, 60, y)
        await _click(mouse, 60, y)
        assert await _wait(pilot, app, lambda: isinstance(app.screen, JobSummaryScreen))
        await _settle(app, pilot)


def test_job_list_clicks_still_click_with_text_selection():
    asyncio.run(_run_job_list_clicks_still_click_with_text_selection())


async def _run_the_preview_s_sub_job_table_selects_text():
    app = CortexTrainingLogTUI(DemoClient(), poll_interval=0.01)
    copied: list[str] = []
    app.copy_to_clipboard = copied.append
    async with app.run_test(size=(130, 40)) as pilot:
        await _demo_list(app, pilot)
        await pilot.press("enter")  # the first job's preview
        assert await _wait(pilot, app, lambda: isinstance(app.screen, JobSummaryScreen)
                           and app.screen.query_one("#subjobs", DataTable).row_count > 2, tries=400)
        await pilot.pause(0.2)
        table = app.screen.query_one("#subjobs", DataTable)
        mouse = _pointer(app, pilot)
        lines = _screen_lines(app)
        y = table.region.y + 2  # the first sub-job's row, under the column names and the "entire job" row
        x = lines[y].index("training")
        await _drag(mouse, (x, y), (x + 5, y), via=[(x + 3, y)])
        assert _highlighted_columns(app, y, table) == list(range(x, x + 6))
        assert await _copied(app, pilot, copied) == "traini"
        assert isinstance(app.screen, JobSummaryScreen)  # the drag opened nothing
        await _settle(app, pilot)


def test_the_preview_s_sub_job_table_selects_text():
    asyncio.run(_run_the_preview_s_sub_job_table_selects_text())


async def _run_a_job_list_row_s_text_is_its_cells_as_drawn():
    # A row of cells that fit their columns is read from its cells' text, not drawn
    # (copying thousands of rows): it must read as the table draws it, every column.
    for width in (130, 80):  # at 80, job ids are cut short ("…"): those rows are drawn
        app = CortexTrainingLogTUI(DemoClient(), poll_interval=0.01)
        async with app.run_test(size=(width, 40)) as pilot:
            _screen, table = await _demo_list(app, pilot)
            read = 0
            for row in table.ordered_rows:
                plain = table._plain_parts(row.key)
                if plain is not None:
                    read += 1
                    assert plain == table._drawn_parts(row.key, 0)
            assert read and (width < 130 or read > table.row_count // 2)
            await _settle(app, pilot)


def test_a_job_list_row_s_text_is_its_cells_as_drawn():
    asyncio.run(_run_a_job_list_row_s_text_is_its_cells_as_drawn())


def test_a_copy_never_waits_for_this_machine_s_clipboard_tool(monkeypatch):
    # A clipboard tool that hangs (xclip waits out its timeout when DISPLAY is gone)
    # froze the screen for as long: it runs on a worker now, and its note follows.
    import cortex_training.tui.app as app_module
    from cortex_training.tui.clipboard import copied_note

    release = threading.Event()
    tried = []

    def hung_tool(text):
        tried.append(text)
        release.wait(10)
        return True

    monkeypatch.setattr(app_module, "copy_local", hung_tool)

    async def run():
        app = CortexTrainingLogTUI(_client(), poll_interval=0.01)
        async with app.run_test() as pilot:
            started = time.monotonic()
            app.copy_out("some \x1b[31mtext", "9 characters")
            assert time.monotonic() - started < 1
            assert app.clipboard == "some text"  # OSC 52 went out at once, without the escape sequence
            assert await _pause_until(pilot, seconds=5.0, predicate=lambda: tried == ["some text"])
            await pilot.pause(0.1)
            assert _notes(app) == []  # nothing is said until the copy is done
            release.set()
            assert await _pause_until(pilot, seconds=5.0, predicate=lambda: _notes(app) == ["copied 9 characters"]), _notes(app)
            app.copy_to_clipboard("abc")  # Textual's own copies (ctrl+c in a text box) are said too
            assert await _pause_until(pilot, seconds=5.0, predicate=lambda: _notes(app) == ["copied 3 characters"]), _notes(app)
            app.copy_out("\x1b[0m", "the escape")  # nothing left to copy: no tool runs, and the note is as it was
            assert _notes(app) == [copied_note("the escape", False)]
            assert tried == ["some text", "abc"]
            await _settle(app, pilot)

    asyncio.run(run())


def test_copies_reach_the_clipboard_in_turn_and_the_last_one_stays(monkeypatch, tmp_path):
    import cortex_training.tui.app as app_module
    from cortex_training.tui.clipboard import copy_local

    clip = tmp_path / "clip.txt"
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CLIPBOARD_FILE", str(clip))  # never the real clipboard
    release = threading.Event()
    tried = []

    def first_one_hangs(text):
        tried.append(text)
        if len(tried) == 1:
            release.wait(10)
        return copy_local(text)

    monkeypatch.setattr(app_module, "copy_local", first_one_hangs)

    async def run():
        app = CortexTrainingLogTUI(_client(), poll_interval=0.01)
        async with app.run_test() as pilot:
            app.copy_out("one", "the first")
            assert await _pause_until(pilot, seconds=5.0, predicate=lambda: tried == ["one"])
            app.copy_out("two", "the second")
            app.copy_out("three", "the third")
            release.set()
            assert await _pause_until(pilot, seconds=5.0, predicate=lambda: _notes(app) == ["copied the third"]), _notes(app)
            assert tried == ["one", "three"]  # "two" was replaced before its turn came
            assert clip.read_text(encoding="utf-8") == "three"
            assert app.clipboard == "three"
            await _settle(app, pilot)

    asyncio.run(run())


def test_quit_does_not_wait_for_a_blocked_worker():
    release = threading.Event()
    c = _client()
    c.download_metrics.side_effect = lambda job_id, output_dir, **_options: release.wait(30) and []

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen) and app.screen._current_source)
            assert ok
            await pilot.press("g")  # Resources: the download blocks, like a request retrying a 5xx
            ok = await _wait(pilot, app, lambda: c.download_metrics.call_count == 1)
            assert ok

    started = time.monotonic()
    try:
        asyncio.run(run())  # asyncio joins the loop's default executor on the way out
        assert time.monotonic() - started < 15, "quitting waited for the blocked worker"
    finally:
        release.set()


def test_daemon_executor_bounds_blocked_workers_and_cancels_queued_work():
    from cortex_training.tui.app import _DaemonThreadExecutor

    release = threading.Event()
    both_started = threading.Event()
    started = []

    def blocked(value):
        started.append(value)
        if len(started) == 2:
            both_started.set()
        release.wait(5)
        return value

    executor = _DaemonThreadExecutor(max_workers=2)
    futures = [executor.submit(blocked, value) for value in range(10)]
    assert both_started.wait(2)
    executor.shutdown()
    assert len(started) == 2
    assert sum(future.cancelled() for future in futures) == 8
    release.set()
    assert [future.result(2) for future in futures[:2]] == [0, 1]
    with pytest.raises(RuntimeError, match="after shutdown"):
        executor.submit(lambda: None)


async def _run_tail_waits_for_logs_tab():
    c = _client()
    app = CortexTrainingLogTUI(c, poll_interval=0.01)
    async with app.run_test(size=(140, 40)) as pilot:
        ok = await _wait(pilot, app, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
        assert ok
        app.push_screen(LogScreen(c, "7", job={"job_id": "7", "status": "RUNNING"}, initial_tab="resource"))
        ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen) and app.screen._current_source)
        assert ok
        for _ in range(20):
            await pilot.pause(0.02)
        assert c.tail_logs.call_count == 0, "the tail polled while Logs was hidden"
        await pilot.press("l")
        ok = await _wait(pilot, app, lambda: c.tail_logs.call_count > 0)
        assert ok, "the tail did not start when Logs was shown"
        await pilot.press("g")
        await pilot.pause(0.5)  # let a request already in flight finish
        calls = c.tail_logs.call_count
        for _ in range(20):
            await pilot.pause(0.02)
        assert c.tail_logs.call_count == calls, "the tail kept polling after Logs was hidden"
        await _settle(app, pilot)


def test_tail_polls_only_while_logs_is_showing():
    asyncio.run(_run_tail_waits_for_logs_tab())


def test_failed_refresh_keeps_the_last_list_and_the_error(monkeypatch):
    c = _client()

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            ok = await _wait(pilot, app, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
            assert ok
            screen = app.screen
            c.list_jobs.side_effect = RuntimeError("503 Service Unavailable")
            await pilot.press("r")
            # The notice has a line of its own, above the list it keeps.
            notice = screen.query_one("#listnotice", Static)
            info = lambda: str(notice.render()) if notice.display else ""  # noqa: E731
            ok = await _wait(pilot, app, lambda: "refresh failed" in info())
            assert ok
            await pilot.press("s")  # any re-render keeps both the list and the error
            await pilot.pause()
            assert len(screen._job_by_item) == 2 and "503 Service Unavailable" in info()
            assert f"showing the list from {screen._last_refreshed[:5]}" in info()
            c.list_jobs.side_effect = None
            await pilot.press("r")
            ok = await _wait(pilot, app, lambda: "refresh failed" not in info() and screen._job_by_item)
            assert ok
            await _settle(app, pilot)

    asyncio.run(run())


def test_first_load_failure_stays_visible():
    c = _client()
    c.list_jobs.side_effect = RuntimeError("403 Forbidden [x]")

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            info = lambda: str(app.screen.query_one("#listinfo", Static).render())  # noqa: E731
            ok = await _wait(pilot, app, lambda: isinstance(app.screen, JobListScreen) and "403 Forbidden [x]" in info())
            assert ok and "[error]" in info()
            await pilot.press("s")
            await pilot.pause()
            assert "403 Forbidden [x]" in info()  # not replaced by "no jobs"
            await _settle(app, pilot)

    asyncio.run(run())


def _old_terminal_job(minutes_ago=30):
    now = datetime.now(timezone.utc)
    return {**_terminal_job(), "created_at": (now - timedelta(hours=3)).isoformat(),
            "state_changed_at": (now - timedelta(minutes=minutes_ago)).isoformat()}


async def _run_saved_console_waits_for_logs_and_is_kept(c):
    app = CortexTrainingLogTUI(c, poll_interval=0.01)
    async with app.run_test(size=(140, 40)) as pilot:
        ok = await _wait(pilot, app, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
        assert ok
        app.push_screen(LogScreen(c, "7", job={"job_id": "7", "status": "FAILED"}, initial_tab="resource"))
        ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen) and app.screen._current_source)
        assert ok
        for _ in range(10):
            await pilot.pause(0.03)
        assert c.download_stdout_logs.call_count == 0, "downloaded the whole console for another tab"
        await pilot.press("l")
        ok = await _wait(pilot, app, lambda: "saved line" in app.screen._shown_lines, tries=200)
        assert ok, "the saved console did not load once Logs was shown"
        await pilot.press("b")
        ok = await _wait(pilot, app, lambda: isinstance(app.screen, JobListScreen))
        assert ok
        # A job that ended long ago keeps its console: reopening replays it without a download.
        app.push_screen(LogScreen(c, "7", job={"job_id": "7", "status": "FAILED"}, initial_tab="logs"))
        ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen) and "saved line" in app.screen._shown_lines)
        assert ok
        assert c.download_stdout_logs.call_count == 1
        await _settle(app, pilot)


def test_saved_console_downloads_only_on_logs_and_is_kept_for_old_jobs():
    c = _client()
    c.get_job.return_value = _old_terminal_job(minutes_ago=30)

    def download(job_id, output_dir):
        _write_stdout(output_dir, "7:training:0", "saved line\n")
        return []

    c.download_stdout_logs.side_effect = download
    asyncio.run(_run_saved_console_waits_for_logs_and_is_kept(c))


@pytest.mark.parametrize("minutes_ago, reused_by", [
    (1, None),  # the job just finished: its last chunks may still be uploading
    (30, "2027-01-01T00:00:00Z"),  # the id was reused by a newer job, which downloads its own console
])
def test_a_console_not_kept_or_kept_for_another_job_is_downloaded_again(minutes_ago, reused_by):
    async def run():
        c = _client()
        c.get_job.return_value = _old_terminal_job(minutes_ago=minutes_ago)
        c.download_stdout_logs.side_effect = lambda job_id, output_dir: _write_stdout(
            output_dir, "7:training:0", "saved line\n") and []
        for again in (False, True):
            if again and reused_by:
                c.get_job.return_value = {**c.get_job.return_value, "created_at": reused_by}
            app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
            async with app.run_test() as pilot:
                ok = await _wait(pilot, app, lambda: "saved line" in app.screen._shown_lines)
                assert ok
                await _settle(app, pilot)
        assert c.download_stdout_logs.call_count == 2

    asyncio.run(run())


def test_refresh_on_logs_keeps_the_selected_source():
    async def run():
        c = _client()
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen) and app.screen._current_source)
            assert ok
            app.screen._start_tail("7:sampling:0")
            calls = c.get_job.call_count
            await pilot.press("r")
            ok = await _wait(pilot, app, lambda: c.get_job.call_count > calls and app.screen._sources_ready)
            assert ok
            for _ in range(5):
                await pilot.pause()
            assert app.screen._current_source == "7:sampling:0"
            await _settle(app, pilot)

    asyncio.run(run())


def test_markup_like_names_and_server_text_render_literally():
    job = {"job_id": "7", "status": "PENDING", "reason": "quota [/x",
           "reason_details": "needs [/b] GPUs", "sub_jobs": [{"sub_job_id": "7:training:0", "job_type": "training",
                                                            "model_name": "org/model [/v2"}]}
    c = _client()
    c.list_jobs.return_value = [job]
    c.get_job.return_value = job

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(160, 45)) as pilot:
            ok = await _wait(pilot, app, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
            assert ok
            app.screen.query_one("#jobfilter", Input).value = "[/b] [x]"  # a search that matches no job
            ok = await _wait(pilot, app, lambda: "“[/b] [x]”" in str(app.screen.query_one("#nomatch", Static).render()))
            assert ok
            app.push_screen(JobSummaryScreen(c, "7", job=job))
            ok = await _wait(pilot, app, lambda: isinstance(app.screen, JobSummaryScreen)
                             and "needs [/b] GPUs" in str(app.screen.query_one("#health", Static).render()))
            assert ok
            assert "org/model [/v2" in str(app.screen.query_one("#preview", Static).render())
            app.push_screen(LogScreen(c, "7", job=job, initial_tab="logs"))
            ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen)
                             and "org/model [/v2" in str(app.screen.query_one("#summary", Static).render()))
            assert ok
            assert app.return_code is None
            await _settle(app, pilot)

    asyncio.run(run())
    assert c.get_job.called


def test_quit_without_job_id_derives_it_from_sub_job_id_and_reports_crashes(monkeypatch):
    seen = {}

    class FakeApp:
        return_code = 1

        def __init__(self, client, job_id, *, sub_job_id=None, poll_interval=1.0):
            seen.update(job_id=job_id, sub_job_id=sub_job_id)

        def run(self):
            pass

    monkeypatch.setattr("cortex_training.tui.app.CortexTrainingLogTUI", FakeApp)
    monkeypatch.setattr("cortex_training._cli.build_client", lambda args, cls: MagicMock())
    rc = tui_main.run(["--sub-job-id", "job-x:sampling:0", "--base-url", "http://localhost:1", "--database", "DB"])
    assert seen == {"job_id": "job-x", "sub_job_id": "job-x:sampling:0"}
    assert rc == 1  # the app's return code, not a blanket 0
    with pytest.raises(SystemExit):
        tui_main.run(["--sub-job-id", "not-a-sub-job-id", "--base-url", "http://localhost:1", "--database", "DB"])


@pytest.mark.parametrize(
    "config, message, usage",
    [
        ({"base_url": "http://localhost:1", "database": "DB", "poll_interval": "2"},
         "config poll_interval must be a number", False),
        ({"base_url": "http://localhost:1", "database": "DB", "verify_ssl": "yes"},
         "config verify_ssl must be a boolean", False),
        ({"base_url": "http://localhost:1", "\x1b]52;c;eA==\x07": 1}, "unknown config key(s): \\x1b]52;c;eA==\\x07",
         False),
        ({"base_url": "http://localhost:1", "bad\nkey": 1}, "unknown config key(s): bad\\nkey", False),
        ("{not json", "invalid JSON in config", False),
        (None, "No such file or directory", False),
        # Never a traceback, this one keeps the usage above its error line, as it always has.
        ({"base_url": "localhost:1", "database": "DB"}, "base_url must start with http:// or https://", True),
    ],
)
def test_a_config_the_tui_cannot_use_is_an_error_not_a_traceback(tmp_path, monkeypatch, capsys, config, message,
                                                                 usage):
    class NeverStarts:
        def __init__(self, *args, **kwargs):
            raise AssertionError("the TUI started")

    for name in ("CORTEX_TRAINING_CONNECTION", "CORTEX_TRAINING_BASE_URL", "CORTEX_TRAINING_HOST", "SNOWFLAKE_HOST",
                 "CORTEX_TRAINING_PAT", "SNOWFLAKE_PAT"):
        monkeypatch.delenv(name, raising=False)  # a PAT would make the bare base_url a host
    monkeypatch.setattr("cortex_training.tui.app.CortexTrainingLogTUI", NeverStarts)
    path = tmp_path / "config.json"
    if config is not None:
        path.write_text(config if isinstance(config, str) else json.dumps(config), encoding="utf-8")
    with pytest.raises(SystemExit) as exit_:
        tui_main.run(["--config", str(path)])
    assert exit_.value.code == 2
    err = capsys.readouterr().err
    lines = err.splitlines()
    if usage:
        assert lines[0].startswith("usage: cortex-training tui"), err
    else:
        assert len(lines) == 1, err  # one line: no usage, no traceback
    assert lines[-1].startswith("cortex-training tui: error: ") and message in lines[-1], err
    assert "\x1b" not in err and "\x07" not in err  # a key from the file can't act on the terminal


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf"])
def test_the_log_poll_interval_must_be_positive_and_finite(value, capsys):
    with pytest.raises(SystemExit) as exit_:
        tui_main.run(["--debug", f"--poll-interval={value}"])
    assert exit_.value.code == 2
    assert "must be a finite number greater than zero" in capsys.readouterr().err


def test_the_poll_interval_is_the_flag_s_alone_and_the_sign_in_names_what_the_config_gives(tmp_path, monkeypatch):
    import cortex_training._cli as cli

    seen = []

    class FakeApp:
        return_code = 0
        sign_in = None

        def __init__(self, client, job_id, *, sub_job_id=None, poll_interval=1.0):
            self.poll_interval = poll_interval

        def run(self):
            seen.append((self.poll_interval, self.sign_in))

    reads = []
    load_config = cli._load_config
    monkeypatch.setattr(cli, "_load_config", lambda path: reads.append(path) or load_config(path))
    monkeypatch.delenv("CORTEX_TRAINING_CONNECTION", raising=False)
    monkeypatch.setattr("cortex_training.tui.app.CortexTrainingLogTUI", FakeApp)
    built = []  # the poll interval each client is built with
    monkeypatch.setattr("cortex_training._cli.build_client", lambda args, cls: built.append(args.poll_interval) or MagicMock())
    path = tmp_path / "config.json"
    # The config's poll_interval is the CLI's result polling: neither the TUI's log polls nor a reason not to start.
    path.write_text(json.dumps({"base_url": "http://localhost:1", "database": "DB", "poll_interval": 10}),
                    encoding="utf-8")
    sign_in = {"kind": "url", "config": str(path.resolve()), "base_url": "http://localhost:1",
               "base_url_in_config": True}
    assert tui_main.run(["--config", str(path)]) == 0
    assert seen == [(1.0, sign_in)]
    assert len(reads) == 2  # the CLI's resolution, then the TUI's one read for what that doesn't say
    assert tui_main.run(["--config", str(path), "--poll-interval", "30", "--base-url", "http://other:2"]) == 0
    assert seen[-1] == (30.0, {**sign_in, "base_url": "http://other:2", "base_url_in_config": False})
    assert built == [1.0, 1.0]  # never the log polls' 30 s: the client refuses more than 6


def test_resource_refresh_skips_the_reparse_when_nothing_new_arrived(monkeypatch):
    import cortex_training.tui.resource_view as tui_app

    loaders = []
    real_loader = tui_app.resource_loader
    monkeypatch.setattr(tui_app, "resource_loader", lambda root: loaders.append(real_loader(root)) or loaders[-1])
    c = _client()
    state = {"new": True}

    def download(job_id, output_dir, *, resume=False, **_kwargs):
        if state["new"]:
            _write_gpu_metrics(output_dir, "7:training:0", gpu_count=2)
        return []

    c.download_metrics.side_effect = download

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen) and app.screen._current_source)
            assert ok
            app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
            ok = await _wait(pilot, app, lambda: app.screen._resource_loaded and not app.screen._resource_loading)
            assert ok
            read = loaders[-1].bytes_read
            assert read > 0
            state["new"] = False
            app.screen.action_refresh_sources()  # r on Resources: nothing new was committed
            ok = await _wait(pilot, app, lambda: c.download_metrics.call_count == 2 and not app.screen._resource_loading)
            assert ok
            assert loaders[-1].bytes_read == read, "re-read the cache although nothing changed"
            await _settle(app, pilot)

    asyncio.run(run())


def _real_download_client(monkeypatch, *, subs, count=3):
    """A TUI client whose downloads go through the real SDK code over a fake
    Snowflake connection (job metadata and tails stay mocked)."""
    from tests.test_client import TestChunkedArtifactDownload

    real, connection, _paths, _sleeps, _opened = TestChunkedArtifactDownload()._setup(
        monkeypatch, subs=subs, count=count
    )
    c = _client()
    c.download_stdout_logs = real.download_stdout_logs
    return c, connection


async def _run_selected_sub_job_without_chunks(c):
    app = CortexTrainingLogTUI(c, "job-1", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen) and app.screen._stdout_ready, tries=300)
        assert ok, "saved-console download did not finish"
        screen = app.screen
        assert screen._current_source == "job-1:training:0"
        assert not screen._stdout_failed
        assert not any("couldn't be downloaded" in line for line in screen._shown_lines)
        ok = await _wait(pilot, app, lambda: "No log output is available." in screen._shown_lines)
        assert ok, "a sub-job without saved output did not show the empty note"
        screen._start_tail("job-1:sampling:0")
        ok = await _wait(pilot, app, lambda: "job-1:sampling:0 line 2" in screen._shown_lines)
        assert ok, "the other sub-job's console was not fetched"
        await _settle(app, pilot)


def test_selected_sub_job_without_chunks_falls_through_to_the_whole_job(monkeypatch):
    c, connection = _real_download_client(monkeypatch, subs=["job-1:sampling:0"])
    c.get_job.return_value = {
        "status": "FAILED",
        "sub_jobs": [
            {"sub_job_id": "job-1:sampling:0", "job_type": "sampling"},
            {"sub_job_id": "job-1:training:0", "job_type": "training"},
        ],
    }
    asyncio.run(_run_selected_sub_job_without_chunks(c))
    gets = [cmd for cmd in connection.commands if cmd.startswith("GET ")]
    assert len(gets) == 3


def test_job_list_loading_state_and_expand_all_toggle():
    release = threading.Event()
    c = _client()
    sub = [{"sub_job_id": "x:training:0", "job_type": "training"}]
    real = [*c.list_jobs.return_value[:1],
            {"job_id": "old", "status": "FAILED", "sub_jobs": sub}, {"job_id": "older", "status": "FAILED", "sub_jobs": sub}]
    c.list_jobs.side_effect = lambda **_bounds: release.wait(10) and real

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            ok = await _wait(pilot, app, lambda: isinstance(app.screen, JobListScreen))
            assert ok
            screen = app.screen
            nomatch = lambda: str(screen.query_one("#nomatch", Static).render())  # noqa: E731
            ok = await _wait(pilot, app, lambda: "Loading jobs" in nomatch())
            assert ok, "the list claimed there were no jobs while still loading"
            release.set()
            ok = await _wait(pilot, app, lambda: screen._job_by_item)
            assert ok
            old = next(job for job in screen._all_jobs if job["job_id"] == "old")
            screen._expanded.add(screen._job_key(old))  # one history job expanded by hand (space) ...
            screen.action_expand_all()  # ... then e expands everything rather than collapsing it
            assert all(screen._is_expanded(j) for j in screen._all_jobs if views_sub_jobs(j))
            await _settle(app, pilot)

    from cortex_training.tui.views import sub_jobs as views_sub_jobs

    try:
        asyncio.run(run())
    finally:
        release.set()


async def _run_recently_ended_hint(state_changed_at, expect_hint):
    from datetime import datetime as _dt
    from datetime import timezone as _tz

    c = _client()
    c.get_job.return_value = {
        **_terminal_job(),
        "state_changed_at": _dt.fromtimestamp(state_changed_at, tz=_tz.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    c.download_stdout_logs.side_effect = lambda job_id, output_dir, **kwargs: []
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen) and app.screen._state_changed_at is not None)
        assert ok
        await _wait(pilot, app, lambda: "FAILED" in (app.sub_title or ""))
        assert await _wait(pilot, app, lambda: app.screen._stdout_ready and not app.screen._repainting)
        hint = "so its saved logs may not all be uploaded yet; r checks again"
        assert any(hint in line for line in app.screen._shown_lines) is expect_hint, app.screen._shown_lines
        await _settle(app, pilot)


def test_a_just_ended_job_says_its_last_output_may_still_be_uploading():
    asyncio.run(_run_recently_ended_hint(time.time() - 30, True))


def test_a_long_ended_job_has_no_upload_hint():
    asyncio.run(_run_recently_ended_hint(time.time() - 3 * 86400, False))


def test_list_failure_is_reported_and_listings_never_stack(monkeypatch):
    monkeypatch.setattr(JobListScreen, "_REFRESH_SECONDS", 0.05)
    release = threading.Event()
    c = _client()

    def failing_list(**_bounds):
        release.wait(10)  # a call still retrying a 5xx
        raise RuntimeError("503 Service Unavailable")

    c.list_jobs.side_effect = failing_list

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            ok = await _wait(pilot, app, lambda: isinstance(app.screen, JobListScreen))
            assert ok
            for _ in range(10):  # several refresh ticks and an r while the first call hangs
                await pilot.pause(0.05)
            await pilot.press("r")
            assert c.list_jobs.call_count == 1
            release.set()
            info = lambda: str(app.screen.query_one("#listinfo", Static).render())  # noqa: E731
            ok = await _wait(pilot, app, lambda: "503 Service Unavailable" in info())
            assert ok, "the failed listing's error was dropped"
            await _settle(app, pilot)

    try:
        asyncio.run(run())
    finally:
        release.set()


def test_live_header_row_stays_visible_after_a_slow_first_load():
    c = _client()
    # Taller than the table, so it can scroll at all (the bug needs a scrollable table).
    real = [{"job_id": f"job-{i:03d}", "status": "RUNNING" if i < 4 else "CANCELLED",
             "sub_jobs": [{"sub_job_id": f"job-{i:03d}:training:0", "job_type": "training"}]} for i in range(60)]
    c.list_jobs.side_effect = lambda **_bounds: time.sleep(1.0) or real

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            ok = await _wait(pilot, app, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item, tries=300)
            assert ok
            await pilot.pause(1.0)
            table = app.screen.query_one("#jobs", DataTable)
            assert table.scroll_y == 0 and table.cursor_row == 1  # header row 0 in view, cursor on the first job
            await _settle(app, pilot)

    asyncio.run(run())


def test_resource_refresh_shows_its_refreshing_state(monkeypatch):
    release = threading.Event()
    c = _client()
    calls = []

    def download(job_id, output_dir, *, resume=False, **_kwargs):
        calls.append(1)
        if len(calls) > 1:
            release.wait(10)  # the refresh is slow
        _write_gpu_metrics(output_dir, "7:training:0", gpu_count=2)
        return []

    c.download_metrics.side_effect = download

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen) and app.screen._current_source)
            assert ok
            app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
            ok = await _wait(pilot, app, lambda: app.screen._resource_loaded and not app.screen._resource_loading)
            assert ok
            app.screen.action_refresh_sources()
            status = lambda: str(app.screen.query_one("#resource-status", Static).render())  # noqa: E731
            ok = await _wait(pilot, app, lambda: "cached · refreshing…" in status())
            assert ok, "a refresh of loaded data showed no refreshing state"
            release.set()
            await _settle(app, pilot)

    try:
        asyncio.run(run())
    finally:
        release.set()


def test_enter_right_after_typing_opens_a_job_from_the_new_matches():
    async def run():
        app = CortexTrainingLogTUI(_client(), poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            ok = await _wait(pilot, app, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
            assert ok
            await pilot.press("/", "o", "l", "d", "enter", "enter")  # faster than the search debounce
            ok = await _wait(pilot, app, lambda: isinstance(app.screen, JobSummaryScreen))
            assert ok and app.screen._job_id == "old"
            await _settle(app, pilot)

    asyncio.run(run())


async def _run_no_end_time_no_upload_hint():
    c = _client()
    c.get_job.return_value = _terminal_job()  # no state_changed_at
    c.download_stdout_logs.side_effect = lambda job_id, output_dir, **kwargs: []
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen) and "FAILED" in (app.sub_title or ""))
        assert ok
        for _ in range(5):
            await pilot.pause()
        assert app.screen._state_changed_at is None
        assert not any("may not all be uploaded" in line for line in app.screen._shown_lines)
        await _settle(app, pilot)


def test_an_ended_job_without_an_end_time_gets_no_upload_hint():
    asyncio.run(_run_no_end_time_no_upload_hint())


# The client methods the TUI calls.
TUI_CALLS = ("get_job", "list_jobs", "list_jobs_page", "for_job", "tail_logs", "download_metrics",
             "download_stdout_logs", "fetch_execution_logs")
# Parameters of the real client the demo leaves out: the TUI never passes them, or
# checks the client's signature for them first (sub_job_ids, on_chunk, stale_after_seconds).
_ENGINE = {"max_workers", "progress", "on_chunk", "sub_job_ids", "strict", "stale_after_seconds",
           "transfer_timeout_seconds"}
NOT_IN_DEMO = {
    "tail_logs": {"sub_job_type"},
    "download_metrics": _ENGINE,
    "download_stdout_logs": _ENGINE,
}


def _parameters(method, leave_out=frozenset()) -> list:
    """``(name, kind, default)`` of each parameter, in order: annotations may differ."""
    return [(p.name, p.kind, p.default) for p in inspect.signature(method).parameters.values()
            if p.name not in leave_out]


@pytest.mark.parametrize("name", TUI_CALLS)
def test_the_demo_client_takes_what_the_real_client_takes(name):
    real = getattr(CortexTrainingClient, name)
    left_out = NOT_IN_DEMO.get(name, set())
    assert left_out <= set(inspect.signature(real).parameters)  # each one left out is still the client's
    assert _parameters(getattr(DemoClient, name)) == _parameters(real, left_out)


ESC, BEL = "\x1b", "\x07"


# ─── the review fixes ─────────────────────────────────────────────────────────


def test_a_cancelled_worker_stops_waiting_out_a_long_poll_interval():
    # A tail sleeps between polls on its own thread. Cancelling a Textual worker can't
    # interrupt time.sleep, so without this the thread (and its executor slot) would be
    # held for the whole interval after the screen closed.
    from cortex_training.tui.widgets import _sleep_until_cancelled

    class Worker:
        is_cancelled = False

    worker = Worker()
    sleep = _sleep_until_cancelled(worker)
    threading.Timer(0.2, lambda: setattr(worker, "is_cancelled", True)).start()
    started = time.monotonic()
    sleep(300.0)
    assert time.monotonic() - started < 5.0  # not 300

    # A short wait is slept in one go, so the tail's own timing is unchanged.
    worker.is_cancelled = False
    started = time.monotonic()
    sleep(0.05)
    assert 0.03 <= time.monotonic() - started < 1.0


def test_an_older_tail_does_not_clear_the_newer_tail_s_in_flight_marker():
    # The status line says "no answer yet" while a tail's read is in flight. A cancelled
    # older tail answering later must not clear the marker the newer read set.
    from cortex_training.tui.widgets import _TimedTail

    class Screen:
        _tail_request = None

    screen = Screen()
    newer = _TimedTail(MagicMock(), screen, 2)

    class Client:
        def tail_logs(self, *_args, **_kwargs):
            # While the older read is in flight, a newer tail starts and claims the marker.
            newer._screen._tail_request = (newer._gen, time.monotonic())
            return {"entries": [], "next_cursor": "", "eof": True}

    older = _TimedTail(Client(), screen, 1)
    older.tail_logs("7", sub_job_id="7:training:0")
    assert screen._tail_request is not None, "the older tail cleared the newer request"
    assert screen._tail_request[0] == 2

    # Its own marker it does clear.
    own = _TimedTail(Client(), screen, 2)
    own.tail_logs("7", sub_job_id="7:training:0")
    assert screen._tail_request is None


def test_one_job_has_one_cache_wherever_it_was_opened_from():
    # The configured schema's list gives the client its configured identifiers; the
    # account-wide list's child client holds the names as stored. Both are the same
    # schema, so the job must not download its logs and telemetry twice.
    from cortex_training.tui.widgets import _cache_job_id

    class Client:
        base_url = "https://ACCOUNT.snowflakecomputing.com"
        endpoint = "cortex-training"

        def __init__(self, database, schema):
            self.database, self.schema = database, schema

    configured, resolved = Client("team_db", "ml"), Client("TEAM_DB", "ML")
    assert _cache_job_id(configured, "J") == _cache_job_id(resolved, "J")
    # A quoted lower-case name is its own identifier, and another schema is another cache.
    assert _cache_job_id(Client("DB", "ONE"), "J") != _cache_job_id(Client("DB", "TWO"), "J")
    assert _cache_job_id(Client("ONE", "S"), "J") != _cache_job_id(Client("TWO", "S"), "J")
    # Nothing to key on (the demo, a test double): the job id stands alone.
    assert _cache_job_id(object(), "J") == "J"


def test_a_location_is_routed_as_stored_and_drawn_made_printable():
    # printable_data cleans everything a job record carries, but these two names are SQL
    # identifiers that route the job's calls: cleaned, they name a different schema and
    # every call for the job answers 404.
    from cortex_training.tui.untrusted import printable_client

    stored = "ml​x"  # a quoted schema whose name holds a zero-width space

    class Service:
        database = schema = "X"

        def list_jobs_page(self, *_args, **_kwargs):
            return {"jobs": [{"job_id": "J", "database_name": "TEAM_DB", "schema_name": stored,
                              "reason": f"boom{ESC}]52;c;cGF5bG9hZA=={BEL}"}]}

        def for_job(self, row):
            return row

    row = printable_client(Service()).list_jobs_page()["jobs"][0]
    assert row["schema_name"] == stored, "the stored identifier was changed"
    assert row["database_name"] == "TEAM_DB"
    assert views.job_key(row) == ("TEAM_DB", stored, "J")  # routed on what the service stored
    assert ESC not in row["reason"] and "cGF5bG9hZA" not in row["reason"]  # other text still cleaned
    assert "​" not in views.location_label(row)  # what is drawn is printable


@pytest.mark.parametrize("kind,command", [("stdout", "cortex-training download-log 7 --log-type stdout"),
                                          ("metrics", "cortex-training download-metrics 7")])
def test_skipped_chunk_hint_names_the_cli_command_for_the_tui_cache(kind, command, tmp_path):
    # The client's own hint says to download again without resume, but the TUI downloads
    # into its cache, not a directory the user chose: the hint names that directory.
    import shlex

    from cortex_training.tui.widgets import _result_warnings

    cache = tmp_path / "cache dir" / kind
    rows = [{"sub_job_id": "7:training:0", "skipped_chunks": ["chunk"],
             "warnings": ["1 damaged chunk(s) skipped by an earlier download are still missing from the "
                          "saved file; download again without resume to retry them"]}]
    (message,) = _result_warnings(rows, kind=kind, job_id="7", output_dir=cache)["7:training:0"]
    assert "download again without resume" not in message
    retry = message.split("`")[1]
    assert retry == f"{command} --sub-job 7:training:0 --output-dir {shlex.quote(str(cache))}"
    # The quoted path survives a shell round trip, so the retry repairs the cached file.
    assert shlex.split(retry)[-1] == str(cache)


async def _run_stdout_warnings(calls):
    c = _client()
    c.get_job.return_value = _terminal_job()

    class _Downloads:
        def download_stdout_logs(self, job_id, output_dir, *, resume=False, on_chunk=None, sub_job_ids=None,
                                 stale_after_seconds=600):
            calls.append(stale_after_seconds)
            _write_stdout(output_dir, "7:training:0", "saved\n")
            return [{"sub_job_id": "7:training:0", "warnings": ["1 damaged chunk was skipped"]},
                    {"sub_job_id": "7:sampling:0", "warnings": ["sampling warning"]}]

    c.download_stdout_logs = _Downloads().download_stdout_logs
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: "warning: 1 damaged chunk was skipped" in (app.sub_title or ""))
        assert ok, "the download warning for the selected sub-job was not shown"
        assert "sampling warning" not in (app.sub_title or "")  # another sub-job's stays there
        app.screen._start_tail("7:sampling:0")
        ok = await _wait(pilot, app, lambda: "warning: sampling warning" in (app.sub_title or ""))
        assert ok, "the warnings did not follow the selected sub-job"
        await _settle(app, pilot)


def test_stdout_download_warnings_show_for_the_selected_sub_job():
    calls = []
    asyncio.run(_run_stdout_warnings(calls))
    # The TUI says itself when a job's output stopped, so the client's stall check is skipped.
    assert calls and all(value is None for value in calls)


async def _run_metrics_warning_shows_in_the_resources_header(tmp_path):
    c = _client()

    def download(job_id, output_dir, **_kwargs):
        _write_gpu_metrics(output_dir, "7:training:0", gpu_count=1)
        return [{"sub_job_id": "7:training:0", "warnings": ["2 chunk(s) could not be downloaded"]}]

    c.download_metrics.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(170, 40)) as pilot:
        ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen) and app.screen._current_source)
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        ok = await _wait(pilot, app, lambda: app.screen._metrics_warnings, tries=400)
        assert ok, "the metrics download's warning was not kept"
        header = str(app.screen.query_one("#resource-status", Static).render())
        assert "warning: 2 chunk(s) could not be downloaded" in header
        await _settle(app, pilot)


def test_metrics_download_warning_shows_in_the_resources_header(tmp_path):
    asyncio.run(_run_metrics_warning_shows_in_the_resources_header(tmp_path))


async def _run_hostile_service_text_never_reaches_the_terminal():
    # Everything the service sends is drawn: a job's reason, its model, a location, a log
    # line. None of it may carry an escape sequence (an OSC 52 would write the clipboard,
    # a CSI would clear the screen) or be read as markup.
    hostile = f"{ESC}]52;c;cGF5bG9hZA=={BEL}{ESC}[2Jboom‮[bold]x[/bold]"
    c = MagicMock()
    c.list_jobs_page.side_effect = AccountJobListUnavailable("not deployed", reason="not_deployed")
    c.list_jobs.return_value = [{"job_id": f"7{hostile}", "status": "RUNNING", "reason": hostile,
                                 "submitted_by": hostile,
                                 "sub_jobs": [{"job_type": "training", "model_name": hostile}]}]
    c.get_job.return_value = {"status": "RUNNING", "reason_details": hostile,
                              "sub_jobs": [{"sub_job_id": "7:training:0", "job_type": "training",
                                            "model_name": hostile}]}
    c.tail_logs.return_value = {"entries": [{"_raw": hostile}], "next_cursor": "", "eof": True}
    app = CortexTrainingLogTUI(printable_client(c), poll_interval=0.01)
    async with app.run_test(size=(140, 40)) as pilot:
        assert await _wait(pilot, app, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
        for stage in ("the job list", "a job's details"):
            if stage == "a job's details":
                await pilot.press("down")
                await pilot.press("enter")
                assert await _wait(pilot, app, lambda: not isinstance(app.screen, JobListScreen))
                await pilot.pause(0.3)
            drawn = "\n".join(strip.text for strip in app.screen._compositor.render_strips())
            drawn += str(app.sub_title or "") + str(app.title or "")
            assert ESC not in drawn, f"an escape sequence was drawn on {stage}"
            assert BEL not in drawn and "cGF5bG9hZA" not in drawn, f"an OSC 52 payload reached {stage}"
            assert "‮" not in drawn, f"a direction override was drawn on {stage}"
        await _settle(app, pilot)


def test_hostile_service_text_never_reaches_the_terminal():
    asyncio.run(_run_hostile_service_text_never_reaches_the_terminal())


async def _run_the_keys_do_what_their_context_means():
    app = CortexTrainingLogTUI(DemoClient(), poll_interval=0.01)
    async with app.run_test(size=(140, 40)) as pilot:
        screen, _table = await _demo_list(app, pilot)
        # a has two bindings on the job list; check_action enables the one for the other list.
        assert screen.check_action("show_list", ("account",)) is True
        assert screen.check_action("show_list", ("schema",)) is False
        before = len(screen._job_by_item)
        await pilot.press("a")
        assert await _pause_until(pilot, lambda: screen._lister.account_wide, seconds=10.0), "a did not widen the list"
        assert screen.check_action("show_list", ("account",)) is False  # now it offers the way back
        await pilot.press("a")
        assert await _pause_until(pilot, lambda: not screen._lister.account_wide, seconds=10.0)
        assert await _pause_until(pilot, lambda: len(screen._job_by_item) == before, seconds=10.0)
        # l belongs to the details screen: on the list it is not a binding at all.
        assert screen.check_action("cycle_level", ()) is True  # not one of the list's actions
        await pilot.press("enter")
        assert await _pause_until(pilot, lambda: not isinstance(app.screen, JobListScreen), seconds=10.0)
        details = app.screen
        if hasattr(details, "query_one"):
            await pilot.pause(0.3)
        # On the details screen l cycles the minimum level, but only while Logs is showing.
        if hasattr(details, "_min_level"):
            details._active_tab = "resource"
            assert details.check_action("cycle_level", ()) is False, "l acts on the Resources tab"
            details._active_tab = "logs"
            assert details.check_action("cycle_level", ()) is True
            level = details._min_level
            await pilot.press("l")
            assert await _pause_until(pilot, lambda: details._min_level != level, seconds=10.0), "l did not cycle"
        await _settle(app, pilot)


def test_the_keys_do_what_their_context_means():
    asyncio.run(_run_the_keys_do_what_their_context_means())


async def _run_the_group_rows_count_the_jobs_under_them():
    app = CortexTrainingLogTUI(DemoClient(), poll_interval=0.01)
    async with app.run_test(size=(140, 44)) as pilot:
        screen, table = await _demo_list(app, pilot)
        await pilot.pause(0.3)
        labels = [str(column.label).strip() for column in table.ordered_columns]
        marker, at = labels.index(""), labels.index("Job ID")
        rows = [(table.get_row_at(i)[marker].plain, table.get_row_at(i)[at].plain)
                for i in range(table.row_count)]
        # Each group row says how many jobs follow it. A job's row carries a status dot;
        # its sub-job rows do not, so they must not be counted.
        counts, group, seen = {}, None, 0
        for dot, text in rows:
            if text.startswith(("LIVE (", "HISTORY (")):
                if group:
                    counts[group] = seen
                group, seen = text, 0
            elif dot.strip():
                seen += 1
        counts[group] = seen
        assert len(counts) == 2, counts
        for header, under in counts.items():
            said = int(header.split("(")[1].rstrip(")"))
            assert said == under, f"{header} has {under} job rows under it"
        await _settle(app, pilot)


def test_the_group_rows_count_the_jobs_under_them():
    asyncio.run(_run_the_group_rows_count_the_jobs_under_them())
